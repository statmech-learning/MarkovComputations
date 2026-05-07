"""First-order Markov ICL model with explicit reaction topology."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .base_icl_model import BaseICLModel
try:
    from topology_metrics import normalize_edges
except ImportError:  # pragma: no cover - used when importing as ICL.models
    from ..topology_metrics import normalize_edges


class TopologyMatrixTreeMarkovICL(BaseICLModel):
    """First-order CRN ICL model on an arbitrary strongly connected digraph.

    Edge tuples use the physical convention ``(source, target)``. Internally,
    the Markov generator follows the existing project convention:
    ``W[target, source]`` is the transition rate from ``source`` to ``target``
    and columns sum to zero.
    """

    def __init__(
        self,
        n_nodes: int = 6,
        z_dim: int = 4,
        L: int = 128,
        N: int = 4,
        edges: Optional[Iterable[Sequence[int]]] = None,
        input_mask: Optional[Sequence[Sequence[float]]] = None,
        use_label_mod: bool = False,
        learn_base_rates: bool = True,
        transform_func: str = "exp",
        init_base: Optional[float] = None,
        base_rate_floor: float = 1e-8,
        print_creation: bool = True,
    ):
        super().__init__(n_nodes=n_nodes, z_dim=z_dim, L=L, N=N)
        self.n_nodes = n_nodes
        self.use_label_mod = use_label_mod
        self.learn_base_rates = learn_base_rates
        self.transform_func = transform_func
        self.base_rate_floor = base_rate_floor

        if edges is None:
            edges = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
        edge_tuple = normalize_edges(n_nodes, edges)
        if not edge_tuple:
            raise ValueError("At least one physical edge is required")
        self.edges = edge_tuple
        self.n_edges = len(edge_tuple)

        sources = torch.tensor([edge[0] for edge in edge_tuple], dtype=torch.long)
        targets = torch.tensor([edge[1] for edge in edge_tuple], dtype=torch.long)
        self.register_buffer("edge_sources", sources)
        self.register_buffer("edge_targets", targets)
        self.register_buffer("edge_rate_multiplier", torch.ones(self.n_edges))

        z_full_dim = (N + 1) * z_dim
        init_scale_K = 0.05 / np.sqrt(max(1, n_nodes))
        init_scale_B = 0.1 / np.sqrt(max(1, N))
        if init_base is None:
            init_base = -2.0 - 0.5 * np.log(max(1, n_nodes))

        self.K_params = nn.Parameter(torch.randn(self.n_edges, z_full_dim) * init_scale_K)
        self.base_log_rates = nn.Parameter(torch.randn(self.n_edges) * 0.1 + init_base)
        self.B = nn.Parameter(torch.randn(n_nodes, N) * init_scale_B)

        if input_mask is None:
            mask = torch.ones(self.n_edges, z_full_dim)
        else:
            mask = torch.as_tensor(input_mask, dtype=torch.float32)
            expected = (self.n_edges, z_full_dim)
            if tuple(mask.shape) != expected:
                raise ValueError(f"input_mask must have shape {expected}, got {tuple(mask.shape)}")
        self.register_buffer("input_mask", mask)

        if self.use_label_mod:
            self.label_modulation = nn.Parameter(
                torch.randn(self.n_edges, N) * init_scale_K * 0.5
            )
        else:
            self.label_modulation = None

        if not learn_base_rates:
            self.base_log_rates.requires_grad = False

        if print_creation:
            print("  Initialized topology-aware first-order Markov ICL model")
            print(f"  Nodes: {n_nodes}, physical edges: {self.n_edges}")
            print(f"  Classes: {L}, context: {N}, z_dim: {z_dim}")
            print(f"  Transform: {transform_func}, base rates learnable: {learn_base_rates}")
            print(f"  Input-coupled parameters: {int(mask.sum().item())}/{mask.numel()}")
            print(f"  Parameters: {self.get_num_parameters():,}")

    def edge_log_rates_from_flat(self, z_batch: torch.Tensor, labels_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        K_masked = self.K_params * self.input_mask
        log_rates = self.base_log_rates.unsqueeze(0) + torch.matmul(z_batch, K_masked.T)
        if self.use_label_mod and labels_batch is not None:
            log_rates = log_rates + torch.matmul(labels_batch.float(), self.label_modulation.T)
        return torch.clamp(log_rates, min=np.log(self.base_rate_floor), max=np.log(1e6))

    def edge_rates_from_flat(self, z_batch: torch.Tensor, labels_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        log_rates = self.edge_log_rates_from_flat(z_batch, labels_batch)
        if self.transform_func == "exp":
            rates = torch.exp(log_rates)
        elif self.transform_func == "softplus":
            rates = torch.nn.functional.softplus(log_rates) + self.base_rate_floor
        elif self.transform_func == "relu":
            rates = torch.relu(log_rates) + self.base_rate_floor
        else:
            raise ValueError(f"Invalid transform function: {self.transform_func}")
        return rates * self.edge_rate_multiplier.unsqueeze(0)

    def compute_rate_matrix_W(self, z_batch: torch.Tensor, labels_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = z_batch.shape[0]
        device = z_batch.device
        rates = self.edge_rates_from_flat(z_batch, labels_batch)

        W_batch = torch.zeros(batch_size, self.n_nodes, self.n_nodes, device=device, dtype=z_batch.dtype)
        targets = self.edge_targets.to(device)
        sources = self.edge_sources.to(device)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.n_edges)
        W_batch[batch_idx, targets.unsqueeze(0), sources.unsqueeze(0)] = rates

        col_sums = W_batch.sum(dim=1)
        W_batch = W_batch - torch.diag_embed(col_sums)
        return W_batch

    def matrix_tree_steady_state(self, W_batch: torch.Tensor) -> torch.Tensor:
        batch_size, n = W_batch.shape[0], self.n_nodes
        device = W_batch.device
        p_batch = torch.zeros(batch_size, n, device=device, dtype=W_batch.dtype)
        for root in range(n):
            keep = [idx for idx in range(n) if idx != root]
            minor = W_batch[:, keep, :][:, :, keep]
            det = torch.det(minor).abs().clamp(min=1e-12, max=1e12)
            p_batch[:, root] = det
        p_batch = p_batch / p_batch.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return p_batch

    def linear_solver_steady_state(self, W_batch: torch.Tensor) -> torch.Tensor:
        batch_size, n = W_batch.shape[0], self.n_nodes
        device = W_batch.device
        ones_row = torch.ones(batch_size, 1, n, device=device, dtype=W_batch.dtype)
        A_augmented = torch.cat([W_batch, ones_row], dim=1)
        b = torch.zeros(batch_size, n + 1, device=device, dtype=W_batch.dtype)
        b[:, -1] = 1.0
        p_batch = torch.linalg.lstsq(A_augmented, b.unsqueeze(-1)).solution.squeeze(-1)
        p_batch = torch.clamp(p_batch, min=0.0)
        return p_batch / p_batch.sum(dim=1, keepdim=True).clamp(min=1e-12)

    def direct_solve_steady_state(self, W_batch: torch.Tensor) -> torch.Tensor:
        batch_size, n = W_batch.shape[0], self.n_nodes
        device = W_batch.device
        W_modified = W_batch.clone()
        W_modified[:, -1, :] = 1.0
        b = torch.zeros(batch_size, n, device=device, dtype=W_batch.dtype)
        b[:, -1] = 1.0
        p_batch = torch.linalg.solve(W_modified, b)
        p_batch = torch.clamp(p_batch, min=0.0)
        return p_batch / p_batch.sum(dim=1, keepdim=True).clamp(min=1e-12)

    def steady_state(self, W_batch: torch.Tensor, method: str = "direct_solve") -> torch.Tensor:
        if method == "matrix_tree":
            return self.matrix_tree_steady_state(W_batch)
        if method == "linear_solver":
            return self.linear_solver_steady_state(W_batch)
        if method == "direct_solve":
            return self.direct_solve_steady_state(W_batch)
        raise ValueError(f"Invalid method: {method}")

    def forward(self, z_seq_batch, labels_seq_batch, method="direct_solve", temperature=1.0):
        batch_size = z_seq_batch.shape[0]
        z_flat = z_seq_batch.reshape(batch_size, -1)
        W_batch = self.compute_rate_matrix_W(z_flat, labels_seq_batch)
        p_batch = self.steady_state(W_batch, method=method)

        q = torch.matmul(p_batch, self.B)
        attention = torch.softmax(q / temperature, dim=1)
        labels_one_hot = torch.nn.functional.one_hot(
            labels_seq_batch.long() - 1,
            num_classes=self.L,
        ).float()
        logits = torch.einsum("bn,bnk->bk", attention, labels_one_hot)
        return torch.log(logits.clamp(min=1e-6, max=1.0))

    def get_dynamics_info(self, z_seq_batch, labels_seq_batch=None, method="direct_solve"):
        batch_size = z_seq_batch.shape[0]
        z_flat = z_seq_batch.reshape(batch_size, -1)
        edge_log_rates = self.edge_log_rates_from_flat(z_flat, labels_seq_batch)
        W_batch = self.compute_rate_matrix_W(z_flat, labels_seq_batch)
        p_batch = self.steady_state(W_batch, method=method)
        return {
            "edge_log_rates": edge_log_rates,
            "edge_rates": torch.exp(edge_log_rates) if self.transform_func == "exp" else self.edge_rates_from_flat(z_flat, labels_seq_batch),
            "W": W_batch,
            "steady_state": p_batch,
            "edges": self.edges,
        }

    def set_edge_rate_multipliers(self, multipliers):
        """Set per-edge physical rate multipliers for ablation diagnostics."""

        tensor = torch.as_tensor(
            multipliers,
            dtype=self.edge_rate_multiplier.dtype,
            device=self.edge_rate_multiplier.device,
        )
        if tuple(tensor.shape) != (self.n_edges,):
            raise ValueError(f"multipliers must have shape ({self.n_edges},)")
        self.edge_rate_multiplier.copy_(tensor)
