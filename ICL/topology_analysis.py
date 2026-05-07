"""Post-training matrix-tree analysis for topology-aware first-order CRNs."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from topology_metrics import topology_matrices


def _roots_from_arborescences(arborescences: Mapping[int, Sequence[Sequence[int]]]) -> np.ndarray:
    roots = []
    for root in sorted(arborescences):
        roots.extend([root] * len(arborescences[root]))
    return np.asarray(roots, dtype=int)


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0:
        return 0.0
    x_values, x_inv = np.unique(x, return_inverse=True)
    y_values, y_inv = np.unique(y, return_inverse=True)
    joint = np.zeros((len(x_values), len(y_values)), dtype=float)
    for xi, yi in zip(x_inv, y_inv):
        joint[xi, yi] += 1.0
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    return float(np.sum(joint[nonzero] * np.log(joint[nonzero] / (px @ py)[nonzero])))


def infer_exact_copy_branch(z_seq_batch: torch.Tensor) -> np.ndarray:
    """Infer query-match context index by nearest context item."""

    context = z_seq_batch[:, :-1, :]
    query = z_seq_batch[:, -1:, :]
    distances = torch.sum((context - query) ** 2, dim=2)
    return distances.argmin(dim=1).detach().cpu().numpy()


def _comparison_difference_bases(n_context: int, z_dim: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build orthonormal bases for context-query difference directions."""

    p = (n_context + 1) * z_dim
    branch_bases = []
    query_offset = n_context * z_dim
    scale = 1.0 / np.sqrt(2.0)
    for branch in range(n_context):
        basis = np.zeros((p, z_dim), dtype=float)
        context_offset = branch * z_dim
        for dim in range(z_dim):
            basis[context_offset + dim, dim] = scale
            basis[query_offset + dim, dim] = -scale
        branch_bases.append(basis)

    if branch_bases:
        raw = np.concatenate(branch_bases, axis=1)
        full_basis, _ = np.linalg.qr(raw, mode="reduced")
    else:
        full_basis = np.zeros((p, 0), dtype=float)
    return full_basis, branch_bases


def _projection_energy_fraction(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    norms_sq = np.sum(vectors * vectors, axis=1)
    if basis.size == 0:
        energy = np.zeros_like(norms_sq)
    else:
        projected = vectors @ basis
        energy = np.sum(projected * projected, axis=1)
    result = np.zeros_like(norms_sq)
    valid = norms_sq > 1e-12
    result[valid] = energy[valid] / norms_sq[valid]
    return result


def tree_projection_geometry(model, matrices: dict) -> dict:
    """Summarize learned tree-sum projection alignment with comparison axes."""

    M_np = np.asarray(matrices["M"], dtype=float)
    K = (model.K_params * model.input_mask).detach().cpu().numpy()
    theta = M_np @ K
    norms = np.sqrt(np.sum(theta * theta, axis=1))
    full_basis, branch_bases = _comparison_difference_bases(model.N, model.z_dim)
    full_fraction = _projection_energy_fraction(theta, full_basis)
    branch_fractions = np.stack(
        [_projection_energy_fraction(theta, basis) for basis in branch_bases],
        axis=1,
    ) if branch_bases else np.zeros((theta.shape[0], 0), dtype=float)

    return {
        "tree_projection_norm": norms,
        "tree_comparison_energy_fraction": full_fraction,
        "tree_branch_comparison_energy_fraction": branch_fractions,
        "tree_projection_norm_mean": float(norms.mean()) if norms.size else 0.0,
        "tree_projection_norm_max": float(norms.max()) if norms.size else 0.0,
        "tree_comparison_energy_fraction_mean": float(full_fraction.mean()) if full_fraction.size else 0.0,
        "tree_comparison_energy_fraction_max": float(full_fraction.max()) if full_fraction.size else 0.0,
    }


def _active_projection_alignment_summary(
    decomposition: dict,
    geometry: dict,
    branch_ids: np.ndarray,
) -> dict:
    active_tree = decomposition["active_tree_by_root"][
        torch.arange(decomposition["active_root"].shape[0], device=decomposition["active_root"].device),
        decomposition["active_root"],
    ].detach().cpu().numpy()
    branch_ids = np.asarray(branch_ids, dtype=int)
    full_fraction = geometry["tree_comparison_energy_fraction"]
    branch_fractions = geometry["tree_branch_comparison_energy_fraction"]

    if active_tree.size == 0 or branch_fractions.shape[1] == 0:
        return {
            "active_tree_comparison_energy_fraction_mean": 0.0,
            "active_tree_matched_comparison_energy_mean": 0.0,
            "active_tree_matched_comparison_gap_mean": 0.0,
            "posterior_comparison_energy_fraction_mean": 0.0,
            "posterior_matched_comparison_energy_mean": 0.0,
            "posterior_matched_comparison_gap_mean": 0.0,
        }

    valid = active_tree >= 0
    clipped_branches = np.clip(branch_ids, 0, branch_fractions.shape[1] - 1)
    active_full = np.zeros(active_tree.shape[0], dtype=float)
    active_matched = np.zeros(active_tree.shape[0], dtype=float)
    active_gap = np.zeros(active_tree.shape[0], dtype=float)
    if np.any(valid):
        active_branch_fraction = branch_fractions[active_tree[valid], :]
        active_full[valid] = full_fraction[active_tree[valid]]
        local_branch_ids = clipped_branches[valid]
        active_matched[valid] = active_branch_fraction[
            np.arange(active_branch_fraction.shape[0]),
            local_branch_ids,
        ]
        if branch_fractions.shape[1] > 1:
            other = active_branch_fraction.copy()
            other[np.arange(other.shape[0]), local_branch_ids] = -np.inf
            active_gap[valid] = active_matched[valid] - np.max(other, axis=1)

    roots = decomposition["roots"].detach().cpu().numpy()
    root_prob = decomposition["root_probabilities"].detach().cpu().numpy()
    tree_post = decomposition["tree_posteriors"].detach().cpu().numpy()
    global_tree_weights = tree_post * root_prob[:, roots]
    posterior_full = global_tree_weights @ full_fraction
    posterior_branch = global_tree_weights @ branch_fractions
    posterior_matched = posterior_branch[
        np.arange(posterior_branch.shape[0]),
        clipped_branches,
    ]
    if branch_fractions.shape[1] > 1:
        posterior_other = posterior_branch.copy()
        posterior_other[np.arange(posterior_other.shape[0]), clipped_branches] = -np.inf
        posterior_gap = posterior_matched - np.max(posterior_other, axis=1)
    else:
        posterior_gap = np.zeros_like(posterior_matched)

    return {
        "active_tree_comparison_energy_fraction_mean": float(active_full.mean()),
        "active_tree_matched_comparison_energy_mean": float(active_matched.mean()),
        "active_tree_matched_comparison_gap_mean": float(active_gap.mean()),
        "posterior_comparison_energy_fraction_mean": float(posterior_full.mean()),
        "posterior_matched_comparison_energy_mean": float(posterior_matched.mean()),
        "posterior_matched_comparison_gap_mean": float(posterior_gap.mean()),
    }


def tree_decomposition_from_edge_logs(
    edge_log_rates: torch.Tensor,
    n_nodes: int,
    edges: Iterable[Sequence[int]],
    matrices: Optional[dict] = None,
) -> dict:
    """Compute tree logits, posteriors, root logits, and active trees."""

    if matrices is None:
        matrices = topology_matrices(n_nodes, edges)
    M_np = matrices["M"]
    roots_np = _roots_from_arborescences(matrices["arborescences"])
    if M_np.shape[0] == 0:
        raise ValueError("No rooted arborescences available for this topology")

    device = edge_log_rates.device
    dtype = edge_log_rates.dtype
    M = torch.as_tensor(M_np, dtype=dtype, device=device)
    roots = torch.as_tensor(roots_np, dtype=torch.long, device=device)

    tree_logits = torch.matmul(edge_log_rates, M.T)
    batch_size = edge_log_rates.shape[0]
    root_logits = torch.full(
        (batch_size, n_nodes),
        -torch.inf,
        dtype=dtype,
        device=device,
    )
    tree_posteriors = torch.zeros_like(tree_logits)
    active_tree_by_root = torch.full(
        (batch_size, n_nodes),
        -1,
        dtype=torch.long,
        device=device,
    )

    for root in range(n_nodes):
        mask = roots == root
        if not torch.any(mask):
            continue
        root_tree_logits = tree_logits[:, mask]
        root_logits[:, root] = torch.logsumexp(root_tree_logits, dim=1)
        tree_posteriors[:, mask] = torch.softmax(root_tree_logits, dim=1)
        local_active = torch.argmax(root_tree_logits, dim=1)
        global_tree_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
        active_tree_by_root[:, root] = global_tree_ids[local_active]

    root_probabilities = torch.softmax(root_logits, dim=1)
    active_root = torch.argmax(root_logits, dim=1)

    return {
        "M": M,
        "roots": roots,
        "tree_logits": tree_logits,
        "tree_posteriors": tree_posteriors,
        "root_logits": root_logits,
        "root_probabilities": root_probabilities,
        "active_tree_by_root": active_tree_by_root,
        "active_root": active_root,
        "matrices": matrices,
    }


def edge_participation_by_root(decomposition: dict, n_nodes: int) -> torch.Tensor:
    """Return A[r,e](z)=Pr(e in T | r,z) for each sample."""

    M = decomposition["M"]
    roots = decomposition["roots"]
    post = decomposition["tree_posteriors"]
    batch_size = post.shape[0]
    n_edges = M.shape[1]
    A = torch.zeros(batch_size, n_nodes, n_edges, device=post.device, dtype=post.dtype)
    for root in range(n_nodes):
        mask = roots == root
        if torch.any(mask):
            A[:, root, :] = torch.matmul(post[:, mask], M[mask, :])
    return A


def context_score_edge_sensitivity(model, decomposition: dict) -> torch.Tensor:
    """Compute d context-position scores / d log k_e.

    The returned tensor has shape ``(batch, N_context, n_edges)`` and uses the
    exact matrix-tree identity

    d log pi_r / d log k_e = A[r,e] - sum_s pi_s A[s,e].
    """

    root_prob = decomposition["root_probabilities"]
    n_nodes = root_prob.shape[1]
    A = edge_participation_by_root(decomposition, n_nodes)
    expected_A = torch.einsum("br,bre->be", root_prob, A)
    dlog_pi = A - expected_A[:, None, :]
    dpi = root_prob[:, :, None] * dlog_pi
    return torch.einsum("rn,bre->bne", model.B, dpi)


def _target_margin(logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
    target_idx = target_labels.long() - 1
    correct = logits.gather(1, target_idx[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target_idx[:, None], -torch.inf)
    incorrect = masked.max(dim=1).values
    return correct - incorrect


def _essential_edge_summary(edge_importance: np.ndarray, fractions=(0.1, 0.2, 0.5)) -> dict:
    if edge_importance.size == 0 or np.allclose(edge_importance.sum(), 0.0):
        summary = {"n_edges": int(edge_importance.size)}
        for frac in fractions:
            summary[f"edges_for_{int(frac * 100)}pct_importance"] = 0
        return summary
    order = np.argsort(edge_importance)[::-1]
    cumulative = np.cumsum(edge_importance[order]) / edge_importance.sum()
    summary = {"n_edges": int(edge_importance.size)}
    for frac in fractions:
        summary[f"edges_for_{int(frac * 100)}pct_importance"] = int(np.searchsorted(cumulative, frac) + 1)
    return summary


def analyze_batch(
    model,
    z_seq_batch,
    labels_seq_batch=None,
    target_labels=None,
    branch_ids=None,
    matrices=None,
    method="direct_solve",
    temperature=1.0,
) -> dict:
    """Analyze a batch from a trained topology-aware first-order model."""

    model.eval()
    with torch.no_grad():
        z_flat = z_seq_batch.reshape(z_seq_batch.shape[0], -1)
        edge_rates = model.edge_rates_from_flat(z_flat, labels_seq_batch)
        edge_log_rates = torch.log(edge_rates.clamp(min=1e-30))
        decomposition = tree_decomposition_from_edge_logs(
            edge_log_rates,
            n_nodes=model.n_nodes,
            edges=model.edges,
            matrices=matrices,
        )
        projection_geometry = tree_projection_geometry(model, decomposition["matrices"])
        sensitivities = context_score_edge_sensitivity(model, decomposition)
        logits = None
        margins = None
        accuracy = None
        if target_labels is not None and labels_seq_batch is not None:
            logits = model(z_seq_batch, labels_seq_batch, method=method, temperature=temperature)
            margins = _target_margin(logits, target_labels)
            preds = logits.argmax(dim=1) + 1
            accuracy = (preds == target_labels.long()).float().mean()

    active_root = decomposition["active_root"].detach().cpu().numpy()
    root_entropy = (
        -decomposition["root_probabilities"]
        * torch.log(decomposition["root_probabilities"].clamp(min=1e-12))
    ).sum(dim=1)
    tree_entropy = (
        -decomposition["tree_posteriors"]
        * torch.log(decomposition["tree_posteriors"].clamp(min=1e-12))
    ).sum(dim=1)
    edge_importance = sensitivities.abs().mean(dim=(0, 1)).detach().cpu().numpy()
    essential = _essential_edge_summary(edge_importance)

    if branch_ids is None:
        branch_ids = infer_exact_copy_branch(z_seq_batch)
    branch_ids = np.asarray(branch_ids)

    active_tree_for_active_root = decomposition["active_tree_by_root"][
        torch.arange(z_seq_batch.shape[0], device=z_seq_batch.device),
        decomposition["active_root"],
    ].detach().cpu().numpy()

    result = {
        "active_root": active_root.tolist(),
        "active_tree": active_tree_for_active_root.tolist(),
        "branch_ids": branch_ids.tolist(),
        "branch_active_root_mi": _mutual_information(branch_ids, active_root),
        "branch_active_tree_mi": _mutual_information(branch_ids, active_tree_for_active_root),
        "root_entropy_mean": float(root_entropy.mean().item()),
        "tree_entropy_mean": float(tree_entropy.mean().item()),
        "edge_importance": edge_importance.tolist(),
        "edge_importance_mean": float(edge_importance.mean()) if edge_importance.size else 0.0,
        "edge_importance_max": float(edge_importance.max()) if edge_importance.size else 0.0,
        "edge_importance_gini": _gini(edge_importance),
        "essential_edges_for_10pct_importance": essential["edges_for_10pct_importance"],
        "essential_edges_for_20pct_importance": essential["edges_for_20pct_importance"],
        "essential_edges_for_50pct_importance": essential["edges_for_50pct_importance"],
        "tree_projection_norm_mean": projection_geometry["tree_projection_norm_mean"],
        "tree_projection_norm_max": projection_geometry["tree_projection_norm_max"],
        "tree_comparison_energy_fraction_mean": projection_geometry["tree_comparison_energy_fraction_mean"],
        "tree_comparison_energy_fraction_max": projection_geometry["tree_comparison_energy_fraction_max"],
    }
    result.update(
        _active_projection_alignment_summary(
            decomposition,
            projection_geometry,
            branch_ids,
        )
    )
    if margins is not None:
        result.update(
            {
                "target_logprob_margin_mean": float(margins.mean().item()),
                "target_logprob_margin_min": float(margins.min().item()),
                "target_accuracy": float(100.0 * accuracy.item()),
            }
        )
    return result


def _accuracy_from_logits(logits: torch.Tensor, target_labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1) + 1
    return float(100.0 * (preds == target_labels.long()).float().mean().item())


def edge_ablation_scan(
    model,
    z_seq_batch,
    labels_seq_batch,
    target_labels,
    mode="input",
    method="direct_solve",
    temperature=1.0,
    physical_epsilon=1e-6,
) -> dict:
    """Measure per-edge novel-task accuracy loss under ablation.

    ``mode='input'`` zeros one row of the input mask, leaving basal chemistry
    intact. ``mode='physical'`` multiplies one edge's physical rate by
    ``physical_epsilon`` while restoring it after each edge.
    """

    if mode not in {"input", "physical"}:
        raise ValueError("mode must be 'input' or 'physical'")

    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        baseline_logits = model(
            z_seq_batch,
            labels_seq_batch,
            method=method,
            temperature=temperature,
        )
        baseline_accuracy = _accuracy_from_logits(baseline_logits, target_labels)

        original_input_mask = model.input_mask.detach().clone()
        original_multipliers = model.edge_rate_multiplier.detach().clone()
        try:
            for edge_idx in range(model.n_edges):
                if mode == "input":
                    model.input_mask.copy_(original_input_mask)
                    model.input_mask[edge_idx, :] = 0.0
                else:
                    model.edge_rate_multiplier.copy_(original_multipliers)
                    model.edge_rate_multiplier[edge_idx] = physical_epsilon

                logits = model(
                    z_seq_batch,
                    labels_seq_batch,
                    method=method,
                    temperature=temperature,
                )
                ablated_accuracy = _accuracy_from_logits(logits, target_labels)
                accuracies.append(ablated_accuracy)
                losses.append(baseline_accuracy - ablated_accuracy)
        finally:
            model.input_mask.copy_(original_input_mask)
            model.edge_rate_multiplier.copy_(original_multipliers)

    order = np.argsort(np.asarray(losses))[::-1]
    return {
        "mode": mode,
        "baseline_accuracy": baseline_accuracy,
        "ablated_accuracy": accuracies,
        "accuracy_loss": losses,
        "top_edges_by_loss": [
            {
                "edge_index": int(idx),
                "edge": list(model.edges[int(idx)]),
                "accuracy_loss": float(losses[int(idx)]),
                "ablated_accuracy": float(accuracies[int(idx)]),
            }
            for idx in order[: min(10, len(order))]
        ],
    }


def _gini(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.allclose(arr.sum(), 0.0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    weights = np.arange(1, n + 1)
    return float((2.0 * np.sum(weights * arr) / (n * arr.sum())) - ((n + 1.0) / n))
