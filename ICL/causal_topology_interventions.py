"""Causal interventions for trained first-order topology ICL mechanisms.

The topology report currently shows strong post-training associations between
ICL and branch-specific active-tree/projection organization.  This script is a
cluster-side probe for the stronger causal question: does ICL collapse when we
scramble learned tree/branch alignment while preserving simpler quantities such
as parameter norms, input-mask counts, or decoder row norms?

The implemented interventions are intentionally mechanical and auditable:

* ``context_block_shuffle`` permutes learned context-item coordinate blocks in
  ``K`` and the input mask, leaving the query block fixed.
* ``all_coordinate_shuffle`` permutes all input coordinates in ``K`` and the
  input mask.
* ``edge_projection_permutation`` permutes learned edge projections and input
  mask rows across physical edges, leaving basal edge biases fixed.
* ``edge_rate_function_permutation`` permutes projections, input-mask rows,
  basal edge biases, and optional label modulation rows across physical edges.
* ``decoder_root_permutation`` permutes decoder rows across species.
* ``randomize_K_direction`` replaces each effective edge projection by a random
  vector on the same support with the same effective row norm.
* ``stat_preserving_branch_alignment_scramble`` is an explicit branch-alignment
  scramble: it preserves the physical graph, root tree counts, ``d_rel``, total
  input-mask density, and per-edge projection row norms while permuting context
  blocks relative to the query block.
* ``stat_preserving_projection_scramble`` preserves the physical graph, root
  tree counts, ``d_rel``, input-mask support, and per-edge projection row norms
  while randomizing projection directions on the existing support.

Each intervention is evaluated on the same sampled novel-class ICL batch as the
baseline.  The output JSON is designed to be joined with mechanism summaries
without treating the intervention as a new training run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from typing import Dict, Iterable, List, Sequence

import numpy as np


DEFAULT_INTERVENTIONS = [
    "context_block_shuffle",
    "edge_projection_permutation",
    "edge_rate_function_permutation",
    "decoder_root_permutation",
    "randomize_K_direction",
    "stat_preserving_branch_alignment_scramble",
    "stat_preserving_projection_scramble",
]


SUMMARY_KEYS = [
    "target_accuracy",
    "target_logprob_margin_mean",
    "target_logprob_margin_min",
    "branch_active_tree_mi",
    "branch_active_root_mi",
    "tree_entropy_mean",
    "root_entropy_mean",
    "active_tree_matched_comparison_gap_mean",
    "posterior_matched_comparison_gap_mean",
]


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_interventions(raw: str) -> List[str]:
    interventions = [item.strip() for item in raw.split(",") if item.strip()]
    if not interventions:
        raise ValueError("At least one intervention is required")
    unknown = [item for item in interventions if item not in DEFAULT_INTERVENTIONS]
    if unknown:
        raise ValueError(f"Unknown interventions: {unknown}")
    return interventions


def permutation(seed: int, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    if n > 1 and np.all(perm == np.arange(n)):
        perm = np.roll(perm, 1)
    return perm.astype(int)


def context_block_permutation(n_context: int, z_dim: int, seed: int, include_query: bool = False) -> np.ndarray:
    """Return a coordinate permutation over flattened ``(N+1)D`` input."""

    if n_context <= 0 or z_dim <= 0:
        raise ValueError("n_context and z_dim must be positive")
    n_blocks = n_context + 1
    block_perm = np.arange(n_blocks)
    if include_query:
        block_perm = permutation(seed, n_blocks)
    elif n_context > 1:
        block_perm[:n_context] = permutation(seed, n_context)
    coords = []
    for block in block_perm:
        start = int(block) * z_dim
        coords.extend(range(start, start + z_dim))
    return np.asarray(coords, dtype=int)


def random_effective_K_with_same_support(effective_K: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
    """Randomize row directions while preserving support and effective row norms."""

    effective = np.asarray(effective_K, dtype=float)
    mask_arr = np.asarray(mask, dtype=float)
    if effective.shape != mask_arr.shape:
        raise ValueError("effective_K and mask must have the same shape")
    rng = np.random.default_rng(seed)
    randomized = np.zeros_like(effective)
    for row_idx in range(effective.shape[0]):
        support = mask_arr[row_idx, :] != 0
        norm = float(np.linalg.norm(effective[row_idx, support]))
        if not support.any() or norm <= 0.0:
            continue
        values = rng.normal(size=int(support.sum()))
        values_norm = float(np.linalg.norm(values))
        if values_norm <= 0.0:
            continue
        randomized[row_idx, support] = values * (norm / values_norm)
    return randomized


def clone_state_dict(model):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def restore_state_dict(model, state):
    model.load_state_dict(state, strict=True)


def _torch_permutation(torch, values: np.ndarray, device) -> object:
    return torch.as_tensor(values, dtype=torch.long, device=device)


def apply_intervention(model, intervention: str, seed: int) -> dict:
    """Apply one intervention in-place and return metadata."""

    import torch

    metadata = {"intervention": intervention, "seed": seed}
    device = model.K_params.device
    if intervention in {"context_block_shuffle", "stat_preserving_branch_alignment_scramble"}:
        perm = context_block_permutation(model.N, model.z_dim, seed, include_query=False)
        perm_t = _torch_permutation(torch, perm, device)
        with torch.no_grad():
            model.K_params.copy_(model.K_params[:, perm_t])
            model.input_mask.copy_(model.input_mask[:, perm_t])
        metadata["coordinate_permutation"] = perm.tolist()
        if intervention == "stat_preserving_branch_alignment_scramble":
            metadata.update(
                {
                    "preserves_physical_graph": True,
                    "preserves_root_tree_counts": True,
                    "preserves_d_rel": True,
                    "preserves_total_input_mask_density": True,
                    "preserves_edge_projection_row_norms": True,
                    "scrambles_branch_context_alignment": True,
                    "query_block_fixed": True,
                }
            )
        return metadata

    if intervention == "all_coordinate_shuffle":
        perm = permutation(seed, model.K_params.shape[1])
        perm_t = _torch_permutation(torch, perm, device)
        with torch.no_grad():
            model.K_params.copy_(model.K_params[:, perm_t])
            model.input_mask.copy_(model.input_mask[:, perm_t])
        metadata["coordinate_permutation"] = perm.tolist()
        return metadata

    if intervention in {"edge_projection_permutation", "edge_rate_function_permutation"}:
        perm = permutation(seed, model.n_edges)
        perm_t = _torch_permutation(torch, perm, device)
        with torch.no_grad():
            model.K_params.copy_(model.K_params[perm_t, :])
            model.input_mask.copy_(model.input_mask[perm_t, :])
            if intervention == "edge_rate_function_permutation":
                model.base_log_rates.copy_(model.base_log_rates[perm_t])
                if getattr(model, "label_modulation", None) is not None:
                    model.label_modulation.copy_(model.label_modulation[perm_t, :])
        metadata["edge_permutation"] = perm.tolist()
        return metadata

    if intervention == "decoder_root_permutation":
        perm = permutation(seed, model.n_nodes)
        perm_t = _torch_permutation(torch, perm, device)
        with torch.no_grad():
            model.B.copy_(model.B[perm_t, :])
        metadata["root_permutation"] = perm.tolist()
        return metadata

    if intervention in {"randomize_K_direction", "stat_preserving_projection_scramble"}:
        with torch.no_grad():
            effective = (model.K_params * model.input_mask).detach().cpu().numpy()
            mask = model.input_mask.detach().cpu().numpy()
            randomized = random_effective_K_with_same_support(effective, mask, seed)
            tensor = torch.as_tensor(randomized, dtype=model.K_params.dtype, device=device)
            model.K_params.copy_(tensor)
        metadata["preserves_effective_row_norms"] = True
        metadata["preserves_input_mask"] = True
        if intervention == "stat_preserving_projection_scramble":
            metadata.update(
                {
                    "preserves_physical_graph": True,
                    "preserves_root_tree_counts": True,
                    "preserves_d_rel": True,
                    "preserves_input_mask_support": True,
                    "preserves_total_input_mask_density": True,
                    "scrambles_projection_direction_on_existing_support": True,
                }
            )
        return metadata

    raise ValueError(f"Unknown intervention {intervention!r}")


def compact_metrics(metrics: dict) -> dict:
    return {
        key: metrics.get(key)
        for key in SUMMARY_KEYS
        if key in metrics
    }


def metric_deltas(metrics: dict, baseline: dict) -> dict:
    deltas = {}
    for key in SUMMARY_KEYS:
        if key in metrics and key in baseline:
            try:
                deltas[f"{key}_delta"] = float(metrics[key]) - float(baseline[key])
            except (TypeError, ValueError):
                continue
    return deltas


def choose_device(torch, name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trained_model(run_dir: str, device):
    import torch

    from models import TopologyMatrixTreeMarkovICL

    with open(os.path.join(run_dir, "config.json")) as handle:
        config = json.load(handle)
    with open(os.path.join(run_dir, "topology.json")) as handle:
        topology = json.load(handle)

    input_mask = np.asarray(topology["input_mask"], dtype=float)
    model = TopologyMatrixTreeMarkovICL(
        n_nodes=topology["n_nodes"],
        z_dim=config["D"],
        L=config["L"],
        N=config["N"],
        edges=topology["edges"],
        input_mask=input_mask,
        use_label_mod=config.get("use_label_mod", False),
        learn_base_rates=config.get("learn_base_rates", True),
        transform_func=config.get("transform_func", "exp"),
        print_creation=False,
    )
    state = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config, topology


def sample_novel_batch(config: dict, n_samples: int, device):
    import torch

    from data_generation import GaussianMixtureModel, generate_icl_gmm_data

    gmm = GaussianMixtureModel(
        K=config["K"],
        D=config["D"],
        L=config["L"],
        epsilon=config["epsilon"],
        seed=config["seed"],
        offset=0.0,
    )
    data = generate_icl_gmm_data(
        gmm,
        n_samples,
        config["N"],
        novel_classes=True,
        exact_copy=config.get("exact_copy", True),
        B=config.get("B", 1),
        L=config["L"],
        shuffle_context=True,
        unique_labels=config.get("unique_labels", False),
    )
    z_seq = torch.stack([item[0] for item in data]).to(device)
    labels = torch.stack([item[1] for item in data]).to(device)
    targets = torch.as_tensor([int(item[2].item()) for item in data], device=device)
    return z_seq, labels, targets


def run_interventions(args) -> dict:
    import torch

    from topology_analysis import analyze_batch
    from topology_metrics import topology_matrices

    interventions = parse_interventions(args.interventions)
    device = choose_device(torch, args.device)
    model, config, topology = load_trained_model(args.run_dir, device)
    z_seq, labels, targets = sample_novel_batch(config, args.n_samples, device)
    matrices = topology_matrices(topology["n_nodes"], topology["edges"])
    method = config.get("method", "direct_solve")
    temperature = config.get("temperature", 1.0)

    baseline_metrics = analyze_batch(
        model,
        z_seq,
        labels,
        target_labels=targets,
        matrices=matrices,
        method=method,
        temperature=temperature,
    )
    baseline_state = clone_state_dict(model)
    rows = []
    for intervention in interventions:
        for repeat in range(args.n_repeats):
            seed = args.seed + repeat + 1009 * interventions.index(intervention)
            restore_state_dict(model, baseline_state)
            metadata = apply_intervention(model, intervention, seed)
            metrics = analyze_batch(
                model,
                z_seq,
                labels,
                target_labels=targets,
                matrices=matrices,
                method=method,
                temperature=temperature,
            )
            row = {
                **metadata,
                "repeat": repeat,
                **compact_metrics(metrics),
                **metric_deltas(metrics, baseline_metrics),
            }
            rows.append(row)
    restore_state_dict(model, baseline_state)

    return {
        "run_dir": os.path.abspath(args.run_dir),
        "topology_name": topology.get("name", "custom"),
        "n_samples": args.n_samples,
        "n_repeats": args.n_repeats,
        "device": str(device),
        "baseline": compact_metrics(baseline_metrics),
        "interventions": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--interventions", default=",".join(DEFAULT_INTERVENTIONS))
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.n_samples <= 0:
        raise SystemExit("--n_samples must be positive")
    if args.n_repeats <= 0:
        raise SystemExit("--n_repeats must be positive")

    result = run_interventions(args)
    output = args.output or os.path.join(args.run_dir, "causal_interventions.json")
    with open(output, "w") as handle:
        json.dump(json_ready(result), handle, indent=2)
    print(f"Wrote causal intervention report to {output}")
    baseline_acc = result["baseline"].get("target_accuracy")
    print(f"Baseline novel-class target accuracy on sampled batch: {baseline_acc}")
    for row in result["interventions"]:
        if row.get("repeat") == 0:
            print(
                f"  {row['intervention']}: "
                f"accuracy_delta={row.get('target_accuracy_delta')}"
            )


if __name__ == "__main__":
    main()
