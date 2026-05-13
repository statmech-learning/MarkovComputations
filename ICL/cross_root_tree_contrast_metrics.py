"""Cross-root tree-contrast metrics for first-order Markov ICL.

The Phase 2 multiplicity metrics measure same-root tree and tree-difference
input co-participation.  This module adds decoder-aware pre-training
diagnostics by comparing trees rooted at different species.  The steady state
normalizes all rooted tree numerators jointly, so cross-root contrasts are a
natural structural proxy for what a learned decoder can compare.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from topology_metrics import normalize_edges, topology_matrices, tree_counts_by_determinant, svd_metrics
from tree_level_multiplicity_metrics import (
    comparison_columns,
    finite_matmul,
    positive_gini,
    sampled_pair_indices,
    tree_table_from_arborescences,
)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def safe_log1p(value: float | None) -> float | None:
    if value is None:
        return None
    return float(math.log1p(max(0.0, value)))


def entropy01(values: Sequence[float]) -> float | None:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0])
    if arr.size == 0:
        return None
    probs = arr / arr.sum()
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(arr.size) if arr.size > 1 else 0.0


def effective_rank_and_condition(matrix: np.ndarray) -> dict[str, Any]:
    metrics = svd_metrics(np.asarray(matrix, dtype=float))
    condition = metrics["condition_number"]
    if isinstance(condition, float) and not math.isfinite(condition):
        condition_log = None
    else:
        condition_log = safe_log1p(float(condition))
    return {
        "rank": int(metrics["rank"]),
        "effective_rank": float(metrics["effective_rank"]),
        "condition_number": condition if math.isfinite(float(condition)) else None,
        "condition_number_log": condition_log,
    }


def _sample_cross_indices(
    n_left: int,
    n_right: int,
    max_pairs: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    total = int(n_left * n_right)
    if total <= 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), total, False
    if max_pairs is None or total <= max_pairs:
        left, right = np.meshgrid(np.arange(n_left), np.arange(n_right), indexing="ij")
        return left.ravel(), right.ravel(), total, False
    rng = np.random.default_rng(seed)
    sampled = rng.choice(total, size=max_pairs, replace=False)
    return sampled // n_right, sampled % n_right, total, True


def _prefix_summary(prefix: str, values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_min": None,
            f"{prefix}_mean": None,
            f"{prefix}_max": None,
            f"{prefix}_gini": None,
        }
    return {
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_gini": positive_gini(arr),
    }


def cross_root_tree_contrast_summary(
    tree_incidence: np.ndarray,
    tree_roots: np.ndarray,
    input_mask: np.ndarray,
    n_context: int,
    z_dim: int,
    max_pairs_per_root_pair: int | None = 50000,
    top_k_root_pairs: int = 3,
) -> dict[str, Any]:
    """Summarize cross-root input participation and controllability proxies."""

    incidence = np.asarray(tree_incidence, dtype=float)
    roots = np.asarray(tree_roots, dtype=int)
    mask = np.asarray(input_mask, dtype=float)
    comp_cols = comparison_columns(n_context, z_dim)

    overlap_norm_values: list[float] = []
    separation_norm_values: list[float] = []
    imbalance_values: list[float] = []
    root_pair_overlap_values: dict[tuple[int, int], list[float]] = {}
    comparison_best_values: dict[tuple[int, int], list[float]] = {
        (branch, dim): [] for branch, dim, _, _ in comp_cols
    }
    signed_rows: list[np.ndarray] = []
    abs_rows: list[np.ndarray] = []
    bottleneck_edge_counts = np.zeros(mask.shape[0], dtype=float)
    per_root_pair: list[dict[str, Any]] = []
    total_possible = 0
    total_sampled = 0
    truncated = False

    unique_roots = sorted(set(int(root) for root in roots.tolist()))
    for r in unique_roots:
        idx_r = np.where(roots == r)[0]
        for s in unique_roots:
            if r == s:
                continue
            idx_s = np.where(roots == s)[0]
            left, right, possible, is_truncated = _sample_cross_indices(
                idx_r.size,
                idx_s.size,
                max_pairs=max_pairs_per_root_pair,
                seed=7919 + 97 * r + s,
            )
            total_possible += possible
            truncated = truncated or is_truncated
            if left.size == 0:
                continue
            left_global = idx_r[left]
            right_global = idx_s[right]
            signed_incidence = incidence[left_global] - incidence[right_global]
            abs_incidence = np.abs(signed_incidence)
            cross_loads = finite_matmul(abs_incidence, mask)
            total_sampled += int(cross_loads.shape[0])
            signed_rows.append(signed_incidence)
            abs_rows.append(abs_incidence)
            bottleneck_edge_counts += np.sum(abs_incidence > 0.0, axis=0)

            pair_values = []
            pair_separation = []
            pair_imbalance = []
            for branch, dim, ctx_col, query_col in comp_cols:
                ctx_active = cross_loads[:, ctx_col] > 0.0
                query_active = cross_loads[:, query_col] > 0.0
                overlap = float(np.mean(ctx_active & query_active))
                separation = float(np.mean(np.logical_xor(ctx_active, query_active)))
                denom = np.maximum(cross_loads[:, ctx_col] + cross_loads[:, query_col], 1.0)
                imbalance = float(np.mean(np.abs(cross_loads[:, ctx_col] - cross_loads[:, query_col]) / denom))
                pair_values.append(overlap)
                pair_separation.append(separation)
                pair_imbalance.append(imbalance)
                comparison_best_values[(branch, dim)].append(overlap)
                overlap_norm_values.append(overlap)
                separation_norm_values.append(separation)
                imbalance_values.append(imbalance)

            root_pair_overlap_values[(r, s)] = pair_values
            per_root_pair.append(
                {
                    "root_left": int(r),
                    "root_right": int(s),
                    "cross_pair_count_sampled": int(cross_loads.shape[0]),
                    "cross_pair_count_possible": int(possible),
                    "cross_pairs_truncated": bool(is_truncated),
                    "cross_overlap_norm_min": float(np.min(pair_values)),
                    "cross_overlap_norm_mean": float(np.mean(pair_values)),
                    "cross_overlap_norm_max": float(np.max(pair_values)),
                    "cross_separation_norm_mean": float(np.mean(pair_separation)),
                    "cross_imbalance_norm_mean": float(np.mean(pair_imbalance)),
                }
            )

    best_by_comparison = []
    topk_by_comparison = []
    usable_root_pair_counts = []
    entropy_by_comparison = []
    for values in comparison_best_values.values():
        if not values:
            continue
        vals = np.asarray(values, dtype=float)
        sorted_vals = np.sort(vals)[::-1]
        best_by_comparison.append(float(sorted_vals[0]))
        topk_by_comparison.append(float(np.mean(sorted_vals[: max(1, min(top_k_root_pairs, sorted_vals.size))])))
        usable_root_pair_counts.append(int(np.sum(vals > 0.0)))
        entropy_by_comparison.append(entropy01(vals) or 0.0)

    if signed_rows:
        signed_matrix = np.vstack(signed_rows)
        abs_matrix = np.vstack(abs_rows)
    else:
        signed_matrix = np.zeros((0, mask.shape[0]), dtype=float)
        abs_matrix = np.zeros((0, mask.shape[0]), dtype=float)

    # A low-dimensional controllability proxy: how many independent masked
    # cross-root tree contrasts can affect each context-query coordinate pair.
    contrast_rank_values = []
    contrast_effective_rank_values = []
    contrast_condition_logs = []
    for _, _, ctx_col, query_col in comp_cols:
        supported = (mask[:, ctx_col] > 0.0) | (mask[:, query_col] > 0.0)
        if not np.any(supported) or signed_matrix.shape[0] == 0:
            contrast_rank_values.append(0.0)
            contrast_effective_rank_values.append(0.0)
            continue
        signed_support = signed_matrix[:, supported]
        rank_info = effective_rank_and_condition(signed_support)
        contrast_rank_values.append(float(rank_info["rank"]))
        contrast_effective_rank_values.append(float(rank_info["effective_rank"]))
        if rank_info["condition_number_log"] is not None:
            contrast_condition_logs.append(float(rank_info["condition_number_log"]))

    all_supported = np.any(mask > 0.0, axis=1)
    all_rank_info = effective_rank_and_condition(signed_matrix[:, all_supported]) if np.any(all_supported) else {
        "rank": 0,
        "effective_rank": 0.0,
        "condition_number": None,
        "condition_number_log": None,
    }

    root_pair_means = [float(np.mean(vals)) for vals in root_pair_overlap_values.values() if vals]
    root_pair_mins = [float(np.min(vals)) for vals in root_pair_overlap_values.values() if vals]
    return {
        **_prefix_summary("cross_overlap_norm", overlap_norm_values),
        **_prefix_summary("cross_separation_norm", separation_norm_values),
        **_prefix_summary("cross_imbalance_norm", imbalance_values),
        **_prefix_summary("cross_root_pair_overlap_norm_mean", root_pair_means),
        **_prefix_summary("cross_root_pair_overlap_norm_min", root_pair_mins),
        "cross_best_root_pair_overlap_norm_min": float(np.min(best_by_comparison)) if best_by_comparison else None,
        "cross_best_root_pair_overlap_norm_mean": float(np.mean(best_by_comparison)) if best_by_comparison else None,
        "cross_topk_root_pair_overlap_norm_mean": float(np.mean(topk_by_comparison)) if topk_by_comparison else None,
        "cross_usable_root_pair_count_min": int(np.min(usable_root_pair_counts)) if usable_root_pair_counts else None,
        "cross_usable_root_pair_count_mean": float(np.mean(usable_root_pair_counts)) if usable_root_pair_counts else None,
        "cross_root_pair_overlap_entropy_mean": float(np.mean(entropy_by_comparison)) if entropy_by_comparison else None,
        "cross_pair_count_sampled": int(total_sampled),
        "cross_pair_count_possible": int(total_possible),
        "cross_pairs_truncated": bool(truncated),
        "cross_pair_count_log": safe_log1p(float(total_sampled)),
        "cross_contrast_rank_mean": float(np.mean(contrast_rank_values)) if contrast_rank_values else None,
        "cross_contrast_rank_min": float(np.min(contrast_rank_values)) if contrast_rank_values else None,
        "cross_contrast_effective_rank_mean": (
            float(np.mean(contrast_effective_rank_values)) if contrast_effective_rank_values else None
        ),
        "cross_contrast_condition_log_mean": (
            float(np.mean(contrast_condition_logs)) if contrast_condition_logs else None
        ),
        "cross_all_supported_rank": all_rank_info["rank"],
        "cross_all_supported_effective_rank": all_rank_info["effective_rank"],
        "cross_all_supported_condition_number_log": all_rank_info["condition_number_log"],
        "cross_edge_participation_gini": positive_gini(bottleneck_edge_counts),
        "cross_bottleneck_edge_fraction_095": (
            float(np.mean(bottleneck_edge_counts >= 0.95 * np.max(bottleneck_edge_counts)))
            if bottleneck_edge_counts.size and np.max(bottleneck_edge_counts) > 0.0
            else 0.0
        ),
        "cross_per_root_pair": per_root_pair,
    }


def cross_root_metrics_for_topology(
    n_nodes: int,
    edges: Sequence[Sequence[int]],
    input_mask: np.ndarray,
    n_context: int,
    z_dim: int,
    max_trees_per_root: int | None = None,
    max_pairs_per_root_pair: int | None = 50000,
) -> dict[str, Any]:
    edge_tuple = normalize_edges(n_nodes, edges)
    mask = np.asarray(input_mask, dtype=float)
    if mask.shape != (len(edge_tuple), (n_context + 1) * z_dim):
        raise ValueError(
            f"input_mask shape {mask.shape} incompatible with edges={len(edge_tuple)}, "
            f"n_context={n_context}, z_dim={z_dim}"
        )
    mats = topology_matrices(n_nodes, edge_tuple, max_trees_per_root=max_trees_per_root)
    tree_roots, tree_incidence = tree_table_from_arborescences(mats["arborescences"], len(edge_tuple))
    exact_counts = tree_counts_by_determinant(n_nodes, edge_tuple)
    enumerated_counts = [int(np.sum(tree_roots == root)) for root in range(n_nodes)]
    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(len(edge_tuple)),
        "n_context": int(n_context),
        "z_dim": int(z_dim),
        "n_trees_enumerated": int(tree_incidence.shape[0]),
        "tree_count_exact_total": int(sum(exact_counts)),
        "tree_count_enumerated_total": int(sum(enumerated_counts)),
        "tree_count_exact_by_root": [int(value) for value in exact_counts],
        "tree_count_enumerated_by_root": enumerated_counts,
        "tree_count_balance_gini": positive_gini(enumerated_counts),
        **cross_root_tree_contrast_summary(
            tree_incidence,
            tree_roots,
            mask,
            n_context=n_context,
            z_dim=z_dim,
            max_pairs_per_root_pair=max_pairs_per_root_pair,
        ),
    }


def _load_topology_payload(topology_path: Path, mask_path: Path | None) -> tuple[int, list[list[int]], np.ndarray]:
    topology = json.loads(topology_path.read_text())
    if mask_path is not None:
        mask_payload = json.loads(mask_path.read_text())
        edges = mask_payload.get("edges", topology.get("edges"))
        return int(mask_payload.get("n_nodes", topology["n_nodes"])), edges, np.asarray(mask_payload["input_mask"], dtype=float)
    return int(topology["n_nodes"]), topology["edges"], np.asarray(topology["input_mask"], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-json", required=True)
    parser.add_argument("--mask-json", default=None)
    parser.add_argument("--n-context", type=int, required=True)
    parser.add_argument("--z-dim", type=int, required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--max-trees-per-root", type=int, default=None)
    parser.add_argument("--max-pairs-per-root-pair", type=int, default=50000)
    args = parser.parse_args()

    n_nodes, edges, mask = _load_topology_payload(Path(args.topology_json), Path(args.mask_json) if args.mask_json else None)
    metrics = cross_root_metrics_for_topology(
        n_nodes,
        edges,
        mask,
        n_context=args.n_context,
        z_dim=args.z_dim,
        max_trees_per_root=args.max_trees_per_root,
        max_pairs_per_root_pair=args.max_pairs_per_root_pair,
    )
    payload = json.dumps(json_ready(metrics), indent=2, sort_keys=True) + "\n"
    if args.out_json:
        Path(args.out_json).write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
