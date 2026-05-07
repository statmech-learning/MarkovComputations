"""Sampled branch-margin capacity probes for first-order topology ICL.

This module is a deliberately conservative bridge between coarse topology
rank metrics and the harder branch-margin theory proposed for the next phase.

``d_rel`` asks how many relative tree directions are available. ICL asks a
more specific question: can the available relative tree directions support
query/context comparison branches?  The exact capacity problem,

    max_{K,B} min_z q_{label*(z)} - max_{label != label*(z)} q_label(z),

is nonconvex once tree max/log-sum-exp structure is included.  The probe here
does not solve that full problem. Instead it:

1. Computes per-branch, per-coordinate common context/query support in the
   relative tree-contrast map.
2. Samples exact-copy branch data.
3. Builds comparison features ``-(z_i - z_q)^2`` only where that branch has
   common tree-contrast support.
4. Reports an oracle comparison margin and a norm-controlled linear ridge
   classifier margin on those topology-gated features.

This is intended as a pre-training capacity proxy to compare against ``d_rel``
and masked tree geometry, not as proof of full CRN expressivity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from input_mask_utils import load_input_mask_json
from topology_metrics import (
    centered_tree_matrix,
    compute_topology_metrics,
    graph_from_family,
    incidence_matrix,
    normalize_edges,
    subspace_intersection_rank,
    topology_matrices,
)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_edge_json(path: str) -> Tuple[int, Tuple[Tuple[int, int], ...], str]:
    with open(path) as handle:
        payload = json.load(handle)
    n_nodes = int(payload["n_nodes"])
    edges = normalize_edges(n_nodes, payload["edges"])
    return n_nodes, edges, str(payload.get("name", Path(path).stem))


def coordinate_common_rank_matrix(
    D_matrix: np.ndarray,
    input_mask: Optional[np.ndarray],
    n_context: int,
    z_dim: int,
    tol: float = 1e-9,
) -> np.ndarray:
    """Return common context/query relative-rank support per branch/dimension."""

    if n_context <= 0 or z_dim <= 0:
        raise ValueError("n_context and z_dim must be positive")
    p = (n_context + 1) * z_dim
    if input_mask is not None:
        mask = np.asarray(input_mask, dtype=float)
        if mask.shape != (D_matrix.shape[1], p):
            raise ValueError(
                f"input_mask shape {mask.shape} incompatible with D {D_matrix.shape} and p={p}"
            )
    else:
        mask = None

    ranks = np.zeros((n_context, z_dim), dtype=int)
    query_offset = n_context * z_dim
    for branch in range(n_context):
        context_offset = branch * z_dim
        for dim in range(z_dim):
            context_idx = context_offset + dim
            query_idx = query_offset + dim
            if mask is None:
                context_map = D_matrix
                query_map = D_matrix
            else:
                context_map = D_matrix * mask[:, context_idx][None, :]
                query_map = D_matrix * mask[:, query_idx][None, :]
            ranks[branch, dim] = subspace_intersection_rank(
                context_map,
                query_map,
                tol=tol,
            )
    return ranks


def sample_exact_copy_branches(
    n_samples: int,
    n_context: int,
    z_dim: int,
    seed: int = 0,
    query_noise: float = 0.0,
    context_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample simple exact-copy branch data in flattened ``(N+1)D`` form."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    contexts = rng.normal(loc=0.0, scale=context_scale, size=(n_samples, n_context, z_dim))
    labels = rng.integers(0, n_context, size=n_samples)
    queries = contexts[np.arange(n_samples), labels].copy()
    if query_noise:
        queries += rng.normal(loc=0.0, scale=query_noise, size=queries.shape)
    z = np.concatenate([contexts.reshape(n_samples, n_context * z_dim), queries], axis=1)
    return z, labels.astype(int)


def comparison_feature_matrix(
    z: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Build topology-gated squared-distance branch comparison features."""

    z = np.asarray(z, dtype=float)
    support = np.asarray(support, dtype=bool)
    if support.ndim != 2:
        raise ValueError("support must have shape (n_context, z_dim)")
    n_context, z_dim = support.shape
    expected_p = (n_context + 1) * z_dim
    if z.ndim != 2 or z.shape[1] != expected_p:
        raise ValueError(f"z must have shape (n_samples, {expected_p})")

    contexts = z[:, : n_context * z_dim].reshape(z.shape[0], n_context, z_dim)
    query = z[:, n_context * z_dim :].reshape(z.shape[0], 1, z_dim)
    squared_diff = (contexts - query) ** 2
    features = -squared_diff * support[None, :, :]
    return features.reshape(z.shape[0], n_context * z_dim)


def weighted_comparison_feature_matrix(
    z: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Build rank-weighted squared-distance branch comparison features."""

    z = np.asarray(z, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError("weights must have shape (n_context, z_dim)")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    n_context, z_dim = weights.shape
    expected_p = (n_context + 1) * z_dim
    if z.ndim != 2 or z.shape[1] != expected_p:
        raise ValueError(f"z must have shape (n_samples, {expected_p})")

    contexts = z[:, : n_context * z_dim].reshape(z.shape[0], n_context, z_dim)
    query = z[:, n_context * z_dim :].reshape(z.shape[0], 1, z_dim)
    squared_diff = (contexts - query) ** 2
    features = -squared_diff * weights[None, :, :]
    return features.reshape(z.shape[0], n_context * z_dim)


def normalized_rank_weights(common_ranks: np.ndarray) -> np.ndarray:
    """Normalize common-rank support to [0, 1] while preserving zeros."""

    ranks = np.asarray(common_ranks, dtype=float)
    if ranks.ndim != 2:
        raise ValueError("common_ranks must have shape (n_context, z_dim)")
    max_rank = float(np.max(ranks)) if ranks.size else 0.0
    if max_rank <= 0:
        return np.zeros_like(ranks, dtype=float)
    return ranks / max_rank


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    min_value = float(np.min(arr))
    if min_value < 0:
        arr = arr - min_value
    total = float(np.sum(arr))
    if total <= 1e-12:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    weighted = float(np.sum((np.arange(n) + 1) * arr))
    return float((2.0 * weighted / (n * total)) - ((n + 1.0) / n))


def rank_geometry_summary(common_ranks: np.ndarray) -> dict:
    """Summarize rank-weight geometry beyond Boolean support."""

    ranks = np.asarray(common_ranks, dtype=float)
    if ranks.ndim != 2:
        raise ValueError("common_ranks must have shape (n_context, z_dim)")
    if ranks.size == 0:
        return {
            "rank_mass_min": 0.0,
            "rank_mass_mean": 0.0,
            "rank_mass_max": 0.0,
            "rank_mass_gini": 0.0,
            "rank_dim_mass_min": 0.0,
            "rank_dim_mass_mean": 0.0,
            "rank_dim_mass_max": 0.0,
            "rank_dim_mass_gini": 0.0,
            "rank_nonzero_fraction": 0.0,
            "rank_weight_entropy": 0.0,
            "rank_weight_effective_entries": 0.0,
        }
    branch_mass = ranks.sum(axis=1)
    dim_mass = ranks.sum(axis=0)
    total = float(ranks.sum())
    if total > 1e-12:
        probs = ranks.reshape(-1) / total
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs)))
        effective = float(np.exp(entropy))
    else:
        entropy = 0.0
        effective = 0.0
    return {
        "rank_mass_min": float(np.min(branch_mass)) if branch_mass.size else 0.0,
        "rank_mass_mean": float(np.mean(branch_mass)) if branch_mass.size else 0.0,
        "rank_mass_max": float(np.max(branch_mass)) if branch_mass.size else 0.0,
        "rank_mass_gini": _gini(branch_mass),
        "rank_dim_mass_min": float(np.min(dim_mass)) if dim_mass.size else 0.0,
        "rank_dim_mass_mean": float(np.mean(dim_mass)) if dim_mass.size else 0.0,
        "rank_dim_mass_max": float(np.max(dim_mass)) if dim_mass.size else 0.0,
        "rank_dim_mass_gini": _gini(dim_mass),
        "rank_nonzero_fraction": float(np.mean(ranks > 0)),
        "rank_weight_entropy": entropy,
        "rank_weight_effective_entries": effective,
    }


def oracle_branch_scores(
    features: np.ndarray,
    n_context: int,
    z_dim: int,
    support: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score each branch by summing its supported comparison features."""

    arr = np.asarray(features, dtype=float).reshape(features.shape[0], n_context, z_dim)
    scores = arr.sum(axis=2)
    if support is not None:
        branch_has_support = np.asarray(support, dtype=bool).any(axis=1)
        scores[:, ~branch_has_support] = -np.inf
    return scores


def multiclass_margins(scores: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.ndim != 2:
        raise ValueError("scores must have shape (n_samples, n_classes)")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have matching sample counts")
    true_scores = scores[np.arange(scores.shape[0]), labels]
    masked = scores.copy()
    masked[np.arange(scores.shape[0]), labels] = -np.inf
    with np.errstate(invalid="ignore"):
        return true_scores - masked.max(axis=1)


def accuracy_from_scores(scores: np.ndarray, labels: Sequence[int]) -> float:
    labels = np.asarray(labels, dtype=int)
    return float(np.mean(np.argmax(scores, axis=1) == labels))


def fit_ridge_multiclass(
    features: np.ndarray,
    labels: Sequence[int],
    n_classes: int,
    ridge: float = 1e-3,
    l2_radius: Optional[float] = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic norm-controlled linear multiclass ridge model."""

    X = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if X.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if labels.shape != (X.shape[0],):
        raise ValueError("labels must have shape (n_samples,)")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    Y = -np.ones((X.shape[0], n_classes), dtype=float) / max(n_classes - 1, 1)
    Y[np.arange(X.shape[0]), labels] = 1.0
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=float)], axis=1)
    gram = X_aug.T @ X_aug
    if ridge:
        regularizer = np.eye(gram.shape[0], dtype=float) * ridge
        regularizer[-1, -1] = 0.0
        gram = gram + regularizer
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        rhs = X_aug.T @ Y
    try:
        weights_aug = np.linalg.solve(gram + 1e-12 * np.eye(gram.shape[0], dtype=float), rhs)
    except np.linalg.LinAlgError:
        weights_aug = np.linalg.lstsq(gram, rhs, rcond=1e-10)[0]
    if not np.all(np.isfinite(weights_aug)):
        weights_aug = np.nan_to_num(weights_aug, nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights_aug[:-1, :]
    bias = weights_aug[-1, :]

    if l2_radius is not None:
        norm = float(np.linalg.norm(weights))
        if norm > l2_radius > 0:
            scale = l2_radius / norm
            weights = weights * scale
            bias = bias * scale
    return weights, bias


def linear_scores(features: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scores = np.asarray(features, dtype=float) @ weights + bias
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


def summarize_margin_scores(scores: np.ndarray, labels: Sequence[int], prefix: str) -> dict:
    margins = multiclass_margins(scores, labels)
    finite = margins[np.isfinite(margins)]
    if finite.size:
        margin_mean = float(np.mean(finite))
        margin_min = float(np.min(finite))
        margin_p10 = float(np.quantile(finite, 0.10))
    else:
        margin_mean = None
        margin_min = None
        margin_p10 = None
    return {
        f"{prefix}_accuracy": accuracy_from_scores(scores, labels),
        f"{prefix}_margin_mean": margin_mean,
        f"{prefix}_margin_min": margin_min,
        f"{prefix}_margin_p10": margin_p10,
        f"{prefix}_margin_finite_fraction": float(finite.size / margins.size) if margins.size else 0.0,
    }


def branch_margin_capacity(
    n_nodes: int,
    edges: Iterable[Sequence[int]],
    n_context: int,
    z_dim: int,
    input_mask: Optional[np.ndarray] = None,
    train_samples: int = 2000,
    test_samples: int = 2000,
    seed: int = 0,
    query_noise: float = 0.0,
    ridge: float = 1e-3,
    l2_radius: Optional[float] = 1.0,
    max_trees_per_root: Optional[int] = None,
) -> dict:
    """Compute topology-gated branch-margin capacity proxy metrics."""

    edge_tuple = normalize_edges(n_nodes, edges)
    p = (n_context + 1) * z_dim
    mats = topology_matrices(
        n_nodes,
        edge_tuple,
        max_trees_per_root=max_trees_per_root,
    )
    D_centered = centered_tree_matrix(mats["M"])
    common_ranks = coordinate_common_rank_matrix(
        D_centered,
        input_mask=input_mask,
        n_context=n_context,
        z_dim=z_dim,
    )
    support = common_ranks > 0

    train_z, train_labels = sample_exact_copy_branches(
        train_samples,
        n_context,
        z_dim,
        seed=seed,
        query_noise=query_noise,
    )
    test_z, test_labels = sample_exact_copy_branches(
        test_samples,
        n_context,
        z_dim,
        seed=seed + 1,
        query_noise=query_noise,
    )
    train_features = comparison_feature_matrix(train_z, support)
    test_features = comparison_feature_matrix(test_z, support)
    rank_weights = normalized_rank_weights(common_ranks)
    train_rank_features = weighted_comparison_feature_matrix(train_z, rank_weights)
    test_rank_features = weighted_comparison_feature_matrix(test_z, rank_weights)

    oracle_train_scores = oracle_branch_scores(train_features, n_context, z_dim, support=support)
    oracle_test_scores = oracle_branch_scores(test_features, n_context, z_dim, support=support)
    weights, bias = fit_ridge_multiclass(
        train_features,
        train_labels,
        n_classes=n_context,
        ridge=ridge,
        l2_radius=l2_radius,
    )
    linear_train_scores = linear_scores(train_features, weights, bias)
    linear_test_scores = linear_scores(test_features, weights, bias)
    weighted_oracle_train_scores = oracle_branch_scores(
        train_rank_features,
        n_context,
        z_dim,
        support=support,
    )
    weighted_oracle_test_scores = oracle_branch_scores(
        test_rank_features,
        n_context,
        z_dim,
        support=support,
    )
    weighted_weights, weighted_bias = fit_ridge_multiclass(
        train_rank_features,
        train_labels,
        n_classes=n_context,
        ridge=ridge,
        l2_radius=l2_radius,
    )
    weighted_linear_train_scores = linear_scores(train_rank_features, weighted_weights, weighted_bias)
    weighted_linear_test_scores = linear_scores(test_rank_features, weighted_weights, weighted_bias)

    topology = compute_topology_metrics(
        n_nodes=n_nodes,
        edges=edge_tuple,
        p=p,
        input_mask=input_mask,
        n_context=n_context,
        z_dim=z_dim,
        max_trees_per_root=max_trees_per_root,
    )
    branch_support_counts = support.sum(axis=1)
    result = {
        "probe_kind": "topology_gated_squared_comparison_margin",
        "n_nodes": n_nodes,
        "n_edges": len(edge_tuple),
        "n_context": n_context,
        "z_dim": z_dim,
        "p": p,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "seed": seed,
        "query_noise": query_noise,
        "ridge": ridge,
        "l2_radius": l2_radius,
        "common_rank_by_branch_dim": common_ranks.tolist(),
        "support_by_branch_dim": support.astype(int).tolist(),
        "rank_weight_by_branch_dim": rank_weights.tolist(),
        "support_fraction": float(np.mean(support)) if support.size else 0.0,
        "support_min": int(branch_support_counts.min()) if branch_support_counts.size else 0,
        "support_mean": float(branch_support_counts.mean()) if branch_support_counts.size else 0.0,
        "support_max": int(branch_support_counts.max()) if branch_support_counts.size else 0,
        "linear_weight_norm": float(np.linalg.norm(weights)),
        "weighted_linear_weight_norm": float(np.linalg.norm(weighted_weights)),
    }
    result.update(rank_geometry_summary(common_ranks))
    for key in [
        "d_rel",
        "rank_D",
        "effective_rank_D",
        "effective_rank_D_masked",
        "condition_number_D_masked",
        "comparison_branch_common_d_rel_min",
        "comparison_branch_common_d_rel_mean",
        "comparison_branch_common_d_rel_gini",
        "comparison_branch_d_rel_min",
        "comparison_branch_d_rel_gini",
    ]:
        result[key] = topology.get(key)
    result.update(summarize_margin_scores(oracle_train_scores, train_labels, "oracle_train"))
    result.update(summarize_margin_scores(oracle_test_scores, test_labels, "oracle_test"))
    result.update(summarize_margin_scores(linear_train_scores, train_labels, "linear_train"))
    result.update(summarize_margin_scores(linear_test_scores, test_labels, "linear_test"))
    result.update(
        summarize_margin_scores(
            weighted_oracle_train_scores,
            train_labels,
            "rank_weighted_oracle_train",
        )
    )
    result.update(
        summarize_margin_scores(
            weighted_oracle_test_scores,
            test_labels,
            "rank_weighted_oracle_test",
        )
    )
    result.update(
        summarize_margin_scores(
            weighted_linear_train_scores,
            train_labels,
            "rank_weighted_linear_train",
        )
    )
    result.update(
        summarize_margin_scores(
            weighted_linear_test_scores,
            test_labels,
            "rank_weighted_linear_test",
        )
    )
    return result


def markdown_report(result: dict) -> str:
    lines = [
        "# Branch-Margin Capacity Probe",
        "",
        "This is a topology-gated comparison-feature proxy, not the full nonconvex CRN capacity.",
        "",
        "## Topology",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| n_nodes | {result['n_nodes']} |",
        f"| n_edges | {result['n_edges']} |",
        f"| n_context | {result['n_context']} |",
        f"| z_dim | {result['z_dim']} |",
        f"| d_rel | {result.get('d_rel')} |",
        f"| common branch d_rel min | {result.get('comparison_branch_common_d_rel_min')} |",
        f"| support fraction | {result['support_fraction']:.3f} |",
        f"| support min/mean/max | {result['support_min']} / {result['support_mean']:.2f} / {result['support_max']} |",
        f"| rank mass min/mean/max | {result['rank_mass_min']:.2f} / {result['rank_mass_mean']:.2f} / {result['rank_mass_max']:.2f} |",
        f"| rank mass gini | {result['rank_mass_gini']:.3f} |",
        "",
        "## Margins",
        "",
        "| model | accuracy | mean margin | p10 margin | min margin |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for prefix, label in [
        ("oracle_test", "oracle comparison"),
        ("rank_weighted_oracle_test", "rank-weighted oracle"),
        ("linear_test", "linear ridge"),
        ("rank_weighted_linear_test", "rank-weighted linear ridge"),
        ("oracle_train", "oracle comparison train"),
        ("linear_train", "linear ridge train"),
    ]:
        lines.append(
            "| "
            + label
            + f" | {100.0 * result[prefix + '_accuracy']:.2f} |"
            + f" {result[prefix + '_margin_mean']:.4f} |"
            + f" {result[prefix + '_margin_p10']:.4f} |"
            + f" {result[prefix + '_margin_min']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "Use this probe as a branch-specific pre-training predictor to compare against `d_rel`.",
            "A high score means the topology/mask supports the exact-copy comparison features used",
            "by the sampled ICL branches. It does not prove that gradient descent will find a CRN",
            "implementation, and it does not apply to nonlinear autocatalytic or WTA CRNs.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge_json", type=str, default=None)
    parser.add_argument(
        "--topology_family",
        type=str,
        default="random_sc",
        choices=[
            "complete",
            "directed_cycle",
            "bidirected_cycle",
            "cycle_chords",
            "random_sc",
            "hub_spoke",
            "two_module",
            "degree_balanced",
            "bottleneck_bridge",
            "redundant_paths",
        ],
    )
    parser.add_argument("--n_nodes", type=int, default=6)
    parser.add_argument("--n_edges", type=int, default=20)
    parser.add_argument("--topology_seed", type=int, default=0)
    parser.add_argument("--input_mask_json", type=str, default=None)
    parser.add_argument("--n_context", type=int, default=4)
    parser.add_argument("--z_dim", type=int, default=4)
    parser.add_argument("--train_samples", type=int, default=2000)
    parser.add_argument("--test_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query_noise", type=float, default=0.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--l2_radius", type=float, default=1.0)
    parser.add_argument("--max_trees_per_root", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.edge_json:
        n_nodes, edges, topology_name = load_edge_json(args.edge_json)
    else:
        n_edges = None if args.topology_family in {"complete", "directed_cycle", "bidirected_cycle"} else args.n_edges
        spec = graph_from_family(
            args.topology_family,
            n_nodes=args.n_nodes,
            n_edges=n_edges,
            seed=args.topology_seed,
        )
        n_nodes, edges, topology_name = spec.n_nodes, spec.edges, spec.name

    p = (args.n_context + 1) * args.z_dim
    input_mask = None
    input_mask_name = "full"
    if args.input_mask_json:
        input_mask, metadata = load_input_mask_json(args.input_mask_json, n_nodes, edges, p)
        input_mask_name = str(metadata.get("name", Path(args.input_mask_json).stem))

    result = branch_margin_capacity(
        n_nodes=n_nodes,
        edges=edges,
        n_context=args.n_context,
        z_dim=args.z_dim,
        input_mask=input_mask,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
        query_noise=args.query_noise,
        ridge=args.ridge,
        l2_radius=args.l2_radius,
        max_trees_per_root=args.max_trees_per_root,
    )
    result["topology_name"] = topology_name
    result["input_mask_name"] = input_mask_name

    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(_json_ready(result), handle, indent=2, sort_keys=True)
            handle.write("\n")
    if args.output_md:
        with open(args.output_md, "w") as handle:
            handle.write(markdown_report(result))
    if not args.output_json and not args.output_md:
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
