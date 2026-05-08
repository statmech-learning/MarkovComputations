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
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from input_mask_utils import load_input_mask_json
from topology_metrics import (
    centered_tree_matrix,
    compute_topology_metrics,
    graph_from_family,
    incidence_matrix,
    normalize_edges,
    subspace_intersection_rank,
    svd_metrics,
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


def rooted_common_rank_tensor(
    arborescences: Mapping[int, Sequence[Sequence[int]]],
    n_edges: int,
    input_mask: Optional[np.ndarray],
    n_context: int,
    z_dim: int,
    tol: float = 1e-9,
) -> np.ndarray:
    """Common context/query support inside each rooted tree polytope.

    Global relative tree rank can be high because it mixes trees from all
    roots.  In the tropical matrix-tree view, however, each concentration
    numerator is controlled by the normal fan of one rooted-tree polytope.
    This tensor measures, root by root, whether that root's tree-difference
    span contains common context/query directions for each ICL branch.
    """

    rows = []
    for root in sorted(arborescences):
        M_root = incidence_matrix({root: arborescences[root]}, n_edges)
        D_root = centered_tree_matrix(M_root)
        rows.append(
            coordinate_common_rank_matrix(
                D_root,
                input_mask=input_mask,
                n_context=n_context,
                z_dim=z_dim,
                tol=tol,
            )
        )
    if not rows:
        return np.zeros((0, n_context, z_dim), dtype=int)
    return np.stack(rows, axis=0).astype(int)


def rooted_polytope_support_summary(rooted_ranks: np.ndarray) -> dict:
    """Summarize rooted-tree-polytope branch support."""

    ranks = np.asarray(rooted_ranks, dtype=float)
    if ranks.ndim != 3:
        raise ValueError("rooted_ranks must have shape (n_roots, n_context, z_dim)")
    if ranks.size == 0 or ranks.shape[0] == 0:
        return {
            "rooted_polytope_n_roots": int(ranks.shape[0]) if ranks.ndim == 3 else 0,
            "rooted_polytope_common_rank_total": 0.0,
            "rooted_polytope_common_rank_mean": 0.0,
            "rooted_polytope_common_rank_max": 0.0,
            "rooted_polytope_supported_branch_dim_fraction": 0.0,
            "rooted_polytope_branch_root_support_min": 0.0,
            "rooted_polytope_branch_root_support_mean": 0.0,
            "rooted_polytope_branch_root_support_max": 0.0,
            "rooted_polytope_branch_root_support_gini": 0.0,
            "rooted_polytope_branch_best_rank_min": 0.0,
            "rooted_polytope_branch_best_rank_mean": 0.0,
            "rooted_polytope_branch_best_rank_max": 0.0,
            "rooted_polytope_branch_best_rank_gini": 0.0,
            "rooted_polytope_root_rank_mass_min": 0.0,
            "rooted_polytope_root_rank_mass_mean": 0.0,
            "rooted_polytope_root_rank_mass_max": 0.0,
            "rooted_polytope_root_rank_mass_gini": 0.0,
            "rooted_polytope_root_rank_mass_effective": 0.0,
        }

    root_branch_mass = ranks.sum(axis=2)
    branch_root_support = np.sum(root_branch_mass > 0, axis=0).astype(float)
    branch_best_rank = np.max(root_branch_mass, axis=0)
    root_mass = ranks.sum(axis=(1, 2))
    branch_dim_best = np.max(ranks, axis=0)
    total = float(root_mass.sum())
    if total > 1e-12:
        probs = root_mass / total
        probs = probs[probs > 0]
        root_effective = float(np.exp(-np.sum(probs * np.log(probs))))
    else:
        root_effective = 0.0

    return {
        "rooted_polytope_n_roots": int(ranks.shape[0]),
        "rooted_polytope_common_rank_total": float(np.sum(ranks)),
        "rooted_polytope_common_rank_mean": float(np.mean(ranks)),
        "rooted_polytope_common_rank_max": float(np.max(ranks)),
        "rooted_polytope_supported_branch_dim_fraction": float(np.mean(branch_dim_best > 0)),
        "rooted_polytope_branch_root_support_min": float(np.min(branch_root_support)),
        "rooted_polytope_branch_root_support_mean": float(np.mean(branch_root_support)),
        "rooted_polytope_branch_root_support_max": float(np.max(branch_root_support)),
        "rooted_polytope_branch_root_support_gini": _gini(branch_root_support),
        "rooted_polytope_branch_best_rank_min": float(np.min(branch_best_rank)),
        "rooted_polytope_branch_best_rank_mean": float(np.mean(branch_best_rank)),
        "rooted_polytope_branch_best_rank_max": float(np.max(branch_best_rank)),
        "rooted_polytope_branch_best_rank_gini": _gini(branch_best_rank),
        "rooted_polytope_root_rank_mass_min": float(np.min(root_mass)),
        "rooted_polytope_root_rank_mass_mean": float(np.mean(root_mass)),
        "rooted_polytope_root_rank_mass_max": float(np.max(root_mass)),
        "rooted_polytope_root_rank_mass_gini": _gini(root_mass),
        "rooted_polytope_root_rank_mass_effective": root_effective,
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


def _stable_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_values = np.max(values, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        shifted = np.exp(values - max_values)
        summed = np.sum(shifted, axis=axis, keepdims=True)
        out = max_values + np.log(summed)
    return np.squeeze(out, axis=axis)


def tropical_root_feature_matrix(
    z: np.ndarray,
    arborescences: Mapping[int, Sequence[Sequence[int]]],
    n_edges: int,
    edge_projections: np.ndarray,
    edge_bias: Optional[np.ndarray] = None,
    mode: str = "max",
    normalize_roots: bool = True,
) -> np.ndarray:
    """Map samples to root log-weight features using sampled tree projections.

    This is a topology-aware random feature map for the tropical/tree-polytope
    view: each root feature is a max or log-sum-exp over rooted tree scores
    ``beta_T + Theta_T^T z`` where ``Theta_T`` is induced by edge projections.
    """

    z = np.asarray(z, dtype=float)
    edge_projections = np.asarray(edge_projections, dtype=float)
    if edge_projections.ndim != 2 or edge_projections.shape[0] != n_edges:
        raise ValueError(f"edge_projections must have shape ({n_edges}, p)")
    if z.ndim != 2 or z.shape[1] != edge_projections.shape[1]:
        raise ValueError("z and edge_projections have incompatible input dimensions")
    if mode not in {"max", "logsumexp"}:
        raise ValueError("mode must be 'max' or 'logsumexp'")
    if edge_bias is None:
        bias = np.zeros(n_edges, dtype=float)
    else:
        bias = np.asarray(edge_bias, dtype=float)
        if bias.shape != (n_edges,):
            raise ValueError(f"edge_bias must have shape ({n_edges},)")

    root_features = []
    for root in sorted(arborescences):
        trees = list(arborescences[root])
        if not trees:
            root_features.append(np.full(z.shape[0], -np.inf, dtype=float))
            continue
        M_root = np.ascontiguousarray(incidence_matrix({root: trees}, n_edges), dtype=float)
        projections = np.ascontiguousarray(edge_projections, dtype=float)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            theta = M_root @ projections
            beta = M_root @ bias
            tree_scores = z @ theta.T + beta[None, :]
        if mode == "max":
            root_features.append(np.max(tree_scores, axis=1))
        else:
            root_features.append(_stable_logsumexp(tree_scores, axis=1))

    features = np.column_stack(root_features)
    if normalize_roots:
        normalizer = _stable_logsumexp(features, axis=1)
        features = features - normalizer[:, None]
    if not np.all(np.isfinite(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=-1e6)
    return features


def tropical_active_tree_assignments(
    z: np.ndarray,
    arborescences: Mapping[int, Sequence[Sequence[int]]],
    n_edges: int,
    edge_projections: np.ndarray,
    edge_bias: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return active roots and active rooted trees for tropical tree scores."""

    z = np.asarray(z, dtype=float)
    edge_projections = np.asarray(edge_projections, dtype=float)
    if edge_projections.ndim != 2 or edge_projections.shape[0] != n_edges:
        raise ValueError(f"edge_projections must have shape ({n_edges}, p)")
    if z.ndim != 2 or z.shape[1] != edge_projections.shape[1]:
        raise ValueError("z and edge_projections have incompatible input dimensions")
    if edge_bias is None:
        bias = np.zeros(n_edges, dtype=float)
    else:
        bias = np.asarray(edge_bias, dtype=float)
        if bias.shape != (n_edges,):
            raise ValueError(f"edge_bias must have shape ({n_edges},)")

    root_scores = []
    active_by_root = []
    tree_offset = 0
    for root in sorted(arborescences):
        trees = list(arborescences[root])
        if not trees:
            root_scores.append(np.full(z.shape[0], -np.inf, dtype=float))
            active_by_root.append(np.full(z.shape[0], -1, dtype=int))
            continue
        M_root = np.ascontiguousarray(incidence_matrix({root: trees}, n_edges), dtype=float)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            theta = M_root @ edge_projections
            beta = M_root @ bias
            tree_scores = z @ theta.T + beta[None, :]
        local_active = np.argmax(tree_scores, axis=1)
        root_scores.append(np.max(tree_scores, axis=1))
        active_by_root.append(local_active.astype(int) + tree_offset)
        tree_offset += len(trees)

    scores = np.column_stack(root_scores)
    active_tree_by_root = np.column_stack(active_by_root)
    active_root = np.argmax(scores, axis=1).astype(int)
    active_tree = active_tree_by_root[np.arange(z.shape[0]), active_root]
    return scores, active_root, active_tree, active_tree_by_root


def _entropy_from_labels(values: Sequence[int]) -> float:
    arr = np.asarray(values, dtype=int).reshape(-1)
    if arr.size == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    probs = counts.astype(float) / float(arr.size)
    return float(-np.sum(probs * np.log(probs)))


def normalized_mutual_information(labels: Sequence[int], assignments: Sequence[int]) -> float:
    """Symmetric normalized mutual information for discrete assignments."""

    y = np.asarray(labels, dtype=int).reshape(-1)
    a = np.asarray(assignments, dtype=int).reshape(-1)
    if y.shape != a.shape:
        raise ValueError("labels and assignments must have matching shape")
    if y.size == 0:
        return 0.0
    hy = _entropy_from_labels(y)
    ha = _entropy_from_labels(a)
    if hy <= 1e-12 or ha <= 1e-12:
        return 0.0
    _, y_inv = np.unique(y, return_inverse=True)
    _, a_inv = np.unique(a, return_inverse=True)
    table = np.zeros((y_inv.max() + 1, a_inv.max() + 1), dtype=float)
    np.add.at(table, (y_inv, a_inv), 1.0)
    joint = table / float(y.size)
    py = joint.sum(axis=1, keepdims=True)
    pa = joint.sum(axis=0, keepdims=True)
    denom = py @ pa
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))
    return float(2.0 * mi / (hy + ha))


def _branch_assignment_count_min(assignments: Sequence[int], labels: Sequence[int]) -> float:
    y = np.asarray(labels, dtype=int).reshape(-1)
    a = np.asarray(assignments, dtype=int).reshape(-1)
    counts = [len(np.unique(a[y == label])) for label in np.unique(y)]
    return float(min(counts)) if counts else 0.0


def tree_normal_fan_coverage(
    arborescences: Mapping[int, Sequence[Sequence[int]]],
    n_edges: int,
    input_dim: int,
    z: np.ndarray,
    labels: Sequence[int],
    input_mask: Optional[np.ndarray] = None,
    n_trials: int = 8,
    seed: int = 0,
    projection_radius: float = 1.0,
    edge_bias_scale: float = 0.0,
) -> dict:
    """Sample whether rooted-tree normal fans produce branch-specific cells."""

    if n_trials <= 0:
        return {
            "normal_fan_trials": 0,
            "normal_fan_branch_root_nmi_mean": None,
            "normal_fan_branch_tree_nmi_mean": None,
            "normal_fan_branch_tree_nmi_max": None,
            "normal_fan_active_root_count_mean": None,
            "normal_fan_active_tree_count_mean": None,
            "normal_fan_branch_active_tree_count_min_mean": None,
        }
    if input_mask is None:
        mask = np.ones((n_edges, input_dim), dtype=float)
    else:
        mask = np.asarray(input_mask, dtype=float)
        if mask.shape != (n_edges, input_dim):
            raise ValueError(f"input_mask must have shape ({n_edges}, {input_dim})")

    rng = np.random.default_rng(seed)
    root_nmi = []
    tree_nmi = []
    root_counts = []
    tree_counts = []
    branch_tree_count_min = []
    for _ in range(n_trials):
        K = rng.normal(loc=0.0, scale=1.0, size=(n_edges, input_dim)) * mask
        norm = float(np.linalg.norm(K))
        if norm > 1e-12 and projection_radius > 0:
            K = K * (projection_radius / norm)
        if edge_bias_scale:
            edge_bias = rng.normal(loc=0.0, scale=edge_bias_scale, size=n_edges)
        else:
            edge_bias = np.zeros(n_edges, dtype=float)
        _, active_root, active_tree, _ = tropical_active_tree_assignments(
            z,
            arborescences,
            n_edges=n_edges,
            edge_projections=K,
            edge_bias=edge_bias,
        )
        root_nmi.append(normalized_mutual_information(labels, active_root))
        tree_nmi.append(normalized_mutual_information(labels, active_tree))
        root_counts.append(float(len(np.unique(active_root))))
        tree_counts.append(float(len(np.unique(active_tree))))
        branch_tree_count_min.append(_branch_assignment_count_min(active_tree, labels))

    out = {"normal_fan_trials": int(n_trials)}
    out.update(_summarize_trial_values(root_nmi, "normal_fan_branch_root_nmi"))
    out.update(_summarize_trial_values(tree_nmi, "normal_fan_branch_tree_nmi"))
    out.update(_summarize_trial_values(root_counts, "normal_fan_active_root_count"))
    out.update(_summarize_trial_values(tree_counts, "normal_fan_active_tree_count"))
    out.update(_summarize_trial_values(branch_tree_count_min, "normal_fan_branch_active_tree_count_min"))
    return out


def _summarize_trial_values(values: Sequence[float], prefix: str) -> dict:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_max": None,
            f"{prefix}_std": None,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_std": float(np.std(finite)),
    }


def tropical_tree_feature_capacity(
    arborescences: Mapping[int, Sequence[Sequence[int]]],
    n_edges: int,
    input_dim: int,
    train_z: np.ndarray,
    train_labels: Sequence[int],
    test_z: np.ndarray,
    test_labels: Sequence[int],
    input_mask: Optional[np.ndarray] = None,
    n_trials: int = 8,
    seed: int = 0,
    projection_radius: float = 1.0,
    edge_bias_scale: float = 0.0,
    mode: str = "max",
    ridge: float = 1e-3,
    l2_radius: Optional[float] = 1.0,
) -> dict:
    """Random tropical rooted-tree feature separability probe.

    Unlike the squared-distance support probe, this uses the actual rooted-tree
    incidence structure.  For each trial it samples norm-controlled edge
    projections, computes root-wise max/log-sum-exp tree features, and fits a
    linear decoder.  It is still a random lower-bound proxy, not an optimized
    CRN capacity.
    """

    if n_trials <= 0:
        return {
            "tropical_feature_trials": 0,
            "tropical_feature_mode": mode,
            "tropical_projection_radius": projection_radius,
            "tropical_edge_bias_scale": edge_bias_scale,
            "tropical_linear_test_accuracy_mean": None,
            "tropical_linear_test_accuracy_max": None,
            "tropical_linear_test_accuracy_std": None,
            "tropical_linear_test_margin_p10_mean": None,
            "tropical_linear_test_margin_p10_max": None,
            "tropical_linear_test_margin_p10_std": None,
            "tropical_root_feature_effective_rank_mean": None,
            "tropical_root_feature_effective_rank_max": None,
            "tropical_root_feature_effective_rank_std": None,
        }

    if input_mask is None:
        mask = np.ones((n_edges, input_dim), dtype=float)
    else:
        mask = np.asarray(input_mask, dtype=float)
        if mask.shape != (n_edges, input_dim):
            raise ValueError(f"input_mask must have shape ({n_edges}, {input_dim})")

    rng = np.random.default_rng(seed)
    test_accuracies = []
    test_margin_p10 = []
    train_accuracies = []
    effective_ranks = []
    variances = []
    n_classes = int(max(np.max(train_labels), np.max(test_labels))) + 1
    for _ in range(n_trials):
        K = rng.normal(loc=0.0, scale=1.0, size=(n_edges, input_dim)) * mask
        norm = float(np.linalg.norm(K))
        if norm > 1e-12 and projection_radius > 0:
            K = K * (projection_radius / norm)
        if edge_bias_scale:
            edge_bias = rng.normal(loc=0.0, scale=edge_bias_scale, size=n_edges)
        else:
            edge_bias = np.zeros(n_edges, dtype=float)

        train_features = tropical_root_feature_matrix(
            train_z,
            arborescences,
            n_edges=n_edges,
            edge_projections=K,
            edge_bias=edge_bias,
            mode=mode,
            normalize_roots=True,
        )
        test_features = tropical_root_feature_matrix(
            test_z,
            arborescences,
            n_edges=n_edges,
            edge_projections=K,
            edge_bias=edge_bias,
            mode=mode,
            normalize_roots=True,
        )
        weights, bias = fit_ridge_multiclass(
            train_features,
            train_labels,
            n_classes=n_classes,
            ridge=ridge,
            l2_radius=l2_radius,
        )
        train_scores = linear_scores(train_features, weights, bias)
        test_scores = linear_scores(test_features, weights, bias)
        train_summary = summarize_margin_scores(train_scores, train_labels, "train")
        test_summary = summarize_margin_scores(test_scores, test_labels, "test")
        train_accuracies.append(train_summary["train_accuracy"])
        test_accuracies.append(test_summary["test_accuracy"])
        test_margin_p10.append(test_summary["test_margin_p10"])
        centered = train_features - train_features.mean(axis=0, keepdims=True)
        effective_ranks.append(svd_metrics(centered)["effective_rank"])
        variances.append(float(np.var(train_features)))

    out = {
        "tropical_feature_trials": int(n_trials),
        "tropical_feature_mode": mode,
        "tropical_projection_radius": float(projection_radius),
        "tropical_edge_bias_scale": float(edge_bias_scale),
    }
    out.update(_summarize_trial_values(train_accuracies, "tropical_linear_train_accuracy"))
    out.update(_summarize_trial_values(test_accuracies, "tropical_linear_test_accuracy"))
    out.update(_summarize_trial_values(test_margin_p10, "tropical_linear_test_margin_p10"))
    out.update(_summarize_trial_values(effective_ranks, "tropical_root_feature_effective_rank"))
    out.update(_summarize_trial_values(variances, "tropical_root_feature_variance"))
    return out


def branch_margin_by_label(scores: np.ndarray, labels: Sequence[int], prefix: str) -> dict:
    """Summarize the branch-wise margin objective in ``min_b margin_b`` form."""

    margins = multiclass_margins(scores, labels)
    labels = np.asarray(labels, dtype=int)
    branch_means = []
    branch_p10s = []
    branch_mins = []
    for label in sorted(np.unique(labels).tolist()):
        values = margins[labels == label]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        branch_means.append(float(np.mean(finite)))
        branch_p10s.append(float(np.quantile(finite, 0.10)))
        branch_mins.append(float(np.min(finite)))
    finite_all = margins[np.isfinite(margins)]
    return {
        f"{prefix}_branch_margin_mean_min": float(np.min(branch_means)) if branch_means else None,
        f"{prefix}_branch_margin_mean_mean": float(np.mean(branch_means)) if branch_means else None,
        f"{prefix}_branch_margin_p10_min": float(np.min(branch_p10s)) if branch_p10s else None,
        f"{prefix}_branch_margin_p10_mean": float(np.mean(branch_p10s)) if branch_p10s else None,
        f"{prefix}_branch_margin_min_min": float(np.min(branch_mins)) if branch_mins else None,
        f"{prefix}_sample_margin_min": float(np.min(finite_all)) if finite_all.size else None,
        f"{prefix}_sample_margin_mean": float(np.mean(finite_all)) if finite_all.size else None,
    }


def bounded_gamma_star_proxy_capacity(
    arborescences: Mapping[int, Sequence[Sequence[int]]],
    n_edges: int,
    input_dim: int,
    train_z: np.ndarray,
    train_labels: Sequence[int],
    test_z: np.ndarray,
    test_labels: Sequence[int],
    input_mask: Optional[np.ndarray] = None,
    n_trials: int = 32,
    seed: int = 0,
    projection_radius: float = 1.0,
    decoder_radius: float = 1.0,
    edge_bias_radius: float = 0.0,
    mode: str = "max",
    ridge: float = 1e-3,
) -> dict:
    """Bounded lower-bound probe for the graph/mask capacity ``gamma*``.

    The target theory object is ``max_{K,B} min_b margin_b`` under norm
    constraints. This routine is not a global optimizer. It samples
    Frobenius-norm-bounded edge projections respecting ``Omega``, optionally
    samples bounded edge-bias vectors, fits a decoder with bounded Frobenius
    norm, and reports the best branch-wise margin found. It is therefore a
    reproducible lower-bound proxy for comparing against ``d_rel``.
    """

    if n_trials <= 0:
        return {
            "gamma_star_proxy_trials": 0,
            "gamma_star_projection_radius": float(projection_radius),
            "gamma_star_decoder_radius": float(decoder_radius),
            "gamma_star_edge_bias_radius": float(edge_bias_radius),
            "gamma_star_mode": mode,
            "gamma_star_selected_train_branch_margin_p10_min": None,
            "gamma_star_selected_test_branch_margin_p10_min": None,
            "gamma_star_selected_test_accuracy": None,
        }
    if input_mask is None:
        mask = np.ones((n_edges, input_dim), dtype=float)
    else:
        mask = np.asarray(input_mask, dtype=float)
        if mask.shape != (n_edges, input_dim):
            raise ValueError(f"input_mask must have shape ({n_edges}, {input_dim})")

    rng = np.random.default_rng(seed)
    n_classes = int(max(np.max(train_labels), np.max(test_labels))) + 1
    selected = None
    train_gamma_p10 = []
    test_gamma_p10 = []
    test_gamma_mean = []
    test_accuracy = []
    decoder_norms = []
    projection_norms = []

    for trial_idx in range(n_trials):
        K = rng.normal(loc=0.0, scale=1.0, size=(n_edges, input_dim)) * mask
        k_norm = float(np.linalg.norm(K))
        if k_norm > 1e-12 and projection_radius > 0:
            K = K * (projection_radius / k_norm)
            k_norm = float(np.linalg.norm(K))
        if edge_bias_radius > 0:
            edge_bias = rng.normal(loc=0.0, scale=1.0, size=n_edges)
            bias_norm = float(np.linalg.norm(edge_bias))
            if bias_norm > 1e-12:
                edge_bias = edge_bias * (edge_bias_radius / bias_norm)
        else:
            edge_bias = np.zeros(n_edges, dtype=float)

        train_features = tropical_root_feature_matrix(
            train_z,
            arborescences,
            n_edges=n_edges,
            edge_projections=K,
            edge_bias=edge_bias,
            mode=mode,
            normalize_roots=True,
        )
        test_features = tropical_root_feature_matrix(
            test_z,
            arborescences,
            n_edges=n_edges,
            edge_projections=K,
            edge_bias=edge_bias,
            mode=mode,
            normalize_roots=True,
        )
        weights, bias = fit_ridge_multiclass(
            train_features,
            train_labels,
            n_classes=n_classes,
            ridge=ridge,
            l2_radius=decoder_radius,
        )
        train_scores = linear_scores(train_features, weights, bias)
        test_scores = linear_scores(test_features, weights, bias)
        train_branch = branch_margin_by_label(train_scores, train_labels, "train")
        test_branch = branch_margin_by_label(test_scores, test_labels, "test")
        train_key = train_branch["train_branch_margin_p10_min"]
        test_key = test_branch["test_branch_margin_p10_min"]
        test_mean_key = test_branch["test_branch_margin_mean_min"]
        acc = accuracy_from_scores(test_scores, test_labels)
        decoder_norm = float(np.linalg.norm(weights))

        if train_key is not None and np.isfinite(train_key):
            train_gamma_p10.append(float(train_key))
        if test_key is not None and np.isfinite(test_key):
            test_gamma_p10.append(float(test_key))
        if test_mean_key is not None and np.isfinite(test_mean_key):
            test_gamma_mean.append(float(test_mean_key))
        test_accuracy.append(float(acc))
        decoder_norms.append(decoder_norm)
        projection_norms.append(k_norm)

        score_for_selection = -np.inf if train_key is None else float(train_key)
        if selected is None or score_for_selection > selected["selection_score"]:
            selected = {
                "selection_score": score_for_selection,
                "trial": int(trial_idx),
                "train": train_branch,
                "test": test_branch,
                "test_accuracy": float(acc),
                "projection_norm": k_norm,
                "decoder_norm": decoder_norm,
            }

    selected = selected or {
        "trial": None,
        "train": {},
        "test": {},
        "test_accuracy": None,
        "projection_norm": None,
        "decoder_norm": None,
    }
    out = {
        "gamma_star_proxy_trials": int(n_trials),
        "gamma_star_projection_radius": float(projection_radius),
        "gamma_star_decoder_radius": float(decoder_radius),
        "gamma_star_edge_bias_radius": float(edge_bias_radius),
        "gamma_star_mode": mode,
        "gamma_star_selected_trial": selected.get("trial"),
        "gamma_star_selected_train_branch_margin_p10_min": selected.get("train", {}).get("train_branch_margin_p10_min"),
        "gamma_star_selected_train_branch_margin_mean_min": selected.get("train", {}).get("train_branch_margin_mean_min"),
        "gamma_star_selected_test_branch_margin_p10_min": selected.get("test", {}).get("test_branch_margin_p10_min"),
        "gamma_star_selected_test_branch_margin_mean_min": selected.get("test", {}).get("test_branch_margin_mean_min"),
        "gamma_star_selected_test_sample_margin_min": selected.get("test", {}).get("test_sample_margin_min"),
        "gamma_star_selected_test_accuracy": selected.get("test_accuracy"),
        "gamma_star_selected_projection_norm": selected.get("projection_norm"),
        "gamma_star_selected_decoder_norm": selected.get("decoder_norm"),
    }
    out.update(_summarize_trial_values(train_gamma_p10, "gamma_star_train_branch_margin_p10_min"))
    out.update(_summarize_trial_values(test_gamma_p10, "gamma_star_test_branch_margin_p10_min"))
    out.update(_summarize_trial_values(test_gamma_mean, "gamma_star_test_branch_margin_mean_min"))
    out.update(_summarize_trial_values(test_accuracy, "gamma_star_test_accuracy"))
    out.update(_summarize_trial_values(projection_norms, "gamma_star_projection_norm"))
    out.update(_summarize_trial_values(decoder_norms, "gamma_star_decoder_norm"))
    return out


def optimized_gamma_star_proxy_capacity(
    arborescences: Mapping[int, Sequence[Sequence[int]]],
    n_edges: int,
    input_dim: int,
    train_z: np.ndarray,
    train_labels: Sequence[int],
    test_z: np.ndarray,
    test_labels: Sequence[int],
    input_mask: Optional[np.ndarray] = None,
    n_restarts: int = 0,
    n_steps: int = 200,
    seed: int = 0,
    projection_radius: float = 1.0,
    decoder_radius: float = 1.0,
    edge_bias_radius: float = 0.0,
    mode: str = "logsumexp",
    learning_rate: float = 0.03,
    branch_softmin_temperature: float = 0.25,
    incorrect_temperature: float = 0.25,
) -> dict:
    """Gradient-optimized lower-bound probe for ``gamma*``.

    This is a sharper companion to ``bounded_gamma_star_proxy_capacity``. It
    directly optimizes bounded edge projections and a bounded decoder against a
    differentiable surrogate of ``max_{K,B} min_b margin_b``. Torch is imported
    lazily so the module remains usable in NumPy-only environments.
    """

    base = {
        "gamma_star_opt_available": False,
        "gamma_star_opt_restarts": int(n_restarts),
        "gamma_star_opt_steps": int(n_steps),
        "gamma_star_opt_learning_rate": float(learning_rate),
        "gamma_star_opt_projection_radius": float(projection_radius),
        "gamma_star_opt_decoder_radius": float(decoder_radius),
        "gamma_star_opt_edge_bias_radius": float(edge_bias_radius),
        "gamma_star_opt_mode": mode,
        "gamma_star_opt_branch_softmin_temperature": float(branch_softmin_temperature),
        "gamma_star_opt_incorrect_temperature": float(incorrect_temperature),
    }
    empty = {
        "gamma_star_opt_selected_restart": None,
        "gamma_star_opt_selected_train_branch_margin_p10_min": None,
        "gamma_star_opt_selected_train_branch_margin_mean_min": None,
        "gamma_star_opt_selected_test_branch_margin_p10_min": None,
        "gamma_star_opt_selected_test_branch_margin_mean_min": None,
        "gamma_star_opt_selected_test_sample_margin_min": None,
        "gamma_star_opt_selected_test_accuracy": None,
        "gamma_star_opt_selected_projection_norm": None,
        "gamma_star_opt_selected_decoder_norm": None,
        "gamma_star_opt_selected_edge_bias_norm": None,
    }
    if n_restarts <= 0 or n_steps <= 0:
        out = dict(base)
        out.update(empty)
        return out
    if mode not in {"max", "logsumexp"}:
        raise ValueError("mode must be 'max' or 'logsumexp'")

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on optional torch install.
        out = dict(base)
        out.update(empty)
        out["gamma_star_opt_unavailable_reason"] = str(exc)
        return out

    if input_mask is None:
        mask = np.ones((n_edges, input_dim), dtype=float)
    else:
        mask = np.asarray(input_mask, dtype=float)
        if mask.shape != (n_edges, input_dim):
            raise ValueError(f"input_mask must have shape ({n_edges}, {input_dim})")

    roots = sorted(arborescences)
    if not roots:
        out = dict(base)
        out.update(empty)
        out["gamma_star_opt_unavailable_reason"] = "no_arborescences"
        return out

    dtype = torch.float64
    mask_t = torch.as_tensor(mask, dtype=dtype)
    train_z_t = torch.as_tensor(np.asarray(train_z, dtype=float), dtype=dtype)
    test_z_t = torch.as_tensor(np.asarray(test_z, dtype=float), dtype=dtype)
    train_labels_np = np.asarray(train_labels, dtype=int)
    test_labels_np = np.asarray(test_labels, dtype=int)
    train_labels_t = torch.as_tensor(train_labels_np, dtype=torch.long)
    test_labels_t = torch.as_tensor(test_labels_np, dtype=torch.long)
    n_classes = int(max(np.max(train_labels_np), np.max(test_labels_np))) + 1
    root_mats = [
        torch.as_tensor(
            np.ascontiguousarray(incidence_matrix({root: list(arborescences[root])}, n_edges), dtype=float),
            dtype=dtype,
        )
        for root in roots
    ]

    def bounded_tensor(raw, radius, mask_tensor=None):
        value = raw if mask_tensor is None else raw * mask_tensor
        if radius is None or radius <= 0:
            return value * 0.0
        norm = torch.linalg.vector_norm(value)
        scale = torch.clamp(torch.as_tensor(float(radius), dtype=dtype) / (norm + 1e-12), max=1.0)
        return value * scale

    def bounded_decoder(raw_w, raw_b):
        if decoder_radius is None or decoder_radius <= 0:
            return raw_w * 0.0, raw_b * 0.0
        norm = torch.sqrt(torch.sum(raw_w * raw_w) + torch.sum(raw_b * raw_b))
        scale = torch.clamp(torch.as_tensor(float(decoder_radius), dtype=dtype) / (norm + 1e-12), max=1.0)
        return raw_w * scale, raw_b * scale

    def root_features(z_tensor, edge_k, edge_b):
        features = []
        for M_root in root_mats:
            if M_root.numel() == 0:
                features.append(torch.full((z_tensor.shape[0],), -1e6, dtype=dtype))
                continue
            theta = M_root @ edge_k
            beta = M_root @ edge_b
            tree_scores = z_tensor @ theta.T + beta[None, :]
            if mode == "max":
                root_score = torch.max(tree_scores, dim=1).values
            else:
                root_score = torch.logsumexp(tree_scores, dim=1)
            features.append(root_score)
        feature_tensor = torch.stack(features, dim=1)
        return feature_tensor - torch.logsumexp(feature_tensor, dim=1, keepdim=True)

    def smooth_branch_min(scores, labels_tensor):
        row_idx = torch.arange(scores.shape[0], dtype=torch.long)
        correct = scores[row_idx, labels_tensor]
        masked = scores.clone()
        masked[row_idx, labels_tensor] = -1e9
        tau = max(float(incorrect_temperature), 1e-6)
        incorrect = tau * torch.logsumexp(masked / tau, dim=1)
        margins = correct - incorrect
        branch_values = []
        for label in torch.unique(labels_tensor):
            branch_values.append(torch.mean(margins[labels_tensor == label]))
        branch_means = torch.stack(branch_values)
        temp = max(float(branch_softmin_temperature), 1e-6)
        return -temp * torch.logsumexp(-branch_means / temp, dim=0), margins

    selected = None
    train_gamma_p10 = []
    test_gamma_p10 = []
    test_gamma_mean = []
    test_accuracy = []
    projection_norms = []
    decoder_norms = []
    edge_bias_norms = []
    rng = np.random.default_rng(seed)

    for restart in range(n_restarts):
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        raw_k = torch.nn.Parameter(0.1 * torch.randn((n_edges, input_dim), dtype=dtype))
        raw_edge_b = torch.nn.Parameter(0.1 * torch.randn((n_edges,), dtype=dtype))
        raw_w = torch.nn.Parameter(0.1 * torch.randn((len(roots), n_classes), dtype=dtype))
        raw_decoder_b = torch.nn.Parameter(torch.zeros((n_classes,), dtype=dtype))
        optimizer = torch.optim.Adam([raw_k, raw_edge_b, raw_w, raw_decoder_b], lr=float(learning_rate))

        for _ in range(n_steps):
            optimizer.zero_grad()
            K = bounded_tensor(raw_k, projection_radius, mask_t)
            edge_b = bounded_tensor(raw_edge_b, edge_bias_radius)
            W, decoder_b = bounded_decoder(raw_w, raw_decoder_b)
            features = root_features(train_z_t, K, edge_b)
            scores = features @ W + decoder_b[None, :]
            branch_min, _ = smooth_branch_min(scores, train_labels_t)
            per_sample_ce = F.cross_entropy(scores, train_labels_t, reduction="none")
            branch_ce = torch.stack([
                torch.mean(per_sample_ce[train_labels_t == label])
                for label in torch.unique(train_labels_t)
            ]).mean()
            loss = branch_ce - branch_min
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            K = bounded_tensor(raw_k, projection_radius, mask_t)
            edge_b = bounded_tensor(raw_edge_b, edge_bias_radius)
            W, decoder_b = bounded_decoder(raw_w, raw_decoder_b)
            train_scores = (root_features(train_z_t, K, edge_b) @ W + decoder_b[None, :]).detach().cpu().numpy()
            test_scores = (root_features(test_z_t, K, edge_b) @ W + decoder_b[None, :]).detach().cpu().numpy()
            train_branch = branch_margin_by_label(train_scores, train_labels_np, "train")
            test_branch = branch_margin_by_label(test_scores, test_labels_np, "test")
            train_key = train_branch["train_branch_margin_p10_min"]
            test_key = test_branch["test_branch_margin_p10_min"]
            test_mean_key = test_branch["test_branch_margin_mean_min"]
            acc = accuracy_from_scores(test_scores, test_labels_np)
            projection_norm = float(torch.linalg.vector_norm(K).detach().cpu())
            edge_bias_norm = float(torch.linalg.vector_norm(edge_b).detach().cpu())
            decoder_norm = float(torch.sqrt(torch.sum(W * W) + torch.sum(decoder_b * decoder_b)).detach().cpu())

        if train_key is not None and np.isfinite(train_key):
            train_gamma_p10.append(float(train_key))
        if test_key is not None and np.isfinite(test_key):
            test_gamma_p10.append(float(test_key))
        if test_mean_key is not None and np.isfinite(test_mean_key):
            test_gamma_mean.append(float(test_mean_key))
        test_accuracy.append(float(acc))
        projection_norms.append(projection_norm)
        decoder_norms.append(decoder_norm)
        edge_bias_norms.append(edge_bias_norm)
        score_for_selection = -np.inf if train_key is None else float(train_key)
        if selected is None or score_for_selection > selected["selection_score"]:
            selected = {
                "selection_score": score_for_selection,
                "restart": int(restart),
                "train": train_branch,
                "test": test_branch,
                "test_accuracy": float(acc),
                "projection_norm": projection_norm,
                "decoder_norm": decoder_norm,
                "edge_bias_norm": edge_bias_norm,
            }

    selected = selected or {"restart": None, "train": {}, "test": {}, "test_accuracy": None}
    out = dict(base)
    out["gamma_star_opt_available"] = True
    out.update({
        "gamma_star_opt_selected_restart": selected.get("restart"),
        "gamma_star_opt_selected_train_branch_margin_p10_min": selected.get("train", {}).get("train_branch_margin_p10_min"),
        "gamma_star_opt_selected_train_branch_margin_mean_min": selected.get("train", {}).get("train_branch_margin_mean_min"),
        "gamma_star_opt_selected_test_branch_margin_p10_min": selected.get("test", {}).get("test_branch_margin_p10_min"),
        "gamma_star_opt_selected_test_branch_margin_mean_min": selected.get("test", {}).get("test_branch_margin_mean_min"),
        "gamma_star_opt_selected_test_sample_margin_min": selected.get("test", {}).get("test_sample_margin_min"),
        "gamma_star_opt_selected_test_accuracy": selected.get("test_accuracy"),
        "gamma_star_opt_selected_projection_norm": selected.get("projection_norm"),
        "gamma_star_opt_selected_decoder_norm": selected.get("decoder_norm"),
        "gamma_star_opt_selected_edge_bias_norm": selected.get("edge_bias_norm"),
    })
    out.update(_summarize_trial_values(train_gamma_p10, "gamma_star_opt_train_branch_margin_p10_min"))
    out.update(_summarize_trial_values(test_gamma_p10, "gamma_star_opt_test_branch_margin_p10_min"))
    out.update(_summarize_trial_values(test_gamma_mean, "gamma_star_opt_test_branch_margin_mean_min"))
    out.update(_summarize_trial_values(test_accuracy, "gamma_star_opt_test_accuracy"))
    out.update(_summarize_trial_values(projection_norms, "gamma_star_opt_projection_norm"))
    out.update(_summarize_trial_values(decoder_norms, "gamma_star_opt_decoder_norm"))
    out.update(_summarize_trial_values(edge_bias_norms, "gamma_star_opt_edge_bias_norm"))
    return out


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
    tree_feature_trials: int = 8,
    tree_feature_mode: str = "max",
    tree_feature_projection_radius: float = 1.0,
    tree_feature_bias_scale: float = 0.0,
    gamma_star_trials: int = 32,
    gamma_star_projection_radius: float = 1.0,
    gamma_star_decoder_radius: float = 1.0,
    gamma_star_edge_bias_radius: float = 0.0,
    gamma_star_opt_restarts: int = 0,
    gamma_star_opt_steps: int = 200,
    gamma_star_opt_lr: float = 0.03,
    gamma_star_opt_branch_softmin_temperature: float = 0.25,
    gamma_star_opt_incorrect_temperature: float = 0.25,
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
    rooted_ranks = rooted_common_rank_tensor(
        mats["arborescences"],
        n_edges=len(edge_tuple),
        input_mask=input_mask,
        n_context=n_context,
        z_dim=z_dim,
    )

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
        "rooted_common_rank_by_root_branch_dim": rooted_ranks.tolist(),
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
    result.update(rooted_polytope_support_summary(rooted_ranks))
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
    result.update(
        tropical_tree_feature_capacity(
            mats["arborescences"],
            n_edges=len(edge_tuple),
            input_dim=p,
            train_z=train_z,
            train_labels=train_labels,
            test_z=test_z,
            test_labels=test_labels,
            input_mask=input_mask,
            n_trials=tree_feature_trials,
            seed=seed + 1009,
            projection_radius=tree_feature_projection_radius,
            edge_bias_scale=tree_feature_bias_scale,
            mode=tree_feature_mode,
            ridge=ridge,
            l2_radius=l2_radius,
        )
    )
    result.update(
        tree_normal_fan_coverage(
            mats["arborescences"],
            n_edges=len(edge_tuple),
            input_dim=p,
            z=test_z,
            labels=test_labels,
            input_mask=input_mask,
            n_trials=tree_feature_trials,
            seed=seed + 2017,
            projection_radius=tree_feature_projection_radius,
            edge_bias_scale=tree_feature_bias_scale,
        )
    )
    result.update(
        bounded_gamma_star_proxy_capacity(
            mats["arborescences"],
            n_edges=len(edge_tuple),
            input_dim=p,
            train_z=train_z,
            train_labels=train_labels,
            test_z=test_z,
            test_labels=test_labels,
            input_mask=input_mask,
            n_trials=gamma_star_trials,
            seed=seed + 3251,
            projection_radius=gamma_star_projection_radius,
            decoder_radius=gamma_star_decoder_radius,
            edge_bias_radius=gamma_star_edge_bias_radius,
            mode=tree_feature_mode,
            ridge=ridge,
        )
    )
    result.update(
        optimized_gamma_star_proxy_capacity(
            mats["arborescences"],
            n_edges=len(edge_tuple),
            input_dim=p,
            train_z=train_z,
            train_labels=train_labels,
            test_z=test_z,
            test_labels=test_labels,
            input_mask=input_mask,
            n_restarts=gamma_star_opt_restarts,
            n_steps=gamma_star_opt_steps,
            seed=seed + 4337,
            projection_radius=gamma_star_projection_radius,
            decoder_radius=gamma_star_decoder_radius,
            edge_bias_radius=gamma_star_edge_bias_radius,
            mode=tree_feature_mode,
            learning_rate=gamma_star_opt_lr,
            branch_softmin_temperature=gamma_star_opt_branch_softmin_temperature,
            incorrect_temperature=gamma_star_opt_incorrect_temperature,
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
        f"| rooted polytope branch-dim support | {result.get('rooted_polytope_supported_branch_dim_fraction'):.3f} |",
        f"| rooted branch root support min/mean/max | {result.get('rooted_polytope_branch_root_support_min'):.2f} / {result.get('rooted_polytope_branch_root_support_mean'):.2f} / {result.get('rooted_polytope_branch_root_support_max'):.2f} |",
        f"| rooted branch best-rank min/mean/max | {result.get('rooted_polytope_branch_best_rank_min'):.2f} / {result.get('rooted_polytope_branch_best_rank_mean'):.2f} / {result.get('rooted_polytope_branch_best_rank_max'):.2f} |",
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
            "## Tropical Tree Random Features",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| trials | {result.get('tropical_feature_trials')} |",
            f"| mode | {result.get('tropical_feature_mode')} |",
            f"| test accuracy mean/max | {result.get('tropical_linear_test_accuracy_mean')} / {result.get('tropical_linear_test_accuracy_max')} |",
            f"| test p10 margin mean/max | {result.get('tropical_linear_test_margin_p10_mean')} / {result.get('tropical_linear_test_margin_p10_max')} |",
            f"| root feature effective rank mean/max | {result.get('tropical_root_feature_effective_rank_mean')} / {result.get('tropical_root_feature_effective_rank_max')} |",
            f"| normal fan branch-tree NMI mean/max | {result.get('normal_fan_branch_tree_nmi_mean')} / {result.get('normal_fan_branch_tree_nmi_max')} |",
            f"| normal fan active tree count mean | {result.get('normal_fan_active_tree_count_mean')} |",
            "",
            "## Bounded Gamma-Star Proxy",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| trials | {result.get('gamma_star_proxy_trials')} |",
            f"| projection radius | {result.get('gamma_star_projection_radius')} |",
            f"| decoder radius | {result.get('gamma_star_decoder_radius')} |",
            f"| selected test branch p10 margin min | {result.get('gamma_star_selected_test_branch_margin_p10_min')} |",
            f"| selected test branch mean margin min | {result.get('gamma_star_selected_test_branch_margin_mean_min')} |",
            f"| selected test accuracy | {result.get('gamma_star_selected_test_accuracy')} |",
            f"| test branch p10 margin min mean/max | {result.get('gamma_star_test_branch_margin_p10_min_mean')} / {result.get('gamma_star_test_branch_margin_p10_min_max')} |",
            f"| optimized available | {result.get('gamma_star_opt_available')} |",
            f"| optimized restarts/steps | {result.get('gamma_star_opt_restarts')} / {result.get('gamma_star_opt_steps')} |",
            f"| optimized selected test branch p10 margin min | {result.get('gamma_star_opt_selected_test_branch_margin_p10_min')} |",
            f"| optimized selected test accuracy | {result.get('gamma_star_opt_selected_test_accuracy')} |",
        ]
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
    parser.add_argument("--tree_feature_trials", type=int, default=8)
    parser.add_argument("--tree_feature_mode", choices=["max", "logsumexp"], default="max")
    parser.add_argument("--tree_feature_projection_radius", type=float, default=1.0)
    parser.add_argument("--tree_feature_bias_scale", type=float, default=0.0)
    parser.add_argument("--gamma_star_trials", type=int, default=32)
    parser.add_argument("--gamma_star_projection_radius", type=float, default=1.0)
    parser.add_argument("--gamma_star_decoder_radius", type=float, default=1.0)
    parser.add_argument("--gamma_star_edge_bias_radius", type=float, default=0.0)
    parser.add_argument("--gamma_star_opt_restarts", type=int, default=0)
    parser.add_argument("--gamma_star_opt_steps", type=int, default=200)
    parser.add_argument("--gamma_star_opt_lr", type=float, default=0.03)
    parser.add_argument("--gamma_star_opt_branch_softmin_temperature", type=float, default=0.25)
    parser.add_argument("--gamma_star_opt_incorrect_temperature", type=float, default=0.25)
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
        tree_feature_trials=args.tree_feature_trials,
        tree_feature_mode=args.tree_feature_mode,
        tree_feature_projection_radius=args.tree_feature_projection_radius,
        tree_feature_bias_scale=args.tree_feature_bias_scale,
        gamma_star_trials=args.gamma_star_trials,
        gamma_star_projection_radius=args.gamma_star_projection_radius,
        gamma_star_decoder_radius=args.gamma_star_decoder_radius,
        gamma_star_edge_bias_radius=args.gamma_star_edge_bias_radius,
        gamma_star_opt_restarts=args.gamma_star_opt_restarts,
        gamma_star_opt_steps=args.gamma_star_opt_steps,
        gamma_star_opt_lr=args.gamma_star_opt_lr,
        gamma_star_opt_branch_softmin_temperature=args.gamma_star_opt_branch_softmin_temperature,
        gamma_star_opt_incorrect_temperature=args.gamma_star_opt_incorrect_temperature,
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
