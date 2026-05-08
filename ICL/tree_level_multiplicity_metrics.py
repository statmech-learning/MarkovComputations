"""Tree-level input multiplicity diagnostics for first-order Markov ICL.

Edge-level multiplicity ``M_alpha = sum_e Omega[e, alpha]`` ignores that a
first-order CRN steady state is built from rooted tree sums.  This module lifts
input-mask support into the tree basis and into same-root tree-difference
contrasts:

``A[T, alpha] = sum_{e in T} Omega[e, alpha]``

``A_diff[T, T', alpha] = sum_e |s_T(e) - s_T'(e)| Omega[e, alpha]``.

The command-line entry point builds the Phase 2 reanalysis artifact requested
in ``MARKOV_ICL_NEXT_PHASE_GOAL.md``.  It uses local hard-sweep topology JSONs
and, when available, reads fixed-m20 topology JSONs from the cluster paths
recorded in the local CSV through ``ssh engaging``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from topology_metrics import (
    gini,
    normalize_edges,
    topology_matrices,
    tree_counts_by_determinant,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "ICL" / "results"
DEFAULT_OUT_DIR = RESULT_ROOT / "next_phase_stats"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[float | None]) -> float | None:
    arr = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(arr)) if arr else None


def std(values: Iterable[float | None]) -> float | None:
    arr = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.std(arr)) if arr else None


def maximum(values: Iterable[float | None]) -> float | None:
    arr = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.max(arr)) if arr else None


def safe_log1p(value: float | None) -> float | None:
    if value is None:
        return None
    return float(math.log1p(max(0.0, value)))


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(item) for item in row) + " |")
    return "\n".join(out)


def tree_table_from_arborescences(
    arborescences: Mapping[int, Sequence[Sequence[int]]],
    n_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    roots: list[int] = []
    rows: list[np.ndarray] = []
    for root in sorted(arborescences):
        for tree in arborescences[root]:
            incidence = np.zeros(n_edges, dtype=float)
            incidence[list(tree)] = 1.0
            roots.append(int(root))
            rows.append(incidence)
    if not rows:
        return np.zeros(0, dtype=int), np.zeros((0, n_edges), dtype=float)
    return np.asarray(roots, dtype=int), np.vstack(rows)


def comparison_columns(n_context: int, z_dim: int) -> list[tuple[int, int, int, int]]:
    """Return ``(branch, dim, context_col, query_col)`` tuples."""

    return [
        (branch, dim, branch * z_dim + dim, n_context * z_dim + dim)
        for branch in range(n_context)
        for dim in range(z_dim)
    ]


def positive_gini(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(gini(np.maximum(arr, 0.0)))


def finite_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Small diagnostic matrix multiply with defensive warning cleanup."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            out = left @ right
    if np.all(np.isfinite(out)):
        return out
    return np.nan_to_num(out, nan=0.0, posinf=1.0e12, neginf=-1.0e12)


def edge_level_multiplicity_summary(
    input_mask: np.ndarray,
    n_context: int,
    z_dim: int,
) -> dict[str, Any]:
    mask = np.asarray(input_mask, dtype=float)
    n_edges, p = mask.shape
    if p != (n_context + 1) * z_dim:
        raise ValueError(f"input width {p} incompatible with n_context={n_context}, z_dim={z_dim}")

    loads = mask.sum(axis=0)
    overlaps = []
    raw_overlaps = []
    imbalances = []
    for _, _, ctx_col, query_col in comparison_columns(n_context, z_dim):
        raw = float(np.sum((mask[:, ctx_col] > 0.0) & (mask[:, query_col] > 0.0)))
        raw_overlaps.append(raw)
        overlaps.append(raw / max(1, n_edges))
        imbalances.append(float(abs(loads[ctx_col] - loads[query_col])))

    return {
        "edge_M_mean": float(np.mean(loads)),
        "edge_M_min": float(np.min(loads)) if loads.size else None,
        "edge_M_max": float(np.max(loads)) if loads.size else None,
        "edge_M_gini": positive_gini(loads),
        "edge_M_zero_fraction": float(np.mean(loads <= 0.0)) if loads.size else None,
        "edge_load_gini": positive_gini(mask.sum(axis=1)),
        "edge_overlap_raw_min": float(np.min(raw_overlaps)) if raw_overlaps else None,
        "edge_overlap_raw_mean": float(np.mean(raw_overlaps)) if raw_overlaps else None,
        "edge_overlap_norm_min": float(np.min(overlaps)) if overlaps else None,
        "edge_overlap_norm_mean": float(np.mean(overlaps)) if overlaps else None,
        "edge_overlap_norm_gini": positive_gini(overlaps),
        "edge_comparison_imbalance_mean": float(np.mean(imbalances)) if imbalances else None,
        "edge_comparison_imbalance_max": float(np.max(imbalances)) if imbalances else None,
    }


def sampled_pair_indices(n: int, max_pairs: int | None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, int, bool]:
    total = n * (n - 1) // 2
    if total <= 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), 0, False
    if max_pairs is None or total <= max_pairs:
        pairs = np.asarray(list(combinations(range(n), 2)), dtype=int)
        return pairs[:, 0], pairs[:, 1], total, False

    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < max_pairs and attempts < max_pairs * 20:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        if i > j:
            i, j = j, i
        pairs.add((i, j))
        attempts += 1
    arr = np.asarray(sorted(pairs), dtype=int)
    return arr[:, 0], arr[:, 1], total, True


def summarize_root_overlaps(
    coordinate_loads: np.ndarray,
    roots: np.ndarray,
    n_context: int,
    z_dim: int,
    prefix: str,
) -> dict[str, Any]:
    norm_values = []
    raw_values = []
    per_root = []
    for root in sorted(set(int(root) for root in roots.tolist())):
        idx = np.where(roots == root)[0]
        root_norm = []
        root_raw = []
        for branch, dim, ctx_col, query_col in comparison_columns(n_context, z_dim):
            raw = float(
                np.sum(
                    (coordinate_loads[idx, ctx_col] > 0.0)
                    & (coordinate_loads[idx, query_col] > 0.0)
                )
            )
            norm = raw / max(1, idx.size)
            raw_values.append(raw)
            norm_values.append(norm)
            root_raw.append(raw)
            root_norm.append(norm)
            per_root.append(
                {
                    "root": root,
                    "branch": branch,
                    "dim": dim,
                    f"{prefix}_overlap_raw": raw,
                    f"{prefix}_overlap_norm": norm,
                    f"{prefix}_denominator": int(idx.size),
                }
            )
    return {
        f"{prefix}_overlap_raw_min": float(np.min(raw_values)) if raw_values else None,
        f"{prefix}_overlap_raw_mean": float(np.mean(raw_values)) if raw_values else None,
        f"{prefix}_overlap_norm_min": float(np.min(norm_values)) if norm_values else None,
        f"{prefix}_overlap_norm_mean": float(np.mean(norm_values)) if norm_values else None,
        f"{prefix}_overlap_norm_gini": positive_gini(norm_values),
        f"{prefix}_per_root_comparison": per_root,
    }


def tree_difference_multiplicity_summary(
    tree_incidence: np.ndarray,
    tree_roots: np.ndarray,
    input_mask: np.ndarray,
    n_context: int,
    z_dim: int,
    max_pairs_per_root: int | None = 50000,
) -> dict[str, Any]:
    pair_norm_values = []
    pair_raw_values = []
    pair_denominators = []
    per_root = []
    total_possible_pairs = 0
    total_sampled_pairs = 0
    truncated = False
    diff_coord_load_totals = np.zeros(input_mask.shape[1], dtype=float)

    for root in sorted(set(int(root) for root in tree_roots.tolist())):
        idx = np.where(tree_roots == root)[0]
        left, right, possible, root_truncated = sampled_pair_indices(
            idx.size, max_pairs=max_pairs_per_root, seed=1009 + root
        )
        total_possible_pairs += possible
        truncated = truncated or root_truncated
        if left.size == 0:
            continue
        left_global = idx[left]
        right_global = idx[right]
        diff_incidence = np.abs(tree_incidence[left_global] - tree_incidence[right_global])
        diff_loads = finite_matmul(diff_incidence, input_mask)
        diff_coord_load_totals += diff_loads.sum(axis=0)
        total_sampled_pairs += int(diff_loads.shape[0])

        for branch, dim, ctx_col, query_col in comparison_columns(n_context, z_dim):
            raw = float(np.sum((diff_loads[:, ctx_col] > 0.0) & (diff_loads[:, query_col] > 0.0)))
            norm = raw / max(1, diff_loads.shape[0])
            pair_raw_values.append(raw)
            pair_norm_values.append(norm)
            pair_denominators.append(int(diff_loads.shape[0]))
            per_root.append(
                {
                    "root": root,
                    "branch": branch,
                    "dim": dim,
                    "diff_overlap_raw": raw,
                    "diff_overlap_norm": norm,
                    "diff_denominator": int(diff_loads.shape[0]),
                    "diff_possible_pairs": possible,
                    "diff_pairs_truncated": root_truncated,
                }
            )

    return {
        "diff_overlap_raw_min": float(np.min(pair_raw_values)) if pair_raw_values else None,
        "diff_overlap_raw_mean": float(np.mean(pair_raw_values)) if pair_raw_values else None,
        "diff_overlap_norm_min": float(np.min(pair_norm_values)) if pair_norm_values else None,
        "diff_overlap_norm_mean": float(np.mean(pair_norm_values)) if pair_norm_values else None,
        "diff_overlap_norm_gini": positive_gini(pair_norm_values),
        "diff_pair_count_sampled": int(total_sampled_pairs),
        "diff_pair_count_possible": int(total_possible_pairs),
        "diff_pairs_truncated": bool(truncated),
        "diff_pair_count_log": safe_log1p(float(total_sampled_pairs)),
        "diff_coord_load_gini": positive_gini(diff_coord_load_totals),
        "diff_per_root_comparison": per_root,
        "diff_denominator_min": int(np.min(pair_denominators)) if pair_denominators else None,
    }


def tree_level_multiplicity_summary(
    n_nodes: int,
    edges: Sequence[Sequence[int]],
    input_mask: np.ndarray,
    n_context: int,
    z_dim: int,
    max_trees_per_root: int | None = None,
    max_pairs_per_root: int | None = 50000,
) -> dict[str, Any]:
    edge_tuple = normalize_edges(n_nodes, edges)
    mask = np.asarray(input_mask, dtype=float)
    if mask.shape != (len(edge_tuple), (n_context + 1) * z_dim):
        raise ValueError(
            f"input_mask shape {mask.shape} incompatible with "
            f"edges={len(edge_tuple)}, n_context={n_context}, z_dim={z_dim}"
        )

    exact_counts = tree_counts_by_determinant(n_nodes, edge_tuple)
    mats = topology_matrices(n_nodes, edge_tuple, max_trees_per_root=max_trees_per_root)
    tree_roots, tree_incidence = tree_table_from_arborescences(mats["arborescences"], len(edge_tuple))
    enumerated_counts = [int(np.sum(tree_roots == root)) for root in range(n_nodes)]
    tree_loads = (
        finite_matmul(tree_incidence, mask)
        if tree_incidence.size
        else np.zeros((0, mask.shape[1]), dtype=float)
    )

    tree_summary = summarize_root_overlaps(tree_loads, tree_roots, n_context, z_dim, "tree")
    tree_coord_load = tree_loads.sum(axis=0) if tree_loads.size else np.zeros(mask.shape[1], dtype=float)
    summary = {
        **edge_level_multiplicity_summary(mask, n_context, z_dim),
        **tree_summary,
        **tree_difference_multiplicity_summary(
            tree_incidence,
            tree_roots,
            mask,
            n_context,
            z_dim,
            max_pairs_per_root=max_pairs_per_root,
        ),
        "n_nodes": int(n_nodes),
        "n_edges": int(len(edge_tuple)),
        "n_context": int(n_context),
        "z_dim": int(z_dim),
        "n_trees_enumerated": int(tree_incidence.shape[0]),
        "tree_count_exact_total": int(sum(exact_counts)),
        "tree_count_enumerated_total": int(sum(enumerated_counts)),
        "tree_enumeration_truncated": bool(any(e < c for e, c in zip(enumerated_counts, exact_counts))),
        "tree_count_log": safe_log1p(float(tree_incidence.shape[0])),
        "tree_count_exact_by_root": [int(value) for value in exact_counts],
        "tree_count_enumerated_by_root": enumerated_counts,
        "tree_coord_load_gini": positive_gini(tree_coord_load),
        "tree_active_fraction_mean": float(np.mean(tree_loads > 0.0)) if tree_loads.size else None,
    }
    return summary


def learned_weighted_tree_multiplicity_summary(
    n_nodes: int,
    edges: Sequence[Sequence[int]],
    input_mask: np.ndarray,
    edge_coordinate_weights_abs: np.ndarray,
    n_context: int,
    z_dim: int,
    max_trees_per_root: int | None = None,
) -> dict[str, Any]:
    """Post-training tree multiplicity using ``|K[e, alpha]| Omega[e, alpha]``.

    Aggregate CSV artifacts do not store learned ``K`` tensors, so the Phase 2
    reanalysis treats trained branch margin as an outcome rather than a learned
    predictor.  This function is provided for direct use on ``model.pt``-level
    analyses.
    """

    mask = np.asarray(input_mask, dtype=float)
    weights = np.asarray(edge_coordinate_weights_abs, dtype=float)
    if weights.shape != mask.shape:
        raise ValueError(f"edge_coordinate_weights_abs must have shape {mask.shape}")
    weighted_mask = mask * np.abs(weights)
    return tree_level_multiplicity_summary(
        n_nodes,
        edges,
        weighted_mask,
        n_context,
        z_dim,
        max_trees_per_root=max_trees_per_root,
    )


def posterior_weighted_tree_overlap(
    tree_loads: np.ndarray,
    tree_roots: np.ndarray,
    tree_posterior_weights: np.ndarray,
    n_context: int,
    z_dim: int,
) -> dict[str, Any]:
    """Post-training overlap weighted by ``P(T | r, z)`` averaged over samples."""

    loads = np.asarray(tree_loads, dtype=float)
    roots = np.asarray(tree_roots, dtype=int)
    weights = np.asarray(tree_posterior_weights, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != loads.shape[0]:
        raise ValueError("tree_posterior_weights must be a vector aligned with tree_loads")
    values = []
    for root in sorted(set(int(root) for root in roots.tolist())):
        idx = np.where(roots == root)[0]
        denom = float(np.sum(weights[idx]))
        if denom <= 0.0:
            continue
        for _, _, ctx_col, query_col in comparison_columns(n_context, z_dim):
            indicator = (loads[idx, ctx_col] > 0.0) & (loads[idx, query_col] > 0.0)
            values.append(float(np.sum(weights[idx] * indicator) / denom))
    return {
        "posterior_tree_overlap_norm_min": float(np.min(values)) if values else None,
        "posterior_tree_overlap_norm_mean": float(np.mean(values)) if values else None,
        "posterior_tree_overlap_norm_gini": positive_gini(values),
    }


def group_rows(rows: Sequence[Mapping[str, Any]], key: str = "topology_name") -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            grouped[str(value)].append(row)
    return dict(grouped)


def aggregate_outcome_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = group_rows(rows, "topology_name")
    out = {}
    for key, members in sorted(groups.items()):
        test_values = [as_float(row, "test_novel_classes") for row in members]
        first = members[0]
        out[key] = {
            "group": key,
            "n_runs": len(members),
            "mean_novel_icl": mean(test_values),
            "best_seed_novel_icl": maximum(test_values),
            "seed_std_novel_icl": std(test_values),
            "run_dir": first.get("run_dir"),
            "physical_topology_name": first.get("physical_topology_name"),
            "input_mask_name": first.get("input_mask_name"),
            "input_mask_family": first.get("input_mask_family"),
            "n_edges": as_float(first, "n_edges"),
            "p": as_float(first, "p"),
        }
    return out


def aggregate_mechanism_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = group_rows(rows, "topology_name")
    out = {}
    for key, members in sorted(groups.items()):
        branch_acc_min = mean(as_float(row, "target_accuracy_branch_mean_min") for row in members)
        out[key] = {
            "trained_branch_margin": mean(
                as_float(row, "target_logprob_margin_branch_mean_min") for row in members
            ),
            "branch_failure_percent": None if branch_acc_min is None else float(100.0 - branch_acc_min),
        }
    return out


def local_topology_path(run_dir: str | None) -> Path | None:
    if not run_dir:
        return None
    path = Path(run_dir) / "topology.json"
    if path.exists():
        return path
    marker = "/ICL/results/"
    if marker in run_dir:
        suffix = run_dir.split(marker, 1)[1]
        candidate = RESULT_ROOT / suffix / "topology.json"
        if candidate.exists():
            return candidate
    return None


def read_remote_json(host: str, remote_path: str, timeout: int = 30) -> dict[str, Any] | None:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host, "cat", remote_path]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=timeout + 10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def load_topology_payload(run_dir: str | None, ssh_host: str | None) -> tuple[dict[str, Any] | None, str]:
    path = local_topology_path(run_dir)
    if path is not None:
        return json.loads(path.read_text()), "local"
    if ssh_host and run_dir:
        remote_path = os.path.join(run_dir, "topology.json")
        payload = read_remote_json(ssh_host, remote_path)
        if payload is not None:
            return payload, f"ssh:{ssh_host}"
    return None, "missing"


def parse_hard_dimensions(path: Path) -> tuple[int, int]:
    match = re.search(r"_N(\d+)_D(\d+)", str(path))
    if not match:
        raise ValueError(f"cannot infer N/D from {path}")
    return int(match.group(1)), int(match.group(2))


def build_dataset(
    name: str,
    topology_csv: Path,
    n_context: int,
    z_dim: int,
    ssh_host: str | None,
    mechanism_csv: Path | None = None,
    max_trees_per_root: int | None = None,
    max_pairs_per_root: int | None = 50000,
) -> dict[str, Any]:
    rows = read_csv(topology_csv)
    outcomes = aggregate_outcome_rows(rows)
    mechanisms = aggregate_mechanism_rows(read_csv(mechanism_csv)) if mechanism_csv and mechanism_csv.exists() else {}
    groups = []
    topology_sources = defaultdict(int)
    missing_topology = []
    for group, outcome in outcomes.items():
        payload, source = load_topology_payload(str(outcome.get("run_dir") or ""), ssh_host)
        topology_sources[source] += 1
        if payload is None:
            missing_topology.append(group)
            continue
        metrics = tree_level_multiplicity_summary(
            int(payload["n_nodes"]),
            payload["edges"],
            np.asarray(payload["input_mask"], dtype=float),
            n_context=n_context,
            z_dim=z_dim,
            max_trees_per_root=max_trees_per_root,
            max_pairs_per_root=max_pairs_per_root,
        )
        groups.append({**outcome, **mechanisms.get(group, {}), **metrics, "topology_source": source})

    return {
        "name": name,
        "topology_csv": str(topology_csv.relative_to(REPO_ROOT)),
        "mechanism_csv": str(mechanism_csv.relative_to(REPO_ROOT)) if mechanism_csv and mechanism_csv.exists() else None,
        "n_context": n_context,
        "z_dim": z_dim,
        "n_seed_rows": len(rows),
        "n_groups_total": len(outcomes),
        "n_groups_with_tree_metrics": len(groups),
        "topology_sources": dict(topology_sources),
        "missing_topology_groups": missing_topology,
        "groups": groups,
    }


def design_matrix(rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], outcome: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        y = row.get(outcome)
        if y is None:
            continue
        vals = []
        ok = True
        for predictor in predictors:
            value = row.get(predictor)
            if value is None:
                ok = False
                break
            try:
                vals.append(float(value))
            except (TypeError, ValueError):
                ok = False
                break
        if ok and math.isfinite(float(y)):
            xs.append(vals)
            ys.append(float(y))
    if not xs:
        return np.zeros((0, len(predictors))), np.zeros(0)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def loo_r2(rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], outcome: str) -> dict[str, Any]:
    X, y = design_matrix(rows, predictors, outcome)
    n, p = X.shape
    if n < max(5, p + 2):
        return {
            "predictors": list(predictors),
            "outcome": outcome,
            "n_groups": int(n),
            "loo_r2": None,
            "reason": "too_few_groups_or_complete_cases",
        }
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1e-12:
        return {
            "predictors": list(predictors),
            "outcome": outcome,
            "n_groups": int(n),
            "loo_r2": None,
            "reason": "constant_outcome",
        }
    preds = []
    for holdout in range(n):
        train = np.arange(n) != holdout
        center = X[train].mean(axis=0)
        scale = X[train].std(axis=0)
        scale[scale <= 1e-12] = 1.0
        X_train = (X[train] - center) / scale
        X_test = (X[holdout : holdout + 1] - center) / scale
        A = np.column_stack([np.ones(X_train.shape[0]), X_train])
        ridge = 1e-6 * np.eye(A.shape[1])
        ridge[0, 0] = 0.0
        coef = np.linalg.solve(A.T @ A + ridge, A.T @ y[train])
        preds.append(float((np.column_stack([np.ones(1), X_test]) @ coef)[0]))
    err = float(np.sum((np.asarray(preds) - y) ** 2))
    return {
        "predictors": list(predictors),
        "outcome": outcome,
        "n_groups": int(n),
        "loo_r2": float(1.0 - err / denom),
    }


PREDICTOR_SETS = {
    "edge_level_multiplicity": [
        "edge_M_mean",
        "edge_M_gini",
        "edge_overlap_norm_min",
        "edge_overlap_norm_mean",
        "edge_comparison_imbalance_mean",
    ],
    "tree_level_multiplicity": [
        "tree_overlap_norm_min",
        "tree_overlap_norm_mean",
        "tree_coord_load_gini",
        "tree_active_fraction_mean",
        "tree_count_log",
    ],
    "tree_difference_multiplicity": [
        "diff_overlap_norm_min",
        "diff_overlap_norm_mean",
        "diff_coord_load_gini",
        "diff_pair_count_log",
    ],
    "tree_and_difference_multiplicity": [
        "tree_overlap_norm_min",
        "tree_overlap_norm_mean",
        "tree_coord_load_gini",
        "diff_overlap_norm_min",
        "diff_overlap_norm_mean",
        "diff_coord_load_gini",
    ],
}

OUTCOMES = [
    "mean_novel_icl",
    "best_seed_novel_icl",
    "seed_std_novel_icl",
    "branch_failure_percent",
    "trained_branch_margin",
]


def analyze_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    groups = dataset.get("groups", [])
    models = []
    for outcome in OUTCOMES:
        for name, predictors in PREDICTOR_SETS.items():
            result = loo_r2(groups, predictors, outcome)
            result["model"] = name
            models.append(result)
    return {
        "name": dataset["name"],
        "n_groups": len(groups),
        "models": models,
    }


def compact_models(analysis: Mapping[str, Any], outcome: str) -> list[list[Any]]:
    rows = []
    for item in analysis.get("models", []):
        if item.get("outcome") == outcome:
            rows.append([item.get("model"), item.get("n_groups"), item.get("loo_r2"), item.get("reason")])
    return rows


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Tree-Level Multiplicity Reanalysis",
        "",
        "## Gate Result",
        "",
        report["gate_result"],
        "",
        "## Metric Definitions",
        "",
        "- Edge-level multiplicity uses `M_alpha = sum_e Omega[e, alpha]` and edge-wise comparison overlap.",
        "- Tree-level multiplicity uses `A[T, alpha] = sum_{e in T} Omega[e, alpha]` and root-conditioned comparison overlap.",
        "- Tree-difference multiplicity uses same-root pairs `(T,T')` and `sum_e |s_T(e)-s_T'(e)| Omega[e, alpha]`.",
        "- Every raw overlap is reported with a normalized overlap because raw counts scale with the number of trees or tree pairs.",
        "- Learned/post-training variants are implemented in `learned_weighted_tree_multiplicity_summary` and `posterior_weighted_tree_overlap`; aggregate reports treat trained margin as an outcome because aggregate CSVs do not store learned `K` tensors.",
        "",
        "## Datasets",
        "",
        markdown_table(
            [
                [
                    dataset["name"],
                    dataset["n_seed_rows"],
                    dataset["n_groups_total"],
                    dataset["n_groups_with_tree_metrics"],
                    dataset["n_context"],
                    dataset["z_dim"],
                    dataset["topology_sources"],
                ]
                for dataset in report["datasets"]
            ],
            ["dataset", "seed rows", "groups", "with tree metrics", "N_c", "D", "topology sources"],
        ),
        "",
    ]
    for analysis in report["analyses"]:
        lines.extend(
            [
                f"## {analysis['name']}: Mean Novel-Class ICL",
                "",
                markdown_table(compact_models(analysis, "mean_novel_icl"), ["model", "groups", "LOO R2", "reason"]),
                "",
                f"## {analysis['name']}: Best Seed ICL",
                "",
                markdown_table(compact_models(analysis, "best_seed_novel_icl"), ["model", "groups", "LOO R2", "reason"]),
                "",
                f"## {analysis['name']}: Seed Standard Deviation",
                "",
                markdown_table(compact_models(analysis, "seed_std_novel_icl"), ["model", "groups", "LOO R2", "reason"]),
                "",
                f"## {analysis['name']}: Branch Failures",
                "",
                markdown_table(compact_models(analysis, "branch_failure_percent"), ["model", "groups", "LOO R2", "reason"]),
                "",
                f"## {analysis['name']}: Trained Branch Margin",
                "",
                markdown_table(compact_models(analysis, "trained_branch_margin"), ["model", "groups", "LOO R2", "reason"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            report["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    datasets = []

    fixed_csv = DEFAULT_OUT_DIR / "pooled_fixed_m20_topology_results.csv"
    if fixed_csv.exists():
        datasets.append(
            build_dataset(
                "fixed_m20_masks_cluster_topology",
                fixed_csv,
                n_context=args.fixed_n_context,
                z_dim=args.fixed_z_dim,
                ssh_host=args.ssh_host,
                max_trees_per_root=args.max_trees_per_root,
                max_pairs_per_root=args.max_pairs_per_root,
            )
        )

    hard_groups: list[dict[str, Any]] = []
    hard_dataset_meta = {
        "name": "hard_full_mask_local",
        "topology_csv": "ICL/results/expanded_hard_sweeps/*/topology_results.csv",
        "mechanism_csv": "ICL/results/expanded_hard_sweeps/*/mechanism_results.csv",
        "n_context": "parsed_per_regime",
        "z_dim": "parsed_per_regime",
        "n_seed_rows": 0,
        "n_groups_total": 0,
        "n_groups_with_tree_metrics": 0,
        "topology_sources": {},
        "missing_topology_groups": [],
    }
    for topology_csv in sorted(RESULT_ROOT.glob("expanded_hard_sweeps/*/topology_results.csv")):
        n_context, z_dim = parse_hard_dimensions(topology_csv)
        dataset = build_dataset(
            topology_csv.parent.name,
            topology_csv,
            n_context=n_context,
            z_dim=z_dim,
            ssh_host=None,
            mechanism_csv=topology_csv.parent / "mechanism_results.csv",
            max_trees_per_root=args.max_trees_per_root,
            max_pairs_per_root=args.max_pairs_per_root,
        )
        hard_groups.extend(dataset["groups"])
        hard_dataset_meta["n_seed_rows"] += dataset["n_seed_rows"]
        hard_dataset_meta["n_groups_total"] += dataset["n_groups_total"]
        hard_dataset_meta["n_groups_with_tree_metrics"] += dataset["n_groups_with_tree_metrics"]
        hard_dataset_meta["missing_topology_groups"].extend(dataset["missing_topology_groups"])
        for key, value in dataset["topology_sources"].items():
            hard_dataset_meta["topology_sources"][key] = hard_dataset_meta["topology_sources"].get(key, 0) + value
    if hard_groups:
        hard_dataset_meta["groups"] = hard_groups
        datasets.append(hard_dataset_meta)

    analyses = [analyze_dataset(dataset) for dataset in datasets]
    return {
        "schema": "tree_level_multiplicity_reanalysis.v1",
        "gate_result": (
            "Phase 2 implements edge-, tree-, and tree-difference-level multiplicity metrics and evaluates "
            "them with grouped leave-one-out regressions.  These are pre-training structural metrics; "
            "post-training learned-weight APIs are implemented but not used as predictors in aggregate CSV "
            "reports because learned K tensors are not stored there."
        ),
        "predictor_sets": PREDICTOR_SETS,
        "outcomes": OUTCOMES,
        "datasets": datasets,
        "analyses": analyses,
        "interpretation": (
            "Use the LOO tables as a screening diagnostic, not as a causal claim.  Fixed-m20 masks are the "
            "right local test bed for input multiplicity because physical edge count is fixed and raw masks "
            "are read from cluster topology JSONs.  Hard full-mask sweeps add trained branch-failure and "
            "branch-margin outcomes, but their full input masks make edge-level multiplicity less diagnostic "
            "within each exact regime."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--ssh-host", default="engaging")
    parser.add_argument("--fixed-n-context", type=int, default=4)
    parser.add_argument("--fixed-z-dim", type=int, default=4)
    parser.add_argument("--max-trees-per-root", type=int, default=None)
    parser.add_argument("--max-pairs-per-root", type=int, default=50000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    json_path = out_dir / "tree_level_multiplicity_reanalysis.json"
    md_path = out_dir / "tree_level_multiplicity_reanalysis.md"
    json_path.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True) + "\n")
    write_markdown(json_ready(report), md_path)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
