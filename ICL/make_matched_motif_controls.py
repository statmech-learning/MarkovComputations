"""Generate matched random controls for extracted essential physical motifs.

The current essential-motif result is mechanism evidence, not a fully matched
causal comparison.  This script builds the first matched-control library for
that comparison.  For each selected extracted physical motif, it samples
strongly connected control graphs with the same ``N_n`` and ``m``, scores them
against the motif on coarse tree-geometry features, and writes a
``selected.csv`` consumable by ``submit_topology_library_sweep.py``.

Supported control kinds:

* ``random_sc``: fresh strongly connected random digraphs with the same edge
  count.
* ``degree_rewire``: directed double-edge swaps that preserve the motif's in-
  and out-degree sequence when possible.

These controls are not a substitute for a true branch-margin capacity theory.
They are a practical baseline for asking whether extracted motifs outperform
graphs with similar first-order spanning-tree statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from topology_metrics import (
    compute_topology_metrics,
    is_strongly_connected,
    normalize_edges,
    random_strongly_connected_digraph,
)


DEFAULT_MATCH_FEATURES = [
    "d_rel",
    "effective_rank_D",
    "root_tree_count_gini",
    "edge_participation_gini",
    "bottleneck_edge_fraction_095",
    "comparison_branch_common_d_rel_min",
]


CSV_FIELDS = [
    "idx",
    "selected",
    "topology_id",
    "topology_name",
    "family",
    "control_kind",
    "seed",
    "source_topology_id",
    "source_topology_name",
    "source_edge_json",
    "source_n_edges",
    "source_d_rel",
    "source_effective_rank_D",
    "source_root_tree_count_gini",
    "source_edge_participation_gini",
    "match_score",
    "n_nodes",
    "n_edges",
    "p",
    "edge_json",
    "d_rel",
    "comparison_branch_common_d_rel_min",
    "comparison_branch_common_d_rel_mean",
    "comparison_branch_common_d_rel_max",
    "comparison_branch_common_d_rel_gini",
    "comparison_branch_d_rel_min",
    "comparison_branch_d_rel_mean",
    "comparison_branch_d_rel_max",
    "comparison_branch_d_rel_gini",
    "rank_D",
    "effective_rank_D",
    "condition_number_D",
    "root_tree_count_gini",
    "edge_participation_gini",
    "bottleneck_edge_fraction_095",
    "mean_shortest_path",
    "in_degree_cv",
    "out_degree_cv",
    "n_trees_total_enum",
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


def parse_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def finite_or_empty(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def read_rows(path: str, selected_only: bool = True) -> List[dict]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if selected_only:
        rows = [row for row in rows if str(row.get("selected", "1")) in {"1", "True", "true"}]
    return rows


def load_edge_json(path: str) -> Tuple[int, Tuple[Tuple[int, int], ...], dict]:
    with open(path) as handle:
        payload = json.load(handle)
    n_nodes = int(payload["n_nodes"])
    edges = normalize_edges(n_nodes, payload["edges"])
    return n_nodes, edges, payload


def metric_value(row: dict, metrics: dict, feature: str) -> Optional[float]:
    value = parse_float(row.get(feature))
    if value is not None:
        return value
    return parse_float(metrics.get(feature))


def target_metrics_from_row(row: dict, n_nodes: int, edges: Sequence[Tuple[int, int]], p: int, n_context: int, z_dim: int) -> dict:
    computed = compute_topology_metrics(n_nodes, edges, p=p, n_context=n_context, z_dim=z_dim)
    target = dict(computed)
    for key, value in row.items():
        parsed = parse_float(value)
        if parsed is not None:
            target[key] = parsed
    return target


def match_score(candidate: dict, target: dict, features: Sequence[str]) -> float:
    total = 0.0
    used = 0
    for feature in features:
        cand = parse_float(candidate.get(feature))
        targ = parse_float(target.get(feature))
        if cand is None or targ is None:
            continue
        scale = max(abs(targ), 1.0)
        total += abs(cand - targ) / scale
        used += 1
    if used == 0:
        return float("inf")
    return total / used


def directed_degree_rewire(
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
    rng: np.random.Generator,
    swap_attempts: int,
) -> Tuple[Tuple[int, int], ...]:
    current = list(normalize_edges(n_nodes, edges))
    edge_set = set(current)
    if len(current) < 2:
        return tuple(current)

    for _ in range(max(0, swap_attempts)):
        idx_a, idx_b = rng.choice(len(current), size=2, replace=False)
        a, b = current[int(idx_a)]
        c, d = current[int(idx_b)]
        new_a = (a, d)
        new_b = (c, b)
        if new_a[0] == new_a[1] or new_b[0] == new_b[1]:
            continue
        if new_a == new_b:
            continue
        old_a = current[int(idx_a)]
        old_b = current[int(idx_b)]
        reduced = edge_set - {old_a, old_b}
        if new_a in reduced or new_b in reduced:
            continue
        current[int(idx_a)] = new_a
        current[int(idx_b)] = new_b
        edge_set = reduced | {new_a, new_b}
    return normalize_edges(n_nodes, current)


def control_edges(
    n_nodes: int,
    source_edges: Sequence[Tuple[int, int]],
    n_edges: int,
    kind: str,
    seed: int,
    swap_attempts: int,
) -> Tuple[Tuple[int, int], ...]:
    if kind == "random_sc":
        return random_strongly_connected_digraph(n_nodes, n_edges, seed=seed).edges
    if kind == "degree_rewire":
        rng = np.random.default_rng(seed)
        return directed_degree_rewire(n_nodes, source_edges, rng, swap_attempts)
    raise ValueError(f"Unknown control kind {kind!r}")


def source_label(row: dict, index: int) -> str:
    for key in ("topology_id", "topology_name", "mask_name"):
        value = row.get(key)
        if value:
            return str(value)
    return f"source_{index:04d}"


def make_candidate_row(
    idx: int,
    source_idx: int,
    source_row: dict,
    source_metrics: dict,
    kind: str,
    seed: int,
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
    p: int,
    n_context: int,
    z_dim: int,
    edge_json: str,
    features: Sequence[str],
) -> dict:
    metrics = compute_topology_metrics(n_nodes, edges, p=p, n_context=n_context, z_dim=z_dim)
    score = match_score(metrics, source_metrics, features)
    source_id = source_label(source_row, source_idx)
    return {
        "idx": idx,
        "selected": 0,
        "topology_id": f"ctrl{idx:04d}_{kind}_for_{source_id}",
        "topology_name": f"matched_control_{kind}_for_{source_id}_seed{seed}",
        "family": "matched_motif_control",
        "control_kind": kind,
        "seed": seed,
        "source_topology_id": source_row.get("topology_id", source_id),
        "source_topology_name": source_row.get("topology_name", source_id),
        "source_edge_json": source_row.get("edge_json", ""),
        "source_n_edges": source_metrics.get("n_edges"),
        "source_d_rel": source_metrics.get("d_rel"),
        "source_effective_rank_D": source_metrics.get("effective_rank_D"),
        "source_root_tree_count_gini": source_metrics.get("root_tree_count_gini"),
        "source_edge_participation_gini": source_metrics.get("edge_participation_gini"),
        "match_score": score,
        "n_nodes": metrics["n_nodes"],
        "n_edges": metrics["n_edges"],
        "p": metrics["p"],
        "edge_json": edge_json,
        "d_rel": metrics["d_rel"],
        "comparison_branch_common_d_rel_min": metrics.get("comparison_branch_common_d_rel_min"),
        "comparison_branch_common_d_rel_mean": metrics.get("comparison_branch_common_d_rel_mean"),
        "comparison_branch_common_d_rel_max": metrics.get("comparison_branch_common_d_rel_max"),
        "comparison_branch_common_d_rel_gini": metrics.get("comparison_branch_common_d_rel_gini"),
        "comparison_branch_d_rel_min": metrics.get("comparison_branch_d_rel_min"),
        "comparison_branch_d_rel_mean": metrics.get("comparison_branch_d_rel_mean"),
        "comparison_branch_d_rel_max": metrics.get("comparison_branch_d_rel_max"),
        "comparison_branch_d_rel_gini": metrics.get("comparison_branch_d_rel_gini"),
        "rank_D": metrics["rank_D"],
        "effective_rank_D": metrics["effective_rank_D"],
        "condition_number_D": metrics["condition_number_D"],
        "root_tree_count_gini": metrics["root_tree_count_gini"],
        "edge_participation_gini": metrics["edge_participation_gini"],
        "bottleneck_edge_fraction_095": metrics["bottleneck_edge_fraction_095"],
        "mean_shortest_path": metrics["mean_shortest_path"],
        "in_degree_cv": metrics["in_degree_cv"],
        "out_degree_cv": metrics["out_degree_cv"],
        "n_trees_total_enum": metrics["n_trees_total_enum"],
    }


def write_edge_json(
    path: str,
    name: str,
    source_row: dict,
    kind: str,
    seed: int,
    n_nodes: int,
    edges: Sequence[Tuple[int, int]],
    metrics: dict,
) -> None:
    payload = {
        "name": name,
        "family": "matched_motif_control",
        "control_kind": kind,
        "seed": seed,
        "source_topology_id": source_row.get("topology_id", ""),
        "source_topology_name": source_row.get("topology_name", ""),
        "source_edge_json": source_row.get("edge_json", ""),
        "n_nodes": n_nodes,
        "edges": [list(edge) for edge in edges],
        "metrics": metrics,
    }
    with open(path, "w") as handle:
        json.dump(json_ready(payload), handle, indent=2)


def build_controls(args) -> Tuple[List[dict], List[dict]]:
    features = [item.strip() for item in args.match_features.split(",") if item.strip()]
    kinds = [item.strip() for item in args.control_kinds.split(",") if item.strip()]
    rows = read_rows(args.source_csv, selected_only=not args.include_unselected)
    if not rows:
        raise ValueError("No source motifs found")

    topology_dir = os.path.join(args.output_root, "topologies")
    os.makedirs(topology_dir, exist_ok=True)

    candidates = []
    selected = []
    seen = set()
    idx = 0
    for source_idx, source_row in enumerate(rows):
        source_edge_json = source_row.get("edge_json")
        if not source_edge_json:
            continue
        source_path = (
            source_edge_json
            if os.path.isabs(source_edge_json)
            else os.path.abspath(os.path.join(os.path.dirname(args.source_csv), source_edge_json))
        )
        n_nodes, source_edges, _ = load_edge_json(source_path)
        p = int(parse_float(source_row.get("p")) or ((args.N + 1) * args.D))
        source_metrics = target_metrics_from_row(
            source_row,
            n_nodes,
            source_edges,
            p=p,
            n_context=args.N,
            z_dim=args.D,
        )
        n_edges = len(source_edges)
        source_candidates = []
        source_edge_key = tuple(source_edges)

        for kind in kinds:
            kind_candidates = []
            for local_idx in range(args.candidates_per_source):
                seed = args.seed + source_idx * 100000 + local_idx * 17 + sum(ord(ch) for ch in kind)
                try:
                    edges = control_edges(
                        n_nodes,
                        source_edges,
                        n_edges,
                        kind,
                        seed,
                        swap_attempts=args.swap_attempts,
                    )
                except RuntimeError:
                    continue
                if tuple(edges) == source_edge_key and not args.allow_identical:
                    continue
                if not is_strongly_connected(n_nodes, edges):
                    continue
                key = (source_idx, kind, tuple(edges))
                if key in seen:
                    continue
                seen.add(key)

                topology_id = f"ctrl{idx:04d}_{kind}_for_{source_label(source_row, source_idx)}"
                edge_json = os.path.abspath(os.path.join(topology_dir, f"{topology_id}.json"))
                row = make_candidate_row(
                    idx,
                    source_idx,
                    source_row,
                    source_metrics,
                    kind,
                    seed,
                    n_nodes,
                    edges,
                    p,
                    args.N,
                    args.D,
                    edge_json,
                    features,
                )
                metrics = compute_topology_metrics(n_nodes, edges, p=p, n_context=args.N, z_dim=args.D)
                write_edge_json(
                    edge_json,
                    row["topology_name"],
                    source_row,
                    kind,
                    seed,
                    n_nodes,
                    edges,
                    metrics,
                )
                candidates.append(row)
                kind_candidates.append(row)
                idx += 1

            kind_candidates.sort(key=lambda item: float(item["match_score"]))
            source_candidates.extend(kind_candidates[: args.controls_per_source])

        for row in source_candidates:
            row["selected"] = 1
        selected.extend(source_candidates)

    return candidates, selected


def write_csv(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: finite_or_empty(row.get(field)) for field in CSV_FIELDS}
            for row in rows
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_csv", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--control_kinds", default="random_sc,degree_rewire")
    parser.add_argument("--controls_per_source", type=int, default=4)
    parser.add_argument("--candidates_per_source", type=int, default=128)
    parser.add_argument("--swap_attempts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include_unselected", action="store_true")
    parser.add_argument("--allow_identical", action="store_true")
    parser.add_argument("--match_features", default=",".join(DEFAULT_MATCH_FEATURES))
    args = parser.parse_args()

    if args.controls_per_source <= 0:
        raise SystemExit("--controls_per_source must be positive")
    if args.candidates_per_source <= 0:
        raise SystemExit("--candidates_per_source must be positive")

    candidates, selected = build_controls(args)
    if not candidates:
        raise SystemExit("No matched controls generated")

    library_csv = os.path.join(args.output_root, "library.csv")
    selected_csv = os.path.join(args.output_root, "selected.csv")
    summary_json = os.path.join(args.output_root, "summary.json")
    write_csv(library_csv, candidates)
    write_csv(selected_csv, selected)
    summary = {
        "source_csv": os.path.abspath(args.source_csv),
        "output_root": os.path.abspath(args.output_root),
        "n_candidates": len(candidates),
        "n_selected_controls": len(selected),
        "control_kinds": [item.strip() for item in args.control_kinds.split(",") if item.strip()],
        "controls_per_source": args.controls_per_source,
        "candidates_per_source": args.candidates_per_source,
        "match_features": [item.strip() for item in args.match_features.split(",") if item.strip()],
        "library_csv": os.path.abspath(library_csv),
        "selected_csv": os.path.abspath(selected_csv),
    }
    with open(summary_json, "w") as handle:
        json.dump(json_ready(summary), handle, indent=2)

    print(f"Source motifs: {os.path.abspath(args.source_csv)}")
    print(f"Generated candidate controls: {len(candidates)}")
    print(f"Selected matched controls: {len(selected)}")
    print(f"Wrote {library_csv}")
    print(f"Wrote {selected_csv}")


if __name__ == "__main__":
    main()
