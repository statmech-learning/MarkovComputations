"""Extract trainable essential-subgraph motifs from trained topology ICL runs.

The post-training mechanism analyzer writes per-edge sensitivity and ablation
scores in ``mechanism_metrics.json``. This script turns those functional edge
scores into physical subgraphs that can be retrained from scratch through the
existing ``submit_topology_library_sweep.py --library_csv`` path.

The raw essential edge set is selected by score coverage or by a fixed top-k.
Because first-order steady states require strong connectivity for the standard
matrix-tree interpretation, the script can greedily augment the raw set with
the next most important edges until the subgraph is strongly connected.
"""

import argparse
import csv
import json
import math
import os
from collections import OrderedDict

import numpy as np

from topology_metrics import compute_topology_metrics, is_strongly_connected, normalize_edges


CSV_FIELDS = [
    "idx",
    "selected",
    "topology_id",
    "topology_name",
    "family",
    "importance_source",
    "coverage_fraction",
    "top_k",
    "score_total",
    "raw_essential_edges",
    "augmented_edges",
    "source_run_count",
    "source_labels",
    "source_topology_names",
    "source_test_novel_classes_max",
    "source_test_novel_classes_mean",
    "source_target_accuracy_max",
    "source_target_accuracy_mean",
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


def parse_float(value):
    if value is None or value == "":
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


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_by_label(path):
    if not path:
        return {}
    with open(path, newline="") as f:
        return {row["label"]: row for row in csv.DictReader(f)}


def iter_run_dirs(input_root):
    for current, _, files in os.walk(input_root):
        if "topology.json" in files and "mechanism_metrics.json" in files:
            yield current


def score_vector(metrics, source):
    if source == "edge_importance":
        return np.asarray(metrics.get("edge_importance", []), dtype=float)
    if source == "input_ablation_loss":
        ablation = metrics.get("input_edge_ablation", {})
        return np.asarray(ablation.get("accuracy_loss", []), dtype=float)
    if source == "physical_ablation_loss":
        ablation = metrics.get("physical_edge_ablation", {})
        return np.asarray(ablation.get("accuracy_loss", []), dtype=float)
    raise ValueError(f"Unknown importance source {source!r}")


def selected_indices(scores, coverage_fraction, top_k, min_edges):
    scores = np.asarray(scores, dtype=float)
    positive = np.maximum(scores, 0.0)
    order = np.argsort(positive)[::-1]
    if top_k is not None:
        k = min(len(order), max(0, int(top_k)))
    elif positive.sum() > 0:
        cumulative = np.cumsum(positive[order]) / positive.sum()
        k = int(np.searchsorted(cumulative, coverage_fraction) + 1)
    else:
        k = 0
    k = min(len(order), max(k, min_edges))
    return list(map(int, order[:k]))


def augment_to_strongly_connected(n_nodes, edges, selected, scores):
    selected_set = set(selected)
    if is_strongly_connected(n_nodes, [edges[idx] for idx in selected]):
        return list(selected)

    positive = np.maximum(np.asarray(scores, dtype=float), 0.0)
    remaining = [idx for idx in np.argsort(positive)[::-1] if int(idx) not in selected_set]
    augmented = list(selected)
    for idx in remaining:
        augmented.append(int(idx))
        selected_set.add(int(idx))
        if is_strongly_connected(n_nodes, [edges[item] for item in augmented]):
            return augmented
    return augmented


def mean_or_none(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.mean(values))


def max_or_none(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.max(values))


def build_candidate(run_dir, args, topology_rows):
    label = os.path.basename(run_dir.rstrip(os.sep))
    topology_row = topology_rows.get(label, {})
    with open(os.path.join(run_dir, "topology.json")) as f:
        topology = json.load(f)
    with open(os.path.join(run_dir, "mechanism_metrics.json")) as f:
        mechanism = json.load(f)

    n_nodes = int(topology["n_nodes"])
    edges = normalize_edges(n_nodes, topology["edges"])
    scores = score_vector(mechanism, args.importance_source)
    if scores.shape != (len(edges),):
        raise ValueError(
            f"{run_dir}: {args.importance_source} has shape {scores.shape}, "
            f"expected ({len(edges)},)"
        )

    raw_selected = selected_indices(
        scores,
        coverage_fraction=args.coverage_fraction,
        top_k=args.top_k,
        min_edges=args.min_raw_edges,
    )
    selected = raw_selected
    if args.ensure_strongly_connected:
        selected = augment_to_strongly_connected(n_nodes, edges, raw_selected, scores)
    subgraph_edges = normalize_edges(n_nodes, [edges[idx] for idx in selected])
    if args.ensure_strongly_connected and not is_strongly_connected(n_nodes, subgraph_edges):
        return None

    test_novel = parse_float(topology_row.get("test_novel_classes"))
    target_accuracy = parse_float(mechanism.get("target_accuracy"))
    return {
        "label": label,
        "run_dir": run_dir,
        "source_topology_name": topology.get("name", mechanism.get("topology_name", "")),
        "n_nodes": n_nodes,
        "edges": subgraph_edges,
        "raw_essential_edges": len(raw_selected),
        "augmented_edges": len(subgraph_edges) - len(raw_selected),
        "score_total": float(np.maximum(scores, 0.0).sum()),
        "source_test_novel_classes": test_novel,
        "source_target_accuracy": target_accuracy,
    }


def merge_candidates(candidates):
    groups = OrderedDict()
    for item in candidates:
        key = (item["n_nodes"], tuple(item["edges"]))
        if key not in groups:
            groups[key] = {
                "n_nodes": item["n_nodes"],
                "edges": item["edges"],
                "raw_essential_edges": [],
                "augmented_edges": [],
                "score_total": [],
                "source_labels": [],
                "source_run_dirs": [],
                "source_topology_names": [],
                "source_test_novel_classes": [],
                "source_target_accuracy": [],
            }
        group = groups[key]
        group["raw_essential_edges"].append(item["raw_essential_edges"])
        group["augmented_edges"].append(item["augmented_edges"])
        group["score_total"].append(item["score_total"])
        group["source_labels"].append(item["label"])
        group["source_run_dirs"].append(item["run_dir"])
        group["source_topology_names"].append(item["source_topology_name"])
        group["source_test_novel_classes"].append(item["source_test_novel_classes"])
        group["source_target_accuracy"].append(item["source_target_accuracy"])
    return list(groups.values())


def selection_score(row):
    primary = parse_float(row.get("source_test_novel_classes_max"))
    secondary = parse_float(row.get("source_target_accuracy_max"))
    primary = -1.0 if primary is None else primary
    secondary = -1.0 if secondary is None else secondary
    return (primary, secondary, int(row.get("source_run_count", 0)), -int(row.get("n_edges", 0)))


def write_outputs(groups, args):
    p = (args.N + 1) * args.D
    topology_dir = os.path.join(args.output_root, "topologies")
    os.makedirs(topology_dir, exist_ok=True)

    rows = []
    for idx, group in enumerate(groups):
        source_test_values = group["source_test_novel_classes"]
        source_target_values = group["source_target_accuracy"]
        source_names = sorted(set(name for name in group["source_topology_names"] if name))
        metrics = compute_topology_metrics(
            group["n_nodes"],
            group["edges"],
            p=p,
            n_context=args.N,
            z_dim=args.D,
        )
        topology_id = f"ess{idx:04d}_{args.importance_source}_m{len(group['edges'])}"
        topology_name = f"essential_{args.importance_source}_n{group['n_nodes']}_m{len(group['edges'])}_{idx:04d}"
        edge_json = os.path.abspath(os.path.join(topology_dir, f"{topology_id}.json"))
        payload = {
            "name": topology_name,
            "family": "essential_subgraph",
            "importance_source": args.importance_source,
            "coverage_fraction": args.coverage_fraction,
            "top_k": args.top_k,
            "n_nodes": group["n_nodes"],
            "edges": [list(edge) for edge in group["edges"]],
            "source_labels": group["source_labels"],
            "source_topology_names": source_names,
            "metrics": metrics,
        }
        with open(edge_json, "w") as f:
            json.dump(json_ready(payload), f, indent=2)

        rows.append(
            {
                "idx": idx,
                "selected": 0,
                "topology_id": topology_id,
                "topology_name": topology_name,
                "family": "essential_subgraph",
                "importance_source": args.importance_source,
                "coverage_fraction": args.coverage_fraction,
                "top_k": "" if args.top_k is None else args.top_k,
                "score_total": mean_or_none(group["score_total"]),
                "raw_essential_edges": mean_or_none(group["raw_essential_edges"]),
                "augmented_edges": mean_or_none(group["augmented_edges"]),
                "source_run_count": len(group["source_labels"]),
                "source_labels": ";".join(group["source_labels"]),
                "source_topology_names": ";".join(source_names),
                "source_test_novel_classes_max": max_or_none(source_test_values),
                "source_test_novel_classes_mean": mean_or_none(source_test_values),
                "source_target_accuracy_max": max_or_none(source_target_values),
                "source_target_accuracy_mean": mean_or_none(source_target_values),
                "n_nodes": metrics["n_nodes"],
                "n_edges": metrics["n_edges"],
                "p": metrics["p"],
                "edge_json": edge_json,
                "d_rel": metrics["d_rel"],
                "comparison_branch_common_d_rel_min": metrics.get(
                    "comparison_branch_common_d_rel_min"
                ),
                "comparison_branch_common_d_rel_mean": metrics.get(
                    "comparison_branch_common_d_rel_mean"
                ),
                "comparison_branch_common_d_rel_max": metrics.get(
                    "comparison_branch_common_d_rel_max"
                ),
                "comparison_branch_common_d_rel_gini": metrics.get(
                    "comparison_branch_common_d_rel_gini"
                ),
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
        )

    ranked = sorted(range(len(rows)), key=lambda idx: selection_score(rows[idx]), reverse=True)
    selected = set(ranked if args.select_topologies <= 0 else ranked[: args.select_topologies])
    for idx, row in enumerate(rows):
        row["selected"] = 1 if idx in selected else 0

    library_csv = os.path.join(args.output_root, "library.csv")
    selected_csv = os.path.join(args.output_root, "selected.csv")
    for path, output_rows in [
        (library_csv, rows),
        (selected_csv, [row for idx, row in enumerate(rows) if idx in selected]),
    ]:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows({field: finite_or_empty(row.get(field)) for field in CSV_FIELDS} for row in output_rows)

    summary = {
        "input_root": os.path.abspath(args.input_root),
        "output_root": os.path.abspath(args.output_root),
        "importance_source": args.importance_source,
        "coverage_fraction": args.coverage_fraction,
        "top_k": args.top_k,
        "n_unique_subgraphs": len(rows),
        "n_selected_subgraphs": len(selected),
        "library_csv": os.path.abspath(library_csv),
        "selected_csv": os.path.abspath(selected_csv),
    }
    with open(os.path.join(args.output_root, "summary.json"), "w") as f:
        json.dump(json_ready(summary), f, indent=2)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--topology_csv", type=str, default=None)
    parser.add_argument(
        "--importance_source",
        type=str,
        default="edge_importance",
        choices=["edge_importance", "input_ablation_loss", "physical_ablation_loss"],
    )
    parser.add_argument("--coverage_fraction", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_raw_edges", type=int, default=0)
    parser.add_argument("--ensure_strongly_connected", action="store_true", default=True)
    parser.add_argument("--allow_not_strongly_connected", dest="ensure_strongly_connected", action="store_false")
    parser.add_argument("--select_topologies", type=int, default=16)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--D", type=int, default=4)
    args = parser.parse_args()

    if args.coverage_fraction <= 0.0 or args.coverage_fraction > 1.0:
        raise SystemExit("--coverage_fraction must be in (0, 1]")

    os.makedirs(args.output_root, exist_ok=True)
    topology_rows = load_by_label(args.topology_csv)
    candidates = []
    for run_dir in sorted(iter_run_dirs(args.input_root)):
        candidate = build_candidate(run_dir, args, topology_rows)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise SystemExit("No candidate essential subgraphs extracted")

    rows, summary = write_outputs(merge_candidates(candidates), args)
    print(f"Input root: {os.path.abspath(args.input_root)}")
    print(f"Output root: {os.path.abspath(args.output_root)}")
    print(f"Candidate runs: {len(candidates)}")
    print(f"Unique subgraphs: {summary['n_unique_subgraphs']}")
    print(f"Selected subgraphs: {summary['n_selected_subgraphs']}")
    print(f"Wrote {summary['library_csv']}")
    print(f"Wrote {summary['selected_csv']}")


if __name__ == "__main__":
    main()
