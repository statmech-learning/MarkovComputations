"""Collect topology ICL run outputs into a flat CSV table."""

import argparse
import csv
import json
import os
import pickle


FIELDS = [
    "run_dir",
    "label",
    "topology_name",
    "seed",
    "topology_seed",
    "n_nodes",
    "n_edges",
    "p",
    "raw_physical_parameter_count",
    "input_coupled_parameter_count",
    "n_req",
    "d_rel",
    "d_rel_minus_n_req",
    "rank_D",
    "effective_rank_D",
    "condition_number_D",
    "root_tree_count_cv",
    "root_tree_count_gini",
    "edge_participation_var",
    "edge_participation_gini",
    "bottleneck_edge_fraction_095",
    "mean_shortest_path",
    "train_acc_final",
    "val_acc_final",
    "icl_acc_final_eval",
    "icl_acc_max_eval",
    "icl_acc_mean_eval",
    "iwl_acc_final_eval",
    "test_in_dist",
    "test_novel_classes",
    "execution_time",
]


def finite_or_empty(value):
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return ""
    return value


def final_non_none(values):
    cleaned = [value for value in values if value is not None]
    return cleaned[-1] if cleaned else None


def load_run(run_dir):
    results_path = os.path.join(run_dir, "results.pkl")
    metrics_path = os.path.join(run_dir, "topology_metrics.json")
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(results_path) or not os.path.exists(metrics_path):
        return None

    with open(results_path, "rb") as f:
        payload = pickle.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    history = payload.get("history", {})
    icl_values = [value for value in history.get("icl_acc", []) if value is not None]
    iwl_values = [value for value in history.get("iwl_acc", []) if value is not None]
    results = payload.get("results", {})

    row = {
        "run_dir": run_dir,
        "label": os.path.basename(run_dir.rstrip(os.sep)),
        "topology_name": metrics.get("topology_name"),
        "seed": config.get("seed"),
        "topology_seed": config.get("topology_seed"),
        "n_nodes": metrics.get("n_nodes"),
        "n_edges": metrics.get("n_edges"),
        "p": metrics.get("p"),
        "raw_physical_parameter_count": metrics.get("raw_physical_parameter_count"),
        "input_coupled_parameter_count": metrics.get("input_coupled_parameter_count"),
        "n_req": metrics.get("n_req"),
        "d_rel": metrics.get("d_rel"),
        "d_rel_minus_n_req": metrics.get("d_rel_minus_n_req"),
        "rank_D": metrics.get("rank_D"),
        "effective_rank_D": metrics.get("effective_rank_D"),
        "condition_number_D": metrics.get("condition_number_D"),
        "root_tree_count_cv": metrics.get("root_tree_count_cv"),
        "root_tree_count_gini": metrics.get("root_tree_count_gini"),
        "edge_participation_var": metrics.get("edge_participation_var"),
        "edge_participation_gini": metrics.get("edge_participation_gini"),
        "bottleneck_edge_fraction_095": metrics.get("bottleneck_edge_fraction_095"),
        "mean_shortest_path": metrics.get("mean_shortest_path"),
        "train_acc_final": final_non_none(history.get("train_acc", [])),
        "val_acc_final": final_non_none(history.get("val_acc", [])),
        "icl_acc_final_eval": icl_values[-1] if icl_values else None,
        "icl_acc_max_eval": max(icl_values) if icl_values else None,
        "icl_acc_mean_eval": sum(icl_values) / len(icl_values) if icl_values else None,
        "iwl_acc_final_eval": iwl_values[-1] if iwl_values else None,
        "test_in_dist": results.get("in_dist"),
        "test_novel_classes": results.get("novel_classes"),
        "execution_time": payload.get("execution_time"),
    }
    return {field: finite_or_empty(row.get(field)) for field in FIELDS}


def iter_run_dirs(root):
    for current, _, files in os.walk(root):
        if "results.pkl" in files and "topology_metrics.json" in files:
            yield current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(iter_run_dirs(args.input_root)):
        row = load_run(run_dir)
        if row is not None:
            rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()

