"""Collect topology ICL run outputs into a flat CSV table."""

import argparse
import csv
import json
import os
import pickle

from topology_metrics import compute_topology_metrics


FIELDS = [
    "run_dir",
    "label",
    "topology_name",
    "physical_topology_name",
    "input_mask_name",
    "input_mask_family",
    "input_mask_seed",
    "seed",
    "topology_seed",
    "n_nodes",
    "n_edges",
    "p",
    "raw_physical_parameter_count",
    "input_coupled_parameter_count",
    "input_coupled_edge_count",
    "input_coupled_coord_count",
    "input_parameter_density",
    "input_edge_density",
    "input_coord_density",
    "input_edge_load_gini",
    "input_coord_load_gini",
    "n_req",
    "d_rel",
    "d_rel_minus_n_req",
    "comparison_branch_d_rel_min",
    "comparison_branch_d_rel_mean",
    "comparison_branch_d_rel_max",
    "comparison_branch_d_rel_gini",
    "comparison_branch_common_d_rel_min",
    "comparison_branch_common_d_rel_mean",
    "comparison_branch_common_d_rel_max",
    "comparison_branch_common_d_rel_gini",
    "comparison_branch_input_count_min",
    "comparison_branch_input_count_mean",
    "comparison_branch_input_count_max",
    "comparison_branch_input_count_gini",
    "comparison_branch_input_overlap_min",
    "comparison_branch_input_overlap_mean",
    "comparison_branch_input_overlap_max",
    "comparison_branch_input_overlap_gini",
    "rank_D",
    "effective_rank_D",
    "condition_number_D",
    "effective_rank_D_masked",
    "condition_number_D_masked",
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

BRANCH_METRIC_FIELDS = [
    "comparison_branch_d_rel_min",
    "comparison_branch_d_rel_mean",
    "comparison_branch_d_rel_max",
    "comparison_branch_d_rel_gini",
    "comparison_branch_common_d_rel_min",
    "comparison_branch_common_d_rel_mean",
    "comparison_branch_common_d_rel_max",
    "comparison_branch_common_d_rel_gini",
    "comparison_branch_input_count_min",
    "comparison_branch_input_count_mean",
    "comparison_branch_input_count_max",
    "comparison_branch_input_count_gini",
    "comparison_branch_input_overlap_min",
    "comparison_branch_input_overlap_mean",
    "comparison_branch_input_overlap_max",
    "comparison_branch_input_overlap_gini",
]

_BRANCH_METRIC_CACHE = {}


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


def _freeze_nested(value):
    if isinstance(value, list):
        return tuple(_freeze_nested(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_nested(item) for item in value)
    return value


def backfill_branch_metrics(metrics, topology_payload, config):
    """Backfill branch-comparison structural metrics for older run artifacts."""

    if all(metrics.get(field) not in (None, "") for field in BRANCH_METRIC_FIELDS):
        return metrics
    try:
        n_context = int(config["N"])
        z_dim = int(config["D"])
        p = int(metrics.get("p") or ((n_context + 1) * z_dim))
        n_nodes = int(topology_payload.get("n_nodes") or metrics["n_nodes"])
        edges = topology_payload.get("edges") or metrics["edges"]
        input_mask = topology_payload.get("input_mask")
    except (KeyError, TypeError, ValueError):
        return metrics

    cache_key = (
        n_nodes,
        tuple(tuple(edge) for edge in edges),
        _freeze_nested(input_mask),
        p,
        n_context,
        z_dim,
    )
    if cache_key not in _BRANCH_METRIC_CACHE:
        try:
            _BRANCH_METRIC_CACHE[cache_key] = compute_topology_metrics(
                n_nodes,
                edges,
                p=p,
                input_mask=input_mask,
                n_context=n_context,
                z_dim=z_dim,
            )
        except (ValueError, TypeError):
            _BRANCH_METRIC_CACHE[cache_key] = {}
    backfilled = _BRANCH_METRIC_CACHE[cache_key]
    for field in BRANCH_METRIC_FIELDS:
        if metrics.get(field) in (None, "") and field in backfilled:
            metrics[field] = backfilled[field]
    return metrics


def load_run(run_dir):
    results_path = os.path.join(run_dir, "results.pkl")
    metrics_path = os.path.join(run_dir, "topology_metrics.json")
    config_path = os.path.join(run_dir, "config.json")
    topology_path = os.path.join(run_dir, "topology.json")
    if not os.path.exists(results_path) or not os.path.exists(metrics_path):
        return None
    if not os.path.exists(config_path):
        print(f"Skipping {run_dir}: missing config.json")
        return None

    with open(results_path, "rb") as f:
        payload = pickle.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(config_path) as f:
        config = json.load(f)
    topology_payload = {}
    if os.path.exists(topology_path):
        with open(topology_path) as f:
            topology_payload = json.load(f)
    input_mask_metadata = topology_payload.get("input_mask_metadata", {})
    metrics = backfill_branch_metrics(metrics, topology_payload, config)

    history = payload.get("history", {})
    icl_values = [value for value in history.get("icl_acc", []) if value is not None]
    iwl_values = [value for value in history.get("iwl_acc", []) if value is not None]
    results = payload.get("results", {})

    row = {
        "run_dir": run_dir,
        "label": os.path.basename(run_dir.rstrip(os.sep)),
        "topology_name": metrics.get("topology_name"),
        "physical_topology_name": metrics.get("physical_topology_name"),
        "input_mask_name": metrics.get("input_mask_name"),
        "input_mask_family": (
            metrics.get("input_mask_family")
            or input_mask_metadata.get("mask_family")
            or input_mask_metadata.get("family")
        ),
        "input_mask_seed": metrics.get("input_mask_seed") or input_mask_metadata.get("seed"),
        "seed": config.get("seed"),
        "topology_seed": config.get("topology_seed"),
        "n_nodes": metrics.get("n_nodes"),
        "n_edges": metrics.get("n_edges"),
        "p": metrics.get("p"),
        "raw_physical_parameter_count": metrics.get("raw_physical_parameter_count"),
        "input_coupled_parameter_count": metrics.get("input_coupled_parameter_count"),
        "input_coupled_edge_count": metrics.get("input_coupled_edge_count"),
        "input_coupled_coord_count": metrics.get("input_coupled_coord_count"),
        "input_parameter_density": metrics.get("input_parameter_density"),
        "input_edge_density": metrics.get("input_edge_density"),
        "input_coord_density": metrics.get("input_coord_density"),
        "input_edge_load_gini": metrics.get("input_edge_load_gini"),
        "input_coord_load_gini": metrics.get("input_coord_load_gini"),
        "n_req": metrics.get("n_req"),
        "d_rel": metrics.get("d_rel"),
        "d_rel_minus_n_req": metrics.get("d_rel_minus_n_req"),
        "comparison_branch_d_rel_min": metrics.get("comparison_branch_d_rel_min"),
        "comparison_branch_d_rel_mean": metrics.get("comparison_branch_d_rel_mean"),
        "comparison_branch_d_rel_max": metrics.get("comparison_branch_d_rel_max"),
        "comparison_branch_d_rel_gini": metrics.get("comparison_branch_d_rel_gini"),
        "comparison_branch_common_d_rel_min": metrics.get("comparison_branch_common_d_rel_min"),
        "comparison_branch_common_d_rel_mean": metrics.get("comparison_branch_common_d_rel_mean"),
        "comparison_branch_common_d_rel_max": metrics.get("comparison_branch_common_d_rel_max"),
        "comparison_branch_common_d_rel_gini": metrics.get("comparison_branch_common_d_rel_gini"),
        "comparison_branch_input_count_min": metrics.get("comparison_branch_input_count_min"),
        "comparison_branch_input_count_mean": metrics.get("comparison_branch_input_count_mean"),
        "comparison_branch_input_count_max": metrics.get("comparison_branch_input_count_max"),
        "comparison_branch_input_count_gini": metrics.get("comparison_branch_input_count_gini"),
        "comparison_branch_input_overlap_min": metrics.get("comparison_branch_input_overlap_min"),
        "comparison_branch_input_overlap_mean": metrics.get("comparison_branch_input_overlap_mean"),
        "comparison_branch_input_overlap_max": metrics.get("comparison_branch_input_overlap_max"),
        "comparison_branch_input_overlap_gini": metrics.get("comparison_branch_input_overlap_gini"),
        "rank_D": metrics.get("rank_D"),
        "effective_rank_D": metrics.get("effective_rank_D"),
        "condition_number_D": metrics.get("condition_number_D"),
        "effective_rank_D_masked": metrics.get("effective_rank_D_masked"),
        "condition_number_D_masked": metrics.get("condition_number_D_masked"),
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
