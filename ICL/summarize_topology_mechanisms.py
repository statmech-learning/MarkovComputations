"""Join topology and mechanism CSVs and summarize topology-use diagnostics."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np


DEFAULT_TARGET = "test_novel_classes"

TOPOLOGY_COLUMNS = [
    "n_edges",
    "input_coupled_parameter_count",
    "d_rel",
    "effective_rank_D",
    "effective_rank_D_masked",
    "condition_number_D",
    "condition_number_D_masked",
    "root_tree_count_gini",
    "edge_participation_gini",
    "bottleneck_edge_fraction_095",
    "mean_shortest_path",
    "input_edge_load_gini",
    "input_coord_load_gini",
]

MECHANISM_COLUMNS = [
    "target_logprob_margin_mean",
    "branch_active_root_mi",
    "branch_active_tree_mi",
    "active_root_unique_count",
    "active_tree_unique_count",
    "branch_active_root_purity_mean",
    "branch_active_tree_purity_mean",
    "branch_active_tree_unique_mean",
    "edge_importance_gini",
    "essential_edges_for_50pct_importance",
    "tree_projection_norm_mean",
    "tree_comparison_energy_fraction_mean",
    "tree_comparison_energy_fraction_max",
    "active_tree_comparison_energy_fraction_mean",
    "active_tree_matched_comparison_energy_mean",
    "active_tree_matched_comparison_gap_mean",
    "posterior_comparison_energy_fraction_mean",
    "posterior_matched_comparison_energy_mean",
    "posterior_matched_comparison_gap_mean",
    "input_ablation_max_loss",
    "physical_ablation_max_loss",
]


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_by_label(path):
    with open(path, newline="") as f:
        return {row["label"]: row for row in csv.DictReader(f)}


def join_rows(topology_rows, mechanism_rows, target):
    rows = []
    for label, topo in topology_rows.items():
        mechanism = mechanism_rows.get(label)
        if mechanism is None:
            continue
        target_value = parse_float(topo.get(target))
        if target_value is None:
            continue
        row = {
            "label": label,
            "topology_name": topo.get("topology_name"),
            "target": target_value,
        }
        for name in TOPOLOGY_COLUMNS:
            row[name] = parse_float(topo.get(name))
        for name in MECHANISM_COLUMNS:
            row[name] = parse_float(mechanism.get(name))
        rows.append(row)
    return rows


def pearson(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2 or ys.size < 2:
        return None
    x_std = xs.std()
    y_std = ys.std()
    if x_std <= 1e-12 or y_std <= 1e-12:
        return None
    return float(np.mean(((xs - xs.mean()) / x_std) * ((ys - ys.mean()) / y_std)))


def correlation_report(rows, predictor_names):
    report = {}
    for name in predictor_names:
        usable = [
            (row[name], row["target"])
            for row in rows
            if row.get(name) is not None
        ]
        if not usable:
            report[name] = None
            continue
        xs, ys = zip(*usable)
        report[name] = pearson(xs, ys)
    return report


def residualize_by_edge_count(rows, key):
    groups = defaultdict(list)
    for idx, row in enumerate(rows):
        if row.get(key) is not None and row.get("n_edges") is not None:
            groups[row["n_edges"]].append(idx)
    residuals = {}
    for idxs in groups.values():
        mean = float(np.mean([rows[idx][key] for idx in idxs]))
        for idx in idxs:
            residuals[idx] = rows[idx][key] - mean
    return residuals


def residual_correlation_report(rows, predictor_names):
    target_residuals = residualize_by_edge_count(rows, "target")
    report = {}
    for name in predictor_names:
        predictor_residuals = residualize_by_edge_count(rows, name)
        common = sorted(set(target_residuals) & set(predictor_residuals))
        if not common:
            report[name] = None
            continue
        report[name] = pearson(
            [predictor_residuals[idx] for idx in common],
            [target_residuals[idx] for idx in common],
        )
    return report


def group_summary(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        key = row.get(group_key)
        if key is not None:
            groups[key].append(row)
    summary = {}
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        values = np.asarray([row["target"] for row in group], dtype=float)
        item = {
            "n": int(values.size),
            "target_mean": float(values.mean()),
            "target_max": float(values.max()),
            "target_std": float(values.std()),
        }
        for metric in [
            "d_rel",
            "effective_rank_D",
            "target_logprob_margin_mean",
            "branch_active_tree_mi",
            "branch_active_tree_purity_mean",
        ]:
            metric_values = [row[metric] for row in group if row.get(metric) is not None]
            if metric_values:
                item[f"{metric}_mean"] = float(np.mean(metric_values))
        summary[str(key)] = item
    return summary


def family_within_edge_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("n_edges") is not None and row.get("topology_name") is not None:
            groups[(row["n_edges"], row["topology_name"])].append(row)
    summary = []
    for (edge_count, family), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        values = np.asarray([row["target"] for row in group], dtype=float)
        def mean_or_none(metric):
            metric_values = [row[metric] for row in group if row.get(metric) is not None]
            return None if not metric_values else float(np.mean(metric_values))

        summary.append(
            {
                "n_edges": edge_count,
                "topology_name": family,
                "n": int(values.size),
                "target_mean": float(values.mean()),
                "target_max": float(values.max()),
                "d_rel_mean": mean_or_none("d_rel"),
                "branch_active_tree_mi_mean": mean_or_none("branch_active_tree_mi"),
                "branch_active_tree_purity_mean": mean_or_none("branch_active_tree_purity_mean"),
                "target_logprob_margin_mean": mean_or_none("target_logprob_margin_mean"),
            }
        )
    return summary


def build_report(rows, target):
    predictors = TOPOLOGY_COLUMNS[1:] + MECHANISM_COLUMNS
    return {
        "target": target,
        "n_joined_rows": len(rows),
        "overall_correlations": correlation_report(rows, predictors),
        "within_edge_count_residual_correlations": residual_correlation_report(rows, predictors),
        "by_edge_count": group_summary(rows, "n_edges"),
        "by_topology_name": group_summary(rows, "topology_name"),
        "family_within_edge_count": family_within_edge_summary(rows),
    }


def format_value(value):
    return "NA" if value is None else f"{value:.3f}"


def print_report(report):
    print(f"Target: {report['target']}")
    print(f"Rows joined: {report['n_joined_rows']}")
    print("\nOverall correlations:")
    for name, value in report["overall_correlations"].items():
        print(f"  corr(target,{name})={format_value(value)}")
    print("\nWithin-edge-count residual correlations:")
    for name, value in report["within_edge_count_residual_correlations"].items():
        print(f"  resid_corr(target,{name})={format_value(value)}")
    if report["by_edge_count"]:
        print("\nBy edge count:")
        for edge_count, stats in report["by_edge_count"].items():
            print(
                f"  m={float(edge_count):.0f} n={stats['n']:3d} "
                f"mean={stats['target_mean']:.2f} max={stats['target_max']:.2f} "
                f"tree_mi_mean={stats.get('branch_active_tree_mi_mean', float('nan')):.3f} "
                f"tree_purity_mean={stats.get('branch_active_tree_purity_mean_mean', float('nan')):.3f}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology_csv", type=str, required=True)
    parser.add_argument("--mechanism_csv", type=str, required=True)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    rows = join_rows(load_by_label(args.topology_csv), load_by_label(args.mechanism_csv), args.target)
    report = build_report(rows, args.target)
    print_report(report)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
