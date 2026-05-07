"""Collect topology mechanism-analysis JSON files into a flat CSV table."""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict


FIELDS = [
    "run_dir",
    "label",
    "topology_name",
    "n_samples",
    "target_accuracy",
    "target_logprob_margin_mean",
    "target_logprob_margin_min",
    "target_logprob_margin_branch_mean_min",
    "target_logprob_margin_branch_mean_mean",
    "target_logprob_margin_branch_mean_gini",
    "target_logprob_margin_by_branch",
    "target_accuracy_branch_mean_min",
    "target_accuracy_branch_mean_mean",
    "target_accuracy_branch_mean_gini",
    "target_accuracy_by_branch",
    "branch_active_root_mi",
    "branch_active_tree_mi",
    "branch_active_root_nmi",
    "branch_active_tree_nmi",
    "n_branches_observed",
    "active_root_unique_count",
    "active_tree_unique_count",
    "branch_active_root_purity_mean",
    "branch_active_root_purity_min",
    "branch_active_root_unique_mean",
    "branch_active_tree_purity_mean",
    "branch_active_tree_purity_min",
    "branch_active_tree_unique_mean",
    "branch_active_root_assignment",
    "branch_active_tree_assignment",
    "root_entropy_mean",
    "tree_entropy_mean",
    "edge_importance_mean",
    "edge_importance_max",
    "edge_importance_gini",
    "essential_edges_for_10pct_importance",
    "essential_edges_for_20pct_importance",
    "essential_edges_for_50pct_importance",
    "tree_projection_norm_mean",
    "tree_projection_norm_max",
    "tree_comparison_energy_fraction_mean",
    "tree_comparison_energy_fraction_max",
    "active_tree_comparison_energy_fraction_mean",
    "active_tree_matched_comparison_energy_mean",
    "active_tree_matched_comparison_gap_mean",
    "posterior_comparison_energy_fraction_mean",
    "posterior_matched_comparison_energy_mean",
    "posterior_matched_comparison_gap_mean",
    "input_ablation_baseline_accuracy",
    "input_ablation_max_loss",
    "input_ablation_mean_loss",
    "input_ablation_top_edge",
    "physical_ablation_baseline_accuracy",
    "physical_ablation_max_loss",
    "physical_ablation_mean_loss",
    "physical_ablation_top_edge",
]


def finite_or_empty(value):
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return ""
    return value


def summarize_ablation(metrics, key, prefix):
    ablation = metrics.get(key)
    if not ablation:
        return {
            f"{prefix}_baseline_accuracy": "",
            f"{prefix}_max_loss": "",
            f"{prefix}_mean_loss": "",
            f"{prefix}_top_edge": "",
        }
    losses = [float(value) for value in ablation.get("accuracy_loss", [])]
    top_edges = ablation.get("top_edges_by_loss", [])
    return {
        f"{prefix}_baseline_accuracy": finite_or_empty(ablation.get("baseline_accuracy")),
        f"{prefix}_max_loss": max(losses) if losses else "",
        f"{prefix}_mean_loss": (sum(losses) / len(losses)) if losses else "",
        f"{prefix}_top_edge": json.dumps(top_edges[0]) if top_edges else "",
    }


def _int_list(values):
    if values is None:
        return []
    result = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            return []
    return result


def _float_list(values):
    if values is None:
        return []
    result = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(parsed):
            return []
        result.append(parsed)
    return result


def _gini(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return ""
    min_value = min(values)
    if min_value < 0:
        values = [value - min_value for value in values]
    total = sum(values)
    if abs(total) <= 1e-12:
        return 0.0
    values = sorted(values)
    n = len(values)
    weighted = sum((idx + 1) * value for idx, value in enumerate(values))
    return float((2.0 * weighted / (n * total)) - ((n + 1.0) / n))


def _branch_value_summary(branches, values, prefix):
    if len(branches) != len(values) or not branches:
        return {
            f"{prefix}_branch_mean_min": "",
            f"{prefix}_branch_mean_mean": "",
            f"{prefix}_branch_mean_gini": "",
            f"{prefix}_by_branch": "",
        }
    rows = []
    means = []
    groups = defaultdict(list)
    for branch, value in zip(branches, values):
        groups[branch].append(value)
    for branch in sorted(groups):
        branch_values = groups[branch]
        mean_value = float(sum(branch_values) / len(branch_values))
        means.append(mean_value)
        rows.append(
            {
                "branch": int(branch),
                "mean": mean_value,
                "min": float(min(branch_values)),
                "n": int(len(branch_values)),
            }
        )
    return {
        f"{prefix}_branch_mean_min": float(min(means)),
        f"{prefix}_branch_mean_mean": float(sum(means) / len(means)),
        f"{prefix}_branch_mean_gini": _gini(means),
        f"{prefix}_by_branch": json.dumps(rows),
    }


def summarize_branch_margins(metrics):
    branches = _int_list(metrics.get("branch_ids"))
    margins = _float_list(metrics.get("target_logprob_margin"))
    correct = _float_list(metrics.get("target_correct"))
    result = {}
    result.update(_branch_value_summary(branches, margins, "target_logprob_margin"))
    result.update(
        _branch_value_summary(
            branches,
            [100.0 * value for value in correct],
            "target_accuracy",
        )
    )
    for key in [
        "target_logprob_margin_branch_mean_min",
        "target_logprob_margin_branch_mean_mean",
        "target_logprob_margin_branch_mean_gini",
        "target_accuracy_branch_mean_min",
        "target_accuracy_branch_mean_mean",
        "target_accuracy_branch_mean_gini",
    ]:
        if result.get(key) == "" and metrics.get(key) not in (None, ""):
            result[key] = metrics.get(key)
    for key in ["target_logprob_margin_by_branch", "target_accuracy_by_branch"]:
        if result.get(key) == "" and metrics.get(key) not in (None, ""):
            result[key] = json.dumps(metrics.get(key))
    return result


def _assignment_summary(branches, assignments, skip_negative=False):
    if len(branches) != len(assignments) or not branches:
        return {
            "n_observed": "",
            "unique_count": "",
            "purity_mean": "",
            "purity_min": "",
            "unique_mean": "",
            "assignment": "",
        }
    groups = defaultdict(list)
    global_values = []
    for branch, assignment in zip(branches, assignments):
        if skip_negative and assignment < 0:
            continue
        groups[branch].append(assignment)
        global_values.append(assignment)
    if not groups:
        return {
            "n_observed": 0,
            "unique_count": 0,
            "purity_mean": "",
            "purity_min": "",
            "unique_mean": "",
            "assignment": "",
        }

    rows = []
    purities = []
    unique_counts = []
    for branch in sorted(groups):
        values = groups[branch]
        counts = Counter(values)
        dominant, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        purity = float(count / len(values))
        unique_count = len(counts)
        purities.append(purity)
        unique_counts.append(unique_count)
        rows.append(
            {
                "branch": int(branch),
                "dominant": int(dominant),
                "fraction": purity,
                "n": int(len(values)),
                "unique": int(unique_count),
            }
        )

    return {
        "n_observed": int(len(rows)),
        "unique_count": int(len(set(global_values))),
        "purity_mean": float(sum(purities) / len(purities)),
        "purity_min": float(min(purities)),
        "unique_mean": float(sum(unique_counts) / len(unique_counts)),
        "assignment": json.dumps(rows),
    }


def _entropy(values):
    if not values:
        return 0.0
    total = float(len(values))
    return float(
        -sum(
            (count / total) * math.log(count / total)
            for count in Counter(values).values()
        )
    )


def _mutual_information(branches, assignments, skip_negative=False):
    pairs = [
        (branch, assignment)
        for branch, assignment in zip(branches, assignments)
        if not (skip_negative and assignment < 0)
    ]
    if not pairs:
        return None
    joint_counts = Counter(pairs)
    branch_counts = Counter(branch for branch, _ in pairs)
    assignment_counts = Counter(assignment for _, assignment in pairs)
    total = float(len(pairs))
    value = 0.0
    for (branch, assignment), count in joint_counts.items():
        pxy = count / total
        px = branch_counts[branch] / total
        py = assignment_counts[assignment] / total
        value += pxy * math.log(pxy / (px * py))
    return float(value)


def _normalized_branch_mi(branches, assignments, skip_negative=False):
    pairs = [
        (branch, assignment)
        for branch, assignment in zip(branches, assignments)
        if not (skip_negative and assignment < 0)
    ]
    if not pairs:
        return ""
    filtered_branches = [branch for branch, _ in pairs]
    branch_entropy = _entropy(filtered_branches)
    if branch_entropy <= 1e-12:
        return ""
    mi = _mutual_information(filtered_branches, [assignment for _, assignment in pairs])
    return "" if mi is None else float(mi / branch_entropy)


def summarize_branch_assignments(metrics):
    branches = _int_list(metrics.get("branch_ids"))
    roots = _int_list(metrics.get("active_root"))
    trees = _int_list(metrics.get("active_tree"))
    root_summary = _assignment_summary(branches, roots)
    tree_summary = _assignment_summary(branches, trees, skip_negative=True)
    return {
        "branch_active_root_nmi": _normalized_branch_mi(branches, roots),
        "branch_active_tree_nmi": _normalized_branch_mi(branches, trees, skip_negative=True),
        "n_branches_observed": root_summary["n_observed"],
        "active_root_unique_count": root_summary["unique_count"],
        "active_tree_unique_count": tree_summary["unique_count"],
        "branch_active_root_purity_mean": root_summary["purity_mean"],
        "branch_active_root_purity_min": root_summary["purity_min"],
        "branch_active_root_unique_mean": root_summary["unique_mean"],
        "branch_active_tree_purity_mean": tree_summary["purity_mean"],
        "branch_active_tree_purity_min": tree_summary["purity_min"],
        "branch_active_tree_unique_mean": tree_summary["unique_mean"],
        "branch_active_root_assignment": root_summary["assignment"],
        "branch_active_tree_assignment": tree_summary["assignment"],
    }


def load_mechanism(path):
    with open(path) as f:
        metrics = json.load(f)
    run_dir = metrics.get("run_dir") or os.path.dirname(path)
    row = {
        "run_dir": run_dir,
        "label": os.path.basename(run_dir.rstrip(os.sep)),
        "topology_name": metrics.get("topology_name"),
        "n_samples": metrics.get("n_samples"),
        "target_accuracy": metrics.get("target_accuracy"),
        "target_logprob_margin_mean": metrics.get("target_logprob_margin_mean"),
        "target_logprob_margin_min": metrics.get("target_logprob_margin_min"),
        "branch_active_root_mi": metrics.get("branch_active_root_mi"),
        "branch_active_tree_mi": metrics.get("branch_active_tree_mi"),
        "root_entropy_mean": metrics.get("root_entropy_mean"),
        "tree_entropy_mean": metrics.get("tree_entropy_mean"),
        "edge_importance_mean": metrics.get("edge_importance_mean"),
        "edge_importance_max": metrics.get("edge_importance_max"),
        "edge_importance_gini": metrics.get("edge_importance_gini"),
        "essential_edges_for_10pct_importance": metrics.get("essential_edges_for_10pct_importance"),
        "essential_edges_for_20pct_importance": metrics.get("essential_edges_for_20pct_importance"),
        "essential_edges_for_50pct_importance": metrics.get("essential_edges_for_50pct_importance"),
        "tree_projection_norm_mean": metrics.get("tree_projection_norm_mean"),
        "tree_projection_norm_max": metrics.get("tree_projection_norm_max"),
        "tree_comparison_energy_fraction_mean": metrics.get("tree_comparison_energy_fraction_mean"),
        "tree_comparison_energy_fraction_max": metrics.get("tree_comparison_energy_fraction_max"),
        "active_tree_comparison_energy_fraction_mean": metrics.get("active_tree_comparison_energy_fraction_mean"),
        "active_tree_matched_comparison_energy_mean": metrics.get("active_tree_matched_comparison_energy_mean"),
        "active_tree_matched_comparison_gap_mean": metrics.get("active_tree_matched_comparison_gap_mean"),
        "posterior_comparison_energy_fraction_mean": metrics.get("posterior_comparison_energy_fraction_mean"),
        "posterior_matched_comparison_energy_mean": metrics.get("posterior_matched_comparison_energy_mean"),
        "posterior_matched_comparison_gap_mean": metrics.get("posterior_matched_comparison_gap_mean"),
    }
    row.update(summarize_branch_margins(metrics))
    row.update(summarize_branch_assignments(metrics))
    row.update(summarize_ablation(metrics, "input_edge_ablation", "input_ablation"))
    row.update(summarize_ablation(metrics, "physical_edge_ablation", "physical_ablation"))
    return {field: finite_or_empty(row.get(field)) for field in FIELDS}


def iter_mechanism_files(root, metrics_filename):
    for current, _, files in os.walk(root):
        if metrics_filename in files:
            yield os.path.join(current, metrics_filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--metrics_filename", type=str, default="mechanism_metrics.json")
    args = parser.parse_args()

    rows = [
        load_mechanism(path)
        for path in sorted(iter_mechanism_files(args.input_root, args.metrics_filename))
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
