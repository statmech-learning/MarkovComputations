"""Aggregate topology ICL runs across training seeds.

This report separates topology-level expressivity from seed-level trainability:

* target_mean: mean novel-class ICL accuracy across training seeds.
* target_max: best-seed novel-class ICL accuracy.
* target_std: seed variability for the same physical topology.

It can also join mechanism metrics and aggregate active-tree/margin/ablation
features across seeds.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np


DEFAULT_TARGET = "test_novel_classes"

STATIC_COLUMNS = [
    "topology_name",
    "physical_topology_name",
    "input_mask_name",
    "input_mask_family",
    "input_mask_seed",
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
    "comparison_branch_input_count_min",
    "comparison_branch_input_count_mean",
    "comparison_branch_input_count_max",
    "comparison_branch_input_count_gini",
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
]

RUN_COLUMNS = [
    "train_acc_final",
    "val_acc_final",
    "icl_acc_final_eval",
    "icl_acc_max_eval",
    "icl_acc_mean_eval",
    "test_in_dist",
    "test_novel_classes",
]

MECHANISM_COLUMNS = [
    "target_accuracy",
    "target_logprob_margin_mean",
    "target_logprob_margin_min",
    "target_logprob_margin_branch_mean_min",
    "target_logprob_margin_branch_mean_mean",
    "target_logprob_margin_branch_mean_gini",
    "target_accuracy_branch_mean_min",
    "target_accuracy_branch_mean_mean",
    "target_accuracy_branch_mean_gini",
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
    "root_entropy_mean",
    "tree_entropy_mean",
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

PREDICTOR_SETS = {
    "rank_only": ["d_rel"],
    "input_count": ["input_coupled_parameter_count"],
    "input_count_plus_drel": ["input_coupled_parameter_count", "d_rel"],
    "tree_geometry": [
        "d_rel",
        "effective_rank_D",
        "condition_number_D",
        "root_tree_count_gini",
        "edge_participation_gini",
        "bottleneck_edge_fraction_095",
        "mean_shortest_path",
    ],
    "masked_tree_geometry": [
        "input_coupled_parameter_count",
        "d_rel",
        "comparison_branch_d_rel_min",
        "comparison_branch_d_rel_gini",
        "effective_rank_D_masked",
        "condition_number_D_masked",
        "input_edge_load_gini",
        "input_coord_load_gini",
    ],
    "mechanism": [
        "target_logprob_margin_mean_mean",
        "target_logprob_margin_branch_mean_min_mean",
        "branch_active_root_mi_mean",
        "branch_active_tree_mi_mean",
        "branch_active_tree_nmi_mean",
        "branch_active_tree_purity_mean_mean",
        "input_ablation_max_loss_mean",
    ],
    "projection_alignment": [
        "active_tree_matched_comparison_energy_mean_mean",
        "active_tree_matched_comparison_gap_mean_mean",
        "posterior_matched_comparison_energy_mean_mean",
        "posterior_matched_comparison_gap_mean_mean",
    ],
}


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


def finite_or_empty(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.mean(values))


def std(values):
    values = [value for value in values if value is not None]
    return None if len(values) < 2 else float(np.std(values))


def max_or_none(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.max(values))


def first_non_empty(rows, column):
    for row in rows:
        value = row.get(column)
        if value not in (None, ""):
            return value
    return ""


def mechanism_by_label(path):
    if not path:
        return {}
    return {row["label"]: row for row in load_rows(path)}


def aggregate_rows(topology_rows, mechanism_rows, group_by, target):
    groups = defaultdict(list)
    for row in topology_rows:
        key = row.get(group_by)
        if key in (None, ""):
            key = row.get("label")
        groups[key].append(row)

    aggregated = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        targets = [parse_float(row.get(target)) for row in rows]
        item = {
            "group": key,
            "group_by": group_by,
            "n_runs": len(rows),
            "labels": ";".join(row.get("label", "") for row in rows),
            "seeds": ";".join(str(row.get("seed", "")) for row in rows),
            "target": target,
            "target_mean": mean(targets),
            "target_max": max_or_none(targets),
            "target_std": std(targets),
        }

        for column in STATIC_COLUMNS:
            parsed = [parse_float(row.get(column)) for row in rows]
            parsed_values = [value for value in parsed if value is not None]
            if parsed_values:
                item[column] = float(np.mean(parsed_values))
            else:
                item[column] = first_non_empty(rows, column)

        for column in RUN_COLUMNS:
            values = [parse_float(row.get(column)) for row in rows]
            item[f"{column}_mean"] = mean(values)
            item[f"{column}_max"] = max_or_none(values)
            item[f"{column}_std"] = std(values)

        if mechanism_rows:
            joined_mechanisms = [
                mechanism_rows[row["label"]]
                for row in rows
                if row.get("label") in mechanism_rows
            ]
            item["n_mechanism_runs"] = len(joined_mechanisms)
            for column in MECHANISM_COLUMNS:
                values = [parse_float(row.get(column)) for row in joined_mechanisms]
                item[f"{column}_mean"] = mean(values)
                item[f"{column}_max"] = max_or_none(values)
                item[f"{column}_std"] = std(values)
        else:
            item["n_mechanism_runs"] = 0

        aggregated.append(item)

    return aggregated


def usable_xy(rows, predictors, outcome):
    usable = []
    for row in rows:
        y = parse_float(row.get(outcome))
        xs = [parse_float(row.get(predictor)) for predictor in predictors]
        if y is not None and all(value is not None for value in xs):
            usable.append((xs, y))
    if not usable:
        return np.zeros((0, len(predictors) + 1)), np.zeros(0)
    X = np.asarray([[1.0] + xs for xs, _ in usable], dtype=float)
    y = np.asarray([y for _, y in usable], dtype=float)
    return X, y


def standardize(X):
    Xs = X.copy()
    if Xs.shape[0] == 0:
        return Xs
    for col in range(1, Xs.shape[1]):
        scale = Xs[:, col].std()
        if scale > 1e-12:
            Xs[:, col] = (Xs[:, col] - Xs[:, col].mean()) / scale
        else:
            Xs[:, col] = 0.0
    return Xs


def fit_ols(X, y):
    if X.shape[0] == 0:
        return {"n": 0, "r2": None, "rmse": None}
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    prediction = X @ coefficients
    residual = y - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(X.shape[0]),
        "r2": None if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "coefficients": coefficients.tolist(),
    }


def leave_one_out_r2(X, y):
    effective_rank = np.linalg.matrix_rank(X) if X.shape[0] else 0
    if X.shape[0] < effective_rank + 2:
        return None
    predictions = np.zeros(X.shape[0], dtype=float)
    for idx in range(X.shape[0]):
        keep = np.arange(X.shape[0]) != idx
        coefficients = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        predictions[idx] = X[idx] @ coefficients
    ss_res = float(np.sum((y - predictions) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return None if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)


def regression_report(rows):
    report = {}
    for outcome in ["target_mean", "target_max", "target_std"]:
        report[outcome] = {}
        for name, predictors in PREDICTOR_SETS.items():
            X, y = usable_xy(rows, predictors, outcome)
            X = standardize(X)
            fit = fit_ols(X, y)
            fit["predictors"] = predictors
            fit["leave_one_out_r2"] = leave_one_out_r2(X, y)
            report[outcome][name] = fit
    return report


def write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_or_empty(row.get(key)) for key in fieldnames})


def print_report(rows, regressions):
    print(f"Topology groups: {len(rows)}")
    if rows:
        target_means = [row["target_mean"] for row in rows if row.get("target_mean") is not None]
        target_max = [row["target_max"] for row in rows if row.get("target_max") is not None]
        target_std = [row["target_std"] for row in rows if row.get("target_std") is not None]
        print(
            "Target summary: "
            f"mean-of-means={mean(target_means):.2f}, "
            f"max-of-max={max_or_none(target_max):.2f}, "
            f"mean-seed-std={mean(target_std):.2f}"
        )
    for outcome, models in regressions.items():
        print(f"\nOutcome: {outcome}")
        for name, fit in models.items():
            r2 = "NA" if fit["r2"] is None else f"{fit['r2']:.3f}"
            loo = "NA" if fit["leave_one_out_r2"] is None else f"{fit['leave_one_out_r2']:.3f}"
            print(f"  {name:14s} n={fit['n']:3d} R2={r2:>7s} LOO_R2={loo:>7s} RMSE={fit['rmse']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology_csv", type=str, required=True)
    parser.add_argument("--mechanism_csv", type=str, default=None)
    parser.add_argument("--group_by", type=str, default="topology_name")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    rows = aggregate_rows(
        load_rows(args.topology_csv),
        mechanism_by_label(args.mechanism_csv),
        args.group_by,
        args.target,
    )
    regressions = regression_report(rows)
    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} topology groups to {args.output_csv}")
    print_report(rows, regressions)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"n_groups": len(rows), "regressions": regressions}, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
