"""Run simple topology-vs-ICL regression diagnostics from a collected CSV.

The intent is not to replace careful statistical analysis. This script gives a
fast, dependency-light first pass at the central question:

    Do topology-derived predictors explain novel-class ICL accuracy beyond
    raw trainable degree count?

It reports ordinary least-squares R^2 for nested predictor sets and a
leave-one-out R^2 when enough runs are available.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np


DEFAULT_TARGET = "test_novel_classes"
PREDICTOR_FALLBACKS = {
    "comparison_branch_common_d_rel_min": "comparison_branch_d_rel_min",
    "comparison_branch_common_d_rel_gini": "comparison_branch_d_rel_gini",
}

PREDICTOR_SETS = {
    "raw_count": [
        "raw_physical_parameter_count",
    ],
    "raw_plus_drel": [
        "raw_physical_parameter_count",
        "d_rel",
    ],
    "input_count": [
        "input_coupled_parameter_count",
    ],
    "input_count_plus_drel": [
        "input_coupled_parameter_count",
        "d_rel",
    ],
    "input_count_plus_branch_drel": [
        "input_coupled_parameter_count",
        "d_rel",
        "comparison_branch_common_d_rel_min",
        "comparison_branch_common_d_rel_gini",
        "comparison_branch_d_rel_min",
        "comparison_branch_d_rel_gini",
    ],
    "tree_geometry": [
        "raw_physical_parameter_count",
        "d_rel",
        "effective_rank_D",
        "root_tree_count_gini",
        "edge_participation_gini",
        "bottleneck_edge_fraction_095",
    ],
    "trainability_geometry": [
        "raw_physical_parameter_count",
        "d_rel",
        "effective_rank_D",
        "condition_number_D",
        "root_tree_count_gini",
        "edge_participation_var",
        "mean_shortest_path",
    ],
    "masked_tree_geometry": [
        "input_coupled_parameter_count",
        "d_rel",
        "comparison_branch_common_d_rel_min",
        "comparison_branch_common_d_rel_gini",
        "comparison_branch_d_rel_min",
        "comparison_branch_d_rel_gini",
        "effective_rank_D_masked",
        "condition_number_D_masked",
        "input_edge_load_gini",
        "input_coord_load_gini",
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


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def row_value(row, name):
    value = row.get(name)
    if value not in (None, ""):
        return value
    fallback = PREDICTOR_FALLBACKS.get(name)
    return row.get(fallback) if fallback else value


def design_matrix(rows, predictors, target):
    usable = []
    for row in rows:
        y = parse_float(row.get(target))
        xs = [parse_float(row_value(row, name)) for name in predictors]
        if y is not None and all(value is not None for value in xs):
            usable.append((row, xs, y))
    if not usable:
        return [], np.zeros((0, len(predictors) + 1)), np.zeros(0)
    X = np.asarray([[1.0] + xs for _, xs, _ in usable], dtype=float)
    y = np.asarray([item[2] for item in usable], dtype=float)
    return [item[0] for item in usable], X, y


def fit_ols(X, y):
    if X.shape[0] == 0:
        return {
            "n": 0,
            "r2": None,
            "coefficients": [],
            "rmse": None,
        }
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    predictions = X @ coefficients
    residual = y - predictions
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = None if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "n": int(X.shape[0]),
        "r2": r2,
        "coefficients": coefficients.tolist(),
        "rmse": rmse,
    }


def standardize_columns(X):
    if X.shape[0] == 0:
        return X
    Xs = X.copy()
    for col in range(1, X.shape[1]):
        values = X[:, col]
        std = values.std()
        if std > 0.0:
            Xs[:, col] = (values - values.mean()) / std
        else:
            Xs[:, col] = 0.0
    return Xs


def leave_one_out_r2(X, y):
    n = X.shape[0]
    effective_rank = np.linalg.matrix_rank(X) if n else 0
    if n < effective_rank + 2:
        return None
    predictions = np.zeros(n, dtype=float)
    for idx in range(n):
        keep = np.arange(n) != idx
        coefficients = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        predictions[idx] = X[idx] @ coefficients
    ss_res = float(np.sum((y - predictions) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return None if ss_tot == 0.0 else 1.0 - ss_res / ss_tot


def summarize_by_edge_count(rows, target):
    groups = defaultdict(list)
    for row in rows:
        edge_count = row.get("n_edges")
        value = parse_float(row.get(target))
        if edge_count not in (None, "") and value is not None:
            groups[edge_count].append(value)
    summary = {}
    for edge_count, values in sorted(groups.items(), key=lambda item: float(item[0])):
        arr = np.asarray(values, dtype=float)
        summary[edge_count] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "max": float(arr.max()),
            "std": float(arr.std()),
        }
    return summary


def run_regressions(rows, target):
    report = {
        "target": target,
        "n_rows_total": len(rows),
        "models": {},
        "by_edge_count": summarize_by_edge_count(rows, target),
    }
    for name, predictors in PREDICTOR_SETS.items():
        usable_rows, X, y = design_matrix(rows, predictors, target)
        Xs = standardize_columns(X)
        fit = fit_ols(Xs, y)
        fit["predictors"] = predictors
        fit["leave_one_out_r2"] = leave_one_out_r2(Xs, y)
        fit["n_topology_families"] = len({row.get("topology_name") for row in usable_rows})
        report["models"][name] = fit
    return report


def print_report(report):
    print(f"Target: {report['target']}")
    print(f"Rows: {report['n_rows_total']}")
    print("\nNested predictor diagnostics:")
    for name, fit in report["models"].items():
        r2 = fit["r2"]
        loo = fit["leave_one_out_r2"]
        r2_text = "NA" if r2 is None else f"{r2:.3f}"
        loo_text = "NA" if loo is None else f"{loo:.3f}"
        print(
            f"  {name:22s} n={fit['n']:3d} "
            f"R2={r2_text:>7s} LOO_R2={loo_text:>7s} RMSE={fit['rmse']}"
        )
    if report["by_edge_count"]:
        print("\nAccuracy by edge count:")
        for edge_count, stats in report["by_edge_count"].items():
            print(
                f"  m={edge_count:>4s} n={stats['n']:3d} "
                f"mean={stats['mean']:.2f} max={stats['max']:.2f} std={stats['std']:.2f}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    report = run_regressions(rows, args.target)
    print_report(report)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
