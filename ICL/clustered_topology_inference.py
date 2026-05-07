"""Cluster-aware topology ICL regression diagnostics.

The existing run-level regressions are useful descriptive summaries, but
training seeds nested inside a topology/mask group are not independent
topologies.  This script provides dependency-light statistical upgrades:

* group-level regressions on topology/mask aggregates,
* clustered bootstrap deltas over topology/mask groups,
* leave-one-family/backbone-out prediction,
* random-intercept style residual decomposition.

It intentionally uses ordinary least squares and bootstrap summaries rather
than a heavyweight mixed-effects dependency, so it can run on the same cluster
control plane as the rest of the topology tooling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from regress_topology_results import (
    PREDICTOR_SETS,
    design_matrix,
    fit_ols,
    leave_one_out_r2,
    parse_float,
    standardize_columns,
)


DEFAULT_TARGET = "test_novel_classes"
DEFAULT_CLUSTER = "topology_name"
DEFAULT_FAMILY = "physical_topology_name"
DEFAULT_MODELS = [
    "raw_count",
    "raw_plus_drel",
    "input_count",
    "input_count_plus_drel",
    "input_count_plus_branch_drel",
    "tree_geometry",
    "masked_tree_geometry",
    "branch_rank_weighted_capacity",
    "branch_rank_weighted_capacity_plus_drel",
    "tropical_tree_capacity",
    "tropical_tree_capacity_plus_drel",
]


def load_rows(path: str) -> List[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def cluster_key(row: dict, cluster_col: str) -> str:
    value = row.get(cluster_col)
    if value in (None, ""):
        value = row.get("label")
    return str(value)


def group_rows(rows: Sequence[dict], cluster_col: str) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        groups[cluster_key(row, cluster_col)].append(row)
    return dict(groups)


def first_non_empty(rows: Sequence[dict], column: str):
    for row in rows:
        value = row.get(column)
        if value not in (None, ""):
            return value
    return ""


def aggregate_seed_groups(
    rows: Sequence[dict],
    target: str = DEFAULT_TARGET,
    cluster_col: str = DEFAULT_CLUSTER,
) -> List[dict]:
    """Aggregate seed-level rows into topology/mask group rows."""

    groups = group_rows(rows, cluster_col)
    aggregate_rows = []
    all_columns = sorted({column for row in rows for column in row})
    for key, items in sorted(groups.items(), key=lambda item: item[0]):
        values = [parse_float(row.get(target)) for row in items]
        values = [value for value in values if value is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        out = {
            "group": key,
            cluster_col: key,
            "n_runs": int(arr.size),
            "target_mean": float(arr.mean()),
            "target_max": float(arr.max()),
            "target_std": float(arr.std()),
        }
        for column in all_columns:
            if column not in out:
                out[column] = first_non_empty(items, column)
        aggregate_rows.append(out)
    return aggregate_rows


def regression_summary(rows: Sequence[dict], target: str, model_names: Sequence[str]) -> dict:
    report = {}
    for name in model_names:
        predictors = PREDICTOR_SETS[name]
        usable_rows, X, y = design_matrix(rows, predictors, target)
        Xs = standardize_columns(X)
        fit = fit_ols(Xs, y)
        fit["predictors"] = predictors
        fit["leave_one_out_r2"] = leave_one_out_r2(Xs, y)
        fit["n_groups_or_rows"] = len(usable_rows)
        report[name] = fit
    return report


def fit_predict(
    train_rows: Sequence[dict],
    test_rows: Sequence[dict],
    predictors: Sequence[str],
    target: str,
) -> Tuple[np.ndarray, np.ndarray]:
    _, X_train, y_train = design_matrix(train_rows, predictors, target)
    usable_test, X_test, y_test = design_matrix(test_rows, predictors, target)
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return np.zeros(0), np.zeros(0)

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    means[0] = 0.0
    stds[0] = 1.0
    stds[stds == 0.0] = 1.0
    X_train_s = (X_train - means) / stds
    X_train_s[:, 0] = 1.0
    X_test_s = (X_test - means) / stds
    X_test_s[:, 0] = 1.0
    coefficients = np.linalg.lstsq(X_train_s, y_train, rcond=None)[0]
    return y_test, X_test_s @ coefficients


def r2_score(y: np.ndarray, predictions: np.ndarray) -> Optional[float]:
    if y.size == 0:
        return None
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return None
    return 1.0 - float(np.sum((y - predictions) ** 2)) / ss_tot


def rmse(y: np.ndarray, predictions: np.ndarray) -> Optional[float]:
    if y.size == 0:
        return None
    return float(np.sqrt(np.mean((y - predictions) ** 2)))


def leave_family_out(
    rows: Sequence[dict],
    target: str,
    family_col: str,
    model_names: Sequence[str],
) -> dict:
    families = sorted({row.get(family_col) for row in rows if row.get(family_col) not in (None, "")})
    report = {}
    for model_name in model_names:
        predictors = PREDICTOR_SETS[model_name]
        family_rows = []
        all_y = []
        all_pred = []
        for family in families:
            train = [row for row in rows if row.get(family_col) != family]
            test = [row for row in rows if row.get(family_col) == family]
            y, pred = fit_predict(train, test, predictors, target)
            family_rows.append(
                {
                    "family": family,
                    "n": int(y.size),
                    "r2": r2_score(y, pred),
                    "rmse": rmse(y, pred),
                }
            )
            all_y.extend(y.tolist())
            all_pred.extend(pred.tolist())
        y_all = np.asarray(all_y, dtype=float)
        pred_all = np.asarray(all_pred, dtype=float)
        report[model_name] = {
            "predictors": predictors,
            "families": family_rows,
            "pooled_r2": r2_score(y_all, pred_all),
            "pooled_rmse": rmse(y_all, pred_all),
        }
    return report


def _bootstrap_r2_delta(
    rows: Sequence[dict],
    target: str,
    cluster_col: str,
    baseline: str,
    candidate: str,
    n_bootstrap: int,
    seed: int,
) -> dict:
    clusters = group_rows(rows, cluster_col)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    deltas = []
    baseline_scores = []
    candidate_scores = []
    for _ in range(n_bootstrap):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        sample = []
        for key in sampled_keys:
            sample.extend(clusters[key])
        _, X0, y0 = design_matrix(sample, PREDICTOR_SETS[baseline], target)
        _, X1, y1 = design_matrix(sample, PREDICTOR_SETS[candidate], target)
        if y0.shape[0] < 3 or y1.shape[0] < 3:
            continue
        score0 = fit_ols(standardize_columns(X0), y0)["r2"]
        score1 = fit_ols(standardize_columns(X1), y1)["r2"]
        if score0 is None or score1 is None:
            continue
        baseline_scores.append(score0)
        candidate_scores.append(score1)
        deltas.append(score1 - score0)
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        return {
            "baseline": baseline,
            "candidate": candidate,
            "n_bootstrap_effective": 0,
            "delta_mean": None,
            "delta_ci95": [None, None],
            "prob_delta_positive": None,
        }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "n_bootstrap_effective": int(arr.size),
        "baseline_r2_mean": float(np.mean(baseline_scores)),
        "candidate_r2_mean": float(np.mean(candidate_scores)),
        "delta_mean": float(arr.mean()),
        "delta_median": float(np.median(arr)),
        "delta_ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "prob_delta_positive": float(np.mean(arr > 0.0)),
    }


def clustered_bootstrap_deltas(
    rows: Sequence[dict],
    target: str,
    cluster_col: str,
    baseline: str,
    candidates: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> dict:
    return {
        candidate: _bootstrap_r2_delta(
            rows,
            target=target,
            cluster_col=cluster_col,
            baseline=baseline,
            candidate=candidate,
            n_bootstrap=n_bootstrap,
            seed=seed + idx,
        )
        for idx, candidate in enumerate(candidates)
    }


def residual_decomposition(
    rows: Sequence[dict],
    target: str,
    cluster_col: str,
    model_name: str,
) -> dict:
    predictors = PREDICTOR_SETS[model_name]
    usable_rows, X, y = design_matrix(rows, predictors, target)
    if X.shape[0] == 0:
        return {"model": model_name, "n": 0}
    Xs = standardize_columns(X)
    coefficients = np.linalg.lstsq(Xs, y, rcond=None)[0]
    residuals = y - Xs @ coefficients
    grouped = defaultdict(list)
    for row, residual in zip(usable_rows, residuals):
        grouped[cluster_key(row, cluster_col)].append(float(residual))
    cluster_means = np.asarray([np.mean(values) for values in grouped.values()], dtype=float)
    within_vars = [
        float(np.var(values))
        for values in grouped.values()
        if len(values) > 1
    ]
    return {
        "model": model_name,
        "n": int(X.shape[0]),
        "n_clusters": len(grouped),
        "residual_std_total": float(np.std(residuals)),
        "residual_std_between_cluster_means": float(np.std(cluster_means)) if cluster_means.size else 0.0,
        "residual_var_within_cluster_mean": float(np.mean(within_vars)) if within_vars else 0.0,
    }


def run_clustered_inference(
    run_rows: Sequence[dict],
    target: str = DEFAULT_TARGET,
    cluster_col: str = DEFAULT_CLUSTER,
    family_col: str = DEFAULT_FAMILY,
    model_names: Sequence[str] = DEFAULT_MODELS,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict:
    aggregate_rows = aggregate_seed_groups(run_rows, target=target, cluster_col=cluster_col)
    bootstrap_candidates = [name for name in model_names if name != "raw_count"]
    return {
        "target": target,
        "cluster_col": cluster_col,
        "family_col": family_col,
        "n_run_rows": len(run_rows),
        "n_clusters": len(group_rows(run_rows, cluster_col)),
        "n_aggregate_rows": len(aggregate_rows),
        "n_families": len({row.get(family_col) for row in run_rows if row.get(family_col) not in (None, "")}),
        "group_level": {
            "target_mean": regression_summary(aggregate_rows, "target_mean", model_names),
            "target_max": regression_summary(aggregate_rows, "target_max", model_names),
            "target_std": regression_summary(aggregate_rows, "target_std", model_names),
        },
        "cluster_bootstrap_run_level": clustered_bootstrap_deltas(
            run_rows,
            target=target,
            cluster_col=cluster_col,
            baseline="raw_count",
            candidates=bootstrap_candidates,
            n_bootstrap=n_bootstrap,
            seed=seed,
        ),
        "leave_family_out_group_target_mean": leave_family_out(
            aggregate_rows,
            target="target_mean",
            family_col=family_col,
            model_names=model_names,
        ),
        "residual_decomposition_run_level": {
            name: residual_decomposition(run_rows, target, cluster_col, name)
            for name in model_names
        },
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def print_summary(report: dict):
    print(f"Target: {report['target']}")
    print(
        f"Rows: {report['n_run_rows']}  "
        f"clusters: {report['n_clusters']}  "
        f"families: {report['n_families']}"
    )
    print("\nGroup-level target_mean LOO R2:")
    for name, fit in report["group_level"]["target_mean"].items():
        loo = fit["leave_one_out_r2"]
        loo_text = "NA" if loo is None else f"{loo:.3f}"
        print(f"  {name:28s} n={fit['n']:3d} LOO_R2={loo_text}")
    print("\nCluster bootstrap run-level delta R2 vs raw_count:")
    for name, item in report["cluster_bootstrap_run_level"].items():
        delta = item["delta_mean"]
        if delta is None:
            print(f"  {name:28s} insufficient")
            continue
        lo, hi = item["delta_ci95"]
        print(
            f"  {name:28s} mean={delta:.3f} "
            f"CI95=[{lo:.3f}, {hi:.3f}] "
            f"P(delta>0)={item['prob_delta_positive']:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_csv", required=True)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--cluster_col", default=DEFAULT_CLUSTER)
    parser.add_argument("--family_col", default=DEFAULT_FAMILY)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--n_bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in model_names if name not in PREDICTOR_SETS]
    if unknown:
        raise SystemExit(f"Unknown model names: {unknown}")
    rows = load_rows(args.run_csv)
    report = run_clustered_inference(
        rows,
        target=args.target,
        cluster_col=args.cluster_col,
        family_col=args.family_col,
        model_names=model_names,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print_summary(report)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(_json_ready(report), handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
