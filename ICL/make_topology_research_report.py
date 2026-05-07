"""Build a consolidated topology-ICL research progress report.

This script intentionally reads the existing analysis artifacts instead of
recomputing model diagnostics. It is meant to make long-running cluster sweeps
auditable: fixed-edge topology sweeps, post-training mechanism summaries,
topology-level seed aggregates, and essential-subgraph retrain comparisons all
land in one Markdown/JSON report.
"""

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from collections import OrderedDict

import numpy as np


DEFAULT_TARGET = "test_novel_classes"

KEY_RUN_MODELS = [
    "raw_count",
    "raw_plus_drel",
    "input_count",
    "input_count_plus_drel",
    "input_count_plus_branch_drel",
    "tree_geometry",
    "masked_tree_geometry",
    "trainability_geometry",
]
KEY_AGG_MODELS = [
    "rank_only",
    "input_count",
    "input_count_plus_drel",
    "tree_geometry",
    "masked_tree_geometry",
    "mechanism",
    "projection_alignment",
]
KEY_CORRELATIONS = [
    "d_rel",
    "comparison_branch_d_rel_min",
    "comparison_branch_d_rel_gini",
    "effective_rank_D",
    "effective_rank_D_masked",
    "condition_number_D",
    "condition_number_D_masked",
    "root_tree_count_gini",
    "edge_participation_gini",
    "input_coupled_parameter_count",
    "input_edge_load_gini",
    "input_coord_load_gini",
    "target_logprob_margin_mean",
    "target_logprob_margin_branch_mean_min",
    "branch_active_root_mi",
    "branch_active_tree_mi",
    "tree_comparison_energy_fraction_mean",
    "posterior_matched_comparison_gap_mean",
    "input_ablation_max_loss",
    "physical_ablation_max_loss",
]

POOLED_RUN_MODELS = OrderedDict(
    [
        ("edge_count", ["n_edges"]),
        ("edge_plus_drel", ["n_edges", "d_rel"]),
        ("input_count", ["input_coupled_parameter_count"]),
        ("input_plus_drel", ["input_coupled_parameter_count", "d_rel"]),
        (
            "input_plus_branch_drel",
            [
                "input_coupled_parameter_count",
                "d_rel",
                "comparison_branch_d_rel_min",
                "comparison_branch_d_rel_gini",
            ],
        ),
        (
            "edge_plus_tree_geometry",
            [
                "n_edges",
                "d_rel",
                "effective_rank_D",
                "condition_number_D",
                "root_tree_count_gini",
                "edge_participation_gini",
                "mean_shortest_path",
            ],
        ),
        (
            "input_plus_masked_geometry",
            [
                "input_coupled_parameter_count",
                "d_rel",
                "comparison_branch_d_rel_min",
                "comparison_branch_d_rel_gini",
                "effective_rank_D_masked",
                "condition_number_D_masked",
                "input_edge_load_gini",
                "input_coord_load_gini",
            ],
        ),
        (
            "edge_plus_mechanism",
            [
                "n_edges",
                "target_logprob_margin_mean",
                "target_logprob_margin_branch_mean_min",
                "branch_active_tree_mi",
                "input_ablation_max_loss",
            ],
        ),
        (
            "edge_plus_projection",
            [
                "n_edges",
                "posterior_matched_comparison_gap_mean",
                "tree_comparison_energy_fraction_mean",
                "active_tree_matched_comparison_gap_mean",
            ],
        ),
    ]
)

POOLED_AGGREGATE_MODELS = OrderedDict(
    [
        ("edge_count", ["n_edges"]),
        ("edge_plus_drel", ["n_edges", "d_rel"]),
        ("input_count", ["input_coupled_parameter_count"]),
        ("input_plus_drel", ["input_coupled_parameter_count", "d_rel"]),
        (
            "input_plus_branch_drel",
            [
                "input_coupled_parameter_count",
                "d_rel",
                "comparison_branch_d_rel_min",
                "comparison_branch_d_rel_gini",
            ],
        ),
        (
            "edge_plus_tree_geometry",
            [
                "n_edges",
                "d_rel",
                "effective_rank_D",
                "condition_number_D",
                "root_tree_count_gini",
                "edge_participation_gini",
                "mean_shortest_path",
            ],
        ),
        (
            "input_plus_masked_geometry",
            [
                "input_coupled_parameter_count",
                "d_rel",
                "comparison_branch_d_rel_min",
                "comparison_branch_d_rel_gini",
                "effective_rank_D_masked",
                "condition_number_D_masked",
                "input_edge_load_gini",
                "input_coord_load_gini",
            ],
        ),
        (
            "edge_plus_mechanism",
            [
                "n_edges",
                "target_logprob_margin_mean_mean",
                "target_logprob_margin_branch_mean_min_mean",
                "branch_active_tree_mi_mean",
                "input_ablation_max_loss_mean",
            ],
        ),
        (
            "edge_plus_projection",
            [
                "n_edges",
                "posterior_matched_comparison_gap_mean_mean",
                "tree_comparison_energy_fraction_mean_mean",
                "active_tree_matched_comparison_gap_mean_mean",
            ],
        ),
    ]
)


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


def fmt(value, digits=3):
    value = parse_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def fmt_acc(value):
    return fmt(value, digits=2)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def maybe_int(value):
    value = parse_float(value)
    if value is None:
        return None
    return int(value)


def mean(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.mean(values))


def max_or_none(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.max(values))


def std(values):
    values = [value for value in values if value is not None]
    return None if len(values) < 2 else float(np.std(values))


def numeric_column(rows, name):
    return [parse_float(row.get(name)) for row in rows]


def standardize_design(X):
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


def design_matrix(rows, predictors, outcome):
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
    return standardize_design(X), y


def fit_ols(X, y):
    if X.shape[0] == 0:
        return {"n": 0, "r2": None, "leave_one_out_r2": None, "rmse": None}
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    prediction = X @ coefficients
    residual = y - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(X.shape[0]),
        "r2": None if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot),
        "leave_one_out_r2": leave_one_out_r2(X, y),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def leave_one_out_r2(X, y):
    if X.shape[0] < X.shape[1] + 2:
        return None
    predictions = np.zeros(X.shape[0], dtype=float)
    for idx in range(X.shape[0]):
        keep = np.arange(X.shape[0]) != idx
        coefficients = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        predictions[idx] = X[idx] @ coefficients
    ss_res = float(np.sum((y - predictions) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return None if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)


def regression_models(rows, model_map, outcome):
    report = OrderedDict()
    for name, predictors in model_map.items():
        X, y = design_matrix(rows, predictors, outcome)
        fit = fit_ols(X, y)
        fit["predictors"] = predictors
        report[name] = fit
    return report


def parse_experiment(raw):
    if "=" in raw:
        name, path = raw.split("=", 1)
    else:
        path = raw
        name = os.path.basename(os.path.abspath(path.rstrip(os.sep)))
    return name, path


def library_paths(root, run_summary):
    candidates = [
        os.path.join(root, "selected.csv"),
        os.path.join(root, "library.csv"),
    ]
    n_edges_values = run_summary.get("n_edges_values") or []
    n_edges = n_edges_values[0] if len(n_edges_values) == 1 else None
    topology_rows = load_csv(os.path.join(root, "topology_results.csv"))
    n_nodes_values = sorted(
        {maybe_int(row.get("n_nodes")) for row in topology_rows if maybe_int(row.get("n_nodes")) is not None}
    )
    n_nodes = n_nodes_values[0] if len(n_nodes_values) == 1 else None
    if n_nodes is not None and n_edges is not None:
        parent = os.path.dirname(os.path.abspath(root.rstrip(os.sep)))
        root_name = os.path.basename(os.path.abspath(root.rstrip(os.sep)))
        input_library_name = (
            root_name.replace("input_mask_fixed", "input_mask_library", 1)
            if root_name.startswith("input_mask_fixed")
            else None
        )
        if input_library_name:
            candidates.extend(
                [
                    os.path.join(parent, input_library_name, "selected.csv"),
                    os.path.join(parent, input_library_name, "library.csv"),
                ]
            )
        candidates.extend(
            [
                os.path.join(parent, f"topology_library_n{n_nodes}_m{n_edges}", "selected.csv"),
                os.path.join(parent, f"topology_library_n{n_nodes}_m{n_edges}", "library.csv"),
            ]
        )
    return [path for path in candidates if os.path.exists(path)]


def fit_summary(fit):
    if not fit:
        return {"n": None, "r2": None, "leave_one_out_r2": None, "rmse": None}
    return {
        "n": fit.get("n"),
        "r2": parse_float(fit.get("r2")),
        "leave_one_out_r2": parse_float(fit.get("leave_one_out_r2")),
        "rmse": parse_float(fit.get("rmse")),
    }


def summarize_run_rows(rows, target):
    if not rows:
        return {}
    target_values = numeric_column(rows, target)
    return {
        "n_runs": len(rows),
        "n_topologies": len({row.get("topology_name") for row in rows if row.get("topology_name")}),
        "n_edges_values": sorted(
            {int(value) for value in numeric_column(rows, "n_edges") if value is not None}
        ),
        "target_mean": mean(target_values),
        "target_max": max_or_none(target_values),
        "target_std": std(target_values),
    }


def summarize_aggregate_rows(rows):
    if not rows:
        return {}
    return {
        "n_topology_groups": len(rows),
        "target_mean_of_means": mean(numeric_column(rows, "target_mean")),
        "target_max_of_max": max_or_none(numeric_column(rows, "target_max")),
        "target_mean_seed_std": mean(numeric_column(rows, "target_std")),
        "n_edges_mean": mean(numeric_column(rows, "n_edges")),
        "n_edges_min": min(value for value in numeric_column(rows, "n_edges") if value is not None)
        if any(value is not None for value in numeric_column(rows, "n_edges"))
        else None,
        "n_edges_max": max_or_none(numeric_column(rows, "n_edges")),
        "d_rel_mean": mean(numeric_column(rows, "d_rel")),
        "effective_rank_D_mean": mean(numeric_column(rows, "effective_rank_D")),
    }


def summarize_library(root, run_summary):
    paths = library_paths(root, run_summary)
    if not paths:
        return {}
    rows_by_path = {path: load_csv(path) for path in paths}
    selected_path = next((path for path in paths if os.path.basename(path) == "selected.csv"), None)
    library_path = next((path for path in paths if os.path.basename(path) == "library.csv"), None)
    selected_rows = rows_by_path.get(selected_path, []) if selected_path else []
    library_rows = rows_by_path.get(library_path, []) if library_path else []
    source_rows = selected_rows or library_rows
    return {
        "selected_csv": selected_path,
        "library_csv": library_path,
        "n_selected": len(selected_rows) if selected_rows else None,
        "n_candidates": len(library_rows) if library_rows else None,
        "families": sorted(
            {
                row.get("family") or row.get("mask_family")
                for row in source_rows
                if row.get("family") or row.get("mask_family")
            }
        ),
        "n_nodes": maybe_int(source_rows[0].get("n_nodes")) if source_rows else None,
        "n_edges": maybe_int(source_rows[0].get("n_edges")) if source_rows else None,
        "input_coupled_parameter_count_values": sorted(
            {
                maybe_int(row.get("input_coupled_parameter_count"))
                for row in source_rows
                if maybe_int(row.get("input_coupled_parameter_count")) is not None
            }
        ),
        "d_rel_values": sorted(
            {maybe_int(row.get("d_rel")) for row in source_rows if maybe_int(row.get("d_rel")) is not None}
        ),
        "effective_rank_D_mean": mean(numeric_column(source_rows, "effective_rank_D")),
        "effective_rank_D_masked_mean": mean(numeric_column(source_rows, "effective_rank_D_masked")),
        "root_tree_count_gini_mean": mean(numeric_column(source_rows, "root_tree_count_gini")),
        "edge_participation_gini_mean": mean(numeric_column(source_rows, "edge_participation_gini")),
    }


def extract_run_regressions(report):
    if not report:
        return {}
    models = report.get("models", {})
    return {name: fit_summary(models.get(name)) for name in KEY_RUN_MODELS}


def extract_aggregate_regressions(report):
    if not report:
        return {}
    regressions = report.get("regressions", {})
    extracted = OrderedDict()
    for outcome in ["target_mean", "target_max", "target_std"]:
        extracted[outcome] = {
            name: fit_summary(regressions.get(outcome, {}).get(name))
            for name in KEY_AGG_MODELS
        }
    return extracted


def extract_correlations(report):
    if not report:
        return {}
    overall = report.get("overall_correlations", {})
    residual = report.get("within_edge_count_residual_correlations", {})
    return {
        name: {
            "overall": parse_float(overall.get(name)),
            "within_edge_residual": parse_float(residual.get(name)),
        }
        for name in KEY_CORRELATIONS
    }


def summarize_essential(root):
    comparison_json = os.path.join(root, "essential_input50", "retrain_comparison.json")
    comparison = load_json(comparison_json)
    retrain_aggregate_csv = os.path.join(root, "essential_input50_retrain", "topology_seed_aggregates.csv")
    retrain_aggregate_json = os.path.join(root, "essential_input50_retrain", "topology_seed_aggregates.json")
    retrain_mechanism_json = os.path.join(root, "essential_input50_retrain", "mechanism_summary.json")
    retrain_rows = load_csv(retrain_aggregate_csv)
    comparison_rows = load_csv(os.path.join(root, "essential_input50", "retrain_comparison.csv"))
    if not comparison and not retrain_rows:
        return {}
    return {
        "comparison_path": comparison_json if os.path.exists(comparison_json) else None,
        "comparison": comparison or {},
        "top_retrained_motifs": comparison_rows[:5],
        "retrain_aggregate": summarize_aggregate_rows(retrain_rows),
        "retrain_aggregate_regressions": extract_aggregate_regressions(load_json(retrain_aggregate_json)),
        "retrain_mechanism_correlations": extract_correlations(load_json(retrain_mechanism_json)),
    }


def summarize_experiment(name, root, target):
    topology_csv = os.path.join(root, "topology_results.csv")
    mechanism_csv = os.path.join(root, "mechanism_results.csv")
    aggregate_csv = os.path.join(root, "topology_seed_aggregates.csv")
    topology_regression_json = os.path.join(root, "topology_regression.json")
    aggregate_json = os.path.join(root, "topology_seed_aggregates.json")
    mechanism_summary_json = os.path.join(root, "mechanism_summary.json")

    topology_rows = load_csv(topology_csv)
    aggregate_rows = load_csv(aggregate_csv)
    mechanism_rows = load_csv(mechanism_csv)
    run_summary = summarize_run_rows(topology_rows, target)
    return {
        "name": name,
        "root": os.path.abspath(root),
        "paths": {
            "topology_csv": topology_csv if os.path.exists(topology_csv) else None,
            "mechanism_csv": mechanism_csv if os.path.exists(mechanism_csv) else None,
            "aggregate_csv": aggregate_csv if os.path.exists(aggregate_csv) else None,
            "topology_regression_json": topology_regression_json
            if os.path.exists(topology_regression_json)
            else None,
            "aggregate_json": aggregate_json if os.path.exists(aggregate_json) else None,
            "mechanism_summary_json": mechanism_summary_json
            if os.path.exists(mechanism_summary_json)
            else None,
        },
        "run_summary": run_summary,
        "library_summary": summarize_library(root, run_summary),
        "mechanism_run_count": len(mechanism_rows),
        "aggregate_summary": summarize_aggregate_rows(aggregate_rows),
        "run_regressions": extract_run_regressions(load_json(topology_regression_json)),
        "aggregate_regressions": extract_aggregate_regressions(load_json(aggregate_json)),
        "mechanism_correlations": extract_correlations(load_json(mechanism_summary_json)),
        "essential_input50": summarize_essential(root),
    }


def rows_with_experiment(experiment, relative_path):
    path = os.path.join(experiment["root"], relative_path)
    rows = []
    for row in load_csv(path):
        item = dict(row)
        item["experiment"] = experiment["name"]
        rows.append(item)
    return rows


def join_mechanism_rows(run_rows, experiment):
    mechanism_rows = {
        row["label"]: row
        for row in load_csv(os.path.join(experiment["root"], "mechanism_results.csv"))
        if row.get("label")
    }
    joined = []
    for row in run_rows:
        mechanism = mechanism_rows.get(row.get("label"))
        item = dict(row)
        if mechanism:
            for key, value in mechanism.items():
                if key not in item:
                    item[key] = value
        joined.append(item)
    return joined


def pooled_report(experiments, target):
    run_rows = []
    aggregate_rows = []
    retrain_rows = []
    for experiment in experiments:
        experiment_run_rows = rows_with_experiment(experiment, "topology_results.csv")
        run_rows.extend(join_mechanism_rows(experiment_run_rows, experiment))
        aggregate_rows.extend(rows_with_experiment(experiment, "topology_seed_aggregates.csv"))
        retrain_rows.extend(rows_with_experiment(experiment, "essential_input50_retrain/topology_seed_aggregates.csv"))

    return {
        "run_rows": len(run_rows),
        "aggregate_groups": len(aggregate_rows),
        "retrain_groups": len(retrain_rows),
        "run_level": regression_models(run_rows, POOLED_RUN_MODELS, target),
        "aggregate_target_mean": regression_models(
            aggregate_rows,
            POOLED_AGGREGATE_MODELS,
            "target_mean",
        ),
        "aggregate_target_max": regression_models(
            aggregate_rows,
            POOLED_AGGREGATE_MODELS,
            "target_max",
        ),
        "retrain_target_mean": regression_models(
            retrain_rows,
            POOLED_AGGREGATE_MODELS,
            "target_mean",
        ),
        "retrain_target_max": regression_models(
            retrain_rows,
            POOLED_AGGREGATE_MODELS,
            "target_max",
        ),
    }


def model_cell(fit, key="leave_one_out_r2"):
    if not fit:
        return "NA"
    return fmt(fit.get(key))


def regression_table(experiments, outcome, model_names, source):
    lines = []
    header = ["experiment"] + [f"{model} R2/LOO" for model in model_names]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for experiment in experiments:
        if source == "aggregate":
            models = experiment.get("aggregate_regressions", {}).get(outcome, {})
        elif source == "essential_retrain":
            models = (
                experiment.get("essential_input50", {})
                .get("retrain_aggregate_regressions", {})
                .get(outcome, {})
            )
        else:
            models = experiment.get("run_regressions", {})
        row = [experiment["name"]]
        for model in model_names:
            fit = models.get(model, {})
            row.append(f"{fmt(fit.get('r2'))}/{fmt(fit.get('leave_one_out_r2'))}")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def summary_table(experiments):
    lines = [
        "| experiment | runs | groups | m values | mean ICL | best ICL | mean seed std |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for experiment in experiments:
        run = experiment.get("run_summary", {})
        agg = experiment.get("aggregate_summary", {})
        m_values = ",".join(str(value) for value in run.get("n_edges_values", [])) or "NA"
        lines.append(
            "| "
            + " | ".join(
                [
                    experiment["name"],
                    str(run.get("n_runs", "NA")),
                    str(agg.get("n_topology_groups", run.get("n_topologies", "NA"))),
                    m_values,
                    fmt_acc(agg.get("target_mean_of_means", run.get("target_mean"))),
                    fmt_acc(agg.get("target_max_of_max", run.get("target_max"))),
                    fmt_acc(agg.get("target_mean_seed_std")),
                ]
            )
            + " |"
        )
    return lines


def library_table(experiments):
    lines = [
        "| experiment | library selected/candidates | families | d_rel values | mean effective rank | mean edge gini |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    for experiment in experiments:
        library = experiment.get("library_summary", {})
        if not library:
            lines.append(f"| {experiment['name']} | NA | NA | NA | NA | NA |")
            continue
        selected = library.get("n_selected")
        candidates = library.get("n_candidates")
        selected_text = (
            f"{selected or 'NA'}/{candidates or 'NA'}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    experiment["name"],
                    selected_text,
                    ", ".join(library.get("families") or []) or "NA",
                    ", ".join(str(value) for value in library.get("d_rel_values") or []) or "NA",
                    fmt(
                        library.get("effective_rank_D_masked_mean")
                        if library.get("effective_rank_D_masked_mean") is not None
                        else library.get("effective_rank_D_mean")
                    ),
                    fmt(library.get("edge_participation_gini_mean")),
                ]
            )
            + " |"
        )
    return lines


def correlation_table(experiments):
    rows = [
        ("d_rel", "relative tree dimension"),
        ("comparison_branch_d_rel_min", "weakest comparison-branch paired rank"),
        ("comparison_branch_d_rel_gini", "comparison-branch rank imbalance"),
        ("effective_rank_D", "tree spectrum effective rank"),
        ("effective_rank_D_masked", "masked relative tree effective rank"),
        ("edge_participation_gini", "bottleneck/participation heterogeneity"),
        ("input_edge_load_gini", "input mask edge-load heterogeneity"),
        ("input_coord_load_gini", "input mask coordinate-load heterogeneity"),
        ("target_logprob_margin_mean", "trained branch margin"),
        ("target_logprob_margin_branch_mean_min", "worst branch mean margin"),
        ("branch_active_tree_mi", "branch-active-tree MI"),
        ("tree_comparison_energy_fraction_mean", "tree-sum comparison alignment"),
        ("posterior_matched_comparison_gap_mean", "posterior matched comparison gap"),
        ("input_ablation_max_loss", "input-coupling ablation max loss"),
        ("physical_ablation_max_loss", "physical ablation max loss"),
    ]
    lines = ["| metric | " + " | ".join(experiment["name"] for experiment in experiments) + " |"]
    lines.append("| " + " | ".join(["---"] * (len(experiments) + 1)) + " |")
    for key, label in rows:
        values = [label]
        for experiment in experiments:
            entry = experiment.get("mechanism_correlations", {}).get(key, {})
            values.append(fmt(entry.get("overall")))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def essential_table(experiments):
    lines = [
        "| source experiment | joined motifs | source mean ICL | retrain mean ICL | retrain best ICL | retention mean/max | motif edges mean/min/max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for experiment in experiments:
        comparison = experiment.get("essential_input50", {}).get("comparison", {})
        if not comparison:
            continue
        edge_text = (
            f"{fmt(comparison.get('n_edges_mean'), 2)}/"
            f"{fmt(comparison.get('n_edges_min'), 0)}/"
            f"{fmt(comparison.get('n_edges_max'), 0)}"
        )
        retention = (
            f"{fmt(comparison.get('retention_mean_mean'))}/"
            f"{fmt(comparison.get('retention_max_mean'))}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    experiment["name"],
                    str(comparison.get("n_joined", "NA")),
                    fmt_acc(comparison.get("source_mean_mean")),
                    fmt_acc(comparison.get("retrain_mean_mean")),
                    fmt_acc(comparison.get("retrain_max_best")),
                    retention,
                    edge_text,
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| none | 0 | NA | NA | NA | NA | NA |")
    return lines


def top_motif_table(experiments):
    lines = [
        "| source experiment | motif | edges | d_rel | source ICL | retrain mean | retrain max | retention mean/max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment in experiments:
        motifs = experiment.get("essential_input50", {}).get("top_retrained_motifs", [])[:3]
        for motif in motifs:
            retention = f"{fmt(motif.get('retrain_retention_mean'))}/{fmt(motif.get('retrain_retention_max'))}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        experiment["name"],
                        motif.get("topology_name", "NA"),
                        fmt(motif.get("n_edges"), 0),
                        fmt(motif.get("d_rel"), 0),
                        fmt_acc(motif.get("source_test_novel_classes_mean")),
                        fmt_acc(motif.get("retrain_target_mean")),
                        fmt_acc(motif.get("retrain_target_max")),
                        retention,
                    ]
                )
                + " |"
            )
    if len(lines) == 2:
        lines.append("| none | NA | NA | NA | NA | NA | NA | NA |")
    return lines


def pooled_regression_table(pooled, key, title):
    lines = [
        title,
        "",
        "| model | n | R2 | LOO_R2 | RMSE | predictors |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, fit in pooled.get(key, {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(fit.get("n", "NA")),
                    fmt(fit.get("r2")),
                    fmt(fit.get("leave_one_out_r2")),
                    fmt(fit.get("rmse")),
                    ", ".join(fit.get("predictors", [])),
                ]
            )
            + " |"
        )
    return lines


def build_markdown(report):
    experiments = report["experiments"]
    pooled = report.get("pooled", {})
    lines = [
        "# Topology-ICL Progress Report",
        "",
        f"Generated from `{os.getcwd()}`.",
        "",
        "## Experiment Coverage",
        "",
        *summary_table(experiments),
        "",
        "## Topology Library Provenance",
        "",
        *library_table(experiments),
        "",
        "## Pooled Fixed-Edge Regime Analysis",
        "",
        f"Rows pooled across supplied regimes: run-level `{pooled.get('run_rows', 0)}`, topology groups `{pooled.get('aggregate_groups', 0)}`, retrained motif groups `{pooled.get('retrain_groups', 0)}`.",
        "",
        "These models test whether tree-geometry and post-training mechanism features explain accuracy beyond edge count when `m` varies across regimes.",
        "",
        *pooled_regression_table(pooled, "run_level", "### Run-Level Novel-Class ICL"),
        "",
        *pooled_regression_table(pooled, "aggregate_target_mean", "### Topology Mean Across Seeds"),
        "",
        *pooled_regression_table(pooled, "aggregate_target_max", "### Topology Best Seed"),
        "",
        "## Fixed-Topology Seed Aggregates",
        "",
        "Values are `R2/LOO_R2` for topology-level regressions. `target_mean` tracks trainability/reliability across seeds; `target_max` tracks best-seed expressivity.",
        "",
        "### Novel-Class ICL Mean Across Seeds",
        "",
        *regression_table(experiments, "target_mean", KEY_AGG_MODELS, "aggregate"),
        "",
        "### Novel-Class ICL Best Seed",
        "",
        *regression_table(experiments, "target_max", KEY_AGG_MODELS, "aggregate"),
        "",
        "## Run-Level Raw Count Control",
        "",
        "Values are `R2/LOO_R2` for run-level regressions. In fixed-edge libraries, raw count is intentionally constant and should not explain residual topology variation.",
        "",
        *regression_table(experiments, "target", KEY_RUN_MODELS, "run"),
        "",
        "## Mechanism Correlations",
        "",
        "Pearson correlations with novel-class ICL at the run level, using completed mechanism analyses.",
        "",
        *correlation_table(experiments),
        "",
        "## Essential Motif Retraining",
        "",
        "Input-ablation 50%-coverage essential subgraphs were extracted with strong-connectivity repair and retrained from scratch.",
        "",
        *essential_table(experiments),
        "",
        "### Top Retrained Motifs",
        "",
        *top_motif_table(experiments),
        "",
        "## Essential Motif Retrain Regressions",
        "",
        "Values are `R2/LOO_R2` on retrained essential motif topology groups.",
        "",
        "### Retrain Mean Across Seeds",
        "",
        *regression_table(experiments, "target_mean", KEY_AGG_MODELS, "essential_retrain"),
        "",
        "### Retrain Best Seed",
        "",
        *regression_table(experiments, "target_max", KEY_AGG_MODELS, "essential_retrain"),
        "",
        "### Pooled Retrained Motifs",
        "",
        *pooled_regression_table(pooled, "retrain_target_mean", "#### Retrain Mean Across Seeds"),
        "",
        *pooled_regression_table(pooled, "retrain_target_max", "#### Retrain Best Seed"),
        "",
        "## Current Interpretation",
        "",
        "- Fixed-edge sweeps directly test topology beyond raw degree count because `n_edges` and raw parameter count are matched within each library.",
        "- Topology-level tree-geometry regressions are the cleanest pre-training test of the matrix-tree hypothesis; mechanism and projection-alignment regressions are post-training explanatory diagnostics.",
        "- Branch-active-tree mutual information, logit margin, comparison-alignment, and ablation losses are functional evidence about what trained models actually used.",
        "- Essential-subgraph retraining separates expressive minimal motifs from dense-graph trainability; retention below 1.0 means dense graphs still helped optimization or supplied redundant pathways.",
        "",
    ]
    return "\n".join(lines)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Experiment root as NAME=PATH or PATH. May be repeated.",
    )
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    experiments = []
    for raw in args.experiment:
        name, path = parse_experiment(raw)
        experiments.append(summarize_experiment(name, path, args.target))

    report = {
        "report_meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "cwd": os.getcwd(),
        },
        "target": args.target,
        "experiments": experiments,
    }
    report["pooled"] = pooled_report(experiments, args.target)
    markdown = build_markdown(report)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write(markdown)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(json_ready(report), f, indent=2)

    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
