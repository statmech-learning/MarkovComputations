"""Build a focused report for fixed-physical-topology input-mask sweeps.

The general topology report intentionally pools physical-edge libraries and
input-mask libraries.  This script keeps the controlled input-encoding question
separate: with the physical edge count, input dimension, decoder, training
protocol, and number of input-coupled parameters fixed, how much variation is
left for the physical backbone, the mask family, masked relative tree geometry,
and post-training functional metrics to explain?
"""

import argparse
import csv
import json
import math
import os
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone

import numpy as np


DEFAULT_TARGET = "test_novel_classes"

RUN_MODELS = OrderedDict(
    [
        ("raw_counts", [("num", "n_edges"), ("num", "input_coupled_parameter_count")]),
        ("physical_backbone", [("cat", "physical_topology_name")]),
        ("mask_family", [("cat", "input_mask_family")]),
        (
            "physical_plus_family",
            [("cat", "physical_topology_name"), ("cat", "input_mask_family")],
        ),
        ("d_rel", [("num", "d_rel")]),
        (
            "masked_geometry",
            [
                ("num", "d_rel"),
                ("num", "effective_rank_D_masked"),
                ("num", "condition_number_D_masked"),
                ("num", "input_edge_load_gini"),
                ("num", "input_coord_load_gini"),
            ],
        ),
        (
            "physical_plus_masked_geometry",
            [
                ("cat", "physical_topology_name"),
                ("num", "d_rel"),
                ("num", "effective_rank_D_masked"),
                ("num", "condition_number_D_masked"),
                ("num", "input_edge_load_gini"),
                ("num", "input_coord_load_gini"),
            ],
        ),
        (
            "physical_family_masked_geometry",
            [
                ("cat", "physical_topology_name"),
                ("cat", "input_mask_family"),
                ("num", "d_rel"),
                ("num", "effective_rank_D_masked"),
                ("num", "condition_number_D_masked"),
                ("num", "input_edge_load_gini"),
                ("num", "input_coord_load_gini"),
            ],
        ),
        (
            "mechanism",
            [
                ("num", "target_logprob_margin_mean"),
                ("num", "branch_active_tree_mi"),
                ("num", "branch_active_tree_purity_mean"),
                ("num", "posterior_matched_comparison_gap_mean"),
                ("num", "input_ablation_max_loss"),
                ("num", "physical_ablation_max_loss"),
            ],
        ),
        (
            "physical_plus_mechanism",
            [
                ("cat", "physical_topology_name"),
                ("num", "target_logprob_margin_mean"),
                ("num", "branch_active_tree_mi"),
                ("num", "branch_active_tree_purity_mean"),
                ("num", "posterior_matched_comparison_gap_mean"),
                ("num", "input_ablation_max_loss"),
                ("num", "physical_ablation_max_loss"),
            ],
        ),
    ]
)

AGGREGATE_MODELS = OrderedDict(
    [
        ("raw_counts", [("num", "n_edges"), ("num", "input_coupled_parameter_count")]),
        ("physical_backbone", [("cat", "physical_topology_name")]),
        ("mask_family", [("cat", "input_mask_family")]),
        (
            "physical_plus_family",
            [("cat", "physical_topology_name"), ("cat", "input_mask_family")],
        ),
        ("d_rel", [("num", "d_rel")]),
        (
            "masked_geometry",
            [
                ("num", "d_rel"),
                ("num", "effective_rank_D_masked"),
                ("num", "condition_number_D_masked"),
                ("num", "input_edge_load_gini"),
                ("num", "input_coord_load_gini"),
            ],
        ),
        (
            "physical_plus_masked_geometry",
            [
                ("cat", "physical_topology_name"),
                ("num", "d_rel"),
                ("num", "effective_rank_D_masked"),
                ("num", "condition_number_D_masked"),
                ("num", "input_edge_load_gini"),
                ("num", "input_coord_load_gini"),
            ],
        ),
        (
            "physical_family_masked_geometry",
            [
                ("cat", "physical_topology_name"),
                ("cat", "input_mask_family"),
                ("num", "d_rel"),
                ("num", "effective_rank_D_masked"),
                ("num", "condition_number_D_masked"),
                ("num", "input_edge_load_gini"),
                ("num", "input_coord_load_gini"),
            ],
        ),
        (
            "mechanism",
            [
                ("num", "target_logprob_margin_mean_mean"),
                ("num", "branch_active_tree_mi_mean"),
                ("num", "branch_active_tree_purity_mean_mean"),
                ("num", "posterior_matched_comparison_gap_mean_mean"),
                ("num", "input_ablation_max_loss_mean"),
                ("num", "physical_ablation_max_loss_mean"),
            ],
        ),
        (
            "physical_plus_mechanism",
            [
                ("cat", "physical_topology_name"),
                ("num", "target_logprob_margin_mean_mean"),
                ("num", "branch_active_tree_mi_mean"),
                ("num", "branch_active_tree_purity_mean_mean"),
                ("num", "posterior_matched_comparison_gap_mean_mean"),
                ("num", "input_ablation_max_loss_mean"),
                ("num", "physical_ablation_max_loss_mean"),
            ],
        ),
    ]
)

CORRELATION_COLUMNS = [
    "d_rel",
    "effective_rank_D_masked",
    "condition_number_D_masked",
    "input_edge_load_gini",
    "input_coord_load_gini",
    "target_logprob_margin_mean",
    "branch_active_root_mi",
    "branch_active_tree_mi",
    "branch_active_root_purity_mean",
    "branch_active_tree_purity_mean",
    "posterior_matched_comparison_gap_mean",
    "input_ablation_max_loss",
    "physical_ablation_max_loss",
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


def fmt(value, digits=3):
    parsed = parse_float(value)
    if parsed is None:
        return "NA"
    return f"{parsed:.{digits}f}"


def fmt_acc(value):
    return fmt(value, 2)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def parse_experiment(raw):
    if "=" in raw:
        name, path = raw.split("=", 1)
    else:
        path = raw
        name = os.path.basename(os.path.abspath(path.rstrip(os.sep)))
    return name, os.path.abspath(path)


def numeric_values(rows, key):
    return [value for value in (parse_float(row.get(key)) for row in rows) if value is not None]


def mean(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.mean(values))


def std(values):
    values = [value for value in values if value is not None]
    return None if len(values) < 2 else float(np.std(values))


def max_or_none(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.max(values))


def min_or_none(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.min(values))


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


def discover_library_paths(root):
    parent = os.path.dirname(root.rstrip(os.sep))
    root_name = os.path.basename(root.rstrip(os.sep))
    candidates = [
        os.path.join(root, "selected.csv"),
        os.path.join(root, "library.csv"),
    ]
    if root_name.startswith("input_mask_fixed"):
        library_name = root_name.replace("input_mask_fixed", "input_mask_library", 1)
        candidates.extend(
            [
                os.path.join(parent, library_name, "selected.csv"),
                os.path.join(parent, library_name, "library.csv"),
            ]
        )
    return [path for path in candidates if os.path.exists(path)]


def load_mask_family_map(root):
    mapping = {}
    selected_path = None
    library_path = None
    for path in discover_library_paths(root):
        if os.path.basename(path) == "selected.csv" and selected_path is None:
            selected_path = path
        elif os.path.basename(path) == "library.csv" and library_path is None:
            library_path = path
    rows = load_csv(selected_path) or load_csv(library_path)
    for row in rows:
        family = row.get("input_mask_family") or row.get("mask_family") or row.get("family")
        if not family:
            continue
        for key in ["mask_name", "topology_id", "topology_name", "input_mask_name"]:
            value = row.get(key)
            if value:
                mapping[value] = family
    return mapping, {
        "selected_csv": selected_path,
        "library_csv": library_path,
        "n_selected": len(load_csv(selected_path)) if selected_path else None,
        "n_candidates": len(load_csv(library_path)) if library_path else None,
    }


def annotate_rows(rows, experiment_name, family_map):
    annotated = []
    for row in rows:
        item = dict(row)
        item["experiment"] = experiment_name
        family = item.get("input_mask_family")
        if not family:
            family = (
                family_map.get(item.get("input_mask_name"))
                or family_map.get(item.get("topology_name"))
                or family_map.get(item.get("group"))
            )
        item["input_mask_family"] = family or "unknown"
        annotated.append(item)
    return annotated


def join_mechanisms(run_rows, mechanism_rows):
    by_label = {row.get("label"): row for row in mechanism_rows if row.get("label")}
    joined = []
    for row in run_rows:
        item = dict(row)
        mechanism = by_label.get(row.get("label"))
        if mechanism:
            for key, value in mechanism.items():
                if key not in item:
                    item[key] = value
        joined.append(item)
    return joined


def standardize_design(X, numeric_columns):
    Xs = X.copy()
    for col in numeric_columns:
        scale = Xs[:, col].std()
        if scale > 1e-12:
            Xs[:, col] = (Xs[:, col] - Xs[:, col].mean()) / scale
        else:
            Xs[:, col] = 0.0
    return Xs


def design_matrix(rows, terms, outcome):
    usable = []
    for row in rows:
        y = parse_float(row.get(outcome))
        if y is None:
            continue
        numeric = []
        ok = True
        for kind, name in terms:
            if kind != "num":
                continue
            value = parse_float(row.get(name))
            if value is None:
                ok = False
                break
            numeric.append(value)
        if ok:
            usable.append((row, numeric, y))
    if not usable:
        return [], np.zeros((0, 1)), np.zeros(0), ["intercept"]

    category_values = OrderedDict()
    for kind, name in terms:
        if kind != "cat":
            continue
        values = sorted({str(row.get(name) or "unknown") for row, _, _ in usable})
        category_values[name] = values

    matrix = []
    columns = ["intercept"]
    numeric_names = [name for kind, name in terms if kind == "num"]
    columns.extend(numeric_names)
    for name, values in category_values.items():
        for value in values[1:]:
            columns.append(f"{name}={value}")

    for row, numeric, _ in usable:
        encoded = [1.0] + numeric
        for name, values in category_values.items():
            current = str(row.get(name) or "unknown")
            encoded.extend(1.0 if current == value else 0.0 for value in values[1:])
        matrix.append(encoded)

    X = np.asarray(matrix, dtype=float)
    y = np.asarray([y for _, _, y in usable], dtype=float)
    numeric_columns = list(range(1, 1 + len(numeric_names)))
    return [row for row, _, _ in usable], standardize_design(X, numeric_columns), y, columns


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


def fit_model(rows, terms, outcome):
    usable_rows, X, y, columns = design_matrix(rows, terms, outcome)
    if X.shape[0] == 0:
        return {
            "n": 0,
            "r2": None,
            "leave_one_out_r2": None,
            "rmse": None,
            "columns": columns,
            "terms": terms,
        }
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    prediction = X @ coefficients
    residual = y - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(X.shape[0]),
        "n_predictors": int(X.shape[1] - 1),
        "n_physical_backbones": len({row.get("physical_topology_name") for row in usable_rows}),
        "n_mask_families": len({row.get("input_mask_family") for row in usable_rows}),
        "r2": None if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot),
        "leave_one_out_r2": leave_one_out_r2(X, y),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "columns": columns,
        "terms": [list(term) for term in terms],
    }


def regression_models(rows, model_map, outcome):
    return OrderedDict((name, fit_model(rows, terms, outcome)) for name, terms in model_map.items())


def correlation_report(rows, outcome, columns):
    report = OrderedDict()
    for column in columns:
        pairs = [
            (parse_float(row.get(column)), parse_float(row.get(outcome)))
            for row in rows
        ]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        report[column] = None if not pairs else pearson([x for x, _ in pairs], [y for _, y in pairs])
    return report


def summarize_rows(rows, target):
    targets = numeric_values(rows, target)
    return {
        "n": len(rows),
        "target_mean": mean(targets),
        "target_max": max_or_none(targets),
        "target_std": std(targets),
        "n_physical_backbones": len({row.get("physical_topology_name") for row in rows if row.get("physical_topology_name")}),
        "n_mask_families": len({row.get("input_mask_family") for row in rows if row.get("input_mask_family")}),
        "n_edges_values": sorted({int(value) for value in numeric_values(rows, "n_edges")}),
        "input_coupled_parameter_count_values": sorted(
            {int(value) for value in numeric_values(rows, "input_coupled_parameter_count")}
        ),
        "d_rel_min": min_or_none(numeric_values(rows, "d_rel")),
        "d_rel_max": max_or_none(numeric_values(rows, "d_rel")),
    }


def summarize_experiment(name, root, target):
    family_map, library = load_mask_family_map(root)
    topology_rows = annotate_rows(load_csv(os.path.join(root, "topology_results.csv")), name, family_map)
    mechanism_rows = load_csv(os.path.join(root, "mechanism_results.csv"))
    aggregate_rows = annotate_rows(load_csv(os.path.join(root, "topology_seed_aggregates.csv")), name, family_map)
    run_rows = join_mechanisms(topology_rows, mechanism_rows)
    return {
        "name": name,
        "root": root,
        "paths": {
            "topology_csv": os.path.join(root, "topology_results.csv"),
            "mechanism_csv": os.path.join(root, "mechanism_results.csv"),
            "aggregate_csv": os.path.join(root, "topology_seed_aggregates.csv"),
        },
        "library": library,
        "run_rows": run_rows,
        "aggregate_rows": aggregate_rows,
        "run_summary": summarize_rows(run_rows, target),
        "aggregate_summary": summarize_rows(aggregate_rows, "target_mean"),
        "run_correlations": correlation_report(run_rows, target, CORRELATION_COLUMNS),
        "aggregate_regressions": regression_models(aggregate_rows, AGGREGATE_MODELS, "target_mean"),
        "essential_inputmask50": summarize_essential_inputmask(root),
    }


def summarize_essential_inputmask(root):
    selected_csv = os.path.join(root, "essential_inputmask50", "selected.csv")
    library_csv = os.path.join(root, "essential_inputmask50", "library.csv")
    comparison_json = os.path.join(root, "essential_inputmask50", "retrain_comparison.json")
    comparison_csv = os.path.join(root, "essential_inputmask50", "retrain_comparison.csv")
    retrain_aggregate_csv = os.path.join(root, "essential_inputmask50_retrain", "topology_seed_aggregates.csv")
    selected_rows = load_csv(selected_csv)
    library_rows = load_csv(library_csv)
    comparison = load_json(comparison_json)
    comparison_rows = load_csv(comparison_csv)
    retrain_rows = load_csv(retrain_aggregate_csv)
    if not selected_rows and not comparison and not retrain_rows:
        return {}
    return {
        "selected_csv": selected_csv if os.path.exists(selected_csv) else None,
        "library_csv": library_csv if os.path.exists(library_csv) else None,
        "selected_summary": summarize_extracted_essential_masks(selected_rows, library_rows),
        "top_extracted_masks": selected_rows[:5],
        "comparison_path": comparison_json if os.path.exists(comparison_json) else None,
        "comparison": comparison or {},
        "top_retrained_masks": comparison_rows[:5],
        "retrain_aggregate": summarize_rows(retrain_rows, "target_mean"),
    }


def summarize_extracted_essential_masks(selected_rows, library_rows):
    if not selected_rows:
        return {}
    return {
        "n_selected": len(selected_rows),
        "n_candidates": len(library_rows) if library_rows else None,
        "input_coupled_parameter_count_min": min_or_none(
            numeric_values(selected_rows, "input_coupled_parameter_count")
        ),
        "input_coupled_parameter_count_mean": mean(
            numeric_values(selected_rows, "input_coupled_parameter_count")
        ),
        "input_coupled_parameter_count_max": max_or_none(
            numeric_values(selected_rows, "input_coupled_parameter_count")
        ),
        "source_input_coupled_parameter_count_mean": mean(
            numeric_values(selected_rows, "source_input_coupled_parameter_count_mean")
        ),
        "raw_essential_edges_mean": mean(numeric_values(selected_rows, "raw_essential_edges")),
        "raw_essential_edges_max": max_or_none(numeric_values(selected_rows, "raw_essential_edges")),
        "d_rel_mean": mean(numeric_values(selected_rows, "d_rel")),
        "effective_rank_D_masked_mean": mean(numeric_values(selected_rows, "effective_rank_D_masked")),
        "source_test_novel_classes_max_mean": mean(
            numeric_values(selected_rows, "source_test_novel_classes_max")
        ),
        "source_test_novel_classes_max_best": max_or_none(
            numeric_values(selected_rows, "source_test_novel_classes_max")
        ),
    }


def group_rows(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return groups


def physical_family_summary(aggregate_rows):
    rows = []
    for (experiment, physical, family), group in sorted(
        group_rows(aggregate_rows, ["experiment", "physical_topology_name", "input_mask_family"]).items()
    ):
        target_means = numeric_values(group, "target_mean")
        rows.append(
            {
                "experiment": experiment,
                "physical_topology_name": physical,
                "input_mask_family": family,
                "n_masks": len(group),
                "target_mean": mean(target_means),
                "target_max": max_or_none(numeric_values(group, "target_max")),
                "target_std_mean": mean(numeric_values(group, "target_std")),
                "d_rel_mean": mean(numeric_values(group, "d_rel")),
                "effective_rank_D_masked_mean": mean(numeric_values(group, "effective_rank_D_masked")),
                "input_edge_load_gini_mean": mean(numeric_values(group, "input_edge_load_gini")),
                "branch_active_tree_mi_mean": mean(numeric_values(group, "branch_active_tree_mi_mean")),
                "branch_active_tree_purity_mean": mean(
                    numeric_values(group, "branch_active_tree_purity_mean_mean")
                ),
                "input_ablation_max_loss_mean": mean(numeric_values(group, "input_ablation_max_loss_mean")),
                "physical_ablation_max_loss_mean": mean(numeric_values(group, "physical_ablation_max_loss_mean")),
            }
        )
    return rows


def top_masks(aggregate_rows, n=8):
    ranked = [
        row for row in aggregate_rows if parse_float(row.get("target_mean")) is not None
    ]
    ranked.sort(key=lambda row: parse_float(row.get("target_mean")), reverse=True)
    return ranked[:n], list(reversed(ranked[-n:]))


def fit_table(models, title):
    lines = [
        title,
        "",
        "| model | n | predictors | R2 | LOO_R2 | RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, fit in models.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(fit.get("n", "NA")),
                    str(fit.get("n_predictors", "NA")),
                    fmt(fit.get("r2")),
                    fmt(fit.get("leave_one_out_r2")),
                    fmt(fit.get("rmse")),
                ]
            )
            + " |"
        )
    return lines


def experiment_table(experiments):
    lines = [
        "| experiment | physical backbone | runs | masks | mask families | input params | d_rel range | mean ICL | best ICL |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for experiment in experiments:
        run = experiment["run_summary"]
        agg = experiment["aggregate_summary"]
        physical = sorted(
            {row.get("physical_topology_name") for row in experiment["run_rows"] if row.get("physical_topology_name")}
        )
        input_counts = ",".join(str(value) for value in run["input_coupled_parameter_count_values"]) or "NA"
        d_rel_range = f"{fmt(run.get('d_rel_min'), 0)}-{fmt(run.get('d_rel_max'), 0)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    experiment["name"],
                    ", ".join(physical) or "NA",
                    str(run["n"]),
                    str(agg["n"]),
                    str(run["n_mask_families"]),
                    input_counts,
                    d_rel_range,
                    fmt_acc(agg["target_mean"]),
                    fmt_acc(max_or_none(numeric_values(experiment["aggregate_rows"], "target_max"))),
                ]
            )
            + " |"
        )
    return lines


def family_table(rows):
    lines = [
        "| experiment | physical backbone | mask family | masks | mean ICL | best ICL | mean seed std | d_rel | tree MI | tree purity | input abl. loss | physical abl. loss |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["experiment"],
                    row["physical_topology_name"],
                    row["input_mask_family"],
                    str(row["n_masks"]),
                    fmt_acc(row["target_mean"]),
                    fmt_acc(row["target_max"]),
                    fmt_acc(row["target_std_mean"]),
                    fmt(row["d_rel_mean"], 0),
                    fmt(row["branch_active_tree_mi_mean"]),
                    fmt(row["branch_active_tree_purity_mean"]),
                    fmt(row["input_ablation_max_loss_mean"]),
                    fmt(row["physical_ablation_max_loss_mean"]),
                ]
            )
            + " |"
        )
    return lines


def mask_table(rows, title):
    lines = [
        title,
        "",
        "| experiment | mask | family | mean ICL | best ICL | seed std | d_rel | eff rank masked | edge gini | tree MI | tree purity |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("experiment", "NA"),
                    row.get("input_mask_name") or row.get("topology_name") or row.get("group", "NA"),
                    row.get("input_mask_family", "NA"),
                    fmt_acc(row.get("target_mean")),
                    fmt_acc(row.get("target_max")),
                    fmt_acc(row.get("target_std")),
                    fmt(row.get("d_rel"), 0),
                    fmt(row.get("effective_rank_D_masked")),
                    fmt(row.get("input_edge_load_gini")),
                    fmt(row.get("branch_active_tree_mi_mean")),
                    fmt(row.get("branch_active_tree_purity_mean_mean")),
                ]
            )
            + " |"
        )
    return lines


def essential_inputmask_table(experiments):
    lines = [
        "| source experiment | joined masks | source mean ICL | retrain mean ICL | retrain best ICL | retention mean/max | retrain input couplings |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment in experiments:
        comparison = experiment.get("essential_inputmask50", {}).get("comparison", {})
        if not comparison:
            continue
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
                    fmt(comparison.get("retrain_input_coupled_parameter_count_mean"), 1),
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| none | 0 | NA | NA | NA | NA | NA |")
    return lines


def extracted_inputmask_table(experiments):
    lines = [
        "| source experiment | selected/candidates | source input couplings | essential input couplings min/mean/max | raw edge rows mean/max | d_rel mean | source best ICL mean/best |",
        "| --- | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for experiment in experiments:
        summary = experiment.get("essential_inputmask50", {}).get("selected_summary", {})
        if not summary:
            continue
        selected = summary.get("n_selected")
        candidates = summary.get("n_candidates")
        selected_text = f"{selected or 'NA'}/{candidates or 'NA'}"
        coupled = (
            f"{fmt(summary.get('input_coupled_parameter_count_min'), 0)}/"
            f"{fmt(summary.get('input_coupled_parameter_count_mean'), 1)}/"
            f"{fmt(summary.get('input_coupled_parameter_count_max'), 0)}"
        )
        raw_edges = (
            f"{fmt(summary.get('raw_essential_edges_mean'), 1)}/"
            f"{fmt(summary.get('raw_essential_edges_max'), 0)}"
        )
        source_icl = (
            f"{fmt_acc(summary.get('source_test_novel_classes_max_mean'))}/"
            f"{fmt_acc(summary.get('source_test_novel_classes_max_best'))}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    experiment["name"],
                    selected_text,
                    fmt(summary.get("source_input_coupled_parameter_count_mean"), 1),
                    coupled,
                    raw_edges,
                    fmt(summary.get("d_rel_mean"), 1),
                    source_icl,
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| none | 0 | NA | NA | NA | NA | NA |")
    return lines


def top_extracted_inputmask_table(experiments):
    lines = [
        "| source experiment | mask | source best ICL | input couplings | source couplings | raw edge rows | d_rel | eff rank masked |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment in experiments:
        masks = experiment.get("essential_inputmask50", {}).get("top_extracted_masks", [])[:3]
        for mask in masks:
            lines.append(
                "| "
                + " | ".join(
                    [
                        experiment["name"],
                        mask.get("topology_name", "NA"),
                        fmt_acc(mask.get("source_test_novel_classes_max")),
                        fmt(mask.get("input_coupled_parameter_count"), 0),
                        fmt(mask.get("source_input_coupled_parameter_count_mean"), 0),
                        fmt(mask.get("raw_essential_edges"), 0),
                        fmt(mask.get("d_rel"), 0),
                        fmt(mask.get("effective_rank_D_masked")),
                    ]
                )
                + " |"
            )
    if len(lines) == 2:
        lines.append("| none | NA | NA | NA | NA | NA | NA | NA |")
    return lines


def top_retrained_inputmask_table(experiments):
    lines = [
        "| source experiment | mask | source ICL | retrain mean | retrain max | retention mean/max | input couplings | d_rel |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment in experiments:
        masks = experiment.get("essential_inputmask50", {}).get("top_retrained_masks", [])[:3]
        for mask in masks:
            retention = f"{fmt(mask.get('retrain_retention_mean'))}/{fmt(mask.get('retrain_retention_max'))}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        experiment["name"],
                        mask.get("topology_name", "NA"),
                        fmt_acc(mask.get("source_test_novel_classes_mean")),
                        fmt_acc(mask.get("retrain_target_mean")),
                        fmt_acc(mask.get("retrain_target_max")),
                        retention,
                        fmt(mask.get("retrain_input_coupled_parameter_count"), 0),
                        fmt(mask.get("d_rel"), 0),
                    ]
                )
                + " |"
            )
    if len(lines) == 2:
        lines.append("| none | NA | NA | NA | NA | NA | NA | NA |")
    return lines


def correlation_table(experiments):
    lines = [
        "| metric | " + " | ".join(experiment["name"] for experiment in experiments) + " |",
        "| " + " | ".join(["---"] * (len(experiments) + 1)) + " |",
    ]
    for column in CORRELATION_COLUMNS:
        row = [column]
        for experiment in experiments:
            row.append(fmt(experiment["run_correlations"].get(column)))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(experiments, target):
    run_rows = []
    aggregate_rows = []
    for experiment in experiments:
        run_rows.extend(experiment["run_rows"])
        aggregate_rows.extend(experiment["aggregate_rows"])

    pooled = {
        "run_summary": summarize_rows(run_rows, target),
        "aggregate_summary": summarize_rows(aggregate_rows, "target_mean"),
        "run_regressions": regression_models(run_rows, RUN_MODELS, target),
        "aggregate_target_mean": regression_models(aggregate_rows, AGGREGATE_MODELS, "target_mean"),
        "aggregate_target_max": regression_models(aggregate_rows, AGGREGATE_MODELS, "target_max"),
        "aggregate_target_std": regression_models(aggregate_rows, AGGREGATE_MODELS, "target_std"),
        "run_correlations": correlation_report(run_rows, target, CORRELATION_COLUMNS),
        "physical_family_summary": physical_family_summary(aggregate_rows),
    }
    best, worst = top_masks(aggregate_rows)
    pooled["best_masks"] = best
    pooled["worst_masks"] = worst
    return {
        "report_meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "cwd": os.getcwd(),
        },
        "target": target,
        "experiments": experiments,
        "pooled": pooled,
    }


def build_markdown(report):
    experiments = report["experiments"]
    pooled = report["pooled"]
    lines = [
        "# Input-Mask Topology-ICL Report",
        "",
        f"Generated from `{os.getcwd()}`.",
        "",
        "## Controlled Regimes",
        "",
        "These sweeps hold the physical edge count and the number of input-coupled parameters fixed inside each experiment. The rows below are mask-level seed aggregates unless noted otherwise.",
        "",
        *experiment_table(experiments),
        "",
        "## Pooled Fixed-Input-Count Regressions",
        "",
        f"Run rows: `{pooled['run_summary']['n']}`. Mask groups: `{pooled['aggregate_summary']['n']}`. Target: `{report['target']}`.",
        "",
        "Raw count controls should be weak here because `n_edges` and `input_coupled_parameter_count` were fixed by construction. Physical-backbone and mask-family terms test controlled topology effects; mechanism terms test what trained models actually used.",
        "",
        *fit_table(pooled["run_regressions"], "### Run-Level Novel-Class ICL"),
        "",
        *fit_table(pooled["aggregate_target_mean"], "### Mask Mean Across Seeds"),
        "",
        *fit_table(pooled["aggregate_target_max"], "### Mask Best Seed"),
        "",
        *fit_table(pooled["aggregate_target_std"], "### Mask Seed Variability"),
        "",
        "## Backbone And Mask-Family Summary",
        "",
        *family_table(pooled["physical_family_summary"]),
        "",
        "## Run-Level Mechanism Correlations",
        "",
        "Pearson correlations with novel-class ICL. Structural masked-geometry rows are pre-training controls; margin, active-tree MI, and ablation losses are post-training functional diagnostics.",
        "",
        *correlation_table(experiments),
        "",
        *mask_table(pooled["best_masks"], "## Best Mask Groups"),
        "",
        *mask_table(pooled["worst_masks"], "## Weakest Mask Groups"),
        "",
        "## Essential Input-Mask Retraining",
        "",
        "Input-ablation 50%-coverage essential masks keep the physical graph fixed and prune only input-coupling rows, then retrain those masks from scratch.",
        "",
        "### Extracted Essential Input Masks",
        "",
        *extracted_inputmask_table(experiments),
        "",
        "### Top Extracted Essential Input Masks",
        "",
        *top_extracted_inputmask_table(experiments),
        "",
        "### Retrain Retention",
        "",
        *essential_inputmask_table(experiments),
        "",
        "### Top Retrained Essential Input Masks",
        "",
        *top_retrained_inputmask_table(experiments),
        "",
        "## Current Interpretation",
        "",
        "- The fixed-count input-mask design removes raw trainable degree count as an explanation for within-regime variation.",
        "- Physical-backbone and mask-family regressions quantify topology effects before using trained-model diagnostics.",
        "- Masked relative tree geometry is a coarse pre-training proxy; weak LOO performance means it should not be overinterpreted as a complete capacity theory.",
        "- Branch-active-tree mutual information, logit margin, and edge ablation losses test whether successful trained models organize ICL through functional tree/edge structure.",
        "",
    ]
    return "\n".join(lines)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items() if key not in {"run_rows", "aggregate_rows"}}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", required=True, help="NAME=PATH or PATH. May be repeated.")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    experiments = [
        summarize_experiment(name, path, args.target)
        for name, path in (parse_experiment(raw) for raw in args.experiment)
    ]
    report = build_report(experiments, args.target)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write(build_markdown(report))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(json_ready(report), f, indent=2)

    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
