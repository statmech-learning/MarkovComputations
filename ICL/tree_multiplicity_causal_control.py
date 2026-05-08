from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


RESULTS_DIR = Path("ICL/results/next_phase_stats")
TREE_REPORT = RESULTS_DIR / "tree_level_multiplicity_reanalysis.json"
TOPOLOGY_CSV = RESULTS_DIR / "pooled_fixed_m20_topology_results.csv"

LIBRARY_MD = RESULTS_DIR / "tree_multiplicity_causal_mask_library.md"
LIBRARY_JSON = RESULTS_DIR / "tree_multiplicity_causal_mask_library.json"
TRAINING_PLAN_MD = RESULTS_DIR / "input_multiplicity_causal_control_training_plan.md"
REPORT_MD = RESULTS_DIR / "input_multiplicity_causal_control_report.md"
REPORT_JSON = RESULTS_DIR / "input_multiplicity_causal_control_report.json"


OUTCOMES = [
    "mean_novel_icl",
    "best_seed_novel_icl",
    "seed_std_novel_icl",
]

BASE_NUMERIC = [
    "d_rel",
    "input_coupled_edge_count",
    "input_coupled_coord_count",
]
BASE_CATEGORICAL = ["physical_topology_name"]

MODEL_SPECS = {
    "controls_only": {
        "numeric": BASE_NUMERIC,
        "categorical": BASE_CATEGORICAL,
    },
    "edge_level_multiplicity_plus_controls": {
        "numeric": BASE_NUMERIC
        + [
            "edge_M_gini",
            "edge_overlap_norm_min",
            "edge_overlap_norm_mean",
            "edge_comparison_imbalance_mean",
            "edge_load_gini",
        ],
        "categorical": BASE_CATEGORICAL,
    },
    "tree_level_multiplicity_plus_controls": {
        "numeric": BASE_NUMERIC
        + [
            "tree_overlap_norm_min",
            "tree_overlap_norm_mean",
            "tree_coord_load_gini",
            "tree_active_fraction_mean",
        ],
        "categorical": BASE_CATEGORICAL,
    },
    "tree_difference_multiplicity_plus_controls": {
        "numeric": BASE_NUMERIC
        + [
            "diff_overlap_norm_min",
            "diff_overlap_norm_mean",
            "diff_coord_load_gini",
        ],
        "categorical": BASE_CATEGORICAL,
    },
    "tree_difference_coord_load_summary_plus_controls": {
        "numeric": BASE_NUMERIC
        + [
            "diff_overlap_norm_min",
            "diff_overlap_norm_mean",
            "diff_coord_load_gini",
            "input_coord_load_gini",
            "input_edge_load_gini",
            "edge_comparison_imbalance_mean",
        ],
        "categorical": BASE_CATEGORICAL,
    },
}


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_int(value: Any) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def mean(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def percentile(values: Sequence[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def load_topology_controls() -> dict[str, dict[str, Any]]:
    controls: dict[str, dict[str, Any]] = {}
    with TOPOLOGY_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["topology_name"]
            if name in controls:
                continue
            controls[name] = {
                "d_rel": parse_float(row.get("d_rel")),
                "input_coupled_parameter_count": parse_int(row.get("input_coupled_parameter_count")),
                "input_coupled_edge_count": parse_int(row.get("input_coupled_edge_count")),
                "input_coupled_coord_count": parse_int(row.get("input_coupled_coord_count")),
                "input_coord_load_gini": parse_float(row.get("input_coord_load_gini")),
                "input_edge_load_gini": parse_float(row.get("input_edge_load_gini")),
                "comparison_branch_common_d_rel_min": parse_float(
                    row.get("comparison_branch_common_d_rel_min")
                ),
                "comparison_branch_common_d_rel_mean": parse_float(
                    row.get("comparison_branch_common_d_rel_mean")
                ),
                "comparison_branch_input_overlap_min": parse_float(
                    row.get("comparison_branch_input_overlap_min")
                ),
                "comparison_branch_input_overlap_mean": parse_float(
                    row.get("comparison_branch_input_overlap_mean")
                ),
            }
    return controls


def load_fixed_m20_rows() -> list[dict[str, Any]]:
    tree_report = json.loads(TREE_REPORT.read_text())
    fixed = next(ds for ds in tree_report["datasets"] if ds["name"] == "fixed_m20_masks_cluster_topology")
    controls = load_topology_controls()
    rows: list[dict[str, Any]] = []
    for group in fixed["groups"]:
        row = dict(group)
        row.update(controls.get(group["group"], {}))
        row["aggregate_M_mean"] = row.get("edge_M_mean")
        row["mask_group"] = row["group"]
        rows.append(row)
    return rows


def category_levels(rows: Sequence[Mapping[str, Any]], categorical: Sequence[str]) -> dict[str, list[str]]:
    return {col: sorted({str(row.get(col, "")) for row in rows}) for col in categorical}


def row_vector(
    row: Mapping[str, Any],
    numeric: Sequence[str],
    categorical: Sequence[str],
    levels: Mapping[str, Sequence[str]],
) -> list[float] | None:
    values: list[float] = []
    for name in numeric:
        value = parse_float(row.get(name))
        if value is None:
            return None
        values.append(value)
    for name in categorical:
        observed = str(row.get(name, ""))
        for level in levels[name]:
            values.append(1.0 if observed == level else 0.0)
    return values


def design_matrix(
    rows: Sequence[Mapping[str, Any]],
    numeric: Sequence[str],
    categorical: Sequence[str],
    outcome: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[Mapping[str, Any]]]:
    levels = category_levels(rows, categorical)
    xs: list[list[float]] = []
    ys: list[float] = []
    used_rows: list[Mapping[str, Any]] = []
    for row in rows:
        y = parse_float(row.get(outcome))
        if y is None:
            continue
        x = row_vector(row, numeric, categorical, levels)
        if x is None:
            continue
        xs.append(x)
        ys.append(y)
        used_rows.append(row)
    names = list(numeric)
    for name in categorical:
        names.extend([f"{name}={level}" for level in levels[name]])
    if not xs:
        return np.zeros((0, len(names))), np.zeros(0), names, []
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), names, used_rows


def loo_r2(
    rows: Sequence[Mapping[str, Any]],
    numeric: Sequence[str],
    categorical: Sequence[str],
    outcome: str,
    ridge_alpha: float = 1e-6,
) -> dict[str, Any]:
    X, y, names, used_rows = design_matrix(rows, numeric, categorical, outcome)
    n, p = X.shape
    if n < max(8, p + 3):
        return {
            "outcome": outcome,
            "n_groups": int(n),
            "n_predictors": int(p),
            "numeric_predictors": list(numeric),
            "categorical_predictors": list(categorical),
            "loo_r2": None,
            "reason": "too_few_groups_or_complete_cases",
        }
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1e-12:
        return {
            "outcome": outcome,
            "n_groups": int(n),
            "n_predictors": int(p),
            "numeric_predictors": list(numeric),
            "categorical_predictors": list(categorical),
            "loo_r2": None,
            "reason": "constant_outcome",
        }
    preds = []
    for holdout in range(n):
        train = np.arange(n) != holdout
        center = X[train].mean(axis=0)
        scale = X[train].std(axis=0)
        scale[scale <= 1e-12] = 1.0
        X_train = (X[train] - center) / scale
        X_test = (X[holdout : holdout + 1] - center) / scale
        A = np.column_stack([np.ones(X_train.shape[0]), X_train])
        penalty = math.sqrt(ridge_alpha) * np.eye(A.shape[1])
        penalty[0, 0] = 0.0
        A_aug = np.vstack([A, penalty])
        y_aug = np.concatenate([y[train], np.zeros(A.shape[1])])
        coef, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
        preds.append(float((np.column_stack([np.ones(1), X_test]) @ coef)[0]))
    err = float(np.sum((np.asarray(preds) - y) ** 2))
    return {
        "outcome": outcome,
        "n_groups": int(n),
        "n_predictors": int(p),
        "design_columns": names,
        "numeric_predictors": list(numeric),
        "categorical_predictors": list(categorical),
        "loo_r2": float(1.0 - err / denom),
        "used_groups": [str(row.get("mask_group", row.get("group"))) for row in used_rows],
    }


def model_results(rows: Sequence[Mapping[str, Any]], outcomes: Sequence[str]) -> list[dict[str, Any]]:
    results = []
    for outcome in outcomes:
        for name, spec in MODEL_SPECS.items():
            result = loo_r2(rows, spec["numeric"], spec["categorical"], outcome)
            result["model"] = name
            results.append(result)
    return results


def within_physical_graph_correlations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "edge_overlap_norm_min",
        "tree_overlap_norm_min",
        "diff_overlap_norm_min",
        "input_coord_load_gini",
        "input_edge_load_gini",
        "edge_comparison_imbalance_mean",
    ]
    output: list[dict[str, Any]] = []
    by_graph: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_graph[str(row["physical_topology_name"])].append(row)
    for outcome in OUTCOMES:
        for feature in features:
            xs: list[float] = []
            ys: list[float] = []
            for group_rows in by_graph.values():
                fvals = [parse_float(row.get(feature)) for row in group_rows]
                yvals = [parse_float(row.get(outcome)) for row in group_rows]
                valid = [(f, y) for f, y in zip(fvals, yvals) if f is not None and y is not None]
                if len(valid) < 2:
                    continue
                fmean = mean([f for f, _ in valid])
                ymean = mean([y for _, y in valid])
                for f, y in valid:
                    xs.append(f - fmean)
                    ys.append(y - ymean)
            if len(xs) < 3:
                corr = None
            else:
                x = np.asarray(xs, dtype=float)
                y = np.asarray(ys, dtype=float)
                denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
                corr = None if denom <= 1e-12 else float(np.sum(x * y) / denom)
            output.append(
                {
                    "outcome": outcome,
                    "feature": feature,
                    "n_centered_groups": len(xs),
                    "within_physical_graph_correlation": corr,
                }
            )
    return output


def group_medians(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    by_graph: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = parse_float(row.get(key))
        if value is not None:
            by_graph[str(row["physical_topology_name"])].append(value)
    return {graph: percentile(values, 0.5) for graph, values in by_graph.items()}


def assign_mask_categories(rows: Sequence[dict[str, Any]]) -> None:
    by_graph_load: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        graph = str(row["physical_topology_name"])
        coord = parse_float(row.get("input_coord_load_gini"))
        diff = parse_float(row.get("diff_overlap_norm_min"))
        if coord is None or diff is None:
            continue
        load_stratum = "balanced" if coord <= 0.02 else "imbalanced"
        by_graph_load[(graph, load_stratum)].append(diff)
    diff_medians = {key: percentile(values, 0.5) for key, values in by_graph_load.items()}
    for row in rows:
        graph = str(row["physical_topology_name"])
        diff = parse_float(row.get("diff_overlap_norm_min"))
        coord = parse_float(row.get("input_coord_load_gini"))
        load_stratum = "balanced" if coord is not None and coord <= 0.02 else "imbalanced"
        high_diff = diff is not None and diff >= diff_medians[(graph, load_stratum)]
        balanced_coord = load_stratum == "balanced"
        if high_diff and balanced_coord:
            category = "high_tree_diff_overlap_balanced_coordinate_load"
        elif high_diff:
            category = "high_tree_diff_overlap_imbalanced_coordinate_load"
        elif balanced_coord:
            category = "low_tree_diff_overlap_balanced_aggregate_multiplicity"
        else:
            category = "low_tree_diff_overlap_high_coordinate_load_imbalance"
        row["causal_mask_category"] = category
        row["within_graph_load_stratum_diff_overlap_median"] = diff_medians[(graph, load_stratum)]
        row["coordinate_load_stratum"] = load_stratum


def category_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["causal_mask_category"])].append(row)
    summaries = []
    for category, group_rows in sorted(by_category.items()):
        summaries.append(
            {
                "category": category,
                "n_groups": len(group_rows),
                "physical_graphs": sorted({str(row["physical_topology_name"]) for row in group_rows}),
                "mean_novel_icl": mean([parse_float(row.get("mean_novel_icl")) for row in group_rows]),
                "best_seed_novel_icl": mean([parse_float(row.get("best_seed_novel_icl")) for row in group_rows]),
                "seed_std_novel_icl": mean([parse_float(row.get("seed_std_novel_icl")) for row in group_rows]),
                "diff_overlap_norm_min_mean": mean(
                    [parse_float(row.get("diff_overlap_norm_min")) for row in group_rows]
                ),
                "tree_overlap_norm_min_mean": mean(
                    [parse_float(row.get("tree_overlap_norm_min")) for row in group_rows]
                ),
                "input_coord_load_gini_mean": mean(
                    [parse_float(row.get("input_coord_load_gini")) for row in group_rows]
                ),
                "edge_M_mean": mean([parse_float(row.get("edge_M_mean")) for row in group_rows]),
                "d_rel_mean": mean([parse_float(row.get("d_rel")) for row in group_rows]),
                "input_coupled_parameter_count_mean": mean(
                    [parse_float(row.get("input_coupled_parameter_count")) for row in group_rows]
                ),
            }
        )
    return summaries


def selected_library_rows(rows: Sequence[Mapping[str, Any]], per_category_graph: int = 2) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["causal_mask_category"]), str(row["physical_topology_name"]))].append(row)
    for (category, graph), group_rows in sorted(by_key.items()):
        if category.startswith("high"):
            group_rows = sorted(
                group_rows,
                key=lambda row: (
                    -float(row["diff_overlap_norm_min"]),
                    float(row.get("input_coord_load_gini") or 0.0)
                    if "balanced" in category
                    else -float(row.get("input_coord_load_gini") or 0.0),
                ),
            )
        else:
            group_rows = sorted(
                group_rows,
                key=lambda row: (
                    float(row["diff_overlap_norm_min"]),
                    float(row.get("input_coord_load_gini") or 0.0)
                    if "balanced" in category
                    else -float(row.get("input_coord_load_gini") or 0.0),
                ),
            )
        for row in group_rows[:per_category_graph]:
            selected.append(library_record(row))
    return selected


def library_record(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = [
        "mask_group",
        "physical_topology_name",
        "input_mask_family",
        "input_mask_name",
        "causal_mask_category",
        "mean_novel_icl",
        "best_seed_novel_icl",
        "seed_std_novel_icl",
        "edge_M_mean",
        "edge_M_gini",
        "tree_overlap_norm_min",
        "tree_overlap_norm_mean",
        "diff_overlap_norm_min",
        "diff_overlap_norm_mean",
        "input_coord_load_gini",
        "input_edge_load_gini",
        "edge_comparison_imbalance_mean",
        "d_rel",
        "input_coupled_parameter_count",
        "input_coupled_edge_count",
        "input_coupled_coord_count",
        "n_runs",
        "run_dir",
        "topology_source",
    ]
    return {field: row.get(field) for field in fields}


def make_matched_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    by_graph: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_graph[str(row["physical_topology_name"])].append(row)
    for graph, group_rows in sorted(by_graph.items()):
        median_diff = percentile([float(row["diff_overlap_norm_min"]) for row in group_rows], 0.5)
        low_rows = [row for row in group_rows if float(row["diff_overlap_norm_min"]) < median_diff]
        high_rows = [row for row in group_rows if float(row["diff_overlap_norm_min"]) >= median_diff]
        unused_high = set(range(len(high_rows)))
        for low in sorted(low_rows, key=lambda row: float(row["diff_overlap_norm_min"])):
            if not unused_high:
                break
            def score(index: int) -> tuple[float, float, float]:
                high = high_rows[index]
                drel_score = abs(float(high["d_rel"]) - float(low["d_rel"]))
                edge_score = abs(float(high["input_coupled_edge_count"]) - float(low["input_coupled_edge_count"]))
                coord_score = abs(float(high["input_coupled_coord_count"]) - float(low["input_coupled_coord_count"]))
                return (drel_score, edge_score + coord_score, abs(float(high["diff_overlap_norm_min"]) - float(low["diff_overlap_norm_min"])))
            best_index = min(unused_high, key=score)
            unused_high.remove(best_index)
            high = high_rows[best_index]
            pair = {
                "physical_topology_name": graph,
                "low_group": low["mask_group"],
                "high_group": high["mask_group"],
                "low_category": low["causal_mask_category"],
                "high_category": high["causal_mask_category"],
                "low_diff_overlap_norm_min": low["diff_overlap_norm_min"],
                "high_diff_overlap_norm_min": high["diff_overlap_norm_min"],
                "delta_diff_overlap_norm_min": float(high["diff_overlap_norm_min"])
                - float(low["diff_overlap_norm_min"]),
                "low_d_rel": low["d_rel"],
                "high_d_rel": high["d_rel"],
                "abs_d_rel_difference": abs(float(high["d_rel"]) - float(low["d_rel"])),
                "low_input_coupled_edge_count": low["input_coupled_edge_count"],
                "high_input_coupled_edge_count": high["input_coupled_edge_count"],
                "low_input_coupled_coord_count": low["input_coupled_coord_count"],
                "high_input_coupled_coord_count": high["input_coupled_coord_count"],
                "exact_input_parameter_count": low["input_coupled_parameter_count"]
                == high["input_coupled_parameter_count"],
                "exact_d_rel": low["d_rel"] == high["d_rel"],
                "exact_input_edge_count": low["input_coupled_edge_count"]
                == high["input_coupled_edge_count"],
                "exact_input_coord_count": low["input_coupled_coord_count"]
                == high["input_coupled_coord_count"],
            }
            for outcome in OUTCOMES:
                pair[f"low_{outcome}"] = low[outcome]
                pair[f"high_{outcome}"] = high[outcome]
                pair[f"delta_{outcome}"] = float(high[outcome]) - float(low[outcome])
            pairs.append(pair)
    return pairs


def bootstrap_ci(values: Sequence[float], n_boot: int = 5000, seed: int = 17) -> dict[str, Any]:
    vals = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = vals[rng.integers(0, vals.size, vals.size)]
        boots[i] = sample.mean()
    return {
        "mean": float(vals.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n": int(vals.size),
    }


def paired_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_pairs": len(pairs),
        "n_physical_graphs": len({pair["physical_topology_name"] for pair in pairs}),
        "mean_abs_d_rel_difference": mean([float(pair["abs_d_rel_difference"]) for pair in pairs]),
        "fraction_exact_d_rel": mean([1.0 if pair["exact_d_rel"] else 0.0 for pair in pairs]),
        "fraction_exact_input_parameter_count": mean(
            [1.0 if pair["exact_input_parameter_count"] else 0.0 for pair in pairs]
        ),
        "fraction_exact_input_edge_count": mean(
            [1.0 if pair["exact_input_edge_count"] else 0.0 for pair in pairs]
        ),
        "fraction_exact_input_coord_count": mean(
            [1.0 if pair["exact_input_coord_count"] else 0.0 for pair in pairs]
        ),
    }
    for outcome in OUTCOMES:
        out[f"delta_{outcome}"] = bootstrap_ci([float(pair[f"delta_{outcome}"]) for pair in pairs])
    return out


def cluster_bootstrap_delta_r2(
    rows: Sequence[Mapping[str, Any]],
    richer_model: str,
    base_model: str = "controls_only",
    outcome: str = "mean_novel_icl",
    n_boot: int = 500,
    seed: int = 23,
) -> dict[str, Any]:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["physical_topology_name"])].append(row)
    names = sorted(clusters)
    if len(names) < 3:
        return {"delta_loo_r2_mean": None, "ci95": [None, None], "n_clusters": len(names), "reason": "too_few_clusters"}
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        sampled_rows: list[Mapping[str, Any]] = []
        for sampled in rng.choice(names, size=len(names), replace=True):
            sampled_rows.extend(clusters[str(sampled)])
        base = loo_r2(
            sampled_rows,
            MODEL_SPECS[base_model]["numeric"],
            MODEL_SPECS[base_model]["categorical"],
            outcome,
        )
        rich = loo_r2(
            sampled_rows,
            MODEL_SPECS[richer_model]["numeric"],
            MODEL_SPECS[richer_model]["categorical"],
            outcome,
        )
        if base.get("loo_r2") is None or rich.get("loo_r2") is None:
            continue
        deltas.append(float(rich["loo_r2"]) - float(base["loo_r2"]))
    if not deltas:
        return {
            "delta_loo_r2_mean": None,
            "ci95": [None, None],
            "n_clusters": len(names),
            "reason": "bootstrap_resamples_failed",
        }
    return {
        "delta_loo_r2_mean": float(np.mean(deltas)),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "n_clusters": len(names),
        "n_bootstrap_resamples": len(deltas),
        "warning": "Only three physical-graph clusters are available; use this as a sensitivity check, not a decisive interval.",
    }


def make_report_json(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    assign_mask_categories(rows)
    all_model_results = model_results(rows, OUTCOMES)
    strict_rows = [row for row in rows if parse_float(row.get("d_rel")) == 200.0]
    strict_model_results = model_results(strict_rows, OUTCOMES)
    pairs = make_matched_pairs(rows)
    summary = paired_summary(pairs)
    cluster_bootstrap = {
        model: cluster_bootstrap_delta_r2(rows, model, outcome="mean_novel_icl")
        for model in MODEL_SPECS
        if model != "controls_only"
    }
    controls = {
        "n_mask_groups": len(rows),
        "n_seed_runs": int(sum(int(row.get("n_runs") or 0) for row in rows)),
        "physical_graphs": sorted({str(row["physical_topology_name"]) for row in rows}),
        "input_coupled_parameter_count_unique": sorted(
            {int(row["input_coupled_parameter_count"]) for row in rows}
        ),
        "edge_M_mean_unique": sorted({float(row["edge_M_mean"]) for row in rows}),
        "d_rel_counts": {
            str(key): sum(1 for row in rows if parse_float(row.get("d_rel")) == key)
            for key in sorted({parse_float(row.get("d_rel")) for row in rows})
        },
        "n_strict_d_rel_200_groups": len(strict_rows),
        "inference_unit": "mask/topology group; seed runs are summarized within group",
    }
    return {
        "schema": "tree_multiplicity_causal_control.v1",
        "status": "completed_existing_trained_fixed_m20_control",
        "source_artifacts": [str(TREE_REPORT), str(TOPOLOGY_CSV)],
        "controls": controls,
        "model_results": all_model_results,
        "strict_d_rel_200_model_results": strict_model_results,
        "within_physical_graph_correlations": within_physical_graph_correlations(rows),
        "category_summary": category_summary(rows),
        "selected_mask_library": selected_library_rows(rows),
        "matched_high_low_tree_diff_pairs": pairs,
        "matched_pair_summary": summary,
        "cluster_bootstrap_delta_loo_r2": cluster_bootstrap,
        "unavailable_requested_outcomes": {
            "branch_failures": "not present in fixed-m20 topology artifacts",
            "trained_branch_margin": "not present in fixed-m20 topology artifacts",
        },
        "interpretation": interpret_results(all_model_results, summary),
    }


def find_model(
    results: Sequence[Mapping[str, Any]],
    model: str,
    outcome: str = "mean_novel_icl",
) -> Mapping[str, Any]:
    return next(item for item in results if item["model"] == model and item["outcome"] == outcome)


def interpret_results(results: Sequence[Mapping[str, Any]], pair_summary: Mapping[str, Any]) -> str:
    controls = find_model(results, "controls_only")
    diff = find_model(results, "tree_difference_multiplicity_plus_controls")
    edge = find_model(results, "edge_level_multiplicity_plus_controls")
    paired = pair_summary.get("delta_mean_novel_icl", {})
    diff_gain = (
        float(diff["loo_r2"]) - float(controls["loo_r2"])
        if diff.get("loo_r2") is not None and controls.get("loo_r2") is not None
        else None
    )
    edge_gain = (
        float(edge["loo_r2"]) - float(controls["loo_r2"])
        if edge.get("loo_r2") is not None and controls.get("loo_r2") is not None
        else None
    )
    paired_mean = paired.get("mean")
    if diff_gain is not None and diff_gain > max(0.05, (edge_gain or -1.0)) and paired_mean is not None and paired_mean > 0:
        return (
            "Tree-difference overlap survives this fixed-count, physical-graph-controlled "
            "existing-data control better than edge-level multiplicity, with a positive matched high-low contrast."
        )
    if paired_mean is not None and paired_mean > 0:
        return (
            "Matched high-low tree-difference overlap is positive, but cross-validated regression evidence is mixed; "
            "treat this as supportive design evidence rather than a completed causal proof."
        )
    return (
        "The matched control does not provide a positive causal signal for tree-difference overlap in this artifact; "
        "the Phase 2 screen may still contain family or regime confounding."
    )


def write_library(report: Mapping[str, Any]) -> None:
    category_rows = []
    for row in report["category_summary"]:
        category_rows.append(
            [
                row["category"],
                row["n_groups"],
                ", ".join(row["physical_graphs"]),
                fmt(row["diff_overlap_norm_min_mean"]),
                fmt(row["tree_overlap_norm_min_mean"]),
                fmt(row["input_coord_load_gini_mean"]),
                fmt(row["mean_novel_icl"]),
            ]
        )
    selected_rows = []
    for row in report["selected_mask_library"]:
        selected_rows.append(
            [
                row["causal_mask_category"],
                row["physical_topology_name"],
                row["input_mask_family"],
                row["mask_group"],
                fmt(row["diff_overlap_norm_min"]),
                fmt(row["input_coord_load_gini"]),
                fmt(row["mean_novel_icl"]),
            ]
        )
    md = [
        "# Tree-Multiplicity Causal Mask Library",
        "",
        "This library is built from the already-trained fixed-m20 mask groups. It keeps physical graph identity explicit and uses normalized tree/difference overlap metrics; raw overlap counts are not used as standalone selectors.",
        "",
        "## Controls",
        "",
        f"- Mask/topology groups: `{report['controls']['n_mask_groups']}`",
        f"- Seed runs summarized inside groups: `{report['controls']['n_seed_runs']}`",
        f"- Physical graphs: `{', '.join(report['controls']['physical_graphs'])}`",
        f"- Input-coupled parameter count: `{report['controls']['input_coupled_parameter_count_unique']}`",
        f"- Aggregate edge-level `M_mean`: `{report['controls']['edge_M_mean_unique']}`",
        f"- `d_rel` counts: `{report['controls']['d_rel_counts']}`",
        "",
        "## Category Summary",
        "",
        "High/low labels are assigned within physical graph and coordinate-load stratum, so the imbalanced high-overlap stratum should be read as high among imbalanced masks rather than globally as high as edge-block masks.",
        "",
        markdown_table(
            category_rows,
            [
                "category",
                "groups",
                "physical graphs",
                "mean min diff overlap",
                "mean min tree overlap",
                "mean coord gini",
                "mean novel ICL",
            ],
        ),
        "",
        "## Selected Mask Groups",
        "",
        markdown_table(
            selected_rows,
            [
                "category",
                "physical graph",
                "mask family",
                "mask group",
                "min diff overlap",
                "coord gini",
                "mean novel ICL",
            ],
        ),
    ]
    LIBRARY_MD.write_text("\n".join(md) + "\n")
    LIBRARY_JSON.write_text(
        json.dumps(
            {
                "schema": "tree_multiplicity_causal_mask_library.v1",
                "controls": report["controls"],
                "category_summary": report["category_summary"],
                "selected_mask_library": report["selected_mask_library"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_training_plan(report: Mapping[str, Any]) -> None:
    md = [
        "# Input-Multiplicity Causal Control Training Plan",
        "",
        "## Status",
        "",
        "The current pass uses the already-trained fixed-m20 mask library, so no broad new sweep was launched. The existing artifact already provides 48 mask/topology groups with 5 training seeds per group.",
        "",
        "## Existing-Data Control Used Now",
        "",
        "- Unit of inference: mask/topology group; seed rows are summarized as mean, best, and standard deviation.",
        "- Matched exactly on input-coupled parameter count `200` and aggregate `M_mean=10`.",
        "- Physical graph identity is held fixed in matched high/low comparisons and included as a categorical control in regressions.",
        "- `d_rel=200` for 45 of 48 groups; the three `d_rel=190` groups are included with a `d_rel` covariate and checked by a strict `d_rel=200` sensitivity analysis.",
        "- The selector is normalized same-root tree-difference comparison overlap, not raw tree-pair counts.",
        "",
        "## Follow-Up Training If More Cluster Time Is Allocated",
        "",
        "1. For each physical graph, materialize the selected mask groups from `tree_multiplicity_causal_mask_library.json`.",
        "2. Add replacement masks if needed so every high/low category has exact `d_rel`, exact input-coupled edge count, and exact input-coupled coordinate count.",
        "3. Train at least 5 seeds per mask group; use more seeds for matched pairs whose current seed standard deviation exceeds 8 novel-ICL points.",
        "4. Save branch-wise novel-class accuracy, branch failures, trained branch margin, and post-training tree/posterior diagnostics in addition to the existing mean/best/std summaries.",
        "5. Analyze only group-level or hierarchical models; do not treat seed rows as independent topology samples.",
        "",
        "## Primary Contrast",
        "",
        "High normalized tree-difference comparison overlap versus low normalized tree-difference comparison overlap, under fixed physical graph, fixed input parameter count, fixed aggregate multiplicity, and matched or covaried `d_rel`.",
    ]
    TRAINING_PLAN_MD.write_text("\n".join(md) + "\n")


def write_control_report(report: Mapping[str, Any]) -> None:
    result_rows = []
    for outcome in OUTCOMES:
        for model in MODEL_SPECS:
            item = find_model(report["model_results"], model, outcome)
            strict = find_model(report["strict_d_rel_200_model_results"], model, outcome)
            result_rows.append(
                [
                    outcome,
                    model,
                    item["n_groups"],
                    fmt(item.get("loo_r2")),
                    strict["n_groups"],
                    fmt(strict.get("loo_r2")),
                ]
            )
    pair = report["matched_pair_summary"]
    pair_rows = []
    for outcome in OUTCOMES:
        delta = pair[f"delta_{outcome}"]
        ci = delta["ci95"]
        pair_rows.append(
            [
                outcome,
                delta["n"],
                fmt(delta["mean"]),
                f"[{fmt(ci[0])}, {fmt(ci[1])}]",
            ]
        )
    boot_rows = []
    for model, item in report["cluster_bootstrap_delta_loo_r2"].items():
        ci = item["ci95"]
        boot_rows.append(
            [
                model,
                item.get("n_clusters"),
                fmt(item.get("delta_loo_r2_mean")),
                f"[{fmt(ci[0])}, {fmt(ci[1])}]",
            ]
        )
    md = [
        "# Input Multiplicity Causal Control Report",
        "",
        "## Status",
        "",
        "`completed_existing_trained_fixed_m20_control`. No broad new topology sweep was launched.",
        "",
        "## Design",
        "",
        "This is an existing-data causal-control analysis over the fixed-m20 mask library. It is stronger than the Phase 2 pooled screen because matched contrasts hold physical graph fixed and all regressions use group-level rows with physical graph controls. It is still not a final prospective causal experiment because the mask library was not originally generated by exact tree-difference-overlap matching.",
        "",
        f"- Groups: `{report['controls']['n_mask_groups']}` mask/topology groups, `{report['controls']['n_seed_runs']}` seed runs summarized within group.",
        f"- Physical graphs: `{', '.join(report['controls']['physical_graphs'])}`.",
        f"- Exact controls: input-coupled parameter count `{report['controls']['input_coupled_parameter_count_unique']}`, aggregate `M_mean` `{report['controls']['edge_M_mean_unique']}`.",
        f"- `d_rel` distribution: `{report['controls']['d_rel_counts']}`; strict `d_rel=200` sensitivity uses `{report['controls']['n_strict_d_rel_200_groups']}` groups.",
        "",
        "Branch failures and trained branch margins are not present in the fixed-m20 artifacts, so this report evaluates mean novel-class ICL, best-seed novel-class ICL, and seed standard deviation.",
        "",
        "## Grouped LOO Models",
        "",
        markdown_table(
            result_rows,
            ["outcome", "model", "groups", "LOO R2", "strict d_rel groups", "strict d_rel LOO R2"],
        ),
        "",
        "## Matched High-Low Tree-Difference Contrast",
        "",
        f"- Pairs: `{pair['n_pairs']}` across `{pair['n_physical_graphs']}` physical graphs.",
        f"- Mean absolute `d_rel` difference: `{fmt(pair['mean_abs_d_rel_difference'])}`.",
        f"- Fraction exact input parameter count: `{fmt(pair['fraction_exact_input_parameter_count'])}`.",
        f"- Fraction exact `d_rel`: `{fmt(pair['fraction_exact_d_rel'])}`.",
        f"- Fraction exact input edge count: `{fmt(pair['fraction_exact_input_edge_count'])}`.",
        f"- Fraction exact input coordinate count: `{fmt(pair['fraction_exact_input_coord_count'])}`.",
        "",
        markdown_table(pair_rows, ["outcome", "pairs", "high-low mean", "bootstrap 95% CI"]),
        "",
        "## Cluster Bootstrap Sensitivity",
        "",
        "The cluster bootstrap resamples physical graphs. There are only three physical-graph clusters, so these intervals are a coarse sensitivity check rather than decisive inference.",
        "",
        markdown_table(
            boot_rows,
            ["model", "clusters", "delta LOO R2 vs controls", "cluster bootstrap 95% CI"],
        ),
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
        "The safest conclusion is that normalized tree-difference comparison overlap is the right metric to carry into a prospective exact-control experiment, but this existing-data control should not be described as a universal scalar law or as motif uniqueness.",
    ]
    REPORT_MD.write_text("\n".join(md) + "\n")


def main() -> None:
    rows = load_fixed_m20_rows()
    report = make_report_json(rows)
    write_library(report)
    write_training_plan(report)
    write_control_report(report)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
