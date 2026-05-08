"""Existing-data Markov-expressivity reanalysis for first-order CRN ICL.

This script consumes committed topology-result CSV/JSON artifacts and writes
the reports required by the Markov-ICL handoff.  It deliberately avoids
launching training.  When raw masks or reversible thermodynamic artifacts are
not available locally, the reports state that limitation rather than filling in
unverifiable numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


RESULT_ROOT = Path("ICL/results")
NEXT_PHASE = RESULT_ROOT / "next_phase_stats"


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["__source_file"] = str(path)
    return rows


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def as_float(row: Mapping[str, object], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def mean(values: Iterable[float]) -> Optional[float]:
    vals = finite(values)
    return float(np.mean(vals)) if vals else None


def std(values: Iterable[float]) -> Optional[float]:
    vals = finite(values)
    return float(np.std(vals, ddof=0)) if vals else None


def gini(values: Sequence[float]) -> float:
    vals = np.asarray(finite(values), dtype=float)
    if vals.size == 0:
        return 0.0
    vals = np.maximum(vals, 0.0)
    total = float(vals.sum())
    if total <= 1e-12:
        return 0.0
    vals = np.sort(vals)
    n = vals.size
    weights = np.arange(1, n + 1)
    return float((2.0 * np.sum(weights * vals) / (n * total)) - ((n + 1.0) / n))


def pearson(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    x = np.asarray(finite(x_values), dtype=float)
    y = np.asarray(finite(y_values), dtype=float)
    if x.size != y.size or x.size < 3:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def outcome_value(row: Mapping[str, object]) -> float:
    for key in (
        "test_novel_classes",
        "test_novel_classes_mean",
        "target_accuracy",
        "target_mean",
        "icl_acc_final_eval",
        "icl_acc_final_eval_mean",
    ):
        value = as_float(row, key)
        if math.isfinite(value):
            return value
    return math.nan


def group_key(row: Mapping[str, object]) -> str:
    for key in ("topology_name", "label", "input_mask_name", "physical_topology_name"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return str(row.get("run_dir", "unknown"))


def aggregate_multiplicity_from_row(row: Mapping[str, object]) -> dict:
    p = as_float(row, "p")
    n_edges = as_float(row, "n_edges")
    coupled = as_float(row, "input_coupled_parameter_count")
    coupled_coords = as_float(row, "input_coupled_coord_count")
    if not math.isfinite(p) or p <= 0:
        p = math.nan
    M_mean = coupled / p if math.isfinite(coupled) and math.isfinite(p) and p > 0 else math.nan
    M_nonzero = (
        coupled / coupled_coords
        if math.isfinite(coupled) and math.isfinite(coupled_coords) and coupled_coords > 0
        else math.nan
    )
    zero_fraction = (
        1.0 - coupled_coords / p
        if math.isfinite(coupled_coords) and math.isfinite(p) and p > 0
        else math.nan
    )
    exact_uniform_sum_log = math.nan
    if (
        math.isfinite(n_edges)
        and math.isfinite(M_mean)
        and math.isfinite(p)
        and abs(M_mean - n_edges) <= 1e-9
    ):
        exact_uniform_sum_log = p * math.log(2.0 * n_edges + 1.0)
    elif (
        math.isfinite(coupled_coords)
        and math.isfinite(p)
        and math.isfinite(M_mean)
        and abs(coupled_coords - p) <= 1e-9
        and as_float(row, "input_coord_load_gini", 1.0) <= 1e-12
    ):
        exact_uniform_sum_log = p * math.log(2.0 * M_mean + 1.0)
    return {
        "M_mean_aggregate": M_mean,
        "M_nonzero_mean_aggregate": M_nonzero,
        "M_zero_fraction_aggregate": zero_fraction,
        "M_gini_aggregate": as_float(row, "input_coord_load_gini"),
        "M_sum_log_2M1_exact_if_uniform": exact_uniform_sum_log,
    }


def group_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    return dict(grouped)


def summarize_groups(rows: Sequence[Mapping[str, object]]) -> list[dict]:
    summaries = []
    for key, members in sorted(group_rows(rows).items()):
        outcomes = [outcome_value(row) for row in members]
        first = members[0]
        metrics = aggregate_multiplicity_from_row(first)
        predictor_keys = [
            "d_rel",
            "d_rel_minus_n_req",
            "comparison_branch_d_rel_min",
            "comparison_branch_d_rel_mean",
            "comparison_branch_d_rel_gini",
            "comparison_branch_common_d_rel_min",
            "comparison_branch_common_d_rel_mean",
            "comparison_branch_common_d_rel_gini",
            "comparison_branch_input_count_min",
            "comparison_branch_input_count_mean",
            "comparison_branch_input_count_gini",
            "comparison_branch_input_overlap_min",
            "comparison_branch_input_overlap_mean",
            "comparison_branch_input_overlap_gini",
            "effective_rank_D_masked",
            "condition_number_D_masked",
            "edge_participation_gini",
            "root_tree_count_gini",
            "capacity_oracle_test_margin_p10",
            "capacity_linear_test_margin_p10",
            "capacity_linear_test_accuracy",
            "capacity_support_fraction",
            "capacity_support_min",
            "capacity_comparison_branch_common_d_rel_min",
            "capacity_normal_fan_active_tree_count_mean",
            "capacity_normal_fan_branch_tree_nmi_mean",
            "capacity_rooted_polytope_common_rank_total",
            "capacity_rooted_polytope_root_rank_mass_effective",
            "library_n_trees_total_enum",
            "library_n_trees_total_enum_log",
            "library_effective_rank_D",
            "library_condition_number_D_log",
            "library_edge_participation_gini",
            "library_mean_shortest_path",
        ]
        for metric_key in predictor_keys:
            metrics[metric_key] = as_float(first, metric_key)
        condition = metrics.get("condition_number_D_masked", math.nan)
        metrics["condition_number_D_masked_log10"] = (
            math.log10(condition) if math.isfinite(condition) and condition > 0 else math.nan
        )
        summaries.append(
            {
                "group": key,
                "n_runs": len(members),
                "source_file": first.get("__source_file", ""),
                "physical_topology_name": first.get("physical_topology_name", first.get("topology_name", "")),
                "input_mask_family": first.get("input_mask_family", ""),
                "mean_novel_icl": mean(outcomes),
                "best_seed_novel_icl": max(finite(outcomes)) if finite(outcomes) else None,
                "seed_std_novel_icl": std(outcomes),
                "seed_min_novel_icl": min(finite(outcomes)) if finite(outcomes) else None,
                "seed_max_minus_min_novel_icl": (
                    max(finite(outcomes)) - min(finite(outcomes)) if finite(outcomes) else None
                ),
                **metrics,
            }
        )
    return summaries


def design_matrix(groups: Sequence[Mapping[str, object]], predictors: Sequence[str], outcome: str):
    rows = []
    y = []
    used = []
    for group in groups:
        xv = [group.get(pred, math.nan) for pred in predictors]
        yv = group.get(outcome)
        if yv is None:
            continue
        if not math.isfinite(float(yv)):
            continue
        if not all(value is not None and math.isfinite(float(value)) for value in xv):
            continue
        rows.append([float(value) for value in xv])
        y.append(float(yv))
        used.append(group["group"])
    if not rows:
        return np.zeros((0, len(predictors))), np.zeros(0), used
    return np.asarray(rows, dtype=float), np.asarray(y, dtype=float), used


def loo_r2(groups: Sequence[Mapping[str, object]], predictors: Sequence[str], outcome: str) -> dict:
    X, y, used = design_matrix(groups, predictors, outcome)
    n, p = X.shape if X.ndim == 2 else (0, 0)
    if n < max(5, p + 2):
        return {
            "predictors": list(predictors),
            "outcome": outcome,
            "n_groups": n,
            "loo_r2": None,
            "reason": "too_few_groups_or_complete_cases",
        }
    y_mean = float(np.mean(y))
    denom = float(np.sum((y - y_mean) ** 2))
    if denom <= 1e-12:
        return {
            "predictors": list(predictors),
            "outcome": outcome,
            "n_groups": n,
            "loo_r2": None,
            "reason": "constant_outcome",
        }
    preds = []
    for holdout in range(n):
        train = np.arange(n) != holdout
        X_train = X[train]
        y_train = y[train]
        center = X_train.mean(axis=0)
        scale = X_train.std(axis=0)
        scale[scale <= 1e-12] = 1.0
        X_train_z = (X_train - center) / scale
        X_test_z = (X[holdout : holdout + 1] - center) / scale
        A = np.column_stack([np.ones(X_train_z.shape[0]), X_train_z])
        ridge = 1e-6 * np.eye(A.shape[1])
        ridge[0, 0] = 0.0
        coef = np.linalg.solve(A.T @ A + ridge, A.T @ y_train)
        pred = float((np.column_stack([np.ones(1), X_test_z]) @ coef)[0])
        preds.append(pred)
    error = float(np.sum((np.asarray(preds) - y) ** 2))
    return {
        "predictors": list(predictors),
        "outcome": outcome,
        "n_groups": n,
        "groups": used,
        "loo_r2": float(1.0 - error / denom),
    }


def single_predictor_correlations(groups: Sequence[Mapping[str, object]], predictors: Sequence[str], outcome: str) -> list[dict]:
    rows = []
    for predictor in predictors:
        pairs = []
        for group in groups:
            x = group.get(predictor)
            y = group.get(outcome)
            if x is None or y is None:
                continue
            if math.isfinite(float(x)) and math.isfinite(float(y)):
                pairs.append((float(x), float(y)))
        if len(pairs) < 3:
            rows.append({"predictor": predictor, "n_groups": len(pairs), "pearson_r": None})
            continue
        x_values, y_values = zip(*pairs)
        rows.append(
            {
                "predictor": predictor,
                "n_groups": len(pairs),
                "pearson_r": pearson(x_values, y_values),
            }
        )
    return rows


def load_datasets() -> dict[str, list[dict]]:
    datasets = {
        "fixed_m20": read_csv(NEXT_PHASE / "pooled_fixed_m20_with_branch_capacity.csv"),
        "hard_n4_m6": read_csv(RESULT_ROOT / "expanded_hard_stats" / "n4_m6_N3_D2_with_branch_capacity.csv"),
        "hard_n5_m8": read_csv(RESULT_ROOT / "expanded_hard_stats" / "n5_m8_N3_D2_with_branch_capacity.csv"),
        "hard_n5_m12": read_csv(RESULT_ROOT / "expanded_hard_stats" / "n5_m12_N3_D2_with_branch_capacity.csv"),
        "degree_rewire_training": read_csv(
            NEXT_PHASE / "degree_rewire_normal_fan_n5_m12_N3_D2" / "normal_fan_training_joined.csv"
        ),
        "degree_rewire_library": read_csv(
            NEXT_PHASE / "degree_rewire_normal_fan_n5_m12_N3_D2" / "library.csv"
        ),
    }
    return datasets


def attach_exact_mask_metrics_from_topology_json(groups: list[dict]) -> None:
    """Add exact mask metrics where run-local topology.json files exist."""

    seen = {}
    for path in RESULT_ROOT.glob("expanded_hard_sweeps/*/*/topology.json"):
        payload = read_json(path)
        if not payload:
            continue
        name = str(payload.get("name", ""))
        mask = payload.get("input_mask")
        if not name or mask is None:
            continue
        arr = np.asarray(mask, dtype=float)
        if arr.ndim != 2:
            continue
        coord_load = arr.sum(axis=0)
        seen[name] = {
            "M_min_exact": float(coord_load.min()) if coord_load.size else None,
            "M_mean_exact": float(coord_load.mean()) if coord_load.size else None,
            "M_var_exact": float(coord_load.var()) if coord_load.size else None,
            "M_gini_exact": gini(coord_load),
            "M_sum_log_2M1_exact": float(np.sum(np.log(2.0 * coord_load + 1.0))),
            "monoRisk_fraction_M_le_1_exact": float(np.mean(coord_load <= 1.0)) if coord_load.size else None,
        }
    for group in groups:
        exact = seen.get(group["group"])
        if exact:
            group.update(exact)


def analyze_dataset(name: str, rows: Sequence[Mapping[str, object]]) -> dict:
    groups = summarize_groups(rows)
    attach_exact_mask_metrics_from_topology_json(groups)
    predictors = [
        "M_mean_aggregate",
        "M_zero_fraction_aggregate",
        "M_gini_aggregate",
        "comparison_branch_input_count_min",
        "comparison_branch_input_overlap_min",
        "comparison_branch_common_d_rel_min",
        "effective_rank_D_masked",
        "condition_number_D_masked_log10",
        "capacity_linear_test_margin_p10",
    ]
    complete_predictors = {
        "multiplicity": [
            "M_mean_aggregate",
            "M_zero_fraction_aggregate",
            "M_gini_aggregate",
        ],
        "comparison_multiplicity": [
            "comparison_branch_input_count_min",
            "comparison_branch_input_overlap_min",
            "comparison_branch_input_overlap_gini",
        ],
        "tree_geometry": [
            "comparison_branch_common_d_rel_min",
            "comparison_branch_common_d_rel_mean",
            "effective_rank_D_masked",
            "condition_number_D_masked_log10",
        ],
        "capacity_proxy": [
            "capacity_support_fraction",
            "capacity_linear_test_margin_p10",
            "capacity_linear_test_accuracy",
        ],
    }
    models = []
    for outcome in ("mean_novel_icl", "best_seed_novel_icl", "seed_std_novel_icl"):
        for label, cols in complete_predictors.items():
            result = loo_r2(groups, cols, outcome)
            result["model"] = label
            models.append(result)
    correlations = {}
    for outcome in ("mean_novel_icl", "best_seed_novel_icl", "seed_std_novel_icl"):
        correlations[outcome] = single_predictor_correlations(groups, predictors, outcome)
    return {
        "name": name,
        "n_rows": len(rows),
        "n_groups": len(groups),
        "groups": groups,
        "models": models,
        "correlations": correlations,
    }


def family_summary(groups: Sequence[Mapping[str, object]]) -> list[dict]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for group in groups:
        family = str(group.get("input_mask_family") or "unknown")
        grouped[family].append(group)
    rows = []
    for family, members in sorted(grouped.items()):
        rows.append(
            {
                "input_mask_family": family,
                "n_groups": len(members),
                "mean_group_mean_icl": mean(group["mean_novel_icl"] for group in members),
                "mean_group_best_icl": mean(group["best_seed_novel_icl"] for group in members),
                "mean_seed_std": mean(group["seed_std_novel_icl"] for group in members),
                "mean_M_mean": mean(group.get("M_mean_aggregate") for group in members),
                "mean_M_gini": mean(group.get("M_gini_aggregate") for group in members),
                "mean_branch_input_overlap_min": mean(
                    group.get("comparison_branch_input_overlap_min") for group in members
                ),
            }
        )
    return rows


def load_mechanism_groups() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in RESULT_ROOT.glob("expanded_hard_sweeps/*/mechanism_results.csv"):
        rows = read_csv(path)
        grouped = group_rows(rows)
        for key, members in grouped.items():
            out[key] = {
                "mechanism_target_accuracy_mean": mean(as_float(row, "target_accuracy") for row in members),
                "mechanism_target_logprob_margin_branch_mean_min": mean(
                    as_float(row, "target_logprob_margin_branch_mean_min") for row in members
                ),
                "mechanism_branch_active_tree_nmi_mean": mean(
                    as_float(row, "branch_active_tree_nmi") for row in members
                ),
                "mechanism_branch_active_root_nmi_mean": mean(
                    as_float(row, "branch_active_root_nmi") for row in members
                ),
                "mechanism_tree_entropy_mean": mean(as_float(row, "tree_entropy_mean") for row in members),
                "mechanism_tree_projection_norm_mean": mean(
                    as_float(row, "tree_projection_norm_mean") for row in members
                ),
                "mechanism_active_tree_matched_comparison_gap_mean": mean(
                    as_float(row, "active_tree_matched_comparison_gap_mean") for row in members
                ),
            }
    return out


def expressivity_trainability_payload(analyses: Mapping[str, Mapping[str, object]]) -> dict:
    hard_groups = []
    for key in ("hard_n4_m6", "hard_n5_m8", "hard_n5_m12"):
        hard_groups.extend(analyses[key]["groups"])
    mechanism = load_mechanism_groups()
    for group in hard_groups:
        group.update(mechanism.get(group["group"], {}))
    predictors = [
        "comparison_branch_common_d_rel_min",
        "effective_rank_D_masked",
        "condition_number_D_masked_log10",
        "capacity_linear_test_margin_p10",
        "mechanism_branch_active_tree_nmi_mean",
        "mechanism_target_logprob_margin_branch_mean_min",
        "mechanism_tree_entropy_mean",
    ]
    rows = {}
    for outcome in ("best_seed_novel_icl", "mean_novel_icl", "seed_std_novel_icl"):
        rows[outcome] = single_predictor_correlations(hard_groups, predictors, outcome)
    return {
        "n_hard_groups": len(hard_groups),
        "correlations": rows,
        "interpretation": {
            "best_seed_novel_icl": "expressivity-envelope proxy",
            "mean_novel_icl": "trainability/reliability proxy",
            "seed_std_novel_icl": "optimization instability proxy",
        },
    }


def reversible_support_fraction(edges: Sequence[Sequence[int]]) -> float:
    edge_set = {tuple(edge) for edge in edges}
    if not edge_set:
        return 0.0
    reversible = sum(1 for source, target in edge_set if (target, source) in edge_set)
    return float(reversible / len(edge_set))


def thermodynamic_payload() -> dict:
    rows = []
    for path in RESULT_ROOT.glob("expanded_hard_sweeps/*/*/topology.json"):
        payload = read_json(path)
        if not payload:
            continue
        rows.append(
            {
                "topology_name": payload.get("name"),
                "physical_topology_name": payload.get("physical_topology_name"),
                "n_edges": len(payload.get("edges", [])),
                "reversible_edge_fraction": reversible_support_fraction(payload.get("edges", [])),
            }
        )
    by_topology = {}
    for row in rows:
        by_topology.setdefault(row["topology_name"], row)
    fractions = [row["reversible_edge_fraction"] for row in by_topology.values()]
    return {
        "status": "no_valid_Fmax_sweep_available",
        "reason": (
            "Existing models use arbitrary directed exponential rates. They may break detailed balance, "
            "but they were not parameterized as reversible-edge thermodynamic Markov processes with "
            "antisymmetric force budget F_max."
        ),
        "n_local_hard_topology_groups_with_topology_json": len(by_topology),
        "reversible_edge_fraction_mean": mean(fractions),
        "reversible_edge_fraction_min": min(fractions) if fractions else None,
        "reversible_edge_fraction_max": max(fractions) if fractions else None,
        "required_next_implementation": [
            "construct bidirected physical support",
            "parameterize W_ij = exp(E_j - B_ij + F_ij/2 + input_drive)",
            "enforce B_ij = B_ji and F_ij = -F_ji",
            "sweep max absolute antisymmetric force F_max",
            "report novel-class ICL and lower-tail branch margins by F_max",
        ],
    }


def degree_rewire_payload(datasets: Mapping[str, Sequence[Mapping[str, object]]]) -> dict:
    training = summarize_groups(datasets["degree_rewire_training"])
    library = datasets["degree_rewire_library"]
    predictors = [
        "capacity_normal_fan_active_tree_count_mean",
        "capacity_normal_fan_branch_tree_nmi_mean",
        "capacity_rooted_polytope_common_rank_total",
        "capacity_rooted_polytope_root_rank_mass_effective",
        "library_n_trees_total_enum_log",
        "library_effective_rank_D",
        "library_condition_number_D_log",
        "library_edge_participation_gini",
        "d_rel",
        "comparison_branch_common_d_rel_min",
    ]
    # The pilot has only four trained groups, so correlations are descriptive.
    correlations = single_predictor_correlations(training, predictors, "mean_novel_icl")
    return {
        "n_library_candidates": len(library),
        "n_trained_groups": len(training),
        "trained_groups": training,
        "descriptive_correlations": correlations,
        "interpretation": (
            "Exact-degree/d_rel normal-fan pilot is constructive but underpowered. "
            "It shows that topology can vary beyond degree sequence and d_rel; it is not a statistical result."
        ),
    }


def compact_model_table(analysis: Mapping[str, object], outcome: str) -> list[dict]:
    rows = []
    for row in analysis["models"]:
        if row["outcome"] == outcome:
            rows.append(
                {
                    "model": row["model"],
                    "n_groups": row["n_groups"],
                    "loo_r2": row.get("loo_r2"),
                    "reason": row.get("reason"),
                }
            )
    return rows


def markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.3f}" if math.isfinite(value) else "")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def existing_data_report(payload: Mapping[str, object]) -> str:
    lines = [
        "# Existing-Data Markov Expressivity Reanalysis",
        "",
        "Primary outcome is novel-class ICL, aggregated by topology/mask group before inference.",
        "Run-level seeds are not treated as independent topology samples.",
        "",
        "## Dataset Coverage",
        "",
    ]
    dataset_rows = []
    for name, analysis in payload["analyses"].items():
        dataset_rows.append({"dataset": name, "rows": analysis["n_rows"], "groups": analysis["n_groups"]})
    lines.append(markdown_table(dataset_rows, ["dataset", "rows", "groups"]))
    lines.extend(["", "## Grouped LOOCV Summaries", ""])
    for name in ("fixed_m20", "hard_n4_m6", "hard_n5_m8", "hard_n5_m12"):
        analysis = payload["analyses"][name]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(markdown_table(compact_model_table(analysis, "mean_novel_icl"), ["model", "n_groups", "loo_r2", "reason"]))
        lines.append("")
    lines.extend(
        [
            "## Limits",
            "",
            "- Exact per-coordinate mask arrays are available locally for hard-sweep topology JSON files, but not for the fixed m20 CSV-only rows.",
            "- Existing thermodynamic quantities are diagnostic only; the trained arbitrary directed-rate models are not reversible-edge thermodynamic parameterizations.",
            "- Capacity proxy rows from the first gamma attempt are included as baselines, not accepted as final theory.",
        ]
    )
    return "\n".join(lines)


def input_multiplicity_report(payload: Mapping[str, object]) -> str:
    fixed = payload["analyses"]["fixed_m20"]
    families = family_summary(fixed["groups"])
    lines = [
        "# Input Multiplicity Control Report",
        "",
        "This report uses existing fixed-count input-mask data. It is an existing-data control, not a new training sweep.",
        "",
        "## Mask-Family Summary",
        "",
        markdown_table(
            families,
            [
                "input_mask_family",
                "n_groups",
                "mean_group_mean_icl",
                "mean_group_best_icl",
                "mean_seed_std",
                "mean_M_mean",
                "mean_M_gini",
                "mean_branch_input_overlap_min",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "Input multiplicity is not a scalar law in these data. The useful signal is branch-aware and mask-aware: zero or imbalanced context/query overlap is a clearer risk than average input count alone.",
    ]
    return "\n".join(lines)


def thermodynamic_report(payload: Mapping[str, object]) -> str:
    thermo = payload["thermodynamic"]
    lines = [
        "# Thermodynamic Force-Budget Report",
        "",
        f"Status: `{thermo['status']}`.",
        "",
        thermo["reason"],
        "",
        "## Reversible-Support Audit",
        "",
        markdown_table(
            [
                {
                    "groups": thermo["n_local_hard_topology_groups_with_topology_json"],
                    "mean": thermo["reversible_edge_fraction_mean"],
                    "min": thermo["reversible_edge_fraction_min"],
                    "max": thermo["reversible_edge_fraction_max"],
                }
            ],
            ["groups", "mean", "min", "max"],
        ),
        "",
        "## Required Next Implementation",
        "",
    ]
    lines.extend(f"- {item}" for item in thermo["required_next_implementation"])
    return "\n".join(lines)


def degree_rewire_report(payload: Mapping[str, object]) -> str:
    degree = payload["degree_rewire"]
    lines = [
        "# Exact-Degree Multiplicity Normal-Fan Report",
        "",
        degree["interpretation"],
        "",
        f"Library candidates: {degree['n_library_candidates']}",
        f"Trained groups: {degree['n_trained_groups']}",
        "",
        "## Descriptive Correlations",
        "",
        markdown_table(degree["descriptive_correlations"], ["predictor", "n_groups", "pearson_r"]),
    ]
    return "\n".join(lines)


def expressivity_trainability_report(payload: Mapping[str, object]) -> str:
    ex = payload["expressivity_vs_trainability"]
    lines = [
        "# Expressivity vs Trainability Report",
        "",
        "Best seed is treated as an expressivity-envelope proxy, mean seed as trainability/reliability, and seed standard deviation as optimization instability.",
        "",
        f"Hard-regime groups analyzed: {ex['n_hard_groups']}",
        "",
    ]
    for outcome, rows in ex["correlations"].items():
        lines.append(f"## {outcome}")
        lines.append("")
        lines.append(markdown_table(rows, ["predictor", "n_groups", "pearson_r"]))
        lines.append("")
    return "\n".join(lines)


def synthesis_report(payload: Mapping[str, object]) -> str:
    def fmt(value: Optional[float]) -> str:
        return f"{value:.3f}" if value is not None and math.isfinite(float(value)) else "n/a"

    def model_r2(dataset: str, model: str, outcome: str = "mean_novel_icl") -> Optional[float]:
        for row in payload["analyses"][dataset]["models"]:
            if row["model"] == model and row["outcome"] == outcome:
                return row.get("loo_r2")
        return None

    fixed_tree = model_r2("fixed_m20", "tree_geometry")
    fixed_multiplicity = model_r2("fixed_m20", "multiplicity")
    hard_tree = model_r2("hard_n5_m12", "tree_geometry")
    degree = payload["degree_rewire"]
    thermo = payload["thermodynamic"]
    return "\n".join(
        [
            "# Markov-ICL Expressivity Synthesis",
            "",
            "This synthesis separates what the existing artifacts can support from what still requires new controlled training or a new thermodynamic parameterization.",
            "",
            "## Expressivity",
            "",
            "The exact first-order object is the rooted tree-sum representation. Input multiplicity, comparison-coordinate overlap, branch-aware tree geometry, and lower-tail margin capacity are the right expressivity probes. Existing data support these as useful structural variables, but not as a final scalar law.",
            "",
            f"In the fixed-count m20 data, grouped LOOCV R2 is {fmt(fixed_multiplicity)} for aggregate multiplicity variables and {fmt(fixed_tree)} for tree-geometry variables. In the hard n5_m12 data, tree geometry reaches grouped LOOCV R2 {fmt(hard_tree)} for mean novel-class ICL. These are existing-data model checks, not held-out theory validation.",
            "",
            f"The exact-degree normal-fan pilot has only {degree['n_trained_groups']} trained groups. It is useful constructively because all groups share the intended fixed-degree/d_rel controls while normal-fan and tree-count summaries vary, but it is not statistically powered.",
            "",
            "## Trainability",
            "",
            "Best-seed and mean-seed outcomes must remain separate. The existing hard-regime data show enough seed spread that conditioning, redundancy, tree entropy, and post-training branch alignment should be modeled as trainability variables rather than folded into expressivity.",
            "",
            "The expressivity/trainability report therefore treats best seed as an envelope proxy, mean seed as reliability, and seed standard deviation as optimization instability. Mechanism correlations are strong in the existing hard data, but they are post-training descriptors and should not be confused with pre-training capacity.",
            "",
            "## Mechanism",
            "",
            "The strongest existing mechanism evidence remains post-training branch/projection/tree organization and statistic-preserving scrambles. Markov-expressivity metrics should be evaluated by whether they predict or explain that organization, not only average accuracy.",
            "",
            "The improved branch-margin capacity probe now reports exact log-sum-exp, tropical, and hard-root lower-tail objectives with branch-wise failures. The smoke run is intentionally small and diagnostic; it validates the measurement path rather than claiming optimized capacity.",
            "",
            "## Physical Thermodynamics",
            "",
            "Existing arbitrary exponential-rate models may be non-equilibrium, but they do not support thermodynamic force-budget claims. A reversible-edge parameterization and explicit F_max sweep are required before physical thermodynamic conclusions can be made.",
            "",
            f"The current thermodynamic report status is `{thermo['status']}`. The local reversible-support audit covers {thermo['n_local_hard_topology_groups_with_topology_json']} hard topology groups, with mean reversible-edge fraction {thermo['reversible_edge_fraction_mean']:.3f}. This is an eligibility audit, not an entropy-production or force-budget result.",
            "",
            "## Next Controlled Work",
            "",
            "The next experiments should be targeted, not broad: input-multiplicity controls with fixed G/count/d_rel; an expanded exact-degree normal-fan panel; a reversible-edge F_max sweep; serial-versus-parallel sharpness controls; and matched expressivity/trainability pairs.",
        ]
    )


def theory_doc() -> str:
    return "\n".join(
        [
            "# Markov-ICL Expressivity Theory",
            "",
            "## Scope",
            "",
            "This theory is for first-order CRNs / Markov jump processes with exponential input-dependent rates. It should not be transferred to autocatalytic or WTA models without a separate derivation.",
            "",
            "Keep the physical reaction graph G separate from the input mask Omega. G controls which rooted spanning trees exist. Omega controls which input coordinates can move which edge rates. Deleting input coupling is therefore not the same operation as deleting a physical edge.",
            "",
            "Novel-class ICL accuracy is the primary behavioral target. Training accuracy and ordinary validation accuracy can diagnose optimization, but they do not establish in-context generalization.",
            "",
            "## Exact First-Order Representation",
            "",
            "For edge e, k_e(z) = exp(b_e + K_e^T z). The matrix-tree theorem gives rooted tree numerators whose exponential projections are tree sums, Theta_T = sum_{e in T} K_e. Therefore the computational basis is the rooted tree-sum basis, not isolated edge projections.",
            "",
            "For root r, the steady-state coordinate is a normalized sum over T in T_r(G), with numerator exp(beta_T + Theta_T^T z). All branch-separation, sharpness, and coefficient-control questions should be asked in this tree-sum feature space.",
            "",
            "## Input Multiplicity",
            "",
            "For a CRN-ICL input mask Omega, define M_alpha = sum_e Omega[e, alpha]. Use the Markov expressivity paper's input multiplicity as a structural measure and hypothesis generator, not as a direct proof of the (2M+1)^D bound for learned continuous K.",
            "",
            "Important ICL multiplicities are context/query paired: for branch i and feature dimension d, compare M_{i,d}, M_{q,d}, their overlap, and their imbalance.",
            "",
            "Useful pre-training summaries are min M_alpha, mean M_alpha, variance, Gini, zero-coordinate fraction, monoRisk fraction for M_alpha <= 1, and sum_alpha log(2M_alpha + 1). For branch decisions, context/query overlap and imbalance matter more directly than the global average.",
            "",
            "## Monotonicity Risk",
            "",
            "Very low multiplicity can limit two-sided branch responses. Pre-training risk metrics are M_alpha <= 1, zero-coordinate fraction, context/query imbalance, and low overlap on comparison coordinates. Post-training effective multiplicity can be computed from learned K using participation-style ratios.",
            "",
            "For learned weights, M_eff_alpha = (sum_e |K_ealpha|)^2 / sum_e K_ealpha^2 is a participation-style diagnostic. It is not a replacement for Omega because optimization can ignore available coordinates.",
            "",
            "## Branch Sharpness",
            "",
            "For branch direction u_b, tree-drive range R_{r,b} = max_T Theta_T^T u_b - min_T Theta_T^T u_b measures whether rooted tree polytopes can create sharp branch margins. Coverage and sharpness are distinct.",
            "",
            "A topology may cover every branch while still producing weak lower-tail margins if the accessible tree-sum directions are poorly separated or badly conditioned.",
            "",
            "## Coefficient Controllability",
            "",
            "Tree-polynomial coefficients are constrained by overlapping spanning trees. Rank can overestimate useful capacity, so coefficient-map effective rank, condition number, entropy, extremal-tree accessibility, and branch-specific concentration should be tracked.",
            "",
            "This is the main reason d_rel should remain a baseline, not a final theory. Two topologies can have the same relative rank while differing in tree-count distribution, normal-fan coverage, extremal-tree accessibility, or coefficient conditioning.",
            "",
            "## Capacity Target",
            "",
            "The correct gamma-style target is lower-tail or worst-branch margin, max_theta min_b LCVaR_alpha[m_theta(z) | z in branch b], under constraints on K, b, decoder B, and the input mask. Average branch accuracy is not the primary objective.",
            "",
            "The three useful finite-sample probes are exact log-sum-exp tree features, tropical max-over-tree features, and hard-root structural compatibility. They should report branch-wise margins and failures, not only one scalar score.",
            "",
            "## Expressivity Versus Trainability",
            "",
            "Best seed is an expressivity-envelope proxy; mean seed is a trainability/reliability proxy; seed variance is an optimization-instability proxy. A valid expressivity metric can predict best-seed behavior without explaining mean-seed reliability.",
            "",
            "## Thermodynamics",
            "",
            "Arbitrary directed exponential rates are not enough for thermodynamic claims. A thermodynamic CRN-ICL variant must use reversible support and a parameterization such as W_ij = exp(E_j - B_ij + F_ij/2 + input drive), with B_ij = B_ji, F_ij = -F_ji, and |F_ij| <= F_max.",
            "",
            "Existing arbitrary directed models can be audited for reversible support, but entropy-production or force-budget claims require the controlled reversible-edge parameterization and an explicit F_max sweep.",
        ]
    )


def run(output_dir: Path) -> dict:
    datasets = load_datasets()
    analyses = {name: analyze_dataset(name, rows) for name, rows in datasets.items()}
    payload = {
        "generated_from": "existing committed CSV/JSON artifacts",
        "analyses": analyses,
        "degree_rewire": degree_rewire_payload(datasets),
        "thermodynamic": thermodynamic_payload(),
    }
    payload["expressivity_vs_trainability"] = expressivity_trainability_payload(analyses)

    write_text(Path("ICL/markov_icl_expressivity_theory.md"), theory_doc())
    write_json(output_dir / "existing_data_markov_expressivity_reanalysis.json", payload)
    write_text(output_dir / "existing_data_markov_expressivity_reanalysis.md", existing_data_report(payload))
    write_json(output_dir / "input_multiplicity_control_report.json", {"fixed_m20": analyses["fixed_m20"], "family_summary": family_summary(analyses["fixed_m20"]["groups"])})
    write_text(output_dir / "input_multiplicity_control_report.md", input_multiplicity_report(payload))
    write_json(output_dir / "thermodynamic_force_budget_report.json", payload["thermodynamic"])
    write_text(output_dir / "thermodynamic_force_budget_report.md", thermodynamic_report(payload))
    write_json(output_dir / "exact_degree_multiplicity_normal_fan_report.json", payload["degree_rewire"])
    write_text(output_dir / "exact_degree_multiplicity_normal_fan_report.md", degree_rewire_report(payload))
    write_json(output_dir / "expressivity_vs_trainability_report.json", payload["expressivity_vs_trainability"])
    write_text(output_dir / "expressivity_vs_trainability_report.md", expressivity_trainability_report(payload))
    write_text(output_dir / "markov_icl_final_synthesis.md", synthesis_report(payload))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=NEXT_PHASE)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = run(args.output_dir)
    print(
        json.dumps(
            {
                "datasets": {
                    key: {
                        "rows": value["n_rows"],
                        "groups": value["n_groups"],
                    }
                    for key, value in payload["analyses"].items()
                },
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
