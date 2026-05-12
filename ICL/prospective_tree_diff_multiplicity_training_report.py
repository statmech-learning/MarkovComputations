"""Analyze the prospective tree-difference multiplicity control training run."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "ICL" / "results"
NEXT_PHASE_DIR = RESULT_ROOT / "next_phase_stats"
LIBRARY_ROOT = RESULT_ROOT / "prospective_tree_diff_multiplicity_n6_m20_c200"
SELECTED_CSV = LIBRARY_ROOT / "selected.csv"
TRAINING_CSV = NEXT_PHASE_DIR / "prospective_tree_diff_multiplicity_training_results.csv"
MECHANISM_CSV = NEXT_PHASE_DIR / "prospective_tree_diff_multiplicity_mechanism_results.csv"
OUT_MD = NEXT_PHASE_DIR / "prospective_tree_diff_multiplicity_causal_report.md"
OUT_JSON = NEXT_PHASE_DIR / "prospective_tree_diff_multiplicity_causal_report.json"

OUTCOMES = ["mean_seed_novel_icl", "best_seed_novel_icl", "seed_std_novel_icl"]
MODEL_SPECS = {
    "controls_only": ["input_coord_load_gini"],
    "edge_level_multiplicity_plus_controls": [
        "input_coord_load_gini",
        "edge_overlap_norm_min",
        "edge_overlap_norm_mean",
    ],
    "tree_level_multiplicity_plus_controls": [
        "input_coord_load_gini",
        "tree_overlap_norm_min",
        "tree_overlap_norm_mean",
    ],
    "tree_difference_multiplicity_plus_controls": [
        "input_coord_load_gini",
        "diff_overlap_norm_min",
        "diff_overlap_norm_mean",
    ],
    "gamma_no_bias_plus_controls": [
        "input_coord_load_gini",
        "gamma_no_bias_exact_lcvar",
        "gamma_no_bias_tropical_lcvar",
    ],
    "gamma_no_bias_plus_tree_difference_multiplicity": [
        "input_coord_load_gini",
        "gamma_no_bias_exact_lcvar",
        "gamma_no_bias_tropical_lcvar",
        "diff_overlap_norm_min",
        "diff_overlap_norm_mean",
    ],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values: Iterable[float | None]) -> float | None:
    arr = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(arr)) if arr else None


def maximum(values: Iterable[float | None]) -> float | None:
    arr = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.max(arr)) if arr else None


def std(values: Iterable[float | None]) -> float | None:
    arr = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.std(arr)) if arr else None


def gini(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    min_value = float(np.min(arr))
    if min_value < 0.0:
        arr = arr - min_value
    total = float(np.sum(arr))
    if abs(total) <= 1e-12:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    weighted = float(np.sum((np.arange(n) + 1.0) * arr))
    return float((2.0 * weighted / (n * total)) - ((n + 1.0) / n))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(item) for item in row) + " |")
    return "\n".join(lines)


def selected_rows_by_topology() -> dict[str, dict[str, Any]]:
    rows = read_csv(SELECTED_CSV)
    out = {}
    for row in rows:
        key = row["topology_id"]
        parsed = dict(row)
        for field, value in list(parsed.items()):
            numeric = parse_float(value)
            if numeric is not None:
                parsed[field] = numeric
        out[key] = parsed
    return out


def topology_key_from_training_row(row: Mapping[str, Any]) -> str:
    return str(row.get("input_mask_name") or row.get("topology_name") or "")


def load_mechanism_by_label() -> dict[str, dict[str, Any]]:
    if not MECHANISM_CSV.exists():
        return {}
    return {row["label"]: row for row in read_csv(MECHANISM_CSV)}


def aggregate_training_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = selected_rows_by_topology()
    if not TRAINING_CSV.exists():
        return [], {
            "status": "missing_training_csv",
            "training_csv": str(TRAINING_CSV.relative_to(REPO_ROOT)),
        }
    training_rows = read_csv(TRAINING_CSV)
    mechanism_by_label = load_mechanism_by_label()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in training_rows:
        key = topology_key_from_training_row(row)
        if key:
            grouped[key].append(row)

    groups = []
    for key, members in sorted(grouped.items()):
        design = selected.get(key)
        if design is None:
            continue
        novel = [parse_float(row.get("test_novel_classes")) for row in members]
        branch_failure_values = []
        branch_margin_values = []
        branch_tree_mi = []
        input_ablation_loss = []
        physical_ablation_loss = []
        for row in members:
            mech = mechanism_by_label.get(str(row.get("label")))
            if mech is None:
                continue
            branch_acc_min = parse_float(mech.get("target_accuracy_branch_mean_min"))
            if branch_acc_min is not None:
                branch_failure_values.append(100.0 - branch_acc_min)
            branch_margin_values.append(parse_float(mech.get("target_logprob_margin_branch_mean_min")))
            branch_tree_mi.append(parse_float(mech.get("branch_active_tree_mi")))
            input_ablation_loss.append(parse_float(mech.get("input_ablation_max_loss")))
            physical_ablation_loss.append(parse_float(mech.get("physical_ablation_max_loss")))
        group = {
            **design,
            "group": key,
            "n_runs": len(members),
            "mean_seed_novel_icl": mean(novel),
            "best_seed_novel_icl": maximum(novel),
            "seed_std_novel_icl": std(novel),
            "branch_failures": mean(branch_failure_values),
            "trained_branch_margin": mean(branch_margin_values),
            "branch_active_tree_mi": mean(branch_tree_mi),
            "input_ablation_max_loss": mean(input_ablation_loss),
            "physical_ablation_max_loss": mean(physical_ablation_loss),
        }
        groups.append(group)

    expected = len(selected) * 5
    status = {
        "status": "complete" if len(training_rows) >= expected else "partial",
        "training_csv": str(TRAINING_CSV.relative_to(REPO_ROOT)),
        "training_rows": len(training_rows),
        "expected_training_rows": expected,
        "groups_with_results": len(groups),
        "expected_groups": len(selected),
        "mechanism_csv_present": MECHANISM_CSV.exists(),
        "mechanism_rows": len(mechanism_by_label),
    }
    return groups, status


def design_matrix(
    rows: Sequence[Mapping[str, Any]],
    predictors: Sequence[str],
    outcome: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        y = parse_float(row.get(outcome))
        if y is None:
            continue
        vals = []
        ok = True
        for predictor in predictors:
            value = parse_float(row.get(predictor))
            if value is None:
                ok = False
                break
            vals.append(value)
        if ok:
            xs.append(vals)
            ys.append(y)
    if not xs:
        return np.zeros((0, len(predictors))), np.zeros(0)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = 1.0) -> float:
    center = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1.0e-12] = 1.0
    x = (train_x - center) / scale
    xt = (test_x.reshape(1, -1) - center) / scale
    design = np.column_stack([np.ones(x.shape[0]), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ train_y
    return float(np.r_[1.0, xt.ravel()] @ coef)


def grouped_loo_r2(rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], outcome: str) -> dict[str, Any]:
    X, y = design_matrix(rows, predictors, outcome)
    if X.shape[0] < 4:
        return {"loo_r2": None, "n_groups": int(X.shape[0]), "reason": "too_few_groups"}
    if np.std(y) <= 1.0e-12:
        return {"loo_r2": None, "n_groups": int(X.shape[0]), "reason": "constant_outcome"}
    preds = []
    for idx in range(X.shape[0]):
        keep = np.ones(X.shape[0], dtype=bool)
        keep[idx] = False
        preds.append(ridge_predict(X[keep], y[keep], X[idx]))
    preds_arr = np.asarray(preds, dtype=float)
    sse = float(np.sum((y - preds_arr) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "loo_r2": None if sst <= 1.0e-12 else float(1.0 - sse / sst),
        "n_groups": int(X.shape[0]),
        "predictors": list(predictors),
    }


def model_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for outcome in OUTCOMES + ["branch_failures", "trained_branch_margin"]:
        items = []
        for model, predictors in MODEL_SPECS.items():
            result = grouped_loo_r2(rows, predictors, outcome)
            items.append({"model": model, **result})
        out[outcome] = items
    return out


def contrast(
    rows: Sequence[Mapping[str, Any]],
    load_stratum: str,
    outcome: str,
    n_boot: int = 5000,
    seed: int = 881,
) -> dict[str, Any]:
    high = [
        parse_float(row.get(outcome))
        for row in rows
        if row.get("load_stratum") == load_stratum and row.get("contrast_level") == "high"
    ]
    low = [
        parse_float(row.get(outcome))
        for row in rows
        if row.get("load_stratum") == load_stratum and row.get("contrast_level") == "low"
    ]
    high = [value for value in high if value is not None]
    low = [value for value in low if value is not None]
    if not high or not low:
        return {"outcome": outcome, "load_stratum": load_stratum, "status": "missing_data"}
    observed = float(np.mean(high) - np.mean(low))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        boot_high = rng.choice(high, size=len(high), replace=True)
        boot_low = rng.choice(low, size=len(low), replace=True)
        boot.append(float(np.mean(boot_high) - np.mean(boot_low)))
    return {
        "outcome": outcome,
        "load_stratum": load_stratum,
        "high_mean": float(np.mean(high)),
        "low_mean": float(np.mean(low)),
        "delta_high_minus_low": observed,
        "bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n_high": len(high),
        "n_low": len(low),
    }


def group_table(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    table = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("load_stratum")), str(row.get("contrast_level")))].append(row)
    for (load_stratum, level), members in sorted(grouped.items()):
        table.append(
            [
                load_stratum,
                level,
                len(members),
                mean(parse_float(row.get("diff_overlap_norm_min")) for row in members),
                mean(parse_float(row.get("mean_seed_novel_icl")) for row in members),
                mean(parse_float(row.get("best_seed_novel_icl")) for row in members),
                mean(parse_float(row.get("seed_std_novel_icl")) for row in members),
                mean(parse_float(row.get("branch_failures")) for row in members),
                mean(parse_float(row.get("trained_branch_margin")) for row in members),
            ]
        )
    return table


def write_reports(rows: Sequence[Mapping[str, Any]], status: Mapping[str, Any]) -> None:
    contrasts = [
        contrast(rows, "balanced_load", "mean_seed_novel_icl"),
        contrast(rows, "balanced_load", "best_seed_novel_icl"),
        contrast(rows, "imbalanced_coord_load", "mean_seed_novel_icl"),
        contrast(rows, "imbalanced_coord_load", "best_seed_novel_icl"),
    ]
    models = model_results(rows)
    primary = contrasts[0]
    supported = (
        primary.get("delta_high_minus_low") is not None
        and primary.get("bootstrap_ci95") is not None
        and primary["bootstrap_ci95"][0] > 0.0
    )
    interpretation = {
        "primary_balanced_mean_delta": primary.get("delta_high_minus_low"),
        "primary_balanced_mean_ci95": primary.get("bootstrap_ci95"),
        "prospective_causal_signal": "positive" if supported else "not_decisive",
        "mechanism_metrics_available": bool(status.get("mechanism_csv_present")),
    }
    payload = {
        "schema": "prospective_tree_diff_multiplicity_causal_report.v1",
        "status": status,
        "group_rows": list(rows),
        "contrasts": contrasts,
        "model_results": models,
        "interpretation": interpretation,
    }
    OUT_JSON.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")

    loo_rows = []
    for outcome, items in models.items():
        for item in items:
            loo_rows.append([outcome, item["model"], item.get("n_groups"), item.get("loo_r2"), item.get("reason")])
    contrast_rows = [
        [
            item.get("load_stratum"),
            item.get("outcome"),
            item.get("n_high"),
            item.get("n_low"),
            item.get("high_mean"),
            item.get("low_mean"),
            item.get("delta_high_minus_low"),
            item.get("bootstrap_ci95"),
        ]
        for item in contrasts
    ]
    lines = [
        "# Prospective Tree-Difference Multiplicity Causal Report",
        "",
        "## Status",
        "",
        f"- Training rows: `{status.get('training_rows')}` of expected `{status.get('expected_training_rows')}`",
        f"- Groups with results: `{status.get('groups_with_results')}` of expected `{status.get('expected_groups')}`",
        f"- Mechanism CSV present: `{status.get('mechanism_csv_present')}`",
        "",
        "## Group Summary",
        "",
        markdown_table(
            group_table(rows),
            [
                "load stratum",
                "contrast",
                "groups",
                "mean min diff overlap",
                "mean ICL",
                "best ICL",
                "seed std",
                "branch failures",
                "trained margin",
            ],
        ),
        "",
        "## High-Low Contrasts",
        "",
        markdown_table(
            contrast_rows,
            ["load stratum", "outcome", "n high", "n low", "high mean", "low mean", "delta", "CI95"],
        ),
        "",
        "## Grouped LOO Models",
        "",
        markdown_table(loo_rows, ["outcome", "model", "groups", "LOO R2", "reason"]),
        "",
        "## Interpretation",
        "",
        f"- Prospective causal signal: `{interpretation['prospective_causal_signal']}`",
        f"- Primary balanced mean-ICL delta: `{fmt(interpretation['primary_balanced_mean_delta'])}`",
        f"- Primary balanced mean-ICL CI95: `{interpretation['primary_balanced_mean_ci95']}`",
        "- This report is prospective because masks were selected before these training outcomes were collected.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows, status = aggregate_training_rows()
    write_reports(rows, status)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(status)


if __name__ == "__main__":
    main()
