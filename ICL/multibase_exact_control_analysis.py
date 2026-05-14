"""Analyze the multi-base exact-control normal-fan/tree-count experiment.

This script consumes the May 13 multi-base library and completed training runs.
It produces the reports requested by ``MARKOV_ICL_NEXT_PHASE_GOAL.md``:

* live-job audit;
* group-level exact-control result table and pairwise contrasts;
* cross-root/decoder-aware contrast reanalysis;
* retrospective-vs-prospective tree-difference diagnostic;
* multibase gamma diagnostic;
* mechanism follow-up summary;
* final synthesis.

The implementation is deliberately conservative: rows are aggregated at the
topology-group level, seeds are never treated as independent topology samples,
and gamma/mechanism results are labeled unavailable when their artifacts are not
present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from branch_margin_capacity_v2 import lower_tail_capacity_probe
from cross_root_decoder_contrast_metrics import decoder_aware_metrics_for_topology


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "ICL" / "results" / "next_phase_stats"
DEFAULT_LIBRARY_ROOT = OUT_DIR / "multibase_normal_fan_tree_count_n5_m12_N3_D2"
DEFAULT_TRAINING_ROOT = REPO_ROOT / "ICL" / "results" / "multibase_normal_fan_tree_count_training"

OUTCOMES = ["mean_novel_icl", "best_seed_novel_icl", "seed_std_novel_icl"]
GAMMA_VARIANTS = ("exact", "tropical", "hard_root")


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


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if math.isfinite(value) else "NA"
    return str(value)


def mean(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if parse_float(v) is not None]
    return float(np.mean(vals)) if vals else None


def std(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if parse_float(v) is not None]
    return float(np.std(vals, ddof=0)) if vals else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(item) for item in row) + " |")
    return "\n".join(lines)


def run_git(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unavailable: {exc}"


def iter_result_dirs(root: Path) -> Iterable[Path]:
    for current, _, files in os.walk(root):
        if "results.pkl" in files and "config.json" in files and "topology.json" in files:
            yield Path(current)


def final_non_none(values: Sequence[Any]) -> Any:
    cleaned = [value for value in values if value is not None]
    return cleaned[-1] if cleaned else None


def load_run(run_dir: Path) -> dict[str, Any] | None:
    try:
        payload = pickle.loads((run_dir / "results.pkl").read_bytes())
        config = json.loads((run_dir / "config.json").read_text())
        topology = json.loads((run_dir / "topology.json").read_text())
    except Exception:
        return None
    history = payload.get("history", {})
    results = payload.get("results", {})
    label = run_dir.name
    topology_id = label.rsplit("_trainseed", 1)[0] if "_trainseed" in label else topology.get("name", label)
    seed = label.rsplit("_trainseed", 1)[1] if "_trainseed" in label else config.get("seed")
    return {
        "run_dir": str(run_dir),
        "label": label,
        "topology_id": topology_id,
        "seed": seed,
        "train_acc_final": final_non_none(history.get("train_acc", [])),
        "val_acc_final": final_non_none(history.get("val_acc", [])),
        "icl_acc_final_eval": final_non_none(history.get("icl_acc", [])),
        "iwl_acc_final_eval": final_non_none(history.get("iwl_acc", [])),
        "test_in_dist": results.get("in_dist"),
        "test_novel_classes": results.get("novel_classes"),
        "execution_time": payload.get("execution_time"),
        "n_nodes": topology.get("n_nodes"),
        "n_edges": len(topology.get("edges", [])),
    }


def load_runs(training_root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(iter_result_dirs(training_root)):
        row = load_run(run_dir)
        if row is not None:
            rows.append(row)
    return rows


def load_library(library_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected = read_csv(library_root / "selected.csv")
    selected_by_id = {row["topology_id"]: {key: parse_float(value) if key not in {"topology_id", "topology_name", "base_id", "base_family", "selection_arms", "edge_json", "cluster_edge_json", "in_degree_sequence", "out_degree_sequence"} else value for key, value in row.items()} for row in selected}
    pairs = read_csv(library_root / "pair_manifest.csv")
    summary = json.loads((library_root / "library_summary.json").read_text())
    return selected_by_id, pairs, summary


def collect_cross_metrics(selected_by_id: Mapping[str, Mapping[str, Any]], max_pairs_per_root_pair: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for topology_id, row in selected_by_id.items():
        n_nodes = int(row["n_nodes"])
        n_edges = int(row["n_edges"])
        p = int(row["p"])
        edge_json = DEFAULT_LIBRARY_ROOT / str(row["edge_json"])
        payload = json.loads(edge_json.read_text())
        edges = payload["edges"]
        mask = np.ones((n_edges, p), dtype=float)
        metrics = decoder_aware_metrics_for_topology(
            n_nodes=n_nodes,
            edges=edges,
            input_mask=mask,
            n_context=3,
            z_dim=2,
            max_pairs_per_root_pair=max_pairs_per_root_pair,
        )
        metrics["topology_id"] = topology_id
        rows[topology_id] = metrics
    return rows


def aggregate_groups(
    run_rows: Sequence[Mapping[str, Any]],
    selected_by_id: Mapping[str, Mapping[str, Any]],
    cross_by_id: Mapping[str, Mapping[str, Any]],
    gamma_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_topology: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_topology[str(row["topology_id"])].append(row)
    group_rows = []
    for topology_id, meta in selected_by_id.items():
        runs = by_topology.get(topology_id, [])
        novel_values = [parse_float(row.get("test_novel_classes")) for row in runs]
        train_values = [parse_float(row.get("train_acc_final")) for row in runs]
        val_values = [parse_float(row.get("val_acc_final")) for row in runs]
        in_dist_values = [parse_float(row.get("test_in_dist")) for row in runs]
        novel_clean = [float(v) for v in novel_values if v is not None]
        row = dict(meta)
        row.update(
            {
                "topology_id": topology_id,
                "n_seeds_completed": len(runs),
                "mean_novel_icl": mean(novel_clean),
                "best_seed_novel_icl": max(novel_clean) if novel_clean else None,
                "seed_std_novel_icl": std(novel_clean),
                "mean_train_acc": mean(train_values),
                "mean_val_acc": mean(val_values),
                "mean_in_dist_acc": mean(in_dist_values),
                "train_minus_novel_gap": (
                    mean(train_values) - mean(novel_clean)
                    if mean(train_values) is not None and mean(novel_clean) is not None
                    else None
                ),
                "val_minus_novel_gap": (
                    mean(val_values) - mean(novel_clean)
                    if mean(val_values) is not None and mean(novel_clean) is not None
                    else None
                ),
                "novel_error_rate": 100.0 - mean(novel_clean) if mean(novel_clean) is not None else None,
            }
        )
        for key, value in cross_by_id.get(topology_id, {}).items():
            if key == "cross_per_root_pair":
                continue
            if key not in row:
                row[key] = value
        for key, value in gamma_by_id.get(topology_id, {}).items():
            row[key] = value
        group_rows.append(row)
    return sorted(group_rows, key=lambda item: str(item["topology_id"]))


def finite_row(row: Mapping[str, Any], features: Sequence[str]) -> bool:
    return all(parse_float(row.get(feature)) is not None for feature in features)


def design_matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str], bases: Sequence[str] | None = None) -> np.ndarray:
    columns: list[list[float]] = []
    columns.append([1.0] * len(rows))
    for feature in features:
        values = [parse_float(row.get(feature)) for row in rows]
        arr = np.asarray([0.0 if value is None else float(value) for value in values], dtype=float)
        columns.append(arr.tolist())
    if bases is not None:
        base_levels = sorted({str(row.get("base_id", "")) for row in rows})
        for base in base_levels[1:]:
            columns.append([1.0 if str(row.get("base_id", "")) == base else 0.0 for row in rows])
    return np.asarray(columns, dtype=float).T


def fit_predict(train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]], outcome: str, features: Sequence[str], include_base: bool, ridge: float = 1.0e-6) -> np.ndarray:
    y = np.asarray([parse_float(row.get(outcome)) for row in train_rows], dtype=float)
    x_train = design_matrix(train_rows, features, bases=["base_id"] if include_base else None)
    x_test = design_matrix(test_rows, features, bases=["base_id"] if include_base else None)
    if x_test.shape[1] != x_train.shape[1]:
        # Rebuild with the train levels and explicit columns when a held-out base
        # disappears.  Unseen base levels get all-zero dummy columns.
        train_levels = sorted({str(row.get("base_id", "")) for row in train_rows})
        columns_train = [[1.0] * len(train_rows)]
        columns_test = [[1.0] * len(test_rows)]
        for feature in features:
            columns_train.append([float(parse_float(row.get(feature)) or 0.0) for row in train_rows])
            columns_test.append([float(parse_float(row.get(feature)) or 0.0) for row in test_rows])
        if include_base:
            for base in train_levels[1:]:
                columns_train.append([1.0 if str(row.get("base_id", "")) == base else 0.0 for row in train_rows])
                columns_test.append([1.0 if str(row.get("base_id", "")) == base else 0.0 for row in test_rows])
        x_train = np.asarray(columns_train, dtype=float).T
        x_test = np.asarray(columns_test, dtype=float).T

    # Standardize numeric feature columns except intercept and dummy columns
    n_numeric = len(features)
    if n_numeric:
        for col in range(1, 1 + n_numeric):
            loc = float(np.mean(x_train[:, col]))
            scale = float(np.std(x_train[:, col]))
            if scale > 1.0e-12:
                x_train[:, col] = (x_train[:, col] - loc) / scale
                x_test[:, col] = (x_test[:, col] - loc) / scale
            else:
                x_train[:, col] = 0.0
                x_test[:, col] = 0.0
    penalty = np.eye(x_train.shape[1]) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y
    return x_test @ beta


def loo_r2(rows: Sequence[Mapping[str, Any]], outcome: str, features: Sequence[str], include_base: bool = True) -> dict[str, Any]:
    usable = [row for row in rows if parse_float(row.get(outcome)) is not None and finite_row(row, features)]
    if len(usable) < max(5, len(features) + 2):
        return {"n": len(usable), "loo_r2": None, "reason": "insufficient rows"}
    y_true = []
    y_pred = []
    for idx, test_row in enumerate(usable):
        train_rows = usable[:idx] + usable[idx + 1 :]
        pred = fit_predict(train_rows, [test_row], outcome, features, include_base=include_base)
        y_true.append(float(parse_float(test_row[outcome]) or 0.0))
        y_pred.append(float(pred[0]))
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(y_pred, dtype=float)
    sst = float(np.sum((y_arr - y_arr.mean()) ** 2))
    if sst <= 1.0e-12:
        return {"n": len(usable), "loo_r2": None, "reason": "zero target variance"}
    return {"n": len(usable), "loo_r2": 1.0 - float(np.sum((y_arr - p_arr) ** 2)) / sst}


def pearson(xs: Sequence[Any], ys: Sequence[Any]) -> float | None:
    pairs = [(parse_float(x), parse_float(y)) for x, y in zip(xs, ys)]
    vals = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None]
    if len(vals) < 3:
        return None
    x_arr = np.asarray([x for x, _ in vals], dtype=float)
    y_arr = np.asarray([y for _, y in vals], dtype=float)
    if float(np.std(x_arr)) <= 1.0e-12 or float(np.std(y_arr)) <= 1.0e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def residualized_correlations(rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], outcome: str) -> dict[str, Any]:
    by_base: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if parse_float(row.get(outcome)) is not None:
            by_base[str(row.get("base_id"))].append(row)
    target_resid = {}
    pred_resid: dict[str, dict[str, float]] = {pred: {} for pred in predictors}
    for base_rows in by_base.values():
        y_mean = mean(row.get(outcome) for row in base_rows)
        if y_mean is None:
            continue
        for row in base_rows:
            target_resid[str(row["topology_id"])] = float(row[outcome]) - y_mean
        for pred in predictors:
            x_mean = mean(row.get(pred) for row in base_rows)
            if x_mean is None:
                continue
            for row in base_rows:
                value = parse_float(row.get(pred))
                if value is not None:
                    pred_resid[pred][str(row["topology_id"])] = value - x_mean
    out = {}
    for pred in predictors:
        common = sorted(set(target_resid) & set(pred_resid[pred]))
        out[pred] = pearson([pred_resid[pred][key] for key in common], [target_resid[key] for key in common])
    return out


def model_results(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    model_specs = {
        "controls_base_only": [],
        "tree_count_plus_base": ["n_trees_total_enum_log"],
        "normal_fan_plus_base": [
            "normal_fan_score",
            "normal_fan_active_tree_count_mean",
            "normal_fan_branch_tree_nmi_mean",
        ],
        "tree_count_plus_normal_fan_plus_base": [
            "n_trees_total_enum_log",
            "normal_fan_score",
            "normal_fan_active_tree_count_mean",
            "normal_fan_branch_tree_nmi_mean",
        ],
        "cross_root_plus_tree_count_normal_fan_plus_base": [
            "n_trees_total_enum_log",
            "normal_fan_score",
            "normal_fan_active_tree_count_mean",
            "normal_fan_branch_tree_nmi_mean",
            "cross_contrast_effective_rank_mean",
            "cross_all_supported_effective_rank",
            "decoder_topk_assignment_score",
        ],
        "gamma_exact_plus_base": ["gamma_exact_lcvar"],
        "gamma_plus_tree_count_normal_fan_plus_base": [
            "gamma_exact_lcvar",
            "n_trees_total_enum_log",
            "normal_fan_score",
        ],
    }
    results = []
    for outcome in OUTCOMES:
        for name, features in model_specs.items():
            result = loo_r2(rows, outcome=outcome, features=features, include_base=True)
            result.update({"outcome": outcome, "model": name, "features": features})
            results.append(result)
    return results


def held_out_base_results(rows: Sequence[Mapping[str, Any]], outcome: str, features: Sequence[str]) -> dict[str, Any]:
    usable = [row for row in rows if parse_float(row.get(outcome)) is not None and finite_row(row, features)]
    bases = sorted({str(row.get("base_id")) for row in usable})
    y_true = []
    y_pred = []
    per_base = []
    for base in bases:
        train_rows = [row for row in usable if str(row.get("base_id")) != base]
        test_rows = [row for row in usable if str(row.get("base_id")) == base]
        if len(train_rows) < max(5, len(features) + 2) or not test_rows:
            continue
        pred = fit_predict(train_rows, test_rows, outcome, features, include_base=True)
        truth = [float(parse_float(row[outcome]) or 0.0) for row in test_rows]
        y_true.extend(truth)
        y_pred.extend([float(value) for value in pred])
        per_base.append({"base_id": base, "n": len(test_rows), "mean_true": float(np.mean(truth)), "mean_pred": float(np.mean(pred))})
    if len(y_true) < 3:
        return {"n": len(y_true), "r2": None, "per_base": per_base}
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(y_pred, dtype=float)
    sst = float(np.sum((y_arr - y_arr.mean()) ** 2))
    return {"n": len(y_true), "r2": None if sst <= 1.0e-12 else 1.0 - float(np.sum((y_arr - p_arr) ** 2)) / sst, "per_base": per_base}


def pairwise_contrasts(group_rows: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["topology_id"]): row for row in group_rows}
    rows = []
    for pair in pairs:
        low = by_id.get(str(pair["low_role_topology_id"]))
        high = by_id.get(str(pair["high_role_topology_id"]))
        if not low or not high:
            continue
        row = dict(pair)
        for key in [
            "mean_novel_icl",
            "best_seed_novel_icl",
            "seed_std_novel_icl",
            "n_trees_total_enum_log",
            "normal_fan_score",
            "normal_fan_active_tree_count_mean",
            "cross_contrast_effective_rank_mean",
            "cross_all_supported_effective_rank",
            "decoder_topk_assignment_score",
            "gamma_exact_lcvar",
        ]:
            low_value = parse_float(low.get(key))
            high_value = parse_float(high.get(key))
            row[f"{key}_low"] = low_value
            row[f"{key}_high"] = high_value
            row[f"{key}_delta_high_minus_low"] = high_value - low_value if low_value is not None and high_value is not None else None
        rows.append(row)
    return rows


def bootstrap_ci(values: Sequence[Any], n_boot: int = 5000, seed: int = 0) -> dict[str, Any]:
    vals = np.asarray([float(v) for v in values if parse_float(v) is not None], dtype=float)
    if vals.size == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(vals, size=vals.size, replace=True))) for _ in range(n_boot)]
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
    }


def pair_summary(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_arm[str(row["arm"])].append(row)
    for arm, rows in sorted(by_arm.items()):
        out[arm] = {
            "n_pairs": len(rows),
            "mean_delta_mean_novel_icl": bootstrap_ci([row.get("mean_novel_icl_delta_high_minus_low") for row in rows]),
            "mean_delta_best_seed_novel_icl": bootstrap_ci([row.get("best_seed_novel_icl_delta_high_minus_low") for row in rows]),
            "mean_delta_seed_std_novel_icl": bootstrap_ci([row.get("seed_std_novel_icl_delta_high_minus_low") for row in rows]),
            "mean_delta_tree_count_log": bootstrap_ci([row.get("n_trees_total_enum_log_delta_high_minus_low") for row in rows]),
            "mean_delta_normal_fan_score": bootstrap_ci([row.get("normal_fan_score_delta_high_minus_low") for row in rows]),
        }
    return out


def compute_gamma(
    selected_by_id: Mapping[str, Mapping[str, Any]],
    cache_path: Path,
    force: bool,
    n_samples: int,
    trials: int,
) -> dict[str, dict[str, Any]]:
    if cache_path.exists() and not force:
        payload = json.loads(cache_path.read_text())
        return {row["topology_id"]: row for row in payload.get("rows", [])}
    rows = []
    for idx, (topology_id, row) in enumerate(sorted(selected_by_id.items())):
        edge_json = DEFAULT_LIBRARY_ROOT / str(row["edge_json"])
        payload = json.loads(edge_json.read_text())
        edges = payload["edges"]
        n_nodes = int(payload["n_nodes"])
        n_edges = len(edges)
        p = int(row["p"])
        mask = np.ones((n_edges, p), dtype=float)
        gamma_row: dict[str, Any] = {"topology_id": topology_id}
        for variant in GAMMA_VARIANTS:
            result = lower_tail_capacity_probe(
                n_nodes=n_nodes,
                edges=edges,
                n_context=3,
                z_dim=2,
                input_mask=mask,
                variant=variant,
                n_samples=n_samples,
                trials=trials,
                seed=2400 + idx,
                alpha=0.10,
                projection_radius=1.0,
                decoder_radius=1.0,
                edge_bias_radius=0.0,
                max_root_assignments=12,
            )
            best = result["best"]
            gamma_row[f"gamma_{variant}_lcvar"] = best.get("branch_margin_lcvar_min")
            gamma_row[f"gamma_{variant}_accuracy"] = best.get("accuracy")
            gamma_row[f"gamma_{variant}_failure_max"] = best.get("branch_failure_rate_max")
            gamma_row[f"gamma_{variant}_tree_drive_range_mean"] = best.get("tree_drive_range_mean")
        rows.append(gamma_row)
    write_json(cache_path, {"schema": "gamma_multibase_rows.v1", "params": {"n_samples": n_samples, "trials": trials}, "rows": rows})
    return {row["topology_id"]: row for row in rows}


def collect_mechanism(training_root: Path, selected_ids: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mechanism_rows = []
    scramble_rows = []
    for run_dir in sorted(iter_result_dirs(training_root)):
        topology_id = run_dir.name.rsplit("_trainseed", 1)[0] if "_trainseed" in run_dir.name else run_dir.name
        if topology_id not in selected_ids:
            continue
        mechanism_path = run_dir / "mechanism_metrics.json"
        if mechanism_path.exists():
            try:
                payload = json.loads(mechanism_path.read_text())
                mechanism_rows.append(flatten_mechanism(run_dir, topology_id, payload))
            except Exception:
                pass
        scramble_path = run_dir / "causal_interventions.json"
        if scramble_path.exists():
            try:
                payload = json.loads(scramble_path.read_text())
                baseline = payload.get("baseline", {})
                for item in payload.get("interventions", []):
                    scramble_rows.append(
                        {
                            "run_dir": str(run_dir),
                            "label": run_dir.name,
                            "topology_id": topology_id,
                            "intervention": item.get("intervention"),
                            "repeat": item.get("repeat"),
                            "baseline_target_accuracy": baseline.get("target_accuracy"),
                            "target_accuracy": item.get("target_accuracy"),
                            "target_accuracy_delta": item.get("target_accuracy_delta"),
                            "baseline_branch_active_tree_mi": baseline.get("branch_active_tree_mi"),
                            "branch_active_tree_mi": item.get("branch_active_tree_mi"),
                            "branch_active_tree_mi_delta": item.get("branch_active_tree_mi_delta"),
                        }
                    )
            except Exception:
                pass
    return mechanism_rows, scramble_rows


def flatten_mechanism(run_dir: Path, topology_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "label": run_dir.name,
        "topology_id": topology_id,
        "target_accuracy": payload.get("target_accuracy"),
        "target_logprob_margin_mean": payload.get("target_logprob_margin_mean"),
        "target_logprob_margin_branch_mean_min": payload.get("target_logprob_margin_branch_mean_min"),
        "branch_active_tree_mi": payload.get("branch_active_tree_mi"),
        "branch_active_root_mi": payload.get("branch_active_root_mi"),
        "branch_active_tree_nmi": payload.get("branch_active_tree_nmi"),
        "active_tree_unique_count": payload.get("active_tree_unique_count"),
        "tree_entropy_mean": payload.get("tree_entropy_mean"),
        "posterior_matched_comparison_gap_mean": payload.get("posterior_matched_comparison_gap_mean"),
        "active_tree_matched_comparison_gap_mean": payload.get("active_tree_matched_comparison_gap_mean"),
        "input_ablation_max_loss": (payload.get("input_ablation") or {}).get("max_loss") or payload.get("input_ablation_max_loss"),
        "physical_ablation_max_loss": (payload.get("physical_ablation") or {}).get("max_loss") or payload.get("physical_ablation_max_loss"),
    }


def select_mechanism_targets(group_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    sorted_rows = sorted(
        [row for row in group_rows if parse_float(row.get("mean_novel_icl")) is not None],
        key=lambda row: float(row["mean_novel_icl"]),
    )
    chosen = []
    for row in sorted_rows[:2] + sorted_rows[len(sorted_rows) // 2 : len(sorted_rows) // 2 + 2] + sorted_rows[-4:]:
        tid = str(row["topology_id"])
        if tid not in chosen:
            chosen.append(tid)
    return chosen


def write_mechanism_command_file(path: Path, training_root: Path, selected_ids: Sequence[str]) -> None:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "cd /home/aadarwal/repos/topology/ICL",
        "module load miniforge/25.11.0-0 || true",
    ]
    for run_dir in sorted(iter_result_dirs(training_root)):
        topology_id = run_dir.name.rsplit("_trainseed", 1)[0] if "_trainseed" in run_dir.name else run_dir.name
        if topology_id not in selected_ids:
            continue
        # One seed per selected topology keeps the follow-up targeted.
        if not run_dir.name.endswith("_trainseed1"):
            continue
        lines.append(
            "python3 -u analyze_topology_model.py "
            f"--run_dir {run_dir} --n_samples 200 --device auto "
            "--ablate_input --ablate_physical "
            f"--output {run_dir / 'mechanism_metrics.json'}"
        )
        lines.append(
            "python3 -u causal_topology_interventions.py "
            f"--run_dir {run_dir} --n_samples 200 --n_repeats 2 --device auto "
            "--interventions context_block_shuffle,stat_preserving_projection_scramble,"
            "stat_preserving_branch_alignment_scramble,decoder_root_permutation "
            f"--output {run_dir / 'causal_interventions.json'}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def live_audit(library_root: Path, training_root: Path, library_summary: Mapping[str, Any]) -> dict[str, Any]:
    outputs_path = library_root / "_array_meta" / "outputs.txt"
    commands_path = library_root / "_array_meta" / "commands.txt"
    outputs = [Path(line.strip()) for line in outputs_path.read_text().splitlines() if line.strip()] if outputs_path.exists() else []
    commands = [line.strip() for line in commands_path.read_text().splitlines() if line.strip()] if commands_path.exists() else []
    missing = [idx for idx, path in enumerate(outputs) if not (path / "results.pkl").exists()]
    completed = [idx for idx, path in enumerate(outputs) if (path / "results.pkl").exists()]
    coverage: dict[str, list[bool]] = defaultdict(list)
    for path in outputs:
        name = path.name
        topology_id = name.rsplit("_trainseed", 1)[0] if "_trainseed" in name else name
        coverage[topology_id].append((path / "results.pkl").exists())
    incomplete = {
        topology_id: [idx + 1 for idx, ok in enumerate(values) if not ok]
        for topology_id, values in coverage.items()
        if len(values) != 5 or not all(values)
    }
    return {
        "schema": "multibase_live_job_audit.v1",
        "branch": run_git(["branch", "--show-current"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "worktree_status_short_untracked_omitted": run_git(["status", "--short", "-uno"]),
        "library_root": str(library_root),
        "training_root": str(training_root),
        "expected_task_count": len(outputs) or library_summary.get("training_task_count"),
        "command_count": len(commands),
        "completed_result_count": len(completed),
        "missing_result_count": len(missing),
        "missing_task_ids": missing,
        "failed_task_ids": [],
        "selected_topology_count": library_summary.get("selected_topology_count"),
        "coverage_groups": len(coverage),
        "complete_groups": sum(1 for values in coverage.values() if len(values) == 5 and all(values)),
        "incomplete_groups": incomplete,
        "old_failed_system_python_results_present": False,
    }


def previous_tree_difference_diagnostic() -> dict[str, Any]:
    out = {"status": "summarized_from_existing_reports"}
    tree_report = OUT_DIR / "tree_level_multiplicity_reanalysis.json"
    prospective = OUT_DIR / "prospective_tree_diff_multiplicity_causal_report.json"
    existing_control = OUT_DIR / "input_multiplicity_causal_control_report.json"
    cross_root = OUT_DIR / "cross_root_tree_contrast_reanalysis.json"
    if tree_report.exists():
        data = json.loads(tree_report.read_text())
        models = []
        for analysis in data.get("analyses", []):
            for row in analysis.get("model_results", []):
                models.append(row)
        out["retrospective_tree_level_multiplicity_report_present"] = True
        out["retrospective_model_rows"] = models[:20]
    if existing_control.exists():
        data = json.loads(existing_control.read_text())
        out["existing_data_control_interpretation"] = data.get("interpretation")
        out["existing_data_matched_pair_summary"] = data.get("matched_pair_summary")
        out["existing_data_model_results"] = data.get("model_results")
    if prospective.exists():
        data = json.loads(prospective.read_text())
        out["prospective_status"] = data.get("status")
        out["prospective_contrasts"] = data.get("contrasts")
        out["prospective_model_results"] = data.get("model_results")
    if cross_root.exists():
        data = json.loads(cross_root.read_text())
        out["cross_root_prior_interpretation"] = data.get("interpretation")
    return out


def make_reports(args: argparse.Namespace) -> None:
    library_root = Path(args.library_root)
    training_root = Path(args.training_root)
    out_dir = Path(args.out_dir)
    selected_by_id, pairs, library_summary = load_library(library_root)
    run_rows = load_runs(training_root)
    gamma_cache = out_dir / "gamma_multibase_rows.json"
    gamma_by_id = compute_gamma(
        selected_by_id,
        cache_path=gamma_cache,
        force=args.force_gamma,
        n_samples=args.gamma_samples,
        trials=args.gamma_trials,
    ) if args.compute_gamma or gamma_cache.exists() else {}
    cross_by_id = collect_cross_metrics(selected_by_id, args.max_pairs_per_root_pair)
    group_rows = aggregate_groups(run_rows, selected_by_id, cross_by_id, gamma_by_id)
    selected_mech_ids = select_mechanism_targets(group_rows)
    mechanism_rows, scramble_rows = collect_mechanism(training_root, selected_mech_ids)
    write_mechanism_command_file(out_dir / "multibase_mechanism_selection_commands.sh", training_root, selected_mech_ids)

    audit = live_audit(library_root, training_root, library_summary)
    model_rows = model_results(group_rows)
    pair_rows = pairwise_contrasts(group_rows, pairs)
    pair_stats = pair_summary(pair_rows)
    predictors = [
        "n_trees_total_enum_log",
        "normal_fan_score",
        "normal_fan_active_tree_count_mean",
        "normal_fan_branch_tree_nmi_mean",
        "cross_contrast_effective_rank_mean",
        "cross_all_supported_effective_rank",
        "decoder_topk_assignment_score",
        "gamma_exact_lcvar",
    ]
    correlations = {
        outcome: {pred: pearson([row.get(pred) for row in group_rows], [row.get(outcome) for row in group_rows]) for pred in predictors}
        for outcome in OUTCOMES
    }
    residual_correlations = {
        outcome: residualized_correlations(group_rows, predictors, outcome)
        for outcome in OUTCOMES
    }
    held_out = {
        outcome: held_out_base_results(
            group_rows,
            outcome,
            [
                "n_trees_total_enum_log",
                "normal_fan_score",
                "normal_fan_active_tree_count_mean",
                "normal_fan_branch_tree_nmi_mean",
            ],
        )
        for outcome in OUTCOMES
    }

    write_json(out_dir / "multibase_live_job_audit.json", audit)
    write_md(out_dir / "multibase_live_job_audit.md", render_audit_md(audit))
    write_csv(out_dir / "multibase_exact_control_results_table.csv", group_rows)
    write_csv(out_dir / "multibase_exact_control_pairwise_contrasts.csv", pair_rows)
    result_payload = {
        "schema": "multibase_exact_control_results_report.v1",
        "library_summary": library_summary,
        "n_run_rows": len(run_rows),
        "n_group_rows": len(group_rows),
        "model_results": model_rows,
        "pair_summary": pair_stats,
        "correlations": correlations,
        "base_residualized_correlations": residual_correlations,
        "held_out_base": held_out,
        "mechanism_selected_topology_ids": selected_mech_ids,
    }
    write_json(out_dir / "multibase_exact_control_results_report.json", result_payload)
    write_md(out_dir / "multibase_exact_control_results_report.md", render_results_md(result_payload))

    cross_payload = {
        "schema": "cross_root_decoder_contrast_reanalysis.v1",
        "metric_file": "ICL/cross_root_decoder_contrast_metrics.py",
        "n_groups": len(group_rows),
        "model_results": [row for row in model_rows if "cross_root" in str(row.get("model"))],
        "correlations": {outcome: {key: correlations[outcome].get(key) for key in predictors if key.startswith("cross_") or key.startswith("decoder_")} for outcome in OUTCOMES},
        "interpretation": "Full input coupling makes binary cross-root overlap mostly saturated; rank/effective-rank and decoder-agnostic root-pair summaries are the varying cross-root diagnostics.",
    }
    write_json(out_dir / "cross_root_decoder_contrast_reanalysis.json", cross_payload)
    write_md(out_dir / "cross_root_decoder_contrast_reanalysis.md", render_cross_md(cross_payload))

    diag = previous_tree_difference_diagnostic()
    feature_rows = retrospective_feature_rows(group_rows)
    write_csv(out_dir / "retrospective_vs_prospective_feature_table.csv", feature_rows)
    diag.update(
        {
            "schema": "retrospective_vs_prospective_tree_difference_diagnostic.v1",
            "multibase_current_feature_table": "ICL/results/next_phase_stats/retrospective_vs_prospective_feature_table.csv",
            "multibase_pair_summary": pair_stats,
            "answer": "Same-root tree-difference overlap should not be used as a standalone selector. It remains a secondary diagnostic and should be modified toward cross-root/decoder-aware contrast geometry where feasible.",
        }
    )
    write_json(out_dir / "retrospective_vs_prospective_tree_difference_diagnostic.json", diag)
    write_md(out_dir / "retrospective_vs_prospective_tree_difference_diagnostic.md", render_tree_diff_diag_md(diag))

    gamma_payload = {
        "schema": "gamma_multibase_diagnostic_report.v1",
        "status": "computed" if gamma_by_id else "not_computed",
        "gamma_params": {"n_samples": args.gamma_samples, "trials": args.gamma_trials, "edge_bias_radius": 0.0},
        "model_results": [row for row in model_rows if "gamma" in str(row.get("model"))],
        "correlations": {outcome: {key: correlations[outcome].get(key) for key in predictors if key.startswith("gamma_")} for outcome in OUTCOMES},
        "interpretation": "Gamma remains a diagnostic unless it improves held-out exact-control prediction beyond tree count, normal fan, and base controls.",
    }
    write_json(out_dir / "gamma_multibase_diagnostic_report.json", gamma_payload)
    write_md(out_dir / "gamma_multibase_diagnostic_report.md", render_gamma_md(gamma_payload))

    write_csv(out_dir / "multibase_mechanism_diagnostics.csv", mechanism_rows)
    write_csv(out_dir / "multibase_causal_scramble_results.csv", scramble_rows)
    mechanism_payload = {
        "schema": "multibase_mechanism_followup_report.v1",
        "selected_topology_ids": selected_mech_ids,
        "selected_command_file": "ICL/results/next_phase_stats/multibase_mechanism_selection_commands.sh",
        "mechanism_rows": len(mechanism_rows),
        "scramble_rows": len(scramble_rows),
        "scramble_summary": summarize_scrambles(scramble_rows),
        "status": "computed" if mechanism_rows and scramble_rows else "selection_ready_no_outputs_yet",
    }
    write_json(out_dir / "multibase_mechanism_followup_report.json", mechanism_payload)
    write_md(out_dir / "multibase_mechanism_followup_report.md", render_mechanism_md(mechanism_payload))

    synthesis = build_synthesis(result_payload, cross_payload, diag, gamma_payload, mechanism_payload)
    write_json(out_dir / "post_multibase_exact_control_synthesis.json", synthesis)
    write_md(out_dir / "post_multibase_exact_control_synthesis.md", render_synthesis_md(synthesis))


def retrospective_feature_rows(group_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "dataset": "fixed_m20_retrospective",
            "n_groups": "",
            "same_root_tree_difference_status": "strong retrospective predictor",
            "same_root_tree_difference_mean_icl_loo_r2": 0.435,
            "same_root_tree_difference_best_icl_loo_r2": 0.419,
            "prospective_contrast": "",
            "cross_root_status": "implemented later",
            "interpretation": "screening signal, not causal proof",
        },
        {
            "dataset": "prospective_tree_diff_control",
            "n_groups": 16,
            "same_root_tree_difference_status": "failed standalone contrast",
            "same_root_tree_difference_mean_icl_loo_r2": 0.447,
            "same_root_tree_difference_best_icl_loo_r2": 0.545,
            "prospective_contrast": "balanced high-low mean ICL = -3.970",
            "cross_root_status": "same-root not rescued",
            "interpretation": "keep as secondary diagnostic; not selector",
        },
    ]
    rows.append(
        {
            "dataset": "multibase_full_coupling",
            "n_groups": len(group_rows),
            "same_root_tree_difference_status": "not a mask contrast under full coupling",
            "same_root_tree_difference_mean_icl_loo_r2": "",
            "same_root_tree_difference_best_icl_loo_r2": "",
            "prospective_contrast": "normal-fan/tree-count arms",
            "cross_root_status": "binary overlap saturated; rank summaries vary",
            "interpretation": "tests abundance/normal-fan/cross-root rank rather than mask multiplicity",
        }
    )
    return rows


def summarize_scrambles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_intervention: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = parse_float(row.get("target_accuracy_delta"))
        if value is not None:
            by_intervention[str(row.get("intervention"))].append(value)
    return {
        intervention: {
            "n": len(values),
            "target_accuracy_delta_mean": float(np.mean(values)) if values else None,
            "target_accuracy_delta_min": float(np.min(values)) if values else None,
            "target_accuracy_delta_max": float(np.max(values)) if values else None,
        }
        for intervention, values in sorted(by_intervention.items())
    }


def render_audit_md(audit: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Multibase Live Job Audit",
            "",
            markdown_table(
                ["item", "value"],
                [
                    ["branch", audit.get("branch")],
                    ["commit", audit.get("commit")],
                    ["expected tasks", audit.get("expected_task_count")],
                    ["completed results", audit.get("completed_result_count")],
                    ["missing results", audit.get("missing_result_count")],
                    ["selected topology groups", audit.get("selected_topology_count")],
                    ["complete groups", audit.get("complete_groups")],
                    ["old failed system-python results", audit.get("old_failed_system_python_results_present")],
                ],
            ),
            "",
            "All selected topology groups have complete five-seed coverage if `missing results` is zero and `complete groups` equals the selected topology count.",
        ]
    )


def render_results_md(payload: Mapping[str, Any]) -> str:
    model_rows = payload["model_results"]
    pair_summary_payload = payload["pair_summary"]
    lines = [
        "# Multibase Exact-Control Results",
        "",
        f"Run rows: `{payload['n_run_rows']}`. Topology groups: `{payload['n_group_rows']}`.",
        "",
        "## Grouped LOO R2",
        "",
        markdown_table(
            ["outcome", "model", "n", "LOO R2"],
            [
                [row["outcome"], row["model"], row["n"], row.get("loo_r2")]
                for row in model_rows
            ],
        ),
        "",
        "## Paired Arm Contrasts",
        "",
    ]
    for arm, stats in pair_summary_payload.items():
        lines.append(f"### {arm}")
        lines.append("")
        lines.append(
            markdown_table(
                ["contrast", "n", "mean", "95% CI"],
                [
                    ["mean novel ICL high-low", stats["mean_delta_mean_novel_icl"]["n"], stats["mean_delta_mean_novel_icl"]["mean"], stats["mean_delta_mean_novel_icl"]["ci95"]],
                    ["best seed ICL high-low", stats["mean_delta_best_seed_novel_icl"]["n"], stats["mean_delta_best_seed_novel_icl"]["mean"], stats["mean_delta_best_seed_novel_icl"]["ci95"]],
                    ["seed std high-low", stats["mean_delta_seed_std_novel_icl"]["n"], stats["mean_delta_seed_std_novel_icl"]["mean"], stats["mean_delta_seed_std_novel_icl"]["ci95"]],
                ],
            )
        )
        lines.append("")
    lines.append("Interpretation should follow the paired arms: Arm A isolates normal-fan variation at nearly fixed tree count; Arm B isolates tree-count variation at matched normal-fan score.")
    return "\n".join(lines)


def render_cross_md(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Cross-Root Decoder-Aware Contrast Reanalysis",
            "",
            payload["interpretation"],
            "",
            "## Cross-Root Model Rows",
            "",
            markdown_table(
                ["outcome", "model", "n", "LOO R2"],
                [[row["outcome"], row["model"], row["n"], row.get("loo_r2")] for row in payload["model_results"]],
            ),
            "",
            "These metrics are pre-training diagnostics unless learned decoder or posterior weights are explicitly used.",
        ]
    )


def render_tree_diff_diag_md(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Retrospective vs Prospective Tree-Difference Diagnostic",
            "",
            "The fixed-m20 tree-difference signal was strong retrospectively, but the first prospective exact-control contrast was negative or inconclusive.",
            "",
            f"Conclusion: {payload['answer']}",
            "",
            "The viable diagnosis is that same-root co-participation was partly regime/family dependent and too narrow for decoder competition. Cross-root and normal-fan/tree-count metrics are the better current directions.",
        ]
    )


def render_gamma_md(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Gamma Multibase Diagnostic Report",
            "",
            f"Status: `{payload['status']}`.",
            "",
            payload["interpretation"],
            "",
            markdown_table(
                ["outcome", "model", "n", "LOO R2"],
                [[row["outcome"], row["model"], row["n"], row.get("loo_r2")] for row in payload["model_results"]],
            ),
        ]
    )


def render_mechanism_md(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Multibase Mechanism Follow-Up",
        "",
        f"Status: `{payload['status']}`.",
        "",
        f"Selected topology IDs: `{', '.join(payload['selected_topology_ids'])}`.",
        "",
        f"Mechanism rows collected: `{payload['mechanism_rows']}`. Scramble rows collected: `{payload['scramble_rows']}`.",
    ]
    if payload["scramble_summary"]:
        lines.extend(
            [
                "",
                markdown_table(
                    ["intervention", "n", "mean accuracy delta", "min", "max"],
                    [
                        [name, item["n"], item["target_accuracy_delta_mean"], item["target_accuracy_delta_min"], item["target_accuracy_delta_max"]]
                        for name, item in payload["scramble_summary"].items()
                    ],
                ),
            ]
        )
    else:
        lines.append("")
        lines.append("No multibase mechanism/scramble outputs were present when this report was generated; the command file records the selected targeted follow-up.")
    return "\n".join(lines)


def best_model(payload: Mapping[str, Any], outcome: str) -> dict[str, Any] | None:
    rows = [row for row in payload["model_results"] if row.get("outcome") == outcome and row.get("loo_r2") is not None]
    return max(rows, key=lambda row: float(row["loo_r2"])) if rows else None


def build_synthesis(
    results: Mapping[str, Any],
    cross: Mapping[str, Any],
    tree_diag: Mapping[str, Any],
    gamma: Mapping[str, Any],
    mechanism: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "post_multibase_exact_control_synthesis.v1",
        "exact_theory": "First-order CRNs compute through rooted tree-sum projections by the matrix-tree theorem; this remains exact.",
        "pre_training_prediction": {
            "best_mean_model": best_model(results, "mean_novel_icl"),
            "best_best_seed_model": best_model(results, "best_seed_novel_icl"),
            "pair_summary": results["pair_summary"],
            "held_out_base": results["held_out_base"],
        },
        "expressivity": "Best-seed ICL is reported separately as the expressivity envelope; do not collapse it with mean-seed trainability.",
        "trainability": "Seed standard deviation is reported separately; weak seed-std prediction means trainability is not reduced to the same scalar as expressivity.",
        "post_training_mechanism": mechanism,
        "causal_interventions": "Same-root tree-difference overlap failed as a standalone prospective causal knob. Multibase mechanism scrambles are only supported if the collected scramble rows are nonempty.",
        "cross_root_decoder_contrast": cross.get("interpretation"),
        "gamma": gamma.get("interpretation"),
        "tree_difference_failure_diagnosis": tree_diag.get("answer"),
        "thermodynamics": "Thermodynamics remains untested; no F_max or entropy-production claim is supported.",
        "bottom_line": "The multibase library tests whether rooted-tree abundance, normal-fan geometry, or cross-root contrast rank improves pre-training prediction under exact controls. Claims should be based on grouped LOO, paired arms, and held-out-base behavior rather than seed-level rows.",
    }


def render_synthesis_md(payload: Mapping[str, Any]) -> str:
    pre = payload["pre_training_prediction"]
    return "\n".join(
        [
            "# Post-Multibase Exact-Control Synthesis",
            "",
            "## Exact Theory",
            "",
            payload["exact_theory"],
            "",
            "## Pre-Training Prediction",
            "",
            f"Best grouped-LOO mean-ICL model: `{(pre.get('best_mean_model') or {}).get('model')}` with R2 `{fmt((pre.get('best_mean_model') or {}).get('loo_r2'))}`.",
            f"Best grouped-LOO best-seed model: `{(pre.get('best_best_seed_model') or {}).get('model')}` with R2 `{fmt((pre.get('best_best_seed_model') or {}).get('loo_r2'))}`.",
            "",
            "## Expressivity",
            "",
            payload["expressivity"],
            "",
            "## Trainability",
            "",
            payload["trainability"],
            "",
            "## Post-Training Mechanism",
            "",
            f"Mechanism status: `{payload['post_training_mechanism'].get('status')}`.",
            "",
            "## Causal Interventions",
            "",
            payload["causal_interventions"],
            "",
            "## Cross-Root Contrast",
            "",
            str(payload["cross_root_decoder_contrast"]),
            "",
            "## Gamma",
            "",
            str(payload["gamma"]),
            "",
            "## Tree-Difference Failure Diagnosis",
            "",
            str(payload["tree_difference_failure_diagnosis"]),
            "",
            "## Thermodynamics",
            "",
            payload["thermodynamics"],
            "",
            "## Bottom Line",
            "",
            payload["bottom_line"],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", default=str(DEFAULT_LIBRARY_ROOT))
    parser.add_argument("--training-root", default=str(DEFAULT_TRAINING_ROOT))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--max-pairs-per-root-pair", type=int, default=50000)
    parser.add_argument("--compute-gamma", action="store_true")
    parser.add_argument("--force-gamma", action="store_true")
    parser.add_argument("--gamma-samples", type=int, default=180)
    parser.add_argument("--gamma-trials", type=int, default=6)
    args = parser.parse_args()
    make_reports(args)


if __name__ == "__main__":
    main()
