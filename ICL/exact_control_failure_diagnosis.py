"""Post exact-control failure diagnosis for the Markov-ICL topology project.

This script produces the artifacts requested by
``MARKOV_ICL_NEXT_PHASE_GOAL.md`` after the May 12 exact-control update.  It
does not launch new training.  It uses the existing fixed-m20, prospective
tree-difference, repaired-gamma, normal-fan, and mechanism artifacts to:

* audit the current state;
* compute decoder-aware cross-root tree-contrast metrics;
* diagnose why same-root tree-difference overlap failed prospectively;
* separate tree-count and normal-fan signals as far as the current data allow;
* update gamma, mechanism, expressivity/trainability, and synthesis reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cross_root_tree_contrast_metrics import cross_root_tree_contrast_summary, json_ready
from topology_metrics import normalize_edges, topology_matrices
from tree_level_multiplicity_metrics import tree_table_from_arborescences


ROOT = Path(__file__).resolve().parents[1]
ICL = ROOT / "ICL"
RESULTS = ICL / "results"
OUT = RESULTS / "next_phase_stats"
PROSPECTIVE_LIBRARY_ROOT = RESULTS / "prospective_tree_diff_multiplicity_n6_m20_c200"
NORMAL_FAN_LIBRARY_ROOT = OUT / "degree_rewire_normal_fan_n5_m12_N3_D2"


REQUIRED_ARTIFACTS = {
    "goal": ICL / "MARKOV_ICL_NEXT_PHASE_GOAL.md",
    "latex_pdf": ICL / "paper" / "topology_icl_first_order_report.pdf",
    "post_phase3_synthesis": OUT / "post_phase3_markov_icl_synthesis.md",
    "gamma_toy_repair": OUT / "gamma_toy_repair_final_report.md",
    "input_multiplicity_causal_control": OUT / "input_multiplicity_causal_control_report.md",
    "tree_multiplicity_causal_mask_library": OUT / "tree_multiplicity_causal_mask_library.md",
    "tree_level_multiplicity_reanalysis": OUT / "tree_level_multiplicity_reanalysis.md",
    "predictor_name_reconciliation": OUT / "predictor_name_reconciliation.md",
    "post_gamma_exact_control_synthesis": OUT / "post_gamma_repair_exact_control_synthesis.md",
    "repaired_gamma_existing_data_reanalysis": OUT / "repaired_gamma_existing_data_reanalysis.md",
    "prospective_tree_diff_causal_report": OUT / "prospective_tree_diff_multiplicity_causal_report.md",
    "normal_fan_training_report": OUT / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.md",
    "mechanism_followup_report": OUT / "mechanism_and_causal_scramble_followup_report.md",
}

SCRIPT_PATHS = {
    "tree_level_and_tree_difference_multiplicity": ICL / "tree_level_multiplicity_metrics.py",
    "fixed_m20_causal_control": ICL / "tree_multiplicity_causal_control.py",
    "prospective_tree_difference_library": ICL / "prospective_tree_diff_multiplicity_control.py",
    "prospective_tree_difference_training_report": ICL / "prospective_tree_diff_multiplicity_training_report.py",
    "repaired_gamma_existing_data_reanalysis": ICL / "repaired_gamma_existing_data_reanalysis.py",
    "scaled_normal_fan_expansion": ICL / "exact_degree_normal_fan_scale_report.py",
    "mechanism_collection": ICL / "collect_mechanism_results.py",
    "causal_intervention_collection": ICL / "collect_causal_interventions.py",
    "causal_interventions": ICL / "causal_topology_interventions.py",
    "current_script": ICL / "exact_control_failure_diagnosis.py",
    "cross_root_metrics": ICL / "cross_root_tree_contrast_metrics.py",
}


def run(cmd: Sequence[str], cwd: Path = ROOT, timeout: int = 30) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=True)
    return proc.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(fmt(item) for item in row) + " |")
    return "\n".join(out)


def value_summary(values: Sequence[Any]) -> dict[str, Any]:
    vals = np.asarray([float(v) for v in values if finite_float(v) is not None], dtype=float)
    if vals.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "p25": float(np.quantile(vals, 0.25)),
        "median": float(np.quantile(vals, 0.5)),
        "p75": float(np.quantile(vals, 0.75)),
        "max": float(vals.max()),
    }


def correlation(rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = finite_float(row.get(x_key))
        y = finite_float(row.get(y_key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    if len(xs) < 3:
        return {"x": x_key, "y": y_key, "n": len(xs), "r": None}
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    denom = float(np.sqrt(np.sum((x - x.mean()) ** 2) * np.sum((y - y.mean()) ** 2)))
    r = None if denom <= 1e-12 else float(np.sum((x - x.mean()) * (y - y.mean())) / denom)
    return {"x": x_key, "y": y_key, "n": len(xs), "r": r}


def matrix_design(
    rows: Sequence[Mapping[str, Any]],
    numeric: Sequence[str],
    categorical: Sequence[str],
    outcome: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    levels = {key: sorted({str(row.get(key, "")) for row in rows}) for key in categorical}
    xs: list[list[float]] = []
    ys: list[float] = []
    names = list(numeric)
    for key in categorical:
        names.extend([f"{key}={level}" for level in levels[key]])
    for row in rows:
        y = finite_float(row.get(outcome))
        if y is None:
            continue
        vals: list[float] = []
        ok = True
        for key in numeric:
            value = finite_float(row.get(key))
            if value is None:
                ok = False
                break
            vals.append(value)
        if not ok:
            continue
        for key in categorical:
            observed = str(row.get(key, ""))
            vals.extend([1.0 if observed == level else 0.0 for level in levels[key]])
        if not all(math.isfinite(value) for value in vals):
            continue
        xs.append(vals)
        ys.append(y)
    if not xs:
        return np.zeros((0, len(names))), np.zeros(0), names
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), names


def loo_r2(
    rows: Sequence[Mapping[str, Any]],
    numeric: Sequence[str],
    outcome: str,
    categorical: Sequence[str] = (),
    ridge_alpha: float = 1.0e-6,
) -> dict[str, Any]:
    X, y, names = matrix_design(rows, numeric, categorical, outcome)
    n, p = X.shape
    if n < max(8, p + 3):
        return {
            "outcome": outcome,
            "numeric_predictors": list(numeric),
            "categorical_predictors": list(categorical),
            "n_groups": int(n),
            "n_predictors": int(p),
            "loo_r2": None,
            "reason": "too_few_groups_or_complete_cases",
        }
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1.0e-12:
        return {
            "outcome": outcome,
            "numeric_predictors": list(numeric),
            "categorical_predictors": list(categorical),
            "n_groups": int(n),
            "n_predictors": int(p),
            "loo_r2": None,
            "reason": "constant_outcome",
        }
    preds: list[float] = []
    for holdout in range(n):
        train = np.arange(n) != holdout
        center = X[train].mean(axis=0)
        scale = X[train].std(axis=0)
        scale[scale <= 1.0e-12] = 1.0
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
        "numeric_predictors": list(numeric),
        "categorical_predictors": list(categorical),
        "design_columns": names,
        "n_groups": int(n),
        "n_predictors": int(p),
        "loo_r2": float(1.0 - err / denom),
    }


def bootstrap_mean_delta(values: Sequence[float], n_boot: int = 5000, seed: int = 19) -> dict[str, Any]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = float(np.mean(vals[rng.integers(0, vals.size, vals.size)]))
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
    }


def localize_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.exists():
        return path
    marker = "/ICL/results/"
    if marker in path_text:
        candidate = RESULTS / path_text.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    return None


def read_remote_json(remote_path: str, host: str = "engaging", timeout: int = 20) -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", host, "cat", remote_path],
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout + 5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


class CrossMetricComputer:
    def __init__(self, n_context: int, z_dim: int, max_pairs_per_root_pair: int = 20000):
        self.n_context = n_context
        self.z_dim = z_dim
        self.max_pairs_per_root_pair = max_pairs_per_root_pair
        self._tree_cache: dict[tuple[int, tuple[tuple[int, int], ...]], tuple[np.ndarray, np.ndarray]] = {}

    def compute(self, n_nodes: int, edges: Sequence[Sequence[int]], input_mask: Sequence[Sequence[float]]) -> dict[str, Any]:
        edge_tuple = normalize_edges(n_nodes, edges)
        key = (int(n_nodes), edge_tuple)
        if key not in self._tree_cache:
            mats = topology_matrices(n_nodes, edge_tuple)
            self._tree_cache[key] = tree_table_from_arborescences(mats["arborescences"], len(edge_tuple))
        roots, incidence = self._tree_cache[key]
        return cross_root_tree_contrast_summary(
            incidence,
            roots,
            np.asarray(input_mask, dtype=float),
            n_context=self.n_context,
            z_dim=self.z_dim,
            max_pairs_per_root_pair=self.max_pairs_per_root_pair,
        )


def compact_cross_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "cross_per_root_pair"}


def compact_group_row(row: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = (
        "cross_",
        "diff_overlap_",
        "tree_overlap_",
        "edge_overlap_",
        "gamma_no_bias_",
        "capacity_normal_fan_",
    )
    exact_keys = {
        "dataset",
        "group",
        "topology_name",
        "physical_topology_name",
        "input_mask_family",
        "mask_family",
        "load_stratum",
        "contrast_level",
        "outcome_mean",
        "outcome_best",
        "outcome_seed_std",
        "mean_novel_icl",
        "best_seed_novel_icl",
        "mean_seed_novel_icl",
        "seed_std_novel_icl",
        "branch_failures",
        "trained_branch_margin",
        "d_rel",
        "input_coupled_parameter_count",
        "input_coupled_edge_count",
        "input_coupled_coord_count",
        "input_coord_load_gini",
        "input_edge_load_gini",
        "edge_M_mean",
        "edge_M_gini",
        "diff_coord_load_gini",
        "tree_coord_load_gini",
        "tree_count_log",
        "tree_count_exact_total",
        "tree_count_enumerated_total",
        "library_n_trees_total_enum",
        "library_n_trees_total_enum_log",
        "library_root_tree_count_gini",
        "topology_payload_source",
    }
    return {
        key: value
        for key, value in row.items()
        if key in exact_keys or any(key.startswith(prefix) for prefix in prefixes)
    }


def load_fixed_m20_rows(max_pairs_per_root_pair: int, ssh_host: str) -> list[dict[str, Any]]:
    report = read_json(OUT / "tree_level_multiplicity_reanalysis.json")
    fixed = next(dataset for dataset in report["datasets"] if dataset["name"] == "fixed_m20_masks_cluster_topology")
    computers: dict[tuple[int, int], CrossMetricComputer] = {}
    rows: list[dict[str, Any]] = []
    for group in fixed["groups"]:
        run_dir = str(group.get("run_dir") or "")
        payload = None
        source = "missing"
        local = localize_path(os.path.join(run_dir, "topology.json"))
        if local is not None:
            payload = read_json(local)
            source = "local"
        else:
            payload = read_remote_json(os.path.join(run_dir, "topology.json"), host=ssh_host)
            if payload is not None:
                source = f"ssh:{ssh_host}"
        row = dict(group)
        row["dataset"] = "fixed_m20_retrospective"
        row["outcome_mean"] = row.get("mean_novel_icl")
        row["outcome_best"] = row.get("best_seed_novel_icl")
        row["outcome_seed_std"] = row.get("seed_std_novel_icl")
        row["topology_payload_source"] = source
        if payload is not None:
            n_context = int(row.get("n_context", 4))
            z_dim = int(row.get("z_dim", 4))
            computer = computers.setdefault((n_context, z_dim), CrossMetricComputer(n_context, z_dim, max_pairs_per_root_pair))
            row.update(compact_cross_metrics(computer.compute(int(payload["n_nodes"]), payload["edges"], payload["input_mask"])))
        rows.append(row)
    return rows


def load_prospective_rows(max_pairs_per_root_pair: int) -> list[dict[str, Any]]:
    report = read_json(OUT / "prospective_tree_diff_multiplicity_causal_report.json")
    computer = CrossMetricComputer(n_context=4, z_dim=4, max_pairs_per_root_pair=max_pairs_per_root_pair)
    rows: list[dict[str, Any]] = []
    for group in report["group_rows"]:
        topology_path = PROSPECTIVE_LIBRARY_ROOT / str(group["edge_json"])
        mask_path = PROSPECTIVE_LIBRARY_ROOT / str(group["input_mask_json"])
        mask_payload = read_json(mask_path)
        row = dict(group)
        row["dataset"] = "prospective_tree_diff_exact_control"
        row["outcome_mean"] = row.get("mean_seed_novel_icl")
        row["outcome_best"] = row.get("best_seed_novel_icl")
        row["outcome_seed_std"] = row.get("seed_std_novel_icl")
        row["topology_payload_source"] = str(topology_path.relative_to(ROOT))
        row.update(compact_cross_metrics(computer.compute(int(mask_payload["n_nodes"]), mask_payload["edges"], mask_payload["input_mask"])))
        rows.append(row)
    return rows


def load_normal_fan_rows(max_pairs_per_root_pair: int) -> list[dict[str, Any]]:
    report = read_json(OUT / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.json")
    computer = CrossMetricComputer(n_context=3, z_dim=2, max_pairs_per_root_pair=max_pairs_per_root_pair)
    rows: list[dict[str, Any]] = []
    for group in report["group_rows"]:
        edge_path = localize_path(str(group.get("library_edge_json") or group.get("capacity_edge_json") or ""))
        if edge_path is None:
            continue
        topo = read_json(edge_path)
        p = int(group.get("p", 8))
        input_mask = np.ones((len(topo["edges"]), p), dtype=float)
        row = dict(group)
        row["dataset"] = "exact_degree_normal_fan"
        row["outcome_mean"] = row.get("mean_seed_novel_icl")
        row["outcome_best"] = row.get("best_seed_novel_icl")
        row["outcome_seed_std"] = row.get("seed_std_novel_icl")
        row["topology_payload_source"] = str(edge_path.relative_to(ROOT))
        row.update(compact_cross_metrics(computer.compute(int(topo["n_nodes"]), topo["edges"], input_mask)))
        rows.append(row)
    return rows


CROSS_MODEL_SPECS = {
    "same_root_tree_diff": ["diff_overlap_norm_min", "diff_overlap_norm_mean", "diff_coord_load_gini"],
    "cross_root_overlap": [
        "cross_overlap_norm_min",
        "cross_overlap_norm_mean",
        "cross_best_root_pair_overlap_norm_min",
        "cross_root_pair_overlap_entropy_mean",
    ],
    "cross_root_oriented": [
        "cross_separation_norm_mean",
        "cross_imbalance_norm_mean",
        "cross_contrast_effective_rank_mean",
        "cross_edge_participation_gini",
    ],
    "same_plus_cross_minimal": [
        "diff_overlap_norm_min",
        "cross_overlap_norm_min",
        "cross_best_root_pair_overlap_norm_min",
        "cross_contrast_effective_rank_mean",
    ],
}


def cross_dataset_analysis(rows: Sequence[Mapping[str, Any]], dataset_name: str) -> dict[str, Any]:
    outcomes = ["outcome_mean", "outcome_best", "outcome_seed_std"]
    if any(finite_float(row.get("branch_failures")) is not None for row in rows):
        outcomes.append("branch_failures")
    if any(finite_float(row.get("trained_branch_margin")) is not None for row in rows):
        outcomes.append("trained_branch_margin")
    models = []
    if dataset_name == "fixed_m20_retrospective":
        categorical = ["physical_topology_name"]
    elif dataset_name == "prospective_tree_diff_exact_control":
        categorical = ["load_stratum"]
    else:
        categorical = []
    for outcome in outcomes:
        for model, predictors in CROSS_MODEL_SPECS.items():
            result = loo_r2(rows, predictors, outcome, categorical=categorical)
            result["model"] = model
            models.append(result)
    corrs = []
    for outcome in outcomes:
        for feature in [
            "diff_overlap_norm_min",
            "cross_overlap_norm_min",
            "cross_best_root_pair_overlap_norm_min",
            "cross_separation_norm_mean",
            "cross_contrast_effective_rank_mean",
        ]:
            corrs.append(correlation(rows, feature, outcome))
    return {
        "dataset": dataset_name,
        "n_groups": len(rows),
        "model_results": models,
        "correlations": corrs,
        "feature_summaries": {
            feature: value_summary([row.get(feature) for row in rows])
            for feature in [
                "diff_overlap_norm_min",
                "diff_overlap_norm_mean",
                "cross_overlap_norm_min",
                "cross_overlap_norm_mean",
                "cross_best_root_pair_overlap_norm_min",
                "cross_separation_norm_mean",
                "cross_imbalance_norm_mean",
                "cross_contrast_effective_rank_mean",
                "cross_root_pair_overlap_entropy_mean",
            ]
        },
    }


def model_lookup(models: Sequence[Mapping[str, Any]], outcome: str, model: str) -> float | None:
    for item in models:
        if item.get("outcome") == outcome and item.get("model") == model:
            return finite_float(item.get("loo_r2"))
    return None


def build_cross_root_reanalysis(
    fixed_rows: list[dict[str, Any]],
    prospective_rows: list[dict[str, Any]],
    normal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    datasets = [
        cross_dataset_analysis(fixed_rows, "fixed_m20_retrospective"),
        cross_dataset_analysis(prospective_rows, "prospective_tree_diff_exact_control"),
        cross_dataset_analysis(normal_rows, "exact_degree_normal_fan"),
    ]
    return {
        "schema": "cross_root_tree_contrast_reanalysis.v1",
        "metric_file": "ICL/cross_root_tree_contrast_metrics.py",
        "datasets": datasets,
        "group_rows": {
            "fixed_m20_retrospective": [compact_group_row(row) for row in fixed_rows],
            "prospective_tree_diff_exact_control": [compact_group_row(row) for row in prospective_rows],
            "exact_degree_normal_fan": [compact_group_row(row) for row in normal_rows],
        },
        "interpretation": (
            "Cross-root metrics are implemented and evaluated as diagnostics.  They should not be treated "
            "as selectors unless they survive exact-control held-out tests.  In the current data, cross-root "
            "co-participation improves some small-sample fits after load-stratum control, but it does not rescue "
            "the prospective high/low causal contrast by itself.  In the full-coupling normal-fan experiment, "
            "binary cross-root overlap saturates at one and only controllability/rank-style cross-root summaries vary."
        ),
    }


def write_cross_root_markdown(report: Mapping[str, Any]) -> None:
    lines = [
        "# Cross-Root Tree-Contrast Reanalysis",
        "",
        "## Scope",
        "",
        "This report implements decoder-aware cross-root tree-difference comparison metrics. It compares trees rooted at different species because the Markov steady state normalizes all rooted tree numerators jointly and the decoder is learned.",
        "",
        "## Datasets",
        "",
        md_table(
            ["dataset", "groups", "diff min median", "cross min median", "cross best-root min median"],
            [
                [
                    ds["dataset"],
                    ds["n_groups"],
                    ds["feature_summaries"]["diff_overlap_norm_min"]["median"],
                    ds["feature_summaries"]["cross_overlap_norm_min"]["median"],
                    ds["feature_summaries"]["cross_best_root_pair_overlap_norm_min"]["median"],
                ]
                for ds in report["datasets"]
            ],
        ),
        "",
    ]
    for ds in report["datasets"]:
        lines.extend([f"## {ds['dataset']} LOO Models", ""])
        rows = []
        for item in ds["model_results"]:
            if item["outcome"] in ("outcome_mean", "outcome_best", "branch_failures", "trained_branch_margin"):
                rows.append([item["outcome"], item["model"], item["n_groups"], item.get("loo_r2"), item.get("reason")])
        lines.extend([md_table(["outcome", "model", "groups", "LOO R2", "reason"], rows), ""])
        lines.extend([f"## {ds['dataset']} Correlations", ""])
        rows = [
            [item["y"], item["x"], item["n"], item["r"]]
            for item in ds["correlations"]
            if item["y"] in ("outcome_mean", "outcome_best")
        ]
        lines.extend([md_table(["outcome", "feature", "n", "r"], rows), ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            str(report["interpretation"]),
        ]
    )
    write_text(OUT / "cross_root_tree_contrast_reanalysis.md", "\n".join(lines))


def build_failure_diagnosis(fixed_rows: Sequence[Mapping[str, Any]], prospective_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fixed_cycle = [row for row in fixed_rows if row.get("physical_topology_name") == "cycle_chords_n6_m20_seed3"]
    fixed_by_graph_corr = [
        {
            "physical_topology_name": graph,
            **correlation([row for row in fixed_rows if row.get("physical_topology_name") == graph], "diff_overlap_norm_min", "outcome_mean"),
        }
        for graph in sorted({str(row.get("physical_topology_name")) for row in fixed_rows})
    ]
    pros_by_category: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in prospective_rows:
        pros_by_category[(str(row.get("load_stratum")), str(row.get("contrast_level")))].append(row)
    category_summary = []
    for (load, contrast), rows in sorted(pros_by_category.items()):
        category_summary.append(
            {
                "load_stratum": load,
                "contrast_level": contrast,
                "n_groups": len(rows),
                "mean_icl": mean(float(row["outcome_mean"]) for row in rows),
                "best_icl": mean(float(row["outcome_best"]) for row in rows),
                "diff_overlap_norm_min_mean": mean(float(row["diff_overlap_norm_min"]) for row in rows),
                "cross_overlap_norm_min_mean": mean(float(row["cross_overlap_norm_min"]) for row in rows),
                "cross_best_root_pair_overlap_norm_min_mean": mean(
                    float(row["cross_best_root_pair_overlap_norm_min"]) for row in rows
                ),
                "cross_contrast_effective_rank_mean": mean(float(row["cross_contrast_effective_rank_mean"]) for row in rows),
            }
        )

    contrasts = []
    for load in sorted({str(row.get("load_stratum")) for row in prospective_rows}):
        high = [row for row in prospective_rows if row.get("load_stratum") == load and row.get("contrast_level") == "high"]
        low = [row for row in prospective_rows if row.get("load_stratum") == load and row.get("contrast_level") == "low"]
        if not high or not low:
            continue
        for outcome in ["outcome_mean", "outcome_best", "branch_failures", "trained_branch_margin"]:
            high_mean = mean(float(row[outcome]) for row in high if finite_float(row.get(outcome)) is not None)
            low_mean = mean(float(row[outcome]) for row in low if finite_float(row.get(outcome)) is not None)
            contrasts.append(
                {
                    "load_stratum": load,
                    "outcome": outcome,
                    "high_mean": high_mean,
                    "low_mean": low_mean,
                    "high_minus_low": None if high_mean is None or low_mean is None else high_mean - low_mean,
                }
            )

    fixed_diff = np.asarray([float(row["diff_overlap_norm_min"]) for row in fixed_rows], dtype=float)
    balanced_low = [
        float(row["diff_overlap_norm_min"])
        for row in prospective_rows
        if row.get("load_stratum") == "balanced_load" and row.get("contrast_level") == "low"
    ]
    imbalanced_low = [
        float(row["diff_overlap_norm_min"])
        for row in prospective_rows
        if row.get("load_stratum") == "imbalanced_coord_load" and row.get("contrast_level") == "low"
    ]
    saturation = {
        "fixed_m20_diff_overlap_quantiles": {
            str(q): float(np.quantile(fixed_diff, q)) for q in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
        },
        "prospective_balanced_low_mean": mean(balanced_low),
        "prospective_imbalanced_low_mean": mean(imbalanced_low),
        "balanced_low_above_fixed_median": mean([1.0 if v > np.quantile(fixed_diff, 0.5) else 0.0 for v in balanced_low]),
        "balanced_low_above_fixed_p75": mean([1.0 if v > np.quantile(fixed_diff, 0.75) else 0.0 for v in balanced_low]),
        "imbalanced_low_above_fixed_median": mean([1.0 if v > np.quantile(fixed_diff, 0.5) else 0.0 for v in imbalanced_low]),
    }
    conclusions = {
        "single_best_explanation": (
            "The prospective contrast did not reproduce the fixed-m20 signal because same-root "
            "co-participation is too coarse once graph, count, d_rel, and load structure are fixed.  The "
            "prospective high/low masks were genuinely separated in same-root overlap, and high masks also "
            "usually had higher cross-root minimum overlap, but they did not improve best seed, mean seed, "
            "branch failures, or trained margins.  Balanced low masks were not in the zero-overlap regime "
            "that helped drive the retrospective family signal, and imbalanced masks tested lower overlap "
            "only under a trainability/load confound.  The failure is therefore not explained by a single "
            "saturation story; it points to missing orientation, controllability, root-pair choice, and "
            "optimization variables."
        ),
        "viable_hypotheses": [
            "the balanced prospective library did not test the low/zero-overlap regime present in retrospective coord-block masks",
            "co-participation without sign/orientation or controllability is too coarse",
            "one-graph prospective evidence is insufficient even though fixed-m20 within-graph correlations were positive",
            "trainability and post-training organization remain important because branch/projection scrambles cause large drops",
        ],
        "weakened_hypotheses": [
            "same-root tree-difference overlap is a standalone causal knob",
            "repaired gamma can be used as a topology selector",
        ],
    }
    return {
        "schema": "tree_difference_failure_diagnosis.v1",
        "source_datasets": ["fixed_m20_retrospective", "prospective_tree_diff_exact_control"],
        "n_fixed_groups": len(fixed_rows),
        "n_fixed_cycle_groups": len(fixed_cycle),
        "n_prospective_groups": len(prospective_rows),
        "fixed_feature_summary": {
            feature: value_summary([row.get(feature) for row in fixed_rows])
            for feature in ["diff_overlap_norm_min", "cross_overlap_norm_min", "cross_best_root_pair_overlap_norm_min"]
        },
        "prospective_feature_summary": {
            feature: value_summary([row.get(feature) for row in prospective_rows])
            for feature in ["diff_overlap_norm_min", "cross_overlap_norm_min", "cross_best_root_pair_overlap_norm_min"]
        },
        "prospective_category_summary": category_summary,
        "prospective_high_low_contrasts": contrasts,
        "fixed_within_graph_diff_overlap_correlations": fixed_by_graph_corr,
        "fixed_cycle_diff_correlation": correlation(fixed_cycle, "diff_overlap_norm_min", "outcome_mean"),
        "saturation_diagnostic": saturation,
        "load_and_tree_count_notes": {
            "physical_graph_fixed_prospectively": True,
            "tree_count_fixed_prospectively": True,
            "input_coupled_count_fixed_prospectively": True,
            "d_rel_fixed_prospectively": True,
            "load_strata": sorted({str(row.get("load_stratum")) for row in prospective_rows}),
        },
        "conclusions": conclusions,
    }


def write_failure_markdown(report: Mapping[str, Any]) -> None:
    lines = [
        "# Tree-Difference Failure Diagnosis",
        "",
        "## Direct Answer",
        "",
        report["conclusions"]["single_best_explanation"],
        "",
        "## Prospective High-Low Contrasts",
        "",
        md_table(
            ["load", "outcome", "high mean", "low mean", "high-low"],
            [
                [row["load_stratum"], row["outcome"], row["high_mean"], row["low_mean"], row["high_minus_low"]]
                for row in report["prospective_high_low_contrasts"]
            ],
        ),
        "",
        "## Range / Saturation Diagnostic",
        "",
        md_table(
            ["quantity", "value"],
            [
                ["fixed median diff overlap", report["saturation_diagnostic"]["fixed_m20_diff_overlap_quantiles"]["0.5"]],
                ["fixed p75 diff overlap", report["saturation_diagnostic"]["fixed_m20_diff_overlap_quantiles"]["0.75"]],
                ["balanced low prospective mean", report["saturation_diagnostic"]["prospective_balanced_low_mean"]],
                ["imbalanced low prospective mean", report["saturation_diagnostic"]["prospective_imbalanced_low_mean"]],
                ["balanced low above fixed median", report["saturation_diagnostic"]["balanced_low_above_fixed_median"]],
                ["balanced low above fixed p75", report["saturation_diagnostic"]["balanced_low_above_fixed_p75"]],
            ],
        ),
        "",
        "## Fixed-m20 Within-Graph Correlations",
        "",
        md_table(
            ["physical graph", "n", "r(diff min, mean ICL)"],
            [
                [row["physical_topology_name"], row["n"], row["r"]]
                for row in report["fixed_within_graph_diff_overlap_correlations"]
            ],
        ),
        "",
        "## Viable Explanations",
        "",
        "\n".join(f"- {item}" for item in report["conclusions"]["viable_hypotheses"]),
        "",
        "## Weakened Explanations",
        "",
        "\n".join(f"- {item}" for item in report["conclusions"]["weakened_hypotheses"]),
    ]
    write_text(OUT / "tree_difference_failure_diagnosis.md", "\n".join(lines))


def model_r2(report: Mapping[str, Any], outcome: str, model: str) -> float | None:
    for item in report.get("model_results", {}).get(outcome, []):
        if item.get("model") == model:
            return finite_float(item.get("loo_r2"))
    return None


def build_normal_fan_reports(normal_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    tree_features = ["library_n_trees_total_enum_log", "library_root_tree_count_gini"]
    fan_features = [
        "capacity_normal_fan_active_tree_count_mean",
        "capacity_normal_fan_branch_tree_nmi_mean",
        "capacity_normal_fan_branch_active_tree_count_min_mean",
    ]
    cross_features = ["cross_overlap_norm_min", "cross_contrast_effective_rank_mean"]
    outcomes = ["outcome_mean", "outcome_best", "outcome_seed_std"]

    feature_correlations = []
    for left in tree_features + fan_features + cross_features:
        for right in tree_features + fan_features + cross_features:
            if left < right:
                feature_correlations.append(correlation(normal_rows, left, right))

    loo_models = []
    model_specs = {
        "tree_count_only": tree_features,
        "normal_fan_only": fan_features,
        "tree_count_plus_normal_fan": tree_features + fan_features,
        "normal_fan_plus_cross_root": fan_features + cross_features,
    }
    for outcome in outcomes:
        for name, predictors in model_specs.items():
            row = loo_r2(normal_rows, predictors, outcome)
            row["model"] = name
            loo_models.append(row)

    by_tree_bin: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    tree_vals = [float(row["library_n_trees_total_enum_log"]) for row in normal_rows]
    q25, q75 = np.quantile(tree_vals, [0.25, 0.75])
    for row in normal_rows:
        val = float(row["library_n_trees_total_enum_log"])
        if val <= q25:
            key = "low_tree_count"
        elif val >= q75:
            key = "high_tree_count"
        else:
            key = "middle_tree_count"
        by_tree_bin[key].append(row)

    library = {
        "schema": "normal_fan_tree_count_separation_library.v1",
        "source": "existing exact-degree normal-fan library and its trained 32 groups",
        "n_groups": len(normal_rows),
        "controls": [
            "N_n=5",
            "m=12",
            "N_c=3",
            "D=2",
            "exact in-degree sequence",
            "exact out-degree sequence",
            "full input coupling",
            "d_rel=88",
        ],
        "arms": {
            "arm_A_fixed_tree_count_variable_normal_fan": (
                "not available cleanly in current one-base library because active-tree count and log rooted-tree count are highly correlated"
            ),
            "arm_B_variable_tree_count_matched_normal_fan": (
                "not available cleanly in current one-base library for the same collinearity reason"
            ),
            "arm_C_multi_base_rewire_libraries": "required next; current library uses one base degree sequence",
        },
        "feature_correlations": feature_correlations,
        "tree_count_bins": {
            key: {
                "n_groups": len(rows),
                "mean_icl": mean(float(row["outcome_mean"]) for row in rows),
                "best_icl": mean(float(row["outcome_best"]) for row in rows),
                "active_tree_mean": mean(float(row["capacity_normal_fan_active_tree_count_mean"]) for row in rows),
                "branch_tree_nmi_mean": mean(float(row["capacity_normal_fan_branch_tree_nmi_mean"]) for row in rows),
            }
            for key, rows in sorted(by_tree_bin.items())
        },
        "selected_group_rows": [
            {
                key: row.get(key)
                for key in [
                    "topology_name",
                    "library_n_trees_total_enum",
                    "library_n_trees_total_enum_log",
                    "capacity_normal_fan_active_tree_count_mean",
                    "capacity_normal_fan_branch_tree_nmi_mean",
                    "cross_overlap_norm_min",
                    "mean_seed_novel_icl",
                    "best_seed_novel_icl",
                ]
            }
            for row in normal_rows
        ],
    }

    training = {
        "schema": "normal_fan_tree_count_training_report.v1",
        "status": "existing_training_reanalyzed_no_new_training_launched",
        "n_trained_groups": len(normal_rows),
        "loo_models": loo_models,
        "correlations": [
            correlation(normal_rows, feature, outcome)
            for feature in tree_features + fan_features + cross_features
            for outcome in outcomes
        ],
        "interpretation": {
            "tree_count_vs_normal_fan": (
                "Current one-base exact-degree data cannot separate total rooted-tree abundance from active-tree/normal-fan geometry. "
                "log rooted-tree count and active-tree count are strongly correlated, so their weak positive predictive signals should be treated as a combined geometry/abundance direction."
            ),
            "next_test": (
                "Build multi-base degree-preserving libraries with arms that hold tree count approximately fixed while varying normal-fan coverage and vice versa."
            ),
        },
    }
    return library, training


def write_normal_fan_markdown(library: Mapping[str, Any], training: Mapping[str, Any]) -> None:
    corr_rows = [
        [item["x"], item["y"], item["n"], item["r"]]
        for item in library["feature_correlations"]
        if item["x"] in ("library_n_trees_total_enum_log", "capacity_normal_fan_active_tree_count_mean")
        or item["y"] in ("library_n_trees_total_enum_log", "capacity_normal_fan_active_tree_count_mean")
    ]
    write_text(
        OUT / "normal_fan_tree_count_separation_library.md",
        "\n".join(
            [
                "# Normal-Fan / Tree-Count Separation Library",
                "",
                "## Status",
                "",
                "The existing one-base exact-degree normal-fan library was reanalyzed. It is useful as a diagnostic library but does not cleanly instantiate the requested fixed-tree-count and matched-normal-fan arms.",
                "",
                "## Controls",
                "",
                "\n".join(f"- {item}" for item in library["controls"]),
                "",
                "## Arm Availability",
                "",
                md_table(["arm", "status"], [[key, value] for key, value in library["arms"].items()]),
                "",
                "## Feature Collinearity",
                "",
                md_table(["feature x", "feature y", "n", "r"], corr_rows),
            ]
        ),
    )

    write_text(
        OUT / "normal_fan_tree_count_training_report.md",
        "\n".join(
            [
                "# Normal-Fan / Tree-Count Training Report",
                "",
                "## Status",
                "",
                str(training["status"]),
                "",
                "No new broad training sweep was launched. The existing 32-group exact-degree exact-d_rel full-coupling experiment was reanalyzed because the goal explicitly forbids broad sweeps without a diagnostic reason.",
                "",
                "## LOO Models",
                "",
                md_table(
                    ["outcome", "model", "groups", "predictors", "LOO R2", "reason"],
                    [
                        [
                            row["outcome"],
                            row["model"],
                            row["n_groups"],
                            row["n_predictors"],
                            row.get("loo_r2"),
                            row.get("reason"),
                        ]
                        for row in training["loo_models"]
                    ],
                ),
                "",
                "## Interpretation",
                "",
                training["interpretation"]["tree_count_vs_normal_fan"],
                "",
                training["interpretation"]["next_test"],
            ]
        ),
    )


def build_gamma_report() -> dict[str, Any]:
    fixed = read_json(OUT / "repaired_gamma_existing_data_reanalysis.json")
    prospective = read_json(OUT / "prospective_tree_diff_multiplicity_causal_report.json")
    normal = read_json(OUT / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.json")
    payload = {
        "schema": "gamma_diagnostic_reanalysis_after_exact_controls.v1",
        "status": "diagnostic_not_selector",
        "fixed_m20": fixed.get("interpretation", {}),
        "prospective_exact_control": {
            "mean_gamma_plus_controls_r2": model_r2(prospective, "mean_seed_novel_icl", "gamma_no_bias_plus_controls"),
            "best_gamma_plus_controls_r2": model_r2(prospective, "best_seed_novel_icl", "gamma_no_bias_plus_controls"),
            "branch_failures_gamma_plus_controls_r2": model_r2(prospective, "branch_failures", "gamma_no_bias_plus_controls"),
            "trained_margin_gamma_plus_controls_r2": model_r2(prospective, "trained_branch_margin", "gamma_no_bias_plus_controls"),
        },
        "normal_fan_exact_control": {
            "mean_gamma_exact_r2": model_r2(normal, "mean_seed_novel_icl", "gamma_no_bias_exact"),
            "best_gamma_exact_r2": model_r2(normal, "best_seed_novel_icl", "gamma_no_bias_exact"),
            "mean_gamma_plus_normal_fan_r2": model_r2(normal, "mean_seed_novel_icl", "gamma_plus_normal_fan"),
            "best_gamma_plus_normal_fan_r2": model_r2(normal, "best_seed_novel_icl", "gamma_plus_normal_fan"),
        },
        "interpretation": (
            "Gamma remains a sanity-checked diagnostic. It passed analytic toys, but exact-control trained data do not support using it as a topology selector."
        ),
    }
    return payload


def write_gamma_markdown(report: Mapping[str, Any]) -> None:
    write_text(
        OUT / "gamma_diagnostic_reanalysis_after_exact_controls.md",
        "\n".join(
            [
                "# Gamma Diagnostic Reanalysis After Exact Controls",
                "",
                "## Status",
                "",
                str(report["status"]),
                "",
                "## Key Values",
                "",
                md_table(
                    ["setting", "metric", "value"],
                    [
                        ["fixed_m20", "best gamma mean LOO R2", report["fixed_m20"].get("fixed_m20_best_gamma_mean_loo_r2")],
                        ["fixed_m20", "tree-diff mean LOO R2", report["fixed_m20"].get("fixed_m20_mean_tree_difference_loo_r2")],
                        ["prospective", "mean gamma+controls LOO R2", report["prospective_exact_control"]["mean_gamma_plus_controls_r2"]],
                        ["prospective", "best gamma+controls LOO R2", report["prospective_exact_control"]["best_gamma_plus_controls_r2"]],
                        ["normal_fan", "mean gamma exact LOO R2", report["normal_fan_exact_control"]["mean_gamma_exact_r2"]],
                        ["normal_fan", "best gamma exact LOO R2", report["normal_fan_exact_control"]["best_gamma_exact_r2"]],
                    ],
                ),
                "",
                "## Interpretation",
                "",
                str(report["interpretation"]),
            ]
        ),
    )


def build_mechanism_report() -> dict[str, Any]:
    old = read_json(OUT / "mechanism_and_causal_scramble_followup_report.json")
    prospective = read_json(OUT / "prospective_tree_diff_multiplicity_causal_report.json")
    normal = read_json(OUT / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.json")
    return {
        "schema": "mechanism_followup_after_exact_controls.v1",
        "source": "mechanism_and_causal_scramble_followup_report.json plus exact-control availability audit",
        "prospective_mechanism_available": prospective.get("interpretation", {}).get("mechanism_metrics_available"),
        "normal_fan_mechanism_available": any(
            finite_float(row.get("branch_active_tree_mi")) is not None for row in normal.get("group_rows", [])
        ),
        "prior_scramble_summary": old.get("causal_interventions", old.get("interventions", old)),
        "interpretation": {
            "pre_training_predictor_claim": "same-root tree-difference overlap and gamma are weak or negative under prospective controls",
            "post_training_mechanism_claim": "successful trained models remain branch/projection/tree dependent",
            "causal_intervention_claim": "statistic-preserving scrambles and edge ablations caused large drops in selected prospective models",
            "normal_fan_gap": "normal-fan trained models have accuracy summaries but not the full mechanism/scramble panel yet",
        },
    }


def write_mechanism_markdown(report: Mapping[str, Any]) -> None:
    interventions = report.get("prior_scramble_summary", {}).get("interventions", {})
    rows = []
    if isinstance(interventions, Mapping):
        for name, item in interventions.items():
            if isinstance(item, Mapping):
                rows.append([name, item.get("n"), item.get("accuracy_delta_mean"), item.get("accuracy_delta_min"), item.get("accuracy_delta_max")])
    write_text(
        OUT / "mechanism_followup_after_exact_controls.md",
        "\n".join(
            [
                "# Mechanism Follow-Up After Exact Controls",
                "",
                "## Availability",
                "",
                md_table(
                    ["dataset", "mechanism panel available"],
                    [
                        ["prospective tree-difference control", report["prospective_mechanism_available"]],
                        ["normal-fan exact-degree training", report["normal_fan_mechanism_available"]],
                    ],
                ),
                "",
                "## Prior Prospective Scrambles",
                "",
                md_table(["intervention", "n", "mean accuracy delta", "min", "max"], rows),
                "",
                "## Claim Separation",
                "",
                f"- Pre-training predictor claim: {report['interpretation']['pre_training_predictor_claim']}",
                f"- Post-training mechanism claim: {report['interpretation']['post_training_mechanism_claim']}",
                f"- Causal intervention claim: {report['interpretation']['causal_intervention_claim']}",
                f"- Gap: {report['interpretation']['normal_fan_gap']}",
            ]
        ),
    )


def build_expressivity_trainability_report(
    cross_report: Mapping[str, Any],
    normal_training: Mapping[str, Any],
) -> dict[str, Any]:
    summary = {
        "schema": "expressivity_vs_trainability_after_exact_controls.v1",
        "datasets": {},
        "interpretation": {
            "best_seed": "best-seed ICL remains the closest available expressivity envelope",
            "mean_seed": "mean-seed ICL mixes expressivity and training reliability",
            "seed_std": "current structural metrics generally do not explain seed variance well",
        },
    }
    for ds in cross_report["datasets"]:
        models = ds["model_results"]
        summary["datasets"][ds["dataset"]] = {
            "best_seed_best_model": best_model(models, "outcome_best"),
            "mean_seed_best_model": best_model(models, "outcome_mean"),
            "seed_std_best_model": best_model(models, "outcome_seed_std"),
        }
    summary["normal_fan_tree_count_models"] = {
        "best_seed_best_model": best_model(normal_training["loo_models"], "outcome_best"),
        "mean_seed_best_model": best_model(normal_training["loo_models"], "outcome_mean"),
        "seed_std_best_model": best_model(normal_training["loo_models"], "outcome_seed_std"),
    }
    return summary


def best_model(models: Sequence[Mapping[str, Any]], outcome: str) -> dict[str, Any] | None:
    candidates = [item for item in models if item.get("outcome") == outcome and finite_float(item.get("loo_r2")) is not None]
    if not candidates:
        return None
    return dict(max(candidates, key=lambda item: float(item["loo_r2"])))


def write_expressivity_markdown(report: Mapping[str, Any]) -> None:
    rows = []
    for dataset, item in report["datasets"].items():
        for label, model in [
            ("mean", item.get("mean_seed_best_model")),
            ("best", item.get("best_seed_best_model")),
            ("seed_std", item.get("seed_std_best_model")),
        ]:
            rows.append([dataset, label, model.get("model") if model else None, model.get("loo_r2") if model else None])
    write_text(
        OUT / "expressivity_vs_trainability_after_exact_controls.md",
        "\n".join(
            [
                "# Expressivity vs Trainability After Exact Controls",
                "",
                "## Best Models By Outcome",
                "",
                md_table(["dataset", "target", "best model", "LOO R2"], rows),
                "",
                "## Interpretation",
                "",
                "\n".join(f"- {key}: {value}" for key, value in report["interpretation"].items()),
            ]
        ),
    )


def build_orientation_audit() -> dict[str, Any]:
    status = run(["git", "status", "--short", "--branch"])
    branch = status.splitlines()[0] if status else ""
    commit = run(["git", "rev-parse", "HEAD"])
    untracked = [line for line in status.splitlines()[1:] if line.startswith("?? ")]
    model_counts = {
        "prospective_tree_diff_local_model_pt": len(list((RESULTS / "prospective_tree_diff_multiplicity_training").glob("*/model.pt"))),
        "normal_fan_local_model_pt": len(list((RESULTS / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training").glob("*/model.pt"))),
    }
    fixed = read_json(OUT / "tree_level_multiplicity_reanalysis.json")
    fixed_ds = next(dataset for dataset in fixed["datasets"] if dataset["name"] == "fixed_m20_masks_cluster_topology")
    remote_sample = fixed_ds["groups"][0]["run_dir"] + "/model.pt"
    fixed_remote_model_available = False
    try:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "engaging", "test", "-f", remote_sample],
            check=True,
            timeout=8,
        )
        fixed_remote_model_available = True
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        fixed_remote_model_available = False

    prospective = read_json(OUT / "prospective_tree_diff_multiplicity_causal_report.json")
    normal = read_json(OUT / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.json")
    return {
        "schema": "current_state_orientation_audit.v1",
        "branch": branch,
        "commit": commit,
        "worktree_status": status,
        "untracked_items": untracked,
        "required_artifacts": {key: {"path": str(path.relative_to(ROOT)), "exists": path.exists()} for key, path in REQUIRED_ARTIFACTS.items()},
        "script_paths": {key: {"path": str(path.relative_to(ROOT)), "exists": path.exists()} for key, path in SCRIPT_PATHS.items()},
        "output_locations": {
            "prospective_tree_diff_training_results": str((OUT / "prospective_tree_diff_multiplicity_training_results.csv").relative_to(ROOT)),
            "normal_fan_training_results": str((OUT / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_results.csv").relative_to(ROOT)),
            "prospective_raw_training_dir": str((RESULTS / "prospective_tree_diff_multiplicity_training").relative_to(ROOT)),
            "normal_fan_raw_training_dir": str((RESULTS / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training").relative_to(ROOT)),
        },
        "learned_tensor_availability": {
            **model_counts,
            "fixed_m20_remote_model_pt_sample_available": fixed_remote_model_available,
            "fixed_m20_aggregate_csv_contains_K_tensors": False,
        },
        "branch_failure_and_margin_availability": {
            "prospective_branch_failures": any(finite_float(row.get("branch_failures")) is not None for row in prospective["group_rows"]),
            "prospective_trained_branch_margin": any(
                finite_float(row.get("trained_branch_margin")) is not None for row in prospective["group_rows"]
            ),
            "normal_fan_branch_failures": any(finite_float(row.get("branch_failures")) is not None for row in normal["group_rows"]),
            "normal_fan_trained_branch_margin": any(
                finite_float(row.get("trained_branch_margin")) is not None for row in normal["group_rows"]
            ),
        },
        "status": "ready_for_diagnostic_reanalysis",
    }


def write_orientation_markdown(report: Mapping[str, Any]) -> None:
    write_text(
        OUT / "current_state_orientation_audit.md",
        "\n".join(
            [
                "# Current State Orientation Audit",
                "",
                "## Git State",
                "",
                md_table(
                    ["field", "value"],
                    [
                        ["branch", report["branch"]],
                        ["commit", report["commit"]],
                        ["untracked items", "; ".join(report["untracked_items"])],
                    ],
                ),
                "",
                "## Required Artifacts",
                "",
                md_table(
                    ["artifact", "path", "exists"],
                    [[key, item["path"], item["exists"]] for key, item in report["required_artifacts"].items()],
                ),
                "",
                "## Script Paths",
                "",
                md_table(
                    ["script", "path", "exists"],
                    [[key, item["path"], item["exists"]] for key, item in report["script_paths"].items()],
                ),
                "",
                "## Availability",
                "",
                md_table(
                    ["field", "value"],
                    [[key, value] for key, value in {**report["learned_tensor_availability"], **report["branch_failure_and_margin_availability"]}.items()],
                ),
            ]
        ),
    )


def build_synthesis(
    failure: Mapping[str, Any],
    cross: Mapping[str, Any],
    normal_training: Mapping[str, Any],
    gamma: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    expressivity: Mapping[str, Any],
) -> dict[str, Any]:
    normal_mean_best = best_model(normal_training["loo_models"], "outcome_mean")
    cross_prospective = next(ds for ds in cross["datasets"] if ds["dataset"] == "prospective_tree_diff_exact_control")
    return {
        "schema": "post_exact_control_failure_diagnosis_synthesis.v1",
        "answers": {
            "why_tree_difference_failed": failure["conclusions"]["single_best_explanation"],
            "same_root_saturated_confounded_or_wrong_object": (
                "Not a pure saturation story.  The balanced prospective arm missed the zero-overlap regime but still had separated high/low masks, and the imbalanced arm reached lower overlap only under a load/trainability confound.  Same-root co-participation is therefore too coarse as a standalone knob."
            ),
            "cross_root_improvement": (
                "Cross-root overlap improved the prospective load-stratum-controlled mean-ICL LOO R2 from "
                f"{fmt(model_lookup(cross_prospective['model_results'], 'outcome_mean', 'same_root_tree_diff'))} "
                "for same-root tree-difference to "
                f"{fmt(model_lookup(cross_prospective['model_results'], 'outcome_mean', 'cross_root_overlap'))}, "
                "but high-overlap masks still lost the direct high-low causal contrast.  Treat cross-root metrics as diagnostics, not selectors."
            ),
            "tree_count_vs_normal_fan": normal_training["interpretation"]["tree_count_vs_normal_fan"],
            "best_pretraining_metric": (
                f"current weak direction: {normal_mean_best.get('model') if normal_mean_best else 'none'} for mean ICL, but only within the one-base exact-degree data"
            ),
            "gamma_status": gamma["interpretation"],
            "next_experiment": normal_training["interpretation"]["next_test"],
            "thermodynamics": "No thermodynamic Fmax claim was tested in this phase; it remains untested.",
        },
        "claim_separation": {
            "expressivity": "First-order tree-sum expressivity remains exact; repaired gamma is analytic-toy valid but not yet predictive in trained data.",
            "trainability": "Mean-vs-best and seed-std analyses still show no strong scalar pre-training trainability law.",
            "mechanism": mechanism["interpretation"]["post_training_mechanism_claim"],
            "causal_evidence": "Prospective same-root tree-difference overlap failed as a standalone causal knob; normal-fan/tree-count remains weak and entangled.",
            "thermodynamic_physics": "untested; no Fmax conclusions.",
        },
        "supported": [
            "matrix-tree rooted tree-sum basis is the correct first-order computational basis",
            "post-training branch/projection/tree dependence is strong in selected trained models",
            "same-root tree-difference overlap is not sufficient as a standalone prospective causal control",
            "gamma is repaired on analytic toys but remains diagnostic, not a selector",
        ],
        "weakened": [
            "retrospective fixed-m20 tree-difference multiplicity as a general causal knob",
            "single scalar pre-training topology law",
            "gamma as an immediate trained-performance predictor",
        ],
        "open": [
            "whether cross-root contrast geometry predicts under multi-base exact controls",
            "whether total rooted-tree count can be separated from task-aligned normal-fan coverage",
            "which variables explain seed variance and trainability",
            "thermodynamic force-budget effects in a reversible-support Markov parameterization",
        ],
        "source_reports": [
            "tree_difference_failure_diagnosis.md",
            "cross_root_tree_contrast_reanalysis.md",
            "normal_fan_tree_count_training_report.md",
            "gamma_diagnostic_reanalysis_after_exact_controls.md",
            "mechanism_followup_after_exact_controls.md",
            "expressivity_vs_trainability_after_exact_controls.md",
        ],
    }


def write_synthesis_markdown(report: Mapping[str, Any]) -> None:
    lines = [
        "# Post Exact-Control Failure Diagnosis Synthesis",
        "",
        "## Direct Answers",
        "",
        md_table(["question", "answer"], [[key, value] for key, value in report["answers"].items()]),
        "",
        "## Claim Separation",
        "",
        md_table(["claim type", "status"], [[key, value] for key, value in report["claim_separation"].items()]),
        "",
        "## Supported",
        "",
        "\n".join(f"- {item}" for item in report["supported"]),
        "",
        "## Weakened",
        "",
        "\n".join(f"- {item}" for item in report["weakened"]),
        "",
        "## Open",
        "",
        "\n".join(f"- {item}" for item in report["open"]),
    ]
    write_text(OUT / "post_exact_control_failure_diagnosis_synthesis.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-host", default="engaging")
    parser.add_argument("--max-pairs-per-root-pair", type=int, default=20000)
    args = parser.parse_args()

    orientation = build_orientation_audit()
    write_json(OUT / "current_state_orientation_audit.json", orientation)
    write_orientation_markdown(orientation)

    fixed_rows = load_fixed_m20_rows(args.max_pairs_per_root_pair, args.ssh_host)
    prospective_rows = load_prospective_rows(args.max_pairs_per_root_pair)
    normal_rows = load_normal_fan_rows(args.max_pairs_per_root_pair)

    cross = build_cross_root_reanalysis(fixed_rows, prospective_rows, normal_rows)
    write_json(OUT / "cross_root_tree_contrast_reanalysis.json", cross)
    write_cross_root_markdown(cross)

    failure = build_failure_diagnosis(fixed_rows, prospective_rows)
    write_json(OUT / "tree_difference_failure_diagnosis.json", failure)
    write_failure_markdown(failure)

    normal_library, normal_training = build_normal_fan_reports(normal_rows)
    write_json(OUT / "normal_fan_tree_count_separation_library.json", normal_library)
    write_json(OUT / "normal_fan_tree_count_training_report.json", normal_training)
    write_normal_fan_markdown(normal_library, normal_training)

    gamma = build_gamma_report()
    write_json(OUT / "gamma_diagnostic_reanalysis_after_exact_controls.json", gamma)
    write_gamma_markdown(gamma)

    mechanism = build_mechanism_report()
    write_json(OUT / "mechanism_followup_after_exact_controls.json", mechanism)
    write_mechanism_markdown(mechanism)

    expressivity = build_expressivity_trainability_report(cross, normal_training)
    write_json(OUT / "expressivity_vs_trainability_after_exact_controls.json", expressivity)
    write_expressivity_markdown(expressivity)

    synthesis = build_synthesis(failure, cross, normal_training, gamma, mechanism, expressivity)
    write_json(OUT / "post_exact_control_failure_diagnosis_synthesis.json", synthesis)
    write_synthesis_markdown(synthesis)

    print(f"wrote exact-control failure diagnosis reports to {OUT}")


if __name__ == "__main__":
    main()
