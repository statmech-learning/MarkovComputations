"""Existing-data reanalysis with repaired no-bias gamma*_ICL probes."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from branch_margin_capacity_v2 import lower_tail_capacity_probe


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "ICL" / "results" / "next_phase_stats"
TREE_REPORT = OUT_DIR / "tree_level_multiplicity_reanalysis.json"
FIXED_TOPOLOGY_CSV = OUT_DIR / "pooled_fixed_m20_topology_results.csv"
GAMMA_ROWS_JSON = OUT_DIR / "repaired_gamma_existing_data_gamma_rows.json"
GAMMA_ROWS_CSV = OUT_DIR / "repaired_gamma_existing_data_gamma_rows.csv"
REPORT_JSON = OUT_DIR / "repaired_gamma_existing_data_reanalysis.json"
REPORT_MD = OUT_DIR / "repaired_gamma_existing_data_reanalysis.md"

GAMMA_PARAMS = {
    "n_samples": 240,
    "trials": 8,
    "alpha": 0.10,
    "projection_radius": 1.0,
    "decoder_radius": 1.0,
    "edge_bias_radius": 0.0,
    "max_root_assignments": 12,
    "seed_base": 771,
}

OUTCOMES = [
    "mean_novel_icl",
    "best_seed_novel_icl",
    "seed_std_novel_icl",
    "branch_failure_percent",
    "trained_branch_margin",
]

KNOWN_FAMILY_PREFIXES = [
    "bottleneck_bridge",
    "degree_balanced",
    "redundant_paths",
    "cycle_chords",
    "directed_cycle",
    "bidirected_cycle",
    "two_module",
    "hub_spoke",
    "random_sc",
    "complete",
]

MODEL_SPECS = {
    "raw_count_structural": ["raw_physical_parameter_count"],
    "raw_plus_drel_structural": ["raw_physical_parameter_count", "d_rel"],
    "masked_tree_geometry_structural": [
        "input_coupled_parameter_count",
        "d_rel",
        "comparison_branch_common_d_rel_min",
        "comparison_branch_common_d_rel_gini",
        "comparison_branch_d_rel_min",
        "comparison_branch_d_rel_gini",
        "effective_rank_D_masked",
        "condition_number_D_masked_log10",
        "input_edge_load_gini",
        "input_coord_load_gini",
    ],
    "tree_geometry_structural_full": [
        "raw_physical_parameter_count",
        "d_rel",
        "effective_rank_D",
        "root_tree_count_gini",
        "edge_participation_gini",
        "bottleneck_edge_fraction_095",
    ],
    "tree_geometry_markov_reanalysis_subset": [
        "comparison_branch_common_d_rel_min",
        "comparison_branch_common_d_rel_mean",
        "effective_rank_D_masked",
        "condition_number_D_masked_log10",
    ],
    "edge_multiplicity_markov_reanalysis": [
        "edge_M_mean",
        "edge_M_gini",
        "edge_overlap_norm_min",
        "edge_overlap_norm_mean",
        "edge_comparison_imbalance_mean",
    ],
    "tree_level_multiplicity": [
        "tree_overlap_norm_min",
        "tree_overlap_norm_mean",
        "tree_coord_load_gini",
        "tree_active_fraction_mean",
        "tree_count_log",
    ],
    "tree_difference_multiplicity": [
        "diff_overlap_norm_min",
        "diff_overlap_norm_mean",
        "diff_coord_load_gini",
        "diff_pair_count_log",
    ],
    "repaired_gamma_no_bias_exact": [
        "gamma_no_bias_exact_lcvar",
        "gamma_no_bias_exact_accuracy",
        "gamma_no_bias_exact_failure_max",
    ],
    "repaired_gamma_no_bias_tropical": [
        "gamma_no_bias_tropical_lcvar",
        "gamma_no_bias_tropical_accuracy",
        "gamma_no_bias_tropical_failure_max",
    ],
    "repaired_gamma_no_bias_hard_root": [
        "gamma_no_bias_hard_root_lcvar",
        "gamma_no_bias_hard_root_accuracy",
        "gamma_no_bias_hard_root_failure_max",
    ],
    "gamma_no_bias_lcvar_all_variants": [
        "gamma_no_bias_exact_lcvar",
        "gamma_no_bias_tropical_lcvar",
        "gamma_no_bias_hard_root_lcvar",
    ],
    "gamma_no_bias_plus_tree_difference_multiplicity": [
        "gamma_no_bias_exact_lcvar",
        "gamma_no_bias_tropical_lcvar",
        "gamma_no_bias_hard_root_lcvar",
        "diff_overlap_norm_min",
        "diff_overlap_norm_mean",
        "diff_coord_load_gini",
    ],
}


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if math.isfinite(value) else "NA"
    return str(value)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def family_name(name: str) -> str:
    for prefix in KNOWN_FAMILY_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return name.split("_n", 1)[0]


TEXT_COLUMNS = {
    "group",
    "group_by",
    "run_dir",
    "label",
    "labels",
    "seeds",
    "target",
    "topology_name",
    "physical_topology_name",
    "input_mask_name",
    "input_mask_family",
    "comparison_branch_common_d_rel_source",
    "comparison_branch_input_overlap_source",
}


def parse_csv_value(key: str, value: Any) -> Any:
    if key in TEXT_COLUMNS or key.endswith("_path") or key.endswith("_dir"):
        return value
    return parse_float(value)


def topology_controls_from_csv(path: Path, key: str = "topology_name") -> dict[str, dict[str, Any]]:
    controls = {}
    for row in read_csv(path):
        group = row.get(key)
        if not group or group in controls:
            continue
        controls[group] = {k: parse_csv_value(k, v) for k, v in row.items()}
    return controls


def hard_controls() -> dict[str, dict[str, Any]]:
    controls = {}
    for path in (REPO_ROOT / "ICL" / "results" / "expanded_hard_sweeps").glob("*/topology_seed_aggregates.csv"):
        for row in read_csv(path):
            group = row.get("topology_name") or row.get("group")
            if not group:
                continue
            out = {k: parse_csv_value(k, v) for k, v in row.items()}
            out["regime"] = path.parent.name
            controls[group] = out
    return controls


def load_tree_datasets() -> dict[str, list[dict[str, Any]]]:
    report = json.loads(TREE_REPORT.read_text())
    fixed_controls = topology_controls_from_csv(FIXED_TOPOLOGY_CSV)
    hard_by_group = hard_controls()
    datasets: dict[str, list[dict[str, Any]]] = {}
    for dataset in report["datasets"]:
        name = dataset["name"]
        rows = []
        for group in dataset["groups"]:
            row = dict(group)
            row["dataset"] = name
            row["mask_group"] = row["group"]
            row["family"] = family_name(str(row["physical_topology_name"]))
            if name == "fixed_m20_masks_cluster_topology":
                row.update(fixed_controls.get(row["group"], {}))
                row["regime"] = "fixed_m20"
            elif name == "hard_full_mask_local":
                row.update(hard_by_group.get(row["group"], {}))
                row["regime"] = row.get("regime") or f"n{row.get('n_nodes')}_m{row.get('n_edges')}"
            if parse_float(row.get("condition_number_D_masked")) is not None:
                row["condition_number_D_masked_log10"] = math.log10(float(row["condition_number_D_masked"]))
            rows.append(row)
        datasets[name] = rows
    return datasets


def localize_run_dir(path: str) -> Path:
    prefix = "/home/aadarwal/repos/topology/"
    if path.startswith(prefix):
        return REPO_ROOT / path[len(prefix) :]
    return Path(path)


def load_topology_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = str(row["run_dir"])
    local = localize_run_dir(run_dir) / "topology.json"
    if local.exists():
        return json.loads(local.read_text())
    remote_path = str(Path(run_dir) / "topology.json")
    data = subprocess.check_output(["ssh", "engaging", "cat", remote_path], text=True)
    return json.loads(data)


def gamma_summary_for_group(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = load_topology_payload(row)
    n_context = int(float(row.get("n_context") or (payload.get("input_mask_metadata", {}).get("p", 0))))
    z_dim = int(float(row.get("z_dim") or 1))
    if n_context <= 0 or z_dim <= 0:
        p = len(payload["input_mask"][0])
        z_dim = int(float(row.get("z_dim") or 1))
        n_context = p // z_dim - 1
    mask = np.asarray(payload["input_mask"], dtype=float)
    out: dict[str, Any] = {
        "dataset": row["dataset"],
        "group": row["group"],
        "physical_topology_name": row["physical_topology_name"],
        "family": row.get("family"),
        "regime": row.get("regime"),
        "gamma_probe_n_samples": GAMMA_PARAMS["n_samples"],
        "gamma_probe_trials": GAMMA_PARAMS["trials"],
        "gamma_probe_edge_bias_radius": GAMMA_PARAMS["edge_bias_radius"],
        "gamma_probe_alpha": GAMMA_PARAMS["alpha"],
    }
    for idx, variant in enumerate(["exact", "tropical", "hard_root"]):
        prefix = f"gamma_no_bias_{variant}"
        group_hash = int(hashlib.sha256(str(row["group"]).encode("utf-8")).hexdigest()[:8], 16)
        seed = GAMMA_PARAMS["seed_base"] + 101 * idx + group_hash % 10000
        try:
            result = lower_tail_capacity_probe(
                n_nodes=int(payload["n_nodes"]),
                edges=payload["edges"],
                n_context=n_context,
                z_dim=z_dim,
                input_mask=mask,
                variant=variant,
                n_samples=GAMMA_PARAMS["n_samples"],
                trials=GAMMA_PARAMS["trials"],
                seed=seed,
                alpha=GAMMA_PARAMS["alpha"],
                projection_radius=GAMMA_PARAMS["projection_radius"],
                decoder_radius=GAMMA_PARAMS["decoder_radius"],
                edge_bias_radius=GAMMA_PARAMS["edge_bias_radius"],
                max_root_assignments=GAMMA_PARAMS["max_root_assignments"],
            )
            best = result["best"]
            out.update(
                {
                    f"{prefix}_available": True,
                    f"{prefix}_lcvar": best.get("branch_margin_lcvar_min"),
                    f"{prefix}_accuracy": best.get("accuracy"),
                    f"{prefix}_failure_max": best.get("branch_failure_rate_max"),
                    f"{prefix}_margin_mean": best.get("margin_mean"),
                    f"{prefix}_margin_p10": best.get("margin_p10"),
                    f"{prefix}_tree_drive_range_mean": best.get("tree_drive_range_mean"),
                    f"{prefix}_n_trees_total": result.get("n_trees_total"),
                    f"{prefix}_total_trials": result.get("total_trials"),
                }
            )
        except Exception as exc:  # pragma: no cover - reported as unavailable.
            out.update({f"{prefix}_available": False, f"{prefix}_error": str(exc)})
    return out


def load_or_compute_gamma_rows(datasets: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    all_rows = [row for rows in datasets.values() for row in rows]
    expected = {(row["dataset"], row["group"]) for row in all_rows}
    if GAMMA_ROWS_JSON.exists():
        cached = json.loads(GAMMA_ROWS_JSON.read_text())
        if cached.get("gamma_params") == GAMMA_PARAMS:
            rows = cached.get("rows", [])
            cached_keys = {(row["dataset"], row["group"]) for row in rows}
            if expected.issubset(cached_keys):
                return rows
    rows = []
    for idx, row in enumerate(all_rows, start=1):
        print(f"[gamma] {idx}/{len(all_rows)} {row['dataset']} {row['group']}", flush=True)
        rows.append(gamma_summary_for_group(row))
    payload = {"schema": "repaired_gamma_existing_data_gamma_rows.v1", "gamma_params": GAMMA_PARAMS, "rows": rows}
    GAMMA_ROWS_JSON.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    write_csv(GAMMA_ROWS_CSV, rows)
    return rows


def merge_gamma(datasets: Mapping[str, list[dict[str, Any]]], gamma_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index = {(row["dataset"], row["group"]): row for row in gamma_rows}
    merged = {}
    for name, rows in datasets.items():
        out = []
        for row in rows:
            joined = dict(row)
            joined.update(index.get((row["dataset"], row["group"]), {}))
            out.append(joined)
        merged[name] = out
    return merged


def complete_rows(rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], outcome: str) -> tuple[np.ndarray, np.ndarray, list[Mapping[str, Any]]]:
    xs = []
    ys = []
    used = []
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
            used.append(row)
    if not xs:
        return np.zeros((0, len(predictors))), np.zeros(0), []
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), used


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    x_train = (train_x - center) / scale
    x_test = (test_x - center) / scale
    A = np.column_stack([np.ones(x_train.shape[0]), x_train])
    penalty = math.sqrt(1e-6) * np.eye(A.shape[1])
    penalty[0, 0] = 0.0
    coef, *_ = np.linalg.lstsq(
        np.vstack([A, penalty]),
        np.concatenate([train_y, np.zeros(A.shape[1])]),
        rcond=None,
    )
    return np.column_stack([np.ones(x_test.shape[0]), x_test]) @ coef


def loo_r2(rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], outcome: str) -> dict[str, Any]:
    X, y, used = complete_rows(rows, predictors, outcome)
    n, p = X.shape
    result = {"predictors": list(predictors), "outcome": outcome, "n_groups": int(n)}
    if n < max(8, p + 3):
        result.update({"loo_r2": None, "reason": "too_few_groups_or_complete_cases"})
        return result
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 1e-12:
        result.update({"loo_r2": None, "reason": "constant_outcome"})
        return result
    preds = []
    for i in range(n):
        train = np.arange(n) != i
        preds.append(float(fit_predict(X[train], y[train], X[i : i + 1])[0]))
    err = float(np.sum((np.asarray(preds) - y) ** 2))
    result.update({"loo_r2": float(1.0 - err / denom), "used_groups": [r["group"] for r in used]})
    return result


def heldout_category_r2(rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], outcome: str, category: str) -> dict[str, Any]:
    X, y, used = complete_rows(rows, predictors, outcome)
    n, p = X.shape
    result = {"predictors": list(predictors), "outcome": outcome, "category": category, "n_groups": int(n)}
    if n < max(8, p + 3):
        result.update({"heldout_r2": None, "reason": "too_few_groups_or_complete_cases"})
        return result
    cats = np.asarray([str(row.get(category, "")) for row in used])
    unique = sorted(set(cats))
    if len(unique) < 2:
        result.update({"heldout_r2": None, "reason": "too_few_categories"})
        return result
    preds = np.full(n, np.nan)
    for cat in unique:
        test = cats == cat
        train = ~test
        if train.sum() < max(5, p + 2):
            continue
        preds[test] = fit_predict(X[train], y[train], X[test])
    valid = np.isfinite(preds)
    if valid.sum() < 3:
        result.update({"heldout_r2": None, "reason": "too_few_valid_predictions"})
        return result
    denom = float(np.sum((y[valid] - y[valid].mean()) ** 2))
    err = float(np.sum((preds[valid] - y[valid]) ** 2))
    result.update({"heldout_r2": float(1.0 - err / denom), "n_categories": len(unique), "n_predicted": int(valid.sum())})
    return result


def model_results(rows: Sequence[Mapping[str, Any]], outcomes: Sequence[str]) -> list[dict[str, Any]]:
    results = []
    for outcome in outcomes:
        for model, predictors in MODEL_SPECS.items():
            item = loo_r2(rows, predictors, outcome)
            item["model"] = model
            results.append(item)
    return results


def strict_drel_subset(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    dvals = [parse_float(row.get("d_rel")) for row in rows]
    finite = [v for v in dvals if v is not None]
    if not finite:
        return []
    counts: dict[float, int] = defaultdict(int)
    for value in finite:
        counts[value] += 1
    mode = max(counts, key=counts.get)
    return [row for row in rows if parse_float(row.get("d_rel")) == mode]


def cluster_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    baseline: str,
    outcome: str,
    cluster_col: str,
    n_boot: int = 300,
    seed: int = 19,
) -> dict[str, Any]:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row.get(cluster_col, ""))].append(row)
    names = sorted(clusters)
    if len(names) < 3:
        return {"delta_loo_r2_mean": None, "ci95": [None, None], "reason": "too_few_clusters", "n_clusters": len(names)}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        sampled = []
        for name in rng.choice(names, size=len(names), replace=True):
            sampled.extend(clusters[str(name)])
        base = loo_r2(sampled, MODEL_SPECS[baseline], outcome).get("loo_r2")
        cand = loo_r2(sampled, MODEL_SPECS[candidate], outcome).get("loo_r2")
        if base is not None and cand is not None:
            deltas.append(float(cand) - float(base))
    if not deltas:
        return {"delta_loo_r2_mean": None, "ci95": [None, None], "reason": "no_valid_resamples", "n_clusters": len(names)}
    return {
        "candidate": candidate,
        "baseline": baseline,
        "outcome": outcome,
        "cluster_col": cluster_col,
        "n_clusters": len(names),
        "n_bootstrap_resamples": len(deltas),
        "delta_loo_r2_mean": float(np.mean(deltas)),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
    }


def residualized_correlations(rows: Sequence[Mapping[str, Any]], outcome: str, group_col: str = "regime") -> list[dict[str, Any]]:
    features = [
        "diff_overlap_norm_min",
        "gamma_no_bias_exact_lcvar",
        "gamma_no_bias_tropical_lcvar",
        "gamma_no_bias_hard_root_lcvar",
        "gamma_no_bias_exact_accuracy",
    ]
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row.get(group_col, ""))].append(row)
    out = []
    for feature in features:
        xs = []
        ys = []
        for members in by_group.values():
            pairs = [(parse_float(row.get(feature)), parse_float(row.get(outcome))) for row in members]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if len(pairs) < 2:
                continue
            mx = mean([x for x, _ in pairs])
            my = mean([y for _, y in pairs])
            for x, y in pairs:
                xs.append(x - mx)
                ys.append(y - my)
        if len(xs) < 3:
            corr = None
        else:
            x = np.asarray(xs)
            y = np.asarray(ys)
            denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
            corr = None if denom <= 1e-12 else float(np.sum(x * y) / denom)
        out.append({"outcome": outcome, "feature": feature, "group_col": group_col, "n": len(xs), "correlation": corr})
    return out


def summarize_dataset(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = [outcome for outcome in OUTCOMES if any(parse_float(row.get(outcome)) is not None for row in rows)]
    primary = model_results(rows, outcomes)
    strict_rows = strict_drel_subset(rows)
    strict = model_results(strict_rows, outcomes) if strict_rows else []
    category_col = "physical_topology_name" if name == "fixed_m20_masks_cluster_topology" else "family"
    heldout = []
    for outcome in outcomes:
        for model in [
            "tree_difference_multiplicity",
            "repaired_gamma_no_bias_exact",
            "repaired_gamma_no_bias_tropical",
            "repaired_gamma_no_bias_hard_root",
            "gamma_no_bias_plus_tree_difference_multiplicity",
        ]:
            heldout.append(heldout_category_r2(rows, MODEL_SPECS[model], outcome, category_col) | {"model": model})
    bootstrap = []
    for outcome in [out for out in outcomes if out in {"mean_novel_icl", "best_seed_novel_icl"}]:
        for candidate in [
            "repaired_gamma_no_bias_exact",
            "repaired_gamma_no_bias_tropical",
            "repaired_gamma_no_bias_hard_root",
            "gamma_no_bias_plus_tree_difference_multiplicity",
        ]:
            bootstrap.append(cluster_bootstrap_delta(rows, candidate, "tree_difference_multiplicity", outcome, category_col))
    return {
        "dataset": name,
        "n_groups": len(rows),
        "outcomes": outcomes,
        "primary_grouped_loo": primary,
        "strict_drel_subset": {
            "n_groups": len(strict_rows),
            "d_rel": parse_float(strict_rows[0].get("d_rel")) if strict_rows else None,
            "grouped_loo": strict,
        },
        "heldout_category": {"category_col": category_col, "models": heldout},
        "cluster_bootstrap_delta_vs_tree_difference": bootstrap,
        "regime_residualized_correlations": (
            residualized_correlations(rows, "mean_novel_icl", "regime") if name == "hard_full_mask_local" else []
        ),
    }


def model_lookup(results: Sequence[Mapping[str, Any]], model: str, outcome: str) -> Mapping[str, Any] | None:
    for item in results:
        if item.get("model") == model and item.get("outcome") == outcome:
            return item
    return None


def interpretation(analyses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fixed = next((item for item in analyses if item["dataset"] == "fixed_m20_masks_cluster_topology"), None)
    hard = next((item for item in analyses if item["dataset"] == "hard_full_mask_local"), None)
    out = {
        "gamma_selector_gate": "not_cleared_for_large_sweeps",
        "reason": "No repaired gamma model is allowed as a broad selector unless it improves over existing structural metrics.",
    }
    if fixed:
        primary = fixed["primary_grouped_loo"]
        tree = model_lookup(primary, "tree_difference_multiplicity", "mean_novel_icl")
        best_gamma = None
        for model in [
            "repaired_gamma_no_bias_exact",
            "repaired_gamma_no_bias_tropical",
            "repaired_gamma_no_bias_hard_root",
            "gamma_no_bias_plus_tree_difference_multiplicity",
        ]:
            item = model_lookup(primary, model, "mean_novel_icl")
            if item and item.get("loo_r2") is not None:
                if best_gamma is None or item["loo_r2"] > best_gamma["loo_r2"]:
                    best_gamma = item
        out["fixed_m20_mean_tree_difference_loo_r2"] = tree.get("loo_r2") if tree else None
        out["fixed_m20_best_gamma_model"] = best_gamma.get("model") if best_gamma else None
        out["fixed_m20_best_gamma_mean_loo_r2"] = best_gamma.get("loo_r2") if best_gamma else None
        if tree and best_gamma and best_gamma["loo_r2"] > tree["loo_r2"] + 0.02:
            out["gamma_selector_gate"] = "candidate_only"
            out["reason"] = "A gamma-containing model improves mean ICL LOO over tree-difference on fixed-m20, but prospective exact-control validation is still required."
    if hard:
        out["hard_full_mask_has_branch_failure_and_trained_margin"] = True
    return out


def write_md(payload: Mapping[str, Any]) -> None:
    lines = [
        "# Repaired Gamma Existing-Data Reanalysis",
        "",
        "## Status",
        "",
        "No new training was launched. This report recomputes no-bias exact, tropical, and hard-root lower-tail gamma probes for existing topology/mask groups and compares them to existing structural metrics.",
        "",
        "## Gamma Probe Settings",
        "",
    ]
    for key, value in GAMMA_PARAMS.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Grouped LOO Summary", ""])
    for analysis in payload["analyses"]:
        lines.append(f"### {analysis['dataset']}")
        rows = []
        for outcome in analysis["outcomes"]:
            for model in [
                "raw_plus_drel_structural",
                "masked_tree_geometry_structural",
                "tree_geometry_structural_full",
                "tree_geometry_markov_reanalysis_subset",
                "edge_multiplicity_markov_reanalysis",
                "tree_level_multiplicity",
                "tree_difference_multiplicity",
                "repaired_gamma_no_bias_exact",
                "repaired_gamma_no_bias_tropical",
                "repaired_gamma_no_bias_hard_root",
                "gamma_no_bias_plus_tree_difference_multiplicity",
            ]:
                item = model_lookup(analysis["primary_grouped_loo"], model, outcome)
                if item:
                    rows.append([outcome, model, item["n_groups"], fmt(item.get("loo_r2")), item.get("reason", "NA")])
        lines.extend([
            "",
            markdown_table(rows, ["outcome", "model", "groups", "LOO R2", "reason"]),
            "",
            f"Strict modal d_rel subset: `{analysis['strict_drel_subset']['d_rel']}` with `{analysis['strict_drel_subset']['n_groups']}` groups.",
        ])
        strict_rows = []
        for item in analysis["strict_drel_subset"]["grouped_loo"]:
            if item["outcome"] in {"mean_novel_icl", "best_seed_novel_icl"} and item["model"] in {
                "tree_difference_multiplicity",
                "repaired_gamma_no_bias_exact",
                "repaired_gamma_no_bias_tropical",
                "repaired_gamma_no_bias_hard_root",
                "gamma_no_bias_plus_tree_difference_multiplicity",
            }:
                strict_rows.append([item["outcome"], item["model"], item["n_groups"], fmt(item.get("loo_r2")), item.get("reason", "NA")])
        lines.append(markdown_table(strict_rows, ["outcome", "model", "groups", "LOO R2", "reason"]))
        lines.append("")
    lines.extend(
        [
            "## Key Answers",
            "",
            f"- Gamma selector gate: `{payload['interpretation']['gamma_selector_gate']}`.",
            f"- Reason: {payload['interpretation']['reason']}",
            "- `gamma_with_bias` was not computed and is not used for no-bias capacity claims.",
            "- Fixed-m20 branch failures and trained branch margins remain unavailable in aggregate artifacts; hard full-mask groups include those outcomes.",
            "",
            "## Interpretation",
            "",
            "Treat repaired gamma as a diagnostic predictor unless it survives the prospective exact-control phase. This reanalysis is existing-data only and does not replace the Track 2 causal mask experiment.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def markdown_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def main() -> None:
    datasets = load_tree_datasets()
    gamma_rows = load_or_compute_gamma_rows(datasets)
    merged = merge_gamma(datasets, gamma_rows)
    analyses = [summarize_dataset(name, rows) for name, rows in merged.items()]
    payload = {
        "schema": "repaired_gamma_existing_data_reanalysis.v1",
        "source_artifacts": [str(TREE_REPORT), str(FIXED_TOPOLOGY_CSV), str(GAMMA_ROWS_JSON)],
        "gamma_params": GAMMA_PARAMS,
        "model_specs": MODEL_SPECS,
        "datasets": {
            name: {
                "n_groups": len(rows),
                "outcomes_available": [outcome for outcome in OUTCOMES if any(parse_float(row.get(outcome)) is not None for row in rows)],
            }
            for name, rows in merged.items()
        },
        "analyses": analyses,
        "interpretation": interpretation(analyses),
    }
    REPORT_JSON.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    write_md(payload)


if __name__ == "__main__":
    main()
