"""Scale report for exact-degree / exact-d_rel / exact-multiplicity normal-fan control."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from branch_margin_capacity_v2 import lower_tail_capacity_probe


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "ICL" / "results"
NEXT_PHASE_DIR = RESULT_ROOT / "next_phase_stats"
NF_DIR = NEXT_PHASE_DIR / "degree_rewire_normal_fan_n5_m12_N3_D2"
LIBRARY_CSV = NF_DIR / "library.csv"
SELECTED_CSV = NF_DIR / "selected.csv"
CAPACITY_CSV = NF_DIR / "branch_margin_capacity.csv"
TRAINING_CSV = NEXT_PHASE_DIR / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_results.csv"
GAMMA_JSON = NF_DIR / "repaired_gamma_no_bias_rows.json"
LIBRARY_MD = NEXT_PHASE_DIR / "exact_degree_exact_drel_exact_multiplicity_normal_fan_library.md"
TRAINING_MD = NEXT_PHASE_DIR / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.md"
TRAINING_JSON = NEXT_PHASE_DIR / "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.json"

GAMMA_PARAMS = {
    "n_samples": 240,
    "trials": 8,
    "alpha": 0.1,
    "projection_radius": 1.0,
    "decoder_radius": 1.0,
    "edge_bias_radius": 0.0,
    "max_root_assignments": 12,
    "seed_base": 2301,
}

PREDICTOR_MODELS = {
    "active_tree_count": ["capacity_normal_fan_active_tree_count_mean"],
    "branch_tree_nmi": ["capacity_normal_fan_branch_tree_nmi_mean"],
    "normal_fan_pair": [
        "capacity_normal_fan_active_tree_count_mean",
        "capacity_normal_fan_branch_tree_nmi_mean",
    ],
    "tree_count": ["library_n_trees_total_enum_log"],
    "conditioning": ["library_condition_number_D_log"],
    "edge_participation": ["library_edge_participation_gini"],
    "gamma_no_bias_exact": ["gamma_no_bias_exact_lcvar"],
    "gamma_no_bias_tropical": ["gamma_no_bias_tropical_lcvar"],
    "gamma_plus_normal_fan": [
        "gamma_no_bias_exact_lcvar",
        "capacity_normal_fan_active_tree_count_mean",
        "capacity_normal_fan_branch_tree_nmi_mean",
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


def mean(values: Iterable[float | None]) -> float | None:
    arr = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(arr)) if arr else None


def maximum(values: Iterable[float | None]) -> float | None:
    arr = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.max(arr)) if arr else None


def std(values: Iterable[float | None]) -> float | None:
    arr = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.std(arr)) if arr else None


def min_max(values: Iterable[float | None]) -> list[float | None]:
    arr = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return [float(np.min(arr)), float(np.max(arr))] if arr else [None, None]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 3 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_library() -> dict[str, dict[str, Any]]:
    rows = read_csv(LIBRARY_CSV)
    out = {}
    for row in rows:
        parsed = dict(row)
        for key, value in list(parsed.items()):
            numeric = parse_float(value)
            if numeric is not None:
                parsed[key] = numeric
        out[str(parsed["topology_name"])] = parsed
    return out


def load_capacity() -> dict[str, dict[str, Any]]:
    rows = read_csv(CAPACITY_CSV)
    out = {}
    for row in rows:
        parsed = {f"capacity_{key}": value for key, value in row.items()}
        for key, value in list(parsed.items()):
            numeric = parse_float(value)
            if numeric is not None:
                parsed[key] = numeric
        out[str(row["topology_name"])] = parsed
    return out


def load_edge_payload(edge_json: str) -> tuple[int, list[list[int]]]:
    path = Path(edge_json)
    if not path.exists() and str(path).startswith("/home/aadarwal/repos/topology/"):
        path = REPO_ROOT / str(path)[len("/home/aadarwal/repos/topology/") :]
    payload = json.loads(path.read_text())
    return int(payload["n_nodes"]), payload["edges"]


def load_or_compute_gamma(library: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if GAMMA_JSON.exists():
        payload = json.loads(GAMMA_JSON.read_text())
        if payload.get("gamma_params") == GAMMA_PARAMS:
            return {row["topology_name"]: row for row in payload.get("rows", [])}
    rows = []
    for idx, (topology_name, row) in enumerate(sorted(library.items()), start=1):
        n_nodes, edges = load_edge_payload(str(row["edge_json"]))
        item: dict[str, Any] = {"topology_name": topology_name}
        for variant_idx, variant in enumerate(["exact", "tropical", "hard_root"]):
            prefix = f"gamma_no_bias_{variant}"
            result = lower_tail_capacity_probe(
                n_nodes=n_nodes,
                edges=edges,
                n_context=3,
                z_dim=2,
                input_mask=None,
                variant=variant,
                n_samples=GAMMA_PARAMS["n_samples"],
                trials=GAMMA_PARAMS["trials"],
                seed=GAMMA_PARAMS["seed_base"] + 101 * variant_idx + idx,
                alpha=GAMMA_PARAMS["alpha"],
                projection_radius=GAMMA_PARAMS["projection_radius"],
                decoder_radius=GAMMA_PARAMS["decoder_radius"],
                edge_bias_radius=GAMMA_PARAMS["edge_bias_radius"],
                max_root_assignments=GAMMA_PARAMS["max_root_assignments"],
            )
            item[f"{prefix}_lcvar"] = result["best"].get("branch_margin_lcvar_min")
            item[f"{prefix}_accuracy"] = result["best"].get("accuracy")
        rows.append(item)
    GAMMA_JSON.write_text(
        json.dumps(json_ready({"schema": "normal_fan_repaired_gamma_no_bias_rows.v1", "gamma_params": GAMMA_PARAMS, "rows": rows}), indent=2, sort_keys=True)
        + "\n"
    )
    return {row["topology_name"]: row for row in rows}


def training_groups() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    library = load_library()
    capacity = load_capacity()
    gamma = load_or_compute_gamma(library)
    if not TRAINING_CSV.exists():
        return [], {
            "status": "missing_training_csv",
            "training_csv": str(TRAINING_CSV.relative_to(REPO_ROOT)),
            "expected_groups": len(library),
            "expected_runs": len(library) * 5,
        }
    rows = read_csv(TRAINING_CSV)
    topology_id_to_name = {
        str(row.get("topology_id")): str(name)
        for name, row in library.items()
        if row.get("topology_id") not in (None, "")
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("topology_name") or "")
        label = str(row.get("label") or "")
        if key not in library and label:
            for topology_id, topology_name in topology_id_to_name.items():
                if label.startswith(f"{topology_id}_trainseed"):
                    key = topology_name
                    break
        if key:
            grouped[key].append(row)
    out = []
    for key, members in sorted(grouped.items()):
        if key not in library:
            continue
        novel = [parse_float(row.get("test_novel_classes")) for row in members]
        group = {
            "group": key,
            "n_runs": len(members),
            "mean_seed_novel_icl": mean(novel),
            "best_seed_novel_icl": maximum(novel),
            "seed_std_novel_icl": std(novel),
            **{f"library_{k}": v for k, v in library[key].items()},
            **capacity.get(key, {}),
            **gamma.get(key, {}),
        }
        out.append(group)
    status = {
        "status": "complete" if len(rows) >= len(library) * 5 else "partial",
        "training_csv": str(TRAINING_CSV.relative_to(REPO_ROOT)),
        "training_rows": len(rows),
        "expected_runs": len(library) * 5,
        "groups_with_results": len(out),
        "expected_groups": len(library),
    }
    return out, status


def design_matrix(rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], outcome: str) -> tuple[np.ndarray, np.ndarray]:
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


def loo_r2(rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], outcome: str) -> dict[str, Any]:
    X, y = design_matrix(rows, predictors, outcome)
    if X.shape[0] < 8:
        return {"loo_r2": None, "n_groups": int(X.shape[0]), "reason": "too_few_groups"}
    if np.std(y) <= 1.0e-12:
        return {"loo_r2": None, "n_groups": int(X.shape[0]), "reason": "constant_outcome"}
    preds = []
    for idx in range(X.shape[0]):
        keep = np.ones(X.shape[0], dtype=bool)
        keep[idx] = False
        preds.append(ridge_predict(X[keep], y[keep], X[idx]))
    preds = np.asarray(preds)
    sse = float(np.sum((y - preds) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"loo_r2": float(1.0 - sse / sst), "n_groups": int(X.shape[0]), "predictors": list(predictors)}


def correlations(rows: Sequence[Mapping[str, Any]], outcome: str) -> list[dict[str, Any]]:
    names = sorted({name for predictors in PREDICTOR_MODELS.values() for name in predictors})
    out = []
    for name in names:
        pairs = [
            (parse_float(row.get(name)), parse_float(row.get(outcome)))
            for row in rows
            if parse_float(row.get(name)) is not None and parse_float(row.get(outcome)) is not None
        ]
        if not pairs:
            out.append({"predictor": name, "n": 0, "pearson_r": None})
            continue
        xs, ys = zip(*pairs)
        out.append({"predictor": name, "n": len(pairs), "pearson_r": pearson(xs, ys)})
    return out


def model_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    results = {}
    for outcome in ["mean_seed_novel_icl", "best_seed_novel_icl", "seed_std_novel_icl"]:
        items = []
        for model, predictors in PREDICTOR_MODELS.items():
            items.append({"model": model, **loo_r2(rows, predictors, outcome)})
        results[outcome] = items
    return results


def write_library_report() -> None:
    library = load_library()
    capacity = load_capacity()
    gamma = load_or_compute_gamma(library)
    rows = list(library.values())
    d_rel_values = sorted({row.get("d_rel") for row in rows})
    in_degrees = []
    out_degrees = []
    first = rows[0]
    n_nodes, edges = load_edge_payload(str(first["edge_json"]))
    for node in range(n_nodes):
        in_degrees.append(sum(1 for src, dst in edges if dst == node))
        out_degrees.append(sum(1 for src, dst in edges if src == node))
    payload = {
        "schema": "exact_degree_exact_drel_exact_multiplicity_normal_fan_library.v1",
        "library_csv": str(LIBRARY_CSV.relative_to(REPO_ROOT)),
        "selected_csv": str(SELECTED_CSV.relative_to(REPO_ROOT)),
        "capacity_csv": str(CAPACITY_CSV.relative_to(REPO_ROOT)),
        "candidate_count": len(rows),
        "selected_count": sum(1 for row in rows if str(row.get("selected")) in {"1", "1.0", "True", "true"}),
        "controls": {
            "n_nodes": 5,
            "n_edges": 12,
            "n_context": 3,
            "z_dim": 2,
            "input_coupled_count": 96,
            "input_mask": "full; exact M_alpha=12 for all 8 input coordinates",
            "d_rel_values": d_rel_values,
            "in_degree_sequence": in_degrees,
            "out_degree_sequence": out_degrees,
        },
        "normal_fan_ranges": {
            "active_tree_count_mean": min_max(
                parse_float(row.get("capacity_normal_fan_active_tree_count_mean")) for row in capacity.values()
            ),
            "branch_tree_nmi_mean": min_max(
                parse_float(row.get("capacity_normal_fan_branch_tree_nmi_mean")) for row in capacity.values()
            ),
            "rooted_tree_count": min_max(parse_float(row.get("n_trees_total_enum")) for row in library.values()),
        },
        "gamma_rows": list(gamma.values()),
    }
    (NEXT_PHASE_DIR / "exact_degree_exact_drel_exact_multiplicity_normal_fan_library.json").write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Exact-Degree Exact-drel Exact-Multiplicity Normal-Fan Library",
        "",
        "## Status",
        "",
        "This library is now ungated for scale-up because the repaired gamma toy gate passed and the prospective tree-difference mask library has been created.",
        "",
        "## Fixed Controls",
        "",
        "- `N_n=5`, `m=12`, `N_c=3`, `D=2`.",
        f"- In-degree sequence: `{in_degrees}`.",
        f"- Out-degree sequence: `{out_degrees}`.",
        f"- `d_rel` values: `{d_rel_values}`.",
        "- Input mask: full coupling, so input-coupled count and multiplicity distribution are exact across all topologies (`M_alpha=12` for all 8 coordinates).",
        f"- Candidate topology groups: `{len(rows)}`.",
        "",
        "## Files",
        "",
        f"- Library CSV: `{LIBRARY_CSV.relative_to(REPO_ROOT)}`",
        f"- Selected CSV: `{SELECTED_CSV.relative_to(REPO_ROOT)}`",
        f"- Capacity/normal-fan CSV: `{CAPACITY_CSV.relative_to(REPO_ROOT)}`",
        f"- Repaired gamma rows: `{GAMMA_JSON.relative_to(REPO_ROOT)}`",
    ]
    LIBRARY_MD.write_text("\n".join(lines) + "\n")


def write_training_report(rows: Sequence[Mapping[str, Any]], status: Mapping[str, Any]) -> None:
    models = model_results(rows)
    corr = {outcome: correlations(rows, outcome) for outcome in ["mean_seed_novel_icl", "best_seed_novel_icl"]}
    payload = {
        "schema": "exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.v1",
        "status": status,
        "group_rows": list(rows),
        "model_results": models,
        "correlations": corr,
        "interpretation": {
            "status": status.get("status"),
            "claim_scope": "statistical if all 32 groups complete; otherwise incomplete",
            "controls": "exact degree sequence, d_rel, input count, and full-coupling multiplicity are fixed",
        },
    }
    TRAINING_JSON.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    group_rows = [
        [
            row.get("group"),
            row.get("mean_seed_novel_icl"),
            row.get("best_seed_novel_icl"),
            row.get("seed_std_novel_icl"),
            row.get("capacity_normal_fan_active_tree_count_mean"),
            row.get("capacity_normal_fan_branch_tree_nmi_mean"),
            row.get("gamma_no_bias_exact_lcvar"),
            row.get("library_n_trees_total_enum"),
        ]
        for row in rows
    ]
    loo_rows = []
    for outcome, items in models.items():
        for item in items:
            loo_rows.append([outcome, item["model"], item.get("n_groups"), item.get("loo_r2"), item.get("reason")])
    corr_rows = []
    for item in corr.get("mean_seed_novel_icl", []):
        corr_rows.append([item["predictor"], item["n"], item["pearson_r"]])
    lines = [
        "# Exact-Degree Exact-drel Exact-Multiplicity Normal-Fan Training Report",
        "",
        "## Status",
        "",
        f"- Training rows: `{status.get('training_rows')}` of expected `{status.get('expected_runs')}`.",
        f"- Groups with results: `{status.get('groups_with_results')}` of expected `{status.get('expected_groups')}`.",
        "- Controls: exact in/out degree sequence, exact `d_rel`, exact full-coupling input count, and exact full-coupling multiplicity distribution.",
        "",
        "## Group Outcomes",
        "",
        markdown_table(
            group_rows,
            ["group", "mean ICL", "best ICL", "seed std", "active trees", "branch-tree NMI", "gamma exact LCVaR", "rooted trees"],
        ),
        "",
        "## Grouped LOO Models",
        "",
        markdown_table(loo_rows, ["outcome", "model", "groups", "LOO R2", "reason"]),
        "",
        "## Mean-ICL Correlations",
        "",
        markdown_table(corr_rows, ["predictor", "groups", "Pearson r"]),
    ]
    TRAINING_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    write_library_report()
    rows, status = training_groups()
    write_training_report(rows, status)
    print(f"Wrote {LIBRARY_MD}")
    print(f"Wrote {TRAINING_MD}")
    print(status)


if __name__ == "__main__":
    main()
