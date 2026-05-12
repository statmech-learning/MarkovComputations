"""Prospective exact-control mask library for tree-difference multiplicity.

This script builds the Track 2 library requested in
``MARKOV_ICL_NEXT_PHASE_GOAL.md``.  It keeps one physical graph fixed and
generates binary input masks with exact input-coupled count, exact edge loads,
and either exact balanced or exact imbalanced coordinate-load distributions.
Masks are then selected by normalized same-root tree-difference comparison
overlap, yielding prospective high/low contrast groups.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from branch_margin_capacity_v2 import lower_tail_capacity_probe
from input_mask_utils import input_mask_summary, validate_input_mask
from topology_metrics import compute_topology_metrics, gini, normalize_edges
from tree_level_multiplicity_metrics import tree_level_multiplicity_summary


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "ICL" / "results"
NEXT_PHASE_DIR = RESULT_ROOT / "next_phase_stats"
TREE_REPORT = NEXT_PHASE_DIR / "tree_level_multiplicity_reanalysis.json"
LIBRARY_ROOT = RESULT_ROOT / "prospective_tree_diff_multiplicity_n6_m20_c200"
SOURCE_PHYSICAL_TOPOLOGY = "cycle_chords_n6_m20_seed3"
SOURCE_RUN_DIR_FALLBACK = (
    "/home/aadarwal/repos/topology/ICL/results/"
    "input_mask_fixed_m20_cycle_chords_seed3_c200/"
    "cycle_chords_n6_m20_seed3__mask0000_entry_random_c200_seed1_trainseed1"
)

GAMMA_PARAMS = {
    "n_samples": 240,
    "trials": 8,
    "alpha": 0.1,
    "projection_radius": 1.0,
    "decoder_radius": 1.0,
    "edge_bias_radius": 0.0,
    "max_root_assignments": 12,
    "seed_base": 1901,
}

CSV_FIELDS = [
    "idx",
    "selected",
    "selection_category",
    "selection_rank",
    "load_stratum",
    "contrast_level",
    "topology_id",
    "topology_name",
    "physical_topology_name",
    "mask_name",
    "mask_family",
    "seed",
    "n_nodes",
    "n_edges",
    "p",
    "edge_json",
    "input_mask_json",
    "input_coupled_parameter_count",
    "input_coupled_edge_count",
    "input_coupled_coord_count",
    "input_parameter_density",
    "input_edge_density",
    "input_coord_density",
    "input_edge_load_gini",
    "input_coord_load_gini",
    "coord_load_distribution",
    "edge_load_distribution",
    "d_rel",
    "d_rel_minus_n_req",
    "comparison_branch_d_rel_min",
    "comparison_branch_d_rel_mean",
    "comparison_branch_d_rel_max",
    "comparison_branch_d_rel_gini",
    "comparison_branch_common_d_rel_min",
    "comparison_branch_common_d_rel_mean",
    "comparison_branch_common_d_rel_max",
    "comparison_branch_common_d_rel_gini",
    "comparison_branch_input_count_min",
    "comparison_branch_input_count_mean",
    "comparison_branch_input_count_max",
    "comparison_branch_input_count_gini",
    "comparison_branch_input_overlap_min",
    "comparison_branch_input_overlap_mean",
    "comparison_branch_input_overlap_max",
    "comparison_branch_input_overlap_gini",
    "rank_D",
    "effective_rank_D",
    "condition_number_D",
    "condition_number_D_masked",
    "condition_number_D_masked_log",
    "root_tree_count_gini",
    "edge_participation_gini",
    "bottleneck_edge_fraction_095",
    "mean_shortest_path",
    "edge_M_mean",
    "edge_M_gini",
    "edge_overlap_norm_min",
    "edge_overlap_norm_mean",
    "tree_overlap_norm_min",
    "tree_overlap_norm_mean",
    "tree_overlap_norm_gini",
    "diff_overlap_norm_min",
    "diff_overlap_norm_mean",
    "diff_overlap_norm_gini",
    "diff_coord_load_gini",
    "diff_pair_count_sampled",
    "diff_pair_count_possible",
    "diff_pairs_truncated",
    "gamma_no_bias_exact_lcvar",
    "gamma_no_bias_tropical_lcvar",
    "gamma_no_bias_hard_root_lcvar",
]


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


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


def read_remote_json(remote_path: str, host: str = "engaging") -> dict[str, Any]:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "cat", remote_path],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(proc.stdout)


def source_run_dir(physical_topology: str) -> str:
    if TREE_REPORT.exists():
        report = read_json(TREE_REPORT)
        for dataset in report.get("datasets", []):
            if dataset.get("name") != "fixed_m20_masks_cluster_topology":
                continue
            for group in dataset.get("groups", []):
                if group.get("physical_topology_name") == physical_topology and group.get("run_dir"):
                    return str(group["run_dir"])
    return SOURCE_RUN_DIR_FALLBACK


def load_source_payload(physical_topology: str) -> tuple[dict[str, Any], str]:
    run_dir = source_run_dir(physical_topology)
    local = Path(run_dir) / "topology.json"
    if local.exists():
        return read_json(local), str(local)
    marker = "/ICL/results/"
    if marker in run_dir:
        suffix = run_dir.split(marker, 1)[1]
        candidate = RESULT_ROOT / suffix / "topology.json"
        if candidate.exists():
            return read_json(candidate), str(candidate)
    return read_remote_json(os.path.join(run_dir, "topology.json")), f"ssh:engaging:{run_dir}/topology.json"


def random_binary_matrix_with_margins(
    row_sums: np.ndarray,
    col_sums: np.ndarray,
    rng: np.random.Generator,
    switch_steps: int = 800,
) -> np.ndarray:
    n_rows = int(row_sums.size)
    n_cols = int(col_sums.size)
    for _attempt in range(400):
        remaining = col_sums.astype(int).copy()
        mat = np.zeros((n_rows, n_cols), dtype=int)
        row_order = sorted(
            rng.permutation(n_rows).tolist(),
            key=lambda row: (-int(row_sums[row]), float(rng.random())),
        )
        ok = True
        for row in row_order:
            need = int(row_sums[row])
            available = np.flatnonzero(remaining > 0)
            if available.size < need:
                ok = False
                break
            jitter = rng.random(available.size) * 1.0e-3
            ranked = available[np.argsort(-(remaining[available].astype(float) + jitter))]
            cols = ranked[:need]
            mat[row, cols] = 1
            remaining[cols] -= 1
            if np.any(remaining < 0):
                ok = False
                break
        if ok and np.all(remaining == 0):
            return switch_randomize(mat, rng, switch_steps=switch_steps)
    raise RuntimeError("could not construct a binary mask with requested margins")


def switch_randomize(mat: np.ndarray, rng: np.random.Generator, switch_steps: int = 800) -> np.ndarray:
    out = mat.copy()
    n_rows, n_cols = out.shape
    for _ in range(switch_steps):
        r1, r2 = rng.choice(n_rows, size=2, replace=False)
        c1, c2 = rng.choice(n_cols, size=2, replace=False)
        if out[r1, c1] and out[r2, c2] and not out[r1, c2] and not out[r2, c1]:
            out[r1, c1] = 0
            out[r2, c2] = 0
            out[r1, c2] = 1
            out[r2, c1] = 1
        elif out[r1, c2] and out[r2, c1] and not out[r1, c1] and not out[r2, c2]:
            out[r1, c2] = 0
            out[r2, c1] = 0
            out[r1, c1] = 1
            out[r2, c2] = 1
    return out


def load_profiles(n_edges: int, p: int, count: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if count % n_edges != 0 or count % p != 0:
        raise ValueError("this exact-control generator requires count divisible by n_edges and p")
    row_sums = np.full(n_edges, count // n_edges, dtype=int)
    balanced_cols = np.full(p, count // p, dtype=int)
    high = count // p + count // (2 * p)
    low = count // p - count // (2 * p)
    if high > n_edges or low < 0:
        raise ValueError("imbalanced coordinate-load profile is infeasible")
    imbalanced_cols = np.asarray([high if idx < p // 2 else low for idx in range(p)], dtype=int)
    if int(imbalanced_cols.sum()) != count:
        raise ValueError("imbalanced coordinate-load profile does not sum to count")
    return {
        "balanced_load": (row_sums, balanced_cols),
        "imbalanced_coord_load": (row_sums, imbalanced_cols),
    }


def safe_log10(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0.0:
        return None
    return float(math.log10(out))


def metric_row(
    idx: int,
    physical_name: str,
    mask: np.ndarray,
    load_stratum: str,
    seed: int,
    n_nodes: int,
    edges: Sequence[Sequence[int]],
    n_context: int,
    z_dim: int,
    coupled_count: int,
    edge_json: str,
    max_pairs_per_root: int,
) -> dict[str, Any]:
    p = (n_context + 1) * z_dim
    n_req = 2 * n_context * (n_context + 1) * z_dim
    summary = input_mask_summary(mask)
    topology_metrics = compute_topology_metrics(
        n_nodes,
        edges,
        p=p,
        input_mask=mask,
        n_context=n_context,
        z_dim=z_dim,
    )
    tree_metrics = tree_level_multiplicity_summary(
        n_nodes,
        edges,
        mask,
        n_context=n_context,
        z_dim=z_dim,
        max_pairs_per_root=max_pairs_per_root,
    )
    mask_name = f"{physical_name}__prospective_{load_stratum}_{idx:04d}_c{coupled_count}_seed{seed}"
    row = {
        "idx": idx,
        "selected": 0,
        "selection_category": "",
        "selection_rank": "",
        "load_stratum": load_stratum,
        "contrast_level": "",
        "topology_id": mask_name,
        "topology_name": mask_name,
        "physical_topology_name": physical_name,
        "mask_name": mask_name,
        "mask_family": f"prospective_tree_diff_{load_stratum}",
        "seed": seed,
        "n_nodes": int(n_nodes),
        "n_edges": len(edges),
        "p": p,
        "edge_json": edge_json,
        "input_mask_json": "",
        "input_coupled_parameter_count": summary["input_coupled_parameter_count"],
        "input_coupled_edge_count": summary["input_coupled_edge_count"],
        "input_coupled_coord_count": summary["input_coupled_coord_count"],
        "input_parameter_density": summary["input_parameter_density"],
        "input_edge_density": summary["input_edge_density"],
        "input_coord_density": summary["input_coord_density"],
        "input_edge_load_gini": summary["input_edge_load_gini"],
        "input_coord_load_gini": summary["input_coord_load_gini"],
        "coord_load_distribution": " ".join(str(int(v)) for v in mask.sum(axis=0).tolist()),
        "edge_load_distribution": " ".join(str(int(v)) for v in mask.sum(axis=1).tolist()),
        "d_rel": topology_metrics["d_rel"],
        "d_rel_minus_n_req": int(topology_metrics["d_rel"] - n_req),
        "comparison_branch_d_rel_min": topology_metrics.get("comparison_branch_d_rel_min"),
        "comparison_branch_d_rel_mean": topology_metrics.get("comparison_branch_d_rel_mean"),
        "comparison_branch_d_rel_max": topology_metrics.get("comparison_branch_d_rel_max"),
        "comparison_branch_d_rel_gini": topology_metrics.get("comparison_branch_d_rel_gini"),
        "comparison_branch_common_d_rel_min": topology_metrics.get("comparison_branch_common_d_rel_min"),
        "comparison_branch_common_d_rel_mean": topology_metrics.get("comparison_branch_common_d_rel_mean"),
        "comparison_branch_common_d_rel_max": topology_metrics.get("comparison_branch_common_d_rel_max"),
        "comparison_branch_common_d_rel_gini": topology_metrics.get("comparison_branch_common_d_rel_gini"),
        "comparison_branch_input_count_min": topology_metrics.get("comparison_branch_input_count_min"),
        "comparison_branch_input_count_mean": topology_metrics.get("comparison_branch_input_count_mean"),
        "comparison_branch_input_count_max": topology_metrics.get("comparison_branch_input_count_max"),
        "comparison_branch_input_count_gini": topology_metrics.get("comparison_branch_input_count_gini"),
        "comparison_branch_input_overlap_min": topology_metrics.get("comparison_branch_input_overlap_min"),
        "comparison_branch_input_overlap_mean": topology_metrics.get("comparison_branch_input_overlap_mean"),
        "comparison_branch_input_overlap_max": topology_metrics.get("comparison_branch_input_overlap_max"),
        "comparison_branch_input_overlap_gini": topology_metrics.get("comparison_branch_input_overlap_gini"),
        "rank_D": topology_metrics["rank_D"],
        "effective_rank_D": topology_metrics["effective_rank_D"],
        "condition_number_D": topology_metrics["condition_number_D"],
        "condition_number_D_masked": topology_metrics["condition_number_D_masked"],
        "condition_number_D_masked_log": safe_log10(topology_metrics["condition_number_D_masked"]),
        "root_tree_count_gini": topology_metrics["root_tree_count_gini"],
        "edge_participation_gini": topology_metrics["edge_participation_gini"],
        "bottleneck_edge_fraction_095": topology_metrics["bottleneck_edge_fraction_095"],
        "mean_shortest_path": topology_metrics["mean_shortest_path"],
        **{key: tree_metrics.get(key) for key in CSV_FIELDS if key in tree_metrics},
        "_input_mask": mask,
    }
    return row


def generate_candidates(
    n_nodes: int,
    edges: Sequence[Sequence[int]],
    physical_name: str,
    n_context: int,
    z_dim: int,
    coupled_count: int,
    candidates_per_stratum: int,
    seed: int,
    edge_json: str,
    max_pairs_per_root: int,
) -> list[dict[str, Any]]:
    p = (n_context + 1) * z_dim
    profiles = load_profiles(len(edges), p, coupled_count)
    rows = []
    seen = set()
    idx = 0
    for load_stratum, (row_sums, col_sums) in profiles.items():
        rng = np.random.default_rng(seed + 10000 * len(rows))
        generated = 0
        attempts = 0
        while generated < candidates_per_stratum and attempts < candidates_per_stratum * 30:
            attempts += 1
            mask = random_binary_matrix_with_margins(row_sums, col_sums, rng)
            key = (load_stratum, mask.tobytes())
            if key in seen:
                continue
            seen.add(key)
            mask = validate_input_mask(mask, len(edges), p)
            rows.append(
                metric_row(
                    idx=idx,
                    physical_name=physical_name,
                    mask=mask,
                    load_stratum=load_stratum,
                    seed=seed + attempts,
                    n_nodes=n_nodes,
                    edges=edges,
                    n_context=n_context,
                    z_dim=z_dim,
                    coupled_count=coupled_count,
                    edge_json=edge_json,
                    max_pairs_per_root=max_pairs_per_root,
                )
            )
            idx += 1
            generated += 1
    return rows


def modal_value(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = [row.get(key) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def select_rows(rows: list[dict[str, Any]], per_category: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    category_specs = [
        ("balanced_load", "high_tree_diff_comparison_overlap_balanced_load", "high"),
        ("balanced_load", "low_tree_diff_comparison_overlap_balanced_load", "low"),
        ("imbalanced_coord_load", "high_tree_diff_comparison_overlap_imbalanced_coord_load", "high"),
        ("imbalanced_coord_load", "low_tree_diff_comparison_overlap_imbalanced_coord_load", "low"),
    ]
    for load_stratum, category, level in category_specs:
        stratum_rows = [row for row in rows if row["load_stratum"] == load_stratum]
        modal_d_rel = modal_value(stratum_rows, "d_rel")
        eligible = [
            row
            for row in stratum_rows
            if row.get("d_rel") == modal_d_rel and row.get("diff_overlap_norm_min") is not None
        ]
        reverse = level == "high"
        ranked = sorted(
            eligible,
            key=lambda row: (
                float(row["diff_overlap_norm_min"]),
                float(row["diff_overlap_norm_mean"]),
            ),
            reverse=reverse,
        )
        if len(ranked) < per_category:
            raise RuntimeError(f"not enough eligible rows for {category}: {len(ranked)}")
        for rank, row in enumerate(ranked[:per_category], start=1):
            row["selected"] = 1
            row["selection_category"] = category
            row["selection_rank"] = rank
            row["contrast_level"] = level
            selected.append(row)
    return selected


def write_selected_masks(
    selected: Sequence[dict[str, Any]],
    n_nodes: int,
    edges: Sequence[Sequence[int]],
    library_root: Path,
) -> None:
    mask_dir = library_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        mask = np.asarray(row["_input_mask"], dtype=int)
        rel_path = Path("masks") / f"{row['mask_name']}.json"
        payload = {
            "name": row["mask_name"],
            "physical_topology_name": row["physical_topology_name"],
            "mask_family": row["mask_family"],
            "selection_category": row["selection_category"],
            "load_stratum": row["load_stratum"],
            "contrast_level": row["contrast_level"],
            "seed": row["seed"],
            "n_nodes": n_nodes,
            "n_edges": len(edges),
            "p": row["p"],
            "edges": [list(edge) for edge in edges],
            "input_mask": mask.tolist(),
            "mask_summary": input_mask_summary(mask),
        }
        write_json(library_root / rel_path, payload)
        row["input_mask_json"] = str(rel_path)


def compute_selected_gamma(
    selected: Sequence[dict[str, Any]],
    n_nodes: int,
    edges: Sequence[Sequence[int]],
    n_context: int,
    z_dim: int,
) -> None:
    for row in selected:
        mask = np.asarray(row["_input_mask"], dtype=float)
        for idx, variant in enumerate(["exact", "tropical", "hard_root"]):
            prefix = f"gamma_no_bias_{variant}"
            try:
                result = lower_tail_capacity_probe(
                    n_nodes=n_nodes,
                    edges=edges,
                    n_context=n_context,
                    z_dim=z_dim,
                    input_mask=mask,
                    variant=variant,
                    n_samples=GAMMA_PARAMS["n_samples"],
                    trials=GAMMA_PARAMS["trials"],
                    seed=GAMMA_PARAMS["seed_base"] + idx * 101 + int(row["idx"]),
                    alpha=GAMMA_PARAMS["alpha"],
                    projection_radius=GAMMA_PARAMS["projection_radius"],
                    decoder_radius=GAMMA_PARAMS["decoder_radius"],
                    edge_bias_radius=GAMMA_PARAMS["edge_bias_radius"],
                    max_root_assignments=GAMMA_PARAMS["max_root_assignments"],
                )
                row[f"{prefix}_lcvar"] = result["best"].get("branch_margin_lcvar_min")
            except Exception as exc:  # pragma: no cover - reported in row.
                row[f"{prefix}_lcvar"] = None
                row[f"{prefix}_error"] = str(exc)


def strip_private(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def group_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    out = {}
    for name, members in grouped.items():
        out[name] = {
            "n": len(members),
            "d_rel_values": sorted({row.get("d_rel") for row in members}),
            "input_count_values": sorted({row.get("input_coupled_parameter_count") for row in members}),
            "edge_load_gini_mean": float(np.mean([float(row["input_edge_load_gini"]) for row in members])),
            "coord_load_gini_mean": float(np.mean([float(row["input_coord_load_gini"]) for row in members])),
            "diff_overlap_norm_min_mean": float(np.mean([float(row["diff_overlap_norm_min"]) for row in members])),
            "diff_overlap_norm_min_range": [
                float(np.min([float(row["diff_overlap_norm_min"]) for row in members])),
                float(np.max([float(row["diff_overlap_norm_min"]) for row in members])),
            ],
            "tree_overlap_norm_min_mean": float(np.mean([float(row["tree_overlap_norm_min"]) for row in members])),
        }
    return out


def write_reports(
    library_root: Path,
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    source: str,
    n_context: int,
    z_dim: int,
    coupled_count: int,
    train_seeds: str,
) -> None:
    selected_public = [strip_private(row) for row in selected]
    all_public = [strip_private(row) for row in rows]
    category_summary = group_summary(selected, "selection_category")
    load_summary = group_summary(selected, "load_stratum")
    payload = {
        "schema": "prospective_tree_diff_multiplicity_mask_library.v1",
        "source_topology": source,
        "library_root": str(library_root.relative_to(REPO_ROOT)),
        "n_context": n_context,
        "z_dim": z_dim,
        "coupled_count": coupled_count,
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "controls": {
            "physical_graph": SOURCE_PHYSICAL_TOPOLOGY,
            "input_coupled_parameter_count": coupled_count,
            "balanced_load": "exact edge load and exact coordinate load are fixed",
            "imbalanced_coord_load": "exact edge load and exact imbalanced coordinate-load distribution are fixed",
            "d_rel": "selected within each load stratum at modal d_rel",
            "raw_overlap_warning": "normalized tree and tree-difference scores are used for selection",
        },
        "category_summary": category_summary,
        "load_summary": load_summary,
        "selected_rows": selected_public,
        "candidate_rows": all_public,
        "gamma_params": GAMMA_PARAMS,
    }
    json_path = NEXT_PHASE_DIR / "prospective_tree_diff_multiplicity_mask_library.json"
    write_json(json_path, payload)

    category_rows = []
    for category, summary in sorted(category_summary.items()):
        category_rows.append(
            [
                category,
                summary["n"],
                summary["d_rel_values"],
                summary["input_count_values"],
                summary["edge_load_gini_mean"],
                summary["coord_load_gini_mean"],
                summary["diff_overlap_norm_min_mean"],
                summary["diff_overlap_norm_min_range"],
            ]
        )
    md_lines = [
        "# Prospective Tree-Difference Multiplicity Mask Library",
        "",
        "## Status",
        "",
        "This is a prospective exact-control mask library. It uses one fixed physical graph and selects masks before new training.",
        "",
        "## Fixed Controls",
        "",
        f"- Physical graph: `{SOURCE_PHYSICAL_TOPOLOGY}`",
        f"- Source topology JSON: `{source}`",
        f"- `N`: `{n_context}`",
        f"- `D`: `{z_dim}`",
        f"- Input-coupled count: `{coupled_count}`",
        "- Balanced primary contrast: exact edge loads and exact coordinate loads are identical across selected masks.",
        "- Imbalanced secondary contrast: exact edge loads and exact imbalanced coordinate-load distribution are identical across selected masks.",
        "- Selection uses normalized same-root tree-difference comparison overlap, not raw tree-pair counts.",
        "",
        "## Selected Categories",
        "",
        markdown_table(
            category_rows,
            [
                "category",
                "masks",
                "d_rel",
                "input counts",
                "edge-load gini",
                "coord-load gini",
                "mean min diff overlap",
                "min diff range",
            ],
        ),
        "",
        "## Training Files",
        "",
        f"- Candidate library CSV: `{library_root.relative_to(REPO_ROOT) / 'library.csv'}`",
        f"- Selected training CSV: `{library_root.relative_to(REPO_ROOT) / 'selected.csv'}`",
        f"- Expected training tasks at seeds `{train_seeds}`: `{len(selected) * len([s for s in train_seeds.split(',') if s.strip()])}`",
        "",
        "## Causal Contrast",
        "",
        "The primary causal contrast is high vs low tree-difference comparison overlap in the balanced-load stratum. The imbalanced stratum is secondary and tests whether the direction survives a fixed but imbalanced coordinate-load profile.",
    ]
    md_path = NEXT_PHASE_DIR / "prospective_tree_diff_multiplicity_mask_library.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    submit_command = (
        "python3 ICL/submit_topology_library_sweep.py "
        f"--library_csv {library_root.relative_to(REPO_ROOT) / 'selected.csv'} "
        f"--output_root {RESULT_ROOT.relative_to(REPO_ROOT) / 'prospective_tree_diff_multiplicity_training'} "
        f"--seeds {train_seeds} "
        f"--manifest_csv {NEXT_PHASE_DIR.relative_to(REPO_ROOT) / 'prospective_tree_diff_multiplicity_training_manifest.csv'} "
        "--array --missing_only --max-concurrent 24"
    )
    plan_lines = [
        "# Prospective Tree-Difference Multiplicity Training Plan",
        "",
        "## Design",
        "",
        f"- Selected masks: `{len(selected)}`",
        f"- Seeds per mask: `{len([s for s in train_seeds.split(',') if s.strip()])}`",
        f"- Expected runs: `{len(selected) * len([s for s in train_seeds.split(',') if s.strip()])}`",
        "- Primary outcome: novel-class ICL accuracy.",
        "- Primary contrast: high vs low same-root tree-difference comparison overlap at fixed graph, count, `d_rel`, edge-load distribution, and coordinate-load distribution.",
        "",
        "## Submit Command",
        "",
        "```bash",
        submit_command,
        "```",
        "",
        "## Post-Training Collection",
        "",
        "```bash",
        "python3 ICL/collect_topology_results.py "
        "--input_root ICL/results/prospective_tree_diff_multiplicity_training "
        "--output_csv ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_training_results.csv",
        "python3 ICL/prospective_tree_diff_multiplicity_training_report.py",
        "```",
    ]
    plan_path = NEXT_PHASE_DIR / "prospective_tree_diff_multiplicity_training_plan.md"
    plan_path.write_text("\n".join(plan_lines) + "\n")


def build_library(args: argparse.Namespace) -> None:
    payload, source = load_source_payload(args.physical_topology)
    n_nodes = int(payload["n_nodes"])
    edges = normalize_edges(n_nodes, payload["edges"])
    library_root = Path(args.library_root)
    topology_dir = library_root / "topologies"
    topology_dir.mkdir(parents=True, exist_ok=True)
    edge_json_rel = Path("topologies") / f"{args.physical_topology}.json"
    edge_payload = {
        "name": args.physical_topology,
        "n_nodes": n_nodes,
        "edges": [list(edge) for edge in edges],
        "source": source,
    }
    write_json(library_root / edge_json_rel, edge_payload)

    rows = generate_candidates(
        n_nodes=n_nodes,
        edges=edges,
        physical_name=args.physical_topology,
        n_context=args.N,
        z_dim=args.D,
        coupled_count=args.coupled_count,
        candidates_per_stratum=args.candidates_per_stratum,
        seed=args.seed,
        edge_json=str(edge_json_rel),
        max_pairs_per_root=args.max_pairs_per_root,
    )
    selected = select_rows(rows, args.selected_per_category)
    write_selected_masks(selected, n_nodes, edges, library_root)
    if args.compute_gamma:
        compute_selected_gamma(selected, n_nodes, edges, args.N, args.D)
    write_csv(library_root / "library.csv", [strip_private(row) for row in rows])
    write_csv(library_root / "selected.csv", [strip_private(row) for row in selected])
    write_reports(
        library_root=library_root,
        rows=rows,
        selected=selected,
        source=source,
        n_context=args.N,
        z_dim=args.D,
        coupled_count=args.coupled_count,
        train_seeds=args.train_seeds,
    )
    print(f"Source: {source}")
    print(f"Candidate masks: {len(rows)}")
    print(f"Selected masks: {len(selected)}")
    print(f"Wrote {library_root / 'selected.csv'}")
    for category, summary in sorted(group_summary(selected, "selection_category").items()):
        print(
            f"{category}: n={summary['n']} d_rel={summary['d_rel_values']} "
            f"min_diff_range={summary['diff_overlap_norm_min_range']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_topology", default=SOURCE_PHYSICAL_TOPOLOGY)
    parser.add_argument("--library_root", default=str(LIBRARY_ROOT))
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--coupled_count", type=int, default=200)
    parser.add_argument("--candidates_per_stratum", type=int, default=160)
    parser.add_argument("--selected_per_category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=60617)
    parser.add_argument("--max_pairs_per_root", type=int, default=50000)
    parser.add_argument("--train_seeds", default="1,2,3,4,5")
    parser.add_argument("--compute_gamma", action="store_true")
    args = parser.parse_args()
    build_library(args)


if __name__ == "__main__":
    main()
