"""Build a multi-base exact-degree library separating tree count and normal fan.

The previous one-base exact-degree normal-fan experiment could not separate
total rooted-tree abundance from active-tree / normal-fan branch geometry. This
script builds the next targeted topology test:

* multiple base graphs / degree sequences;
* degree-preserving directed rewires within each base;
* fixed ``N_n, m, N_c, D``, full input multiplicity, and ``d_rel`` filter;
* Arm A pairs: nearly fixed tree count, variable normal-fan score;
* Arm B pairs: nearly fixed normal-fan score, variable tree count;
* a cluster-ready training manifest, but no automatic submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from branch_margin_capacity import sample_exact_copy_branches, tree_normal_fan_coverage
from cross_root_tree_contrast_metrics import cross_root_metrics_for_topology
from make_matched_motif_controls import directed_degree_rewire
from topology_metrics import compute_topology_metrics, graph_from_family, is_strongly_connected, normalize_edges, topology_matrices


REPO_ROOT = Path(__file__).resolve().parents[1]
ICL_DIR = REPO_ROOT / "ICL"
RESULT_ROOT = ICL_DIR / "results" / "next_phase_stats" / "multibase_normal_fan_tree_count_n5_m12_N3_D2"
CLUSTER_ROOT = Path("/home/aadarwal/repos/topology")

BASE_SPECS = [
    ("degree_balanced", 9),
    ("degree_balanced", 60),
    ("random_sc", 2),
    ("random_sc", 92),
    ("cycle_chords", 41),
    ("redundant_paths", 1),
    ("two_module", 2),
]

CANDIDATE_FIELDS = [
    "idx",
    "selected",
    "selection_arms",
    "topology_id",
    "topology_name",
    "base_id",
    "base_family",
    "base_seed",
    "rewire_seed",
    "n_nodes",
    "n_edges",
    "p",
    "edge_json",
    "cluster_edge_json",
    "d_rel",
    "rank_D",
    "effective_rank_D",
    "condition_number_D",
    "condition_number_D_log",
    "root_tree_count_gini",
    "edge_participation_gini",
    "bottleneck_edge_fraction_095",
    "mean_shortest_path",
    "in_degree_sequence",
    "out_degree_sequence",
    "in_degree_cv",
    "out_degree_cv",
    "n_trees_total_enum",
    "n_trees_total_enum_log",
    "normal_fan_branch_root_nmi_mean",
    "normal_fan_branch_tree_nmi_mean",
    "normal_fan_active_tree_count_mean",
    "normal_fan_branch_active_tree_count_min_mean",
    "normal_fan_score",
    "normal_fan_score_within_base_z",
    "tree_count_within_base_z",
    "cross_contrast_effective_rank_mean",
    "cross_all_supported_effective_rank",
]

PAIR_FIELDS = [
    "pair_id",
    "arm",
    "base_id",
    "low_role_topology_id",
    "high_role_topology_id",
    "low_role",
    "high_role",
    "tree_count_low",
    "tree_count_high",
    "tree_count_delta",
    "normal_fan_score_low",
    "normal_fan_score_high",
    "normal_fan_score_delta",
    "active_tree_low",
    "active_tree_high",
    "branch_tree_nmi_low",
    "branch_tree_nmi_high",
]

TRAINING_FIELDS = [
    "topology_index",
    "topology_id",
    "topology_name",
    "base_id",
    "selection_arms",
    "train_seed",
    "output",
    "results_path",
    "completed",
    "command",
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


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def safe_log10(value: float) -> float:
    return float(math.log10(max(1.0, float(value))))


def cv(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean()) if arr.size else 0.0
    if abs(mean) <= 1.0e-12:
        return 0.0
    return float(arr.std() / abs(mean))


def degree_sequences(n_nodes: int, edges: Sequence[Sequence[int]]) -> tuple[list[int], list[int]]:
    indeg = [0] * n_nodes
    outdeg = [0] * n_nodes
    for source, target in edges:
        outdeg[int(source)] += 1
        indeg[int(target)] += 1
    return indeg, outdeg


def zscore(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    scale = float(arr.std())
    if scale <= 1.0e-12:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / scale


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.3f}" if math.isfinite(value) else "NA"
        return str(value)

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(item) for item in row) + " |")
    return "\n".join(out)


def edge_payload_path(output_root: Path, topology_id: str) -> Path:
    return output_root / "topologies" / f"{topology_id}.json"


def cluster_path(local_path: Path) -> str:
    rel = local_path.relative_to(REPO_ROOT)
    return str(CLUSTER_ROOT / rel)


def normal_fan_metrics(
    n_nodes: int,
    edges: Sequence[Sequence[int]],
    p: int,
    n_context: int,
    z_dim: int,
    trials: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    mats = topology_matrices(n_nodes, edges)
    z, labels = sample_exact_copy_branches(samples, n_context, z_dim, seed=seed)
    return tree_normal_fan_coverage(
        mats["arborescences"],
        n_edges=len(edges),
        input_dim=p,
        z=z,
        labels=labels,
        input_mask=None,
        n_trials=trials,
        seed=seed + 1009,
        projection_radius=1.0,
        edge_bias_scale=0.0,
    )


def make_candidate_row(
    idx: int,
    base_id: str,
    base_family: str,
    base_seed: int,
    rewire_seed: int,
    n_nodes: int,
    edges: Sequence[Sequence[int]],
    n_context: int,
    z_dim: int,
    output_root: Path,
    normal_fan_trials: int,
    normal_fan_samples: int,
) -> dict[str, Any]:
    p = (n_context + 1) * z_dim
    metrics = compute_topology_metrics(n_nodes, edges, p=p, n_context=n_context, z_dim=z_dim)
    fan = normal_fan_metrics(
        n_nodes,
        edges,
        p=p,
        n_context=n_context,
        z_dim=z_dim,
        trials=normal_fan_trials,
        samples=normal_fan_samples,
        seed=19_000 + idx,
    )
    full_mask = np.ones((len(edges), p), dtype=float)
    cross = cross_root_metrics_for_topology(
        n_nodes,
        edges,
        full_mask,
        n_context=n_context,
        z_dim=z_dim,
        max_pairs_per_root_pair=50000,
    )
    indeg, outdeg = degree_sequences(n_nodes, edges)
    topology_id = f"mb{idx:04d}_{base_id}_rw{rewire_seed}"
    edge_json = edge_payload_path(output_root, topology_id)
    payload = {
        "name": f"multibase_{base_id}_rewire_seed{rewire_seed}",
        "family": "multibase_degree_rewire_exact_degree",
        "base_id": base_id,
        "base_family": base_family,
        "base_seed": base_seed,
        "rewire_seed": rewire_seed,
        "n_nodes": n_nodes,
        "edges": [list(edge) for edge in edges],
        "metrics": metrics,
        "normal_fan_metrics": fan,
        "cross_root_metrics": {
            "cross_contrast_effective_rank_mean": cross.get("cross_contrast_effective_rank_mean"),
            "cross_all_supported_effective_rank": cross.get("cross_all_supported_effective_rank"),
        },
    }
    edge_json.parent.mkdir(parents=True, exist_ok=True)
    edge_json.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    condition = finite_float(metrics.get("condition_number_D")) or 1.0
    row = {
        "idx": idx,
        "selected": 0,
        "selection_arms": "",
        "topology_id": topology_id,
        "topology_name": payload["name"],
        "base_id": base_id,
        "base_family": base_family,
        "base_seed": base_seed,
        "rewire_seed": rewire_seed,
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "p": p,
        "edge_json": str(edge_json.relative_to(output_root)),
        "cluster_edge_json": cluster_path(edge_json),
        "d_rel": metrics.get("d_rel"),
        "rank_D": metrics.get("rank_D"),
        "effective_rank_D": metrics.get("effective_rank_D"),
        "condition_number_D": metrics.get("condition_number_D"),
        "condition_number_D_log": safe_log10(condition),
        "root_tree_count_gini": metrics.get("root_tree_count_gini"),
        "edge_participation_gini": metrics.get("edge_participation_gini"),
        "bottleneck_edge_fraction_095": metrics.get("bottleneck_edge_fraction_095"),
        "mean_shortest_path": metrics.get("mean_shortest_path"),
        "in_degree_sequence": " ".join(str(value) for value in indeg),
        "out_degree_sequence": " ".join(str(value) for value in outdeg),
        "in_degree_cv": cv(indeg),
        "out_degree_cv": cv(outdeg),
        "n_trees_total_enum": metrics.get("n_trees_total_enum"),
        "n_trees_total_enum_log": safe_log10(float(metrics.get("n_trees_total_enum") or 0.0)),
        "normal_fan_branch_root_nmi_mean": fan.get("normal_fan_branch_root_nmi_mean"),
        "normal_fan_branch_tree_nmi_mean": fan.get("normal_fan_branch_tree_nmi_mean"),
        "normal_fan_active_tree_count_mean": fan.get("normal_fan_active_tree_count_mean"),
        "normal_fan_branch_active_tree_count_min_mean": fan.get("normal_fan_branch_active_tree_count_min_mean"),
        "cross_contrast_effective_rank_mean": cross.get("cross_contrast_effective_rank_mean"),
        "cross_all_supported_effective_rank": cross.get("cross_all_supported_effective_rank"),
    }
    return row


def generate_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_global: set[tuple[tuple[int, int], ...]] = set()
    idx = 0
    output_root = Path(args.output_root)
    for base_index, (family, seed) in enumerate(BASE_SPECS):
        spec = graph_from_family(family, args.n_nodes, args.n_edges, seed=seed)
        base_edges = normalize_edges(args.n_nodes, spec.edges)
        base_id = f"base{base_index:02d}_{family}_seed{seed}"
        rng = np.random.default_rng(args.seed_base + 1000 * base_index)
        per_base = 0
        attempts = 0
        while per_base < args.candidates_per_base and attempts < args.max_attempts_per_base:
            attempts += 1
            rewire_seed = int(rng.integers(1, 10_000_000))
            edges = directed_degree_rewire(args.n_nodes, base_edges, np.random.default_rng(rewire_seed), args.swap_attempts)
            edge_key = tuple(edges)
            if edge_key in seen_global:
                continue
            if not is_strongly_connected(args.n_nodes, edges):
                continue
            metrics = compute_topology_metrics(args.n_nodes, edges, p=(args.n_context + 1) * args.z_dim, n_context=args.n_context, z_dim=args.z_dim)
            if args.target_d_rel is not None and int(metrics["d_rel"]) != int(args.target_d_rel):
                continue
            seen_global.add(edge_key)
            rows.append(
                make_candidate_row(
                    idx=idx,
                    base_id=base_id,
                    base_family=family,
                    base_seed=seed,
                    rewire_seed=rewire_seed,
                    n_nodes=args.n_nodes,
                    edges=edges,
                    n_context=args.n_context,
                    z_dim=args.z_dim,
                    output_root=output_root,
                    normal_fan_trials=args.normal_fan_trials,
                    normal_fan_samples=args.normal_fan_samples,
                )
            )
            idx += 1
            per_base += 1
    add_scores(rows)
    return rows


def add_scores(rows: list[dict[str, Any]]) -> None:
    active = np.asarray([float(row["normal_fan_active_tree_count_mean"]) for row in rows], dtype=float)
    nmi = np.asarray([float(row["normal_fan_branch_tree_nmi_mean"]) for row in rows], dtype=float)
    branch_min = np.asarray([float(row["normal_fan_branch_active_tree_count_min_mean"]) for row in rows], dtype=float)
    nf_score = zscore(active) + zscore(nmi) + zscore(branch_min)
    for row, score in zip(rows, nf_score):
        row["normal_fan_score"] = float(score)

    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_base[str(row["base_id"])].append(row)
    for base_rows in by_base.values():
        tree_z = zscore([float(row["n_trees_total_enum_log"]) for row in base_rows])
        nf_z = zscore([float(row["normal_fan_score"]) for row in base_rows])
        for row, t_z, n_z in zip(base_rows, tree_z, nf_z):
            row["tree_count_within_base_z"] = float(t_z)
            row["normal_fan_score_within_base_z"] = float(n_z)


def pair_record(pair_id: str, arm: str, base_id: str, low: Mapping[str, Any], high: Mapping[str, Any], low_role: str, high_role: str) -> dict[str, Any]:
    tree_low = float(low["n_trees_total_enum"])
    tree_high = float(high["n_trees_total_enum"])
    nf_low = float(low["normal_fan_score"])
    nf_high = float(high["normal_fan_score"])
    return {
        "pair_id": pair_id,
        "arm": arm,
        "base_id": base_id,
        "low_role_topology_id": low["topology_id"],
        "high_role_topology_id": high["topology_id"],
        "low_role": low_role,
        "high_role": high_role,
        "tree_count_low": tree_low,
        "tree_count_high": tree_high,
        "tree_count_delta": tree_high - tree_low,
        "normal_fan_score_low": nf_low,
        "normal_fan_score_high": nf_high,
        "normal_fan_score_delta": nf_high - nf_low,
        "active_tree_low": low["normal_fan_active_tree_count_mean"],
        "active_tree_high": high["normal_fan_active_tree_count_mean"],
        "branch_tree_nmi_low": low["normal_fan_branch_tree_nmi_mean"],
        "branch_tree_nmi_high": high["normal_fan_branch_tree_nmi_mean"],
    }


def choose_pairs(rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_base[str(row["base_id"])].append(row)
    pairs: list[dict[str, Any]] = []
    selected_by_id: dict[str, set[str]] = defaultdict(set)

    for base_id, base_rows in sorted(by_base.items()):
        # Arm A: fixed tree count / variable normal-fan geometry.
        candidates = []
        for i, left in enumerate(base_rows):
            for right in base_rows[i + 1 :]:
                tree_delta = abs(float(left["n_trees_total_enum"]) - float(right["n_trees_total_enum"]))
                if tree_delta > args.arm_a_tree_tolerance:
                    continue
                nf_delta = abs(float(left["normal_fan_score"]) - float(right["normal_fan_score"]))
                if nf_delta < args.min_normal_fan_delta:
                    continue
                low, high = sorted([left, right], key=lambda row: float(row["normal_fan_score"]))
                candidates.append((nf_delta, -tree_delta, low, high))
        used: set[str] = set()
        for rank, (_, _, low, high) in enumerate(sorted(candidates, reverse=True)[: args.pairs_per_arm_per_base], start=1):
            if low["topology_id"] in used or high["topology_id"] in used:
                continue
            pair_id = f"{base_id}_armA_{rank:02d}"
            pairs.append(pair_record(pair_id, "arm_A_fixed_tree_count_variable_normal_fan", base_id, low, high, "low_normal_fan", "high_normal_fan"))
            selected_by_id[str(low["topology_id"])].add("arm_A_low_normal_fan")
            selected_by_id[str(high["topology_id"])].add("arm_A_high_normal_fan")
            used.update([str(low["topology_id"]), str(high["topology_id"])])

        # Arm B: variable tree count / matched normal-fan geometry.
        candidates = []
        for i, left in enumerate(base_rows):
            for right in base_rows[i + 1 :]:
                nf_delta = abs(float(left["normal_fan_score"]) - float(right["normal_fan_score"]))
                if nf_delta > args.arm_b_normal_fan_tolerance:
                    continue
                tree_delta = abs(float(left["n_trees_total_enum"]) - float(right["n_trees_total_enum"]))
                if tree_delta < args.min_tree_count_delta:
                    continue
                low, high = sorted([left, right], key=lambda row: float(row["n_trees_total_enum"]))
                candidates.append((tree_delta, -nf_delta, low, high))
        used = set()
        for rank, (_, _, low, high) in enumerate(sorted(candidates, reverse=True)[: args.pairs_per_arm_per_base], start=1):
            if low["topology_id"] in used or high["topology_id"] in used:
                continue
            pair_id = f"{base_id}_armB_{rank:02d}"
            pairs.append(pair_record(pair_id, "arm_B_variable_tree_count_matched_normal_fan", base_id, low, high, "low_tree_count", "high_tree_count"))
            selected_by_id[str(low["topology_id"])].add("arm_B_low_tree_count")
            selected_by_id[str(high["topology_id"])].add("arm_B_high_tree_count")
            used.update([str(low["topology_id"]), str(high["topology_id"])])

    for row in rows:
        arms = sorted(selected_by_id.get(str(row["topology_id"]), set()))
        row["selected"] = 1 if arms else 0
        row["selection_arms"] = ";".join(arms)
    return pairs


def command_for(row: Mapping[str, Any], train_seed: int, output_root: str, args: argparse.Namespace) -> tuple[str, str, str]:
    label = f"{row['topology_id']}_trainseed{train_seed}"
    output = str(Path(output_root) / label)
    results = str(Path(output) / "results.pkl")
    command = (
        "python3 -u run_topology_icl.py "
        f"--output {output} "
        f"--edge_json {row['cluster_edge_json']} "
        f"--seed {train_seed} --input_mask_seed {train_seed} --no_progress "
        "--K 128 --L 128 --D 2 --N 3 --B 1 --epsilon 0.001 --epochs 100 "
        "--lr 0.0025 --batch_size 50 --train_samples 5000 --val_samples 1000 "
        "--eval_frequency 10 --n_eval_samples 200 --test_samples 500 "
        "--method direct_solve --temperature 1.0 --device auto"
    )
    return command, output, results


def build_training_manifest(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row.get("selected", 0)) == 1]
    records = []
    for topology_index, row in enumerate(selected):
        for train_seed in range(1, args.train_seeds + 1):
            command, output, results = command_for(row, train_seed, args.cluster_training_output_root, args)
            records.append(
                {
                    "topology_index": topology_index,
                    "topology_id": row["topology_id"],
                    "topology_name": row["topology_name"],
                    "base_id": row["base_id"],
                    "selection_arms": row["selection_arms"],
                    "train_seed": train_seed,
                    "output": output,
                    "results_path": results,
                    "completed": False,
                    "command": command,
                }
            )
    return records


def write_array_files(output_root: Path, training_rows: Sequence[Mapping[str, Any]], max_concurrent: int) -> None:
    meta = output_root / "_array_meta"
    meta.mkdir(parents=True, exist_ok=True)
    commands = meta / "commands.txt"
    outputs = meta / "outputs.txt"
    commands.write_text("\n".join(str(row["command"]) for row in training_rows) + "\n")
    outputs.write_text("\n".join(str(row["output"]) for row in training_rows) + "\n")
    script = meta / "run_task.sh"
    script.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=mb_nf_tree
#SBATCH --output={cluster_path(meta)}/task_%a.out
#SBATCH --error={cluster_path(meta)}/task_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-{len(training_rows) - 1}%{max_concurrent}

set -euo pipefail
cd {CLUSTER_ROOT / 'ICL'}
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
CMD=$(sed -n "${{LINE_NUM}}p" {cluster_path(commands)})
OUT=$(sed -n "${{LINE_NUM}}p" {cluster_path(outputs)})
mkdir -p "$OUT"
if [ -f "$OUT/results.pkl" ]; then
    echo "Skipping $OUT (results.pkl exists)"
    exit 0
fi
echo "$CMD"
eval "$CMD"
"""
    )
    script.chmod(0o755)


def summary_payload(rows: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]], training: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_base = defaultdict(list)
    for row in rows:
        by_base[str(row["base_id"])].append(row)
    selected = [row for row in rows if int(row.get("selected", 0)) == 1]
    pair_by_arm = Counter(str(pair["arm"]) for pair in pairs)
    return {
        "schema": "multibase_normal_fan_tree_count_separation_library.v1",
        "status": "library_and_training_manifest_ready_not_submitted",
        "controls": {
            "n_nodes": args.n_nodes,
            "n_edges": args.n_edges,
            "n_context": args.n_context,
            "z_dim": args.z_dim,
            "p": (args.n_context + 1) * args.z_dim,
            "target_d_rel": args.target_d_rel,
            "input_mask": "full coupling; exact multiplicity M_alpha=m for every input coordinate",
            "degree_control": "exact in/out degree sequence within each base_id",
        },
        "candidate_count": len(rows),
        "selected_topology_count": len(selected),
        "training_task_count": len(training),
        "train_seeds_per_topology": args.train_seeds,
        "pair_count": len(pairs),
        "pair_count_by_arm": dict(pair_by_arm),
        "base_summary": {
            base: {
                "candidate_count": len(items),
                "selected_count": sum(int(row.get("selected", 0)) for row in items),
                "tree_count_range": [
                    min(float(row["n_trees_total_enum"]) for row in items),
                    max(float(row["n_trees_total_enum"]) for row in items),
                ],
                "normal_fan_score_range": [
                    min(float(row["normal_fan_score"]) for row in items),
                    max(float(row["normal_fan_score"]) for row in items),
                ],
                "in_degree_sequence": items[0]["in_degree_sequence"],
                "out_degree_sequence": items[0]["out_degree_sequence"],
            }
            for base, items in sorted(by_base.items())
        },
        "arm_quality": {
            arm: {
                "n_pairs": len(items),
                "mean_abs_tree_delta": float(np.mean([abs(float(item["tree_count_delta"])) for item in items])) if items else None,
                "mean_abs_normal_fan_delta": float(np.mean([abs(float(item["normal_fan_score_delta"])) for item in items])) if items else None,
            }
            for arm, items in {
                arm: [pair for pair in pairs if pair["arm"] == arm]
                for arm in sorted({str(pair["arm"]) for pair in pairs})
            }.items()
        },
        "files": {
            "candidate_library_csv": str((Path(args.output_root) / "candidate_library.csv").relative_to(REPO_ROOT)),
            "selected_csv": str((Path(args.output_root) / "selected.csv").relative_to(REPO_ROOT)),
            "pair_manifest_csv": str((Path(args.output_root) / "pair_manifest.csv").relative_to(REPO_ROOT)),
            "training_manifest_csv": str((Path(args.output_root) / "training_manifest.csv").relative_to(REPO_ROOT)),
            "array_script": str((Path(args.output_root) / "_array_meta" / "run_task.sh").relative_to(REPO_ROOT)),
        },
    }


def write_reports(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    write_json(output_root / "library_summary.json", payload)
    write_json(ICL_DIR / "results" / "next_phase_stats" / "multibase_normal_fan_tree_count_separation_library.json", payload)

    base_rows = [
        [
            base,
            item["candidate_count"],
            item["selected_count"],
            item["tree_count_range"],
            item["normal_fan_score_range"],
        ]
        for base, item in payload["base_summary"].items()
    ]
    arm_rows = [
        [arm, item["n_pairs"], item["mean_abs_tree_delta"], item["mean_abs_normal_fan_delta"]]
        for arm, item in payload["arm_quality"].items()
    ]
    report = "\n".join(
        [
            "# Multi-Base Normal-Fan / Tree-Count Separation Library",
            "",
            "## Status",
            "",
            "Library and training manifest are ready. No training jobs were submitted by this generator.",
            "",
            "## Controls",
            "",
            markdown_table(["control", "value"], [[key, value] for key, value in payload["controls"].items()]),
            "",
            "## Counts",
            "",
            markdown_table(
                ["quantity", "value"],
                [
                    ["candidate topologies", payload["candidate_count"]],
                    ["selected topologies", payload["selected_topology_count"]],
                    ["matched pairs", payload["pair_count"]],
                    ["training tasks", payload["training_task_count"]],
                ],
            ),
            "",
            "## Base Summary",
            "",
            markdown_table(["base", "candidates", "selected", "tree count range", "normal-fan score range"], base_rows),
            "",
            "## Arm Quality",
            "",
            markdown_table(["arm", "pairs", "mean abs tree delta", "mean abs normal-fan delta"], arm_rows),
            "",
            "## Files",
            "",
            "\n".join(f"- `{path}`" for path in payload["files"].values()),
            "",
            "## Training Command",
            "",
            f"Submit on Engaging with: `sbatch {cluster_path(output_root / '_array_meta' / 'run_task.sh')}`",
        ]
    )
    (output_root / "multibase_normal_fan_tree_count_separation_library.md").write_text(report + "\n")
    (ICL_DIR / "results" / "next_phase_stats" / "multibase_normal_fan_tree_count_separation_library.md").write_text(report + "\n")

    plan_payload = {
        "schema": "multibase_normal_fan_tree_count_training_plan.v1",
        "status": "ready_to_submit",
        "primary_models": [
            "controls/base_id only",
            "tree_count_only + base_id",
            "normal_fan_only + base_id",
            "tree_count + normal_fan + base_id",
            "cross_root_rank + normal_fan + tree_count + base_id",
        ],
        "primary_outcomes": ["mean novel-class ICL", "best-seed novel-class ICL", "seed std"],
        "inference": ["grouped LOO", "held-out-base checks", "matched-pair contrasts", "clustered bootstrap by base_id"],
        "manifest": payload["files"]["training_manifest_csv"],
        "array_script": payload["files"]["array_script"],
        "submission_command": f"sbatch {cluster_path(output_root / '_array_meta' / 'run_task.sh')}",
    }
    write_json(ICL_DIR / "results" / "next_phase_stats" / "multibase_normal_fan_tree_count_training_plan.json", plan_payload)
    plan_md = "\n".join(
        [
            "# Multi-Base Normal-Fan / Tree-Count Training Plan",
            "",
            "## Status",
            "",
            "Ready to submit. This is a targeted exact-control test, not a broad sweep.",
            "",
            "## Primary Models",
            "",
            "\n".join(f"- {item}" for item in plan_payload["primary_models"]),
            "",
            "## Outcomes",
            "",
            "\n".join(f"- {item}" for item in plan_payload["primary_outcomes"]),
            "",
            "## Inference",
            "",
            "\n".join(f"- {item}" for item in plan_payload["inference"]),
            "",
            "## Submit",
            "",
            f"`{plan_payload['submission_command']}`",
        ]
    )
    (ICL_DIR / "results" / "next_phase_stats" / "multibase_normal_fan_tree_count_training_plan.md").write_text(plan_md + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(RESULT_ROOT))
    parser.add_argument("--n-nodes", type=int, default=5)
    parser.add_argument("--n-edges", type=int, default=12)
    parser.add_argument("--n-context", type=int, default=3)
    parser.add_argument("--z-dim", type=int, default=2)
    parser.add_argument("--target-d-rel", type=int, default=88)
    parser.add_argument("--candidates-per-base", type=int, default=80)
    parser.add_argument("--max-attempts-per-base", type=int, default=500)
    parser.add_argument("--swap-attempts", type=int, default=400)
    parser.add_argument("--normal-fan-trials", type=int, default=8)
    parser.add_argument("--normal-fan-samples", type=int, default=600)
    parser.add_argument("--pairs-per-arm-per-base", type=int, default=3)
    parser.add_argument("--arm-a-tree-tolerance", type=float, default=2.0)
    parser.add_argument("--arm-b-normal-fan-tolerance", type=float, default=0.25)
    parser.add_argument("--min-normal-fan-delta", type=float, default=1.0)
    parser.add_argument("--min-tree-count-delta", type=float, default=8.0)
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument(
        "--cluster-training-output-root",
        default="/home/aadarwal/repos/topology/ICL/results/multibase_normal_fan_tree_count_training",
    )
    parser.add_argument("--max-concurrent", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=7301)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = generate_candidates(args)
    pairs = choose_pairs(rows, args)
    selected = [row for row in rows if int(row["selected"]) == 1]
    training = build_training_manifest(rows, args)

    write_csv(output_root / "candidate_library.csv", rows, CANDIDATE_FIELDS)
    write_csv(output_root / "selected.csv", selected, CANDIDATE_FIELDS)
    write_csv(output_root / "pair_manifest.csv", pairs, PAIR_FIELDS)
    write_csv(output_root / "training_manifest.csv", training, TRAINING_FIELDS)
    write_array_files(output_root, training, max_concurrent=args.max_concurrent)

    payload = summary_payload(rows, pairs, training, args)
    write_reports(payload, rows, pairs, args)
    print(f"candidate topologies: {len(rows)}")
    print(f"selected topologies: {len(selected)}")
    print(f"matched pairs: {len(pairs)}")
    print(f"training tasks: {len(training)}")
    print(f"wrote {output_root}")


if __name__ == "__main__":
    main()
