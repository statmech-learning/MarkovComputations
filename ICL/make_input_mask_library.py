"""Build a fixed-physical-topology input-mask library.

This is the input-encoding analogue of ``make_topology_library.py``.  The
physical reaction graph is held fixed, while binary masks Omega choose where
input coordinates may modulate edge rates.  Rows in ``selected.csv`` can be
submitted directly with ``submit_topology_library_sweep.py`` because they carry
both ``edge_json`` and ``input_mask_json``.
"""

import argparse
import csv
import json
import math
import os

import numpy as np

from input_mask_utils import input_mask_summary, validate_input_mask
from topology_metrics import compute_topology_metrics, normalize_edges


DEFAULT_FEATURES = [
    "d_rel",
    "comparison_branch_d_rel_min",
    "comparison_branch_d_rel_gini",
    "effective_rank_D_masked",
    "condition_number_D_masked_log",
    "input_coupled_edge_count",
    "input_coupled_coord_count",
    "input_edge_load_gini",
    "input_coord_load_gini",
]


CSV_FIELDS = [
    "idx",
    "selected",
    "topology_id",
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
    "d_rel",
    "d_rel_minus_n_req",
    "comparison_branch_d_rel_min",
    "comparison_branch_d_rel_mean",
    "comparison_branch_d_rel_max",
    "comparison_branch_d_rel_gini",
    "comparison_branch_input_count_min",
    "comparison_branch_input_count_mean",
    "comparison_branch_input_count_max",
    "comparison_branch_input_count_gini",
    "rank_D",
    "effective_rank_D",
    "effective_rank_D_masked",
    "condition_number_D",
    "condition_number_D_masked",
    "condition_number_D_masked_log",
    "root_tree_count_gini",
    "edge_participation_gini",
    "bottleneck_edge_fraction_095",
    "mean_shortest_path",
]


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_edge_payload(path):
    with open(path) as f:
        payload = json.load(f)
    if "edges" not in payload:
        raise ValueError(f"{path} does not contain 'edges'")
    n_nodes = int(payload["n_nodes"])
    edges = normalize_edges(n_nodes, payload["edges"])
    return payload, n_nodes, edges


def parse_int_set(raw):
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [int(item) for item in part.split(":")]
            if len(pieces) == 2:
                start, stop = pieces
                step = 1
            elif len(pieces) == 3:
                start, stop, step = pieces
            else:
                raise ValueError(f"Invalid range {part!r}")
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(part))
    return values


def parse_coupled_counts(raw, total):
    counts = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "." in part:
            value = float(part)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Fractional coupled count must be in [0, 1], got {part}")
            count = int(round(value * total))
        else:
            count = int(part)
        if not 0 <= count <= total:
            raise ValueError(f"Coupled count {count} outside [0, {total}]")
        counts.append(count)
    return sorted(set(counts))


def finite_float(value, default=0.0):
    if value is None:
        return default
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def fill_edges(mask, edge_order, count, rng):
    p = mask.shape[1]
    remaining = count
    for edge in edge_order:
        if remaining <= 0:
            break
        take = min(p, remaining)
        coords = np.arange(p)
        rng.shuffle(coords)
        mask[edge, coords[:take]] = 1
        remaining -= take
    return mask


def mask_entry_random(n_edges, p, count, rng, edge_participation):
    del edge_participation
    mask = np.zeros((n_edges, p), dtype=int)
    choices = rng.choice(n_edges * p, size=count, replace=False) if count else []
    for flat in choices:
        mask[int(flat) // p, int(flat) % p] = 1
    return mask


def mask_edge_block(n_edges, p, count, rng, edge_participation):
    del edge_participation
    mask = np.zeros((n_edges, p), dtype=int)
    edge_order = np.arange(n_edges)
    rng.shuffle(edge_order)
    return fill_edges(mask, edge_order, count, rng)


def mask_coord_block(n_edges, p, count, rng, edge_participation):
    del edge_participation
    mask = np.zeros((n_edges, p), dtype=int)
    coord_order = np.arange(p)
    rng.shuffle(coord_order)
    remaining = count
    for coord in coord_order:
        if remaining <= 0:
            break
        take = min(n_edges, remaining)
        edges = np.arange(n_edges)
        rng.shuffle(edges)
        mask[edges[:take], coord] = 1
        remaining -= take
    return mask


def mask_balanced(n_edges, p, count, rng, edge_participation):
    del edge_participation
    mask = np.zeros((n_edges, p), dtype=int)
    edge_load = np.zeros(n_edges, dtype=int)
    coord_load = np.zeros(p, dtype=int)
    for _ in range(count):
        available = np.argwhere(mask == 0)
        scores = edge_load[available[:, 0]] + coord_load[available[:, 1]]
        best_score = scores.min()
        best = available[scores == best_score]
        edge, coord = best[int(rng.integers(0, len(best)))]
        mask[edge, coord] = 1
        edge_load[edge] += 1
        coord_load[coord] += 1
    return mask


def mask_high_participation_edges(n_edges, p, count, rng, edge_participation):
    order = np.argsort(-np.asarray(edge_participation, dtype=float))
    return fill_edges(np.zeros((n_edges, p), dtype=int), order, count, rng)


def mask_low_participation_edges(n_edges, p, count, rng, edge_participation):
    order = np.argsort(np.asarray(edge_participation, dtype=float))
    return fill_edges(np.zeros((n_edges, p), dtype=int), order, count, rng)


MASK_BUILDERS = {
    "entry_random": mask_entry_random,
    "edge_block": mask_edge_block,
    "coord_block": mask_coord_block,
    "balanced": mask_balanced,
    "high_participation_edges": mask_high_participation_edges,
    "low_participation_edges": mask_low_participation_edges,
}


def metric_row(idx, mask_name, family, seed, metrics, summary, edge_json, input_mask_json):
    condition_masked = metrics.get("condition_number_D_masked")
    return {
        "idx": idx,
        "selected": 0,
        "topology_id": mask_name,
        "physical_topology_name": metrics["physical_topology_name"],
        "mask_name": mask_name,
        "mask_family": family,
        "seed": seed,
        "n_nodes": metrics["n_nodes"],
        "n_edges": metrics["n_edges"],
        "p": metrics["p"],
        "edge_json": edge_json,
        "input_mask_json": input_mask_json,
        "input_coupled_parameter_count": summary["input_coupled_parameter_count"],
        "input_coupled_edge_count": summary["input_coupled_edge_count"],
        "input_coupled_coord_count": summary["input_coupled_coord_count"],
        "input_parameter_density": summary["input_parameter_density"],
        "input_edge_density": summary["input_edge_density"],
        "input_coord_density": summary["input_coord_density"],
        "input_edge_load_gini": summary["input_edge_load_gini"],
        "input_coord_load_gini": summary["input_coord_load_gini"],
        "d_rel": metrics["d_rel"],
        "d_rel_minus_n_req": metrics["d_rel_minus_n_req"],
        "comparison_branch_d_rel_min": metrics.get("comparison_branch_d_rel_min", ""),
        "comparison_branch_d_rel_mean": metrics.get("comparison_branch_d_rel_mean", ""),
        "comparison_branch_d_rel_max": metrics.get("comparison_branch_d_rel_max", ""),
        "comparison_branch_d_rel_gini": metrics.get("comparison_branch_d_rel_gini", ""),
        "comparison_branch_input_count_min": metrics.get("comparison_branch_input_count_min", ""),
        "comparison_branch_input_count_mean": metrics.get("comparison_branch_input_count_mean", ""),
        "comparison_branch_input_count_max": metrics.get("comparison_branch_input_count_max", ""),
        "comparison_branch_input_count_gini": metrics.get("comparison_branch_input_count_gini", ""),
        "rank_D": metrics["rank_D"],
        "effective_rank_D": metrics["effective_rank_D"],
        "effective_rank_D_masked": metrics["effective_rank_D_masked"],
        "condition_number_D": metrics["condition_number_D"],
        "condition_number_D_masked": condition_masked,
        "condition_number_D_masked_log": math.log10(finite_float(condition_masked, 1.0)),
        "root_tree_count_gini": metrics["root_tree_count_gini"],
        "edge_participation_gini": metrics["edge_participation_gini"],
        "bottleneck_edge_fraction_095": metrics["bottleneck_edge_fraction_095"],
        "mean_shortest_path": metrics["mean_shortest_path"],
    }


def standardized_matrix(rows, features):
    matrix = np.asarray(
        [[finite_float(row.get(feature)) for feature in features] for row in rows],
        dtype=float,
    )
    if matrix.size == 0:
        return matrix
    center = matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (matrix - center) / scale


def select_diverse_rows(rows, k, features, family_key=None):
    if k <= 0 or k >= len(rows):
        return set(range(len(rows)))

    z = standardized_matrix(rows, features)
    selected = []
    if family_key:
        families = sorted({row.get(family_key) for row in rows if row.get(family_key)})
        for family in families:
            candidates = [
                idx for idx, row in enumerate(rows) if row.get(family_key) == family and idx not in selected
            ]
            if not candidates:
                continue
            selected.append(max(candidates, key=lambda idx: float(np.linalg.norm(z[idx, :]))))
            if len(selected) >= k:
                return set(selected)

    for col in range(z.shape[1]):
        for idx in (int(np.argmin(z[:, col])), int(np.argmax(z[:, col]))):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= k:
                return set(selected)

    while len(selected) < k:
        selected_z = z[selected, :]
        distances = np.sqrt(((z[:, None, :] - selected_z[None, :, :]) ** 2).sum(axis=2))
        min_distances = distances.min(axis=1)
        min_distances[selected] = -1.0
        selected.append(int(np.argmax(min_distances)))
    return set(selected)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(args):
    edge_payload, n_nodes, edges = load_edge_payload(args.edge_json)
    p = (args.N + 1) * args.D
    n_req = 2 * args.N * (args.N + 1) * args.D
    total = len(edges) * p
    counts = parse_coupled_counts(args.coupled_counts, total)
    if args.include_full_baseline:
        counts = sorted(set(counts + [total]))
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    unknown = [family for family in families if family not in MASK_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown mask families: {unknown}")
    seeds = parse_int_set(args.candidate_seeds)

    mask_dir = os.path.join(args.output_root, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    physical_name = edge_payload.get("name", os.path.splitext(os.path.basename(args.edge_json))[0])
    base_metrics = compute_topology_metrics(n_nodes, edges, p=p, n_context=args.N, z_dim=args.D)
    edge_participation = np.asarray(base_metrics["edge_participation"], dtype=float)

    rows = []
    seen = set()
    idx = 0
    for count in counts:
        for family in families:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                mask = MASK_BUILDERS[family](len(edges), p, count, rng, edge_participation)
                mask = validate_input_mask(mask, len(edges), p)
                key = mask.tobytes()
                if key in seen:
                    continue
                seen.add(key)

                mask_name = f"{physical_name}__mask{idx:04d}_{family}_c{count}_seed{seed}"
                input_mask_json = os.path.abspath(os.path.join(mask_dir, f"{mask_name}.json"))
                summary = input_mask_summary(mask)
                metrics = compute_topology_metrics(
                    n_nodes,
                    edges,
                    p=p,
                    input_mask=mask,
                    n_context=args.N,
                    z_dim=args.D,
                )
                metrics["physical_topology_name"] = physical_name
                metrics["d_rel_minus_n_req"] = int(metrics["d_rel"] - n_req)

                payload = {
                    "name": mask_name,
                    "physical_topology_name": physical_name,
                    "mask_family": family,
                    "seed": seed,
                    "n_nodes": n_nodes,
                    "n_edges": len(edges),
                    "p": p,
                    "coupled_parameter_count": count,
                    "edges": [list(edge) for edge in edges],
                    "input_mask": mask.tolist(),
                    "mask_summary": summary,
                    "topology_metrics": metrics,
                }
                with open(input_mask_json, "w") as f:
                    json.dump(json_ready(payload), f, indent=2)

                rows.append(
                    metric_row(
                        idx,
                        mask_name,
                        family,
                        seed,
                        metrics,
                        summary,
                        os.path.abspath(args.edge_json),
                        input_mask_json,
                    )
                )
                idx += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge_json", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--N", type=int, default=4, help="Number of context items.")
    parser.add_argument("--D", type=int, default=4, help="Input dimension.")
    parser.add_argument(
        "--families",
        type=str,
        default="entry_random,edge_block,coord_block,balanced,high_participation_edges,low_participation_edges",
    )
    parser.add_argument(
        "--coupled_counts",
        type=str,
        default="0.5",
        help="Comma-separated absolute counts or fractions of n_edges*p, e.g. '200' or '0.5'.",
    )
    parser.add_argument("--candidate_seeds", type=str, default="1:40")
    parser.add_argument("--select_masks", type=int, default=16)
    parser.add_argument("--include_full_baseline", action="store_true")
    parser.add_argument(
        "--selection_features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
    )
    args = parser.parse_args()

    rows = build_rows(args)
    if not rows:
        raise SystemExit("No input masks generated")

    features = [item.strip() for item in args.selection_features.split(",") if item.strip()]
    selected = select_diverse_rows(rows, args.select_masks, features, family_key="mask_family")
    for idx, row in enumerate(rows):
        row["selected"] = 1 if idx in selected else 0

    library_csv = os.path.join(args.output_root, "library.csv")
    selected_csv = os.path.join(args.output_root, "selected.csv")
    write_csv(library_csv, rows)
    write_csv(selected_csv, [row for idx, row in enumerate(rows) if idx in selected])

    print(f"Physical topology: {os.path.abspath(args.edge_json)}")
    print(f"Generated candidate masks: {len(rows)}")
    print(f"Selected masks: {len(selected)}")
    print(f"Wrote {library_csv}")
    print(f"Wrote {selected_csv}")
    print("Selected summary:")
    for row in [row for idx, row in enumerate(rows) if idx in selected]:
        print(
            f"  {row['topology_id']} family={row['mask_family']} "
            f"coupled={row['input_coupled_parameter_count']} "
            f"d_rel={row['d_rel']} eff_rank_masked={float(row['effective_rank_D_masked']):.3f} "
            f"edge_load_gini={float(row['input_edge_load_gini']):.3f} "
            f"coord_load_gini={float(row['input_coord_load_gini']):.3f}"
        )


if __name__ == "__main__":
    main()
