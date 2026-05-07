"""Build a fixed-edge-count topology library stratified by tree geometry."""

import argparse
import csv
import json
import math
import os

import numpy as np

from topology_metrics import compute_topology_metrics, graph_from_family


DEFAULT_FEATURES = [
    "effective_rank_D",
    "condition_number_D_log",
    "root_tree_count_gini",
    "edge_participation_gini",
    "bottleneck_edge_fraction_095",
    "mean_shortest_path",
    "in_degree_cv",
    "out_degree_cv",
    "n_trees_total_enum_log",
]


CSV_FIELDS = [
    "idx",
    "selected",
    "topology_id",
    "topology_name",
    "family",
    "seed",
    "n_nodes",
    "n_edges",
    "p",
    "edge_json",
    "d_rel",
    "comparison_branch_common_d_rel_min",
    "comparison_branch_common_d_rel_mean",
    "comparison_branch_common_d_rel_max",
    "comparison_branch_common_d_rel_gini",
    "comparison_branch_d_rel_min",
    "comparison_branch_d_rel_mean",
    "comparison_branch_d_rel_max",
    "comparison_branch_d_rel_gini",
    "rank_D",
    "effective_rank_D",
    "condition_number_D",
    "condition_number_D_log",
    "root_tree_count_gini",
    "edge_participation_gini",
    "bottleneck_edge_fraction_095",
    "mean_shortest_path",
    "in_degree_cv",
    "out_degree_cv",
    "n_trees_total_enum",
    "n_trees_total_enum_log",
]


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


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


def finite_float(value, default=0.0):
    if value is None:
        return default
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def metric_row(idx, family, seed, spec, metrics, edge_json):
    condition_number = metrics.get("condition_number_D")
    total_trees = metrics.get("n_trees_total_enum", 0)
    return {
        "idx": idx,
        "selected": 0,
        "topology_id": f"g{idx:04d}_{family}_seed{seed}",
        "topology_name": spec.name,
        "family": family,
        "seed": seed,
        "n_nodes": metrics["n_nodes"],
        "n_edges": metrics["n_edges"],
        "p": metrics["p"],
        "edge_json": edge_json,
        "d_rel": metrics["d_rel"],
        "comparison_branch_common_d_rel_min": metrics.get("comparison_branch_common_d_rel_min"),
        "comparison_branch_common_d_rel_mean": metrics.get("comparison_branch_common_d_rel_mean"),
        "comparison_branch_common_d_rel_max": metrics.get("comparison_branch_common_d_rel_max"),
        "comparison_branch_common_d_rel_gini": metrics.get("comparison_branch_common_d_rel_gini"),
        "comparison_branch_d_rel_min": metrics.get("comparison_branch_d_rel_min"),
        "comparison_branch_d_rel_mean": metrics.get("comparison_branch_d_rel_mean"),
        "comparison_branch_d_rel_max": metrics.get("comparison_branch_d_rel_max"),
        "comparison_branch_d_rel_gini": metrics.get("comparison_branch_d_rel_gini"),
        "rank_D": metrics["rank_D"],
        "effective_rank_D": metrics["effective_rank_D"],
        "condition_number_D": condition_number,
        "condition_number_D_log": math.log10(finite_float(condition_number, 1.0)),
        "root_tree_count_gini": metrics["root_tree_count_gini"],
        "edge_participation_gini": metrics["edge_participation_gini"],
        "bottleneck_edge_fraction_095": metrics["bottleneck_edge_fraction_095"],
        "mean_shortest_path": metrics["mean_shortest_path"],
        "in_degree_cv": metrics["in_degree_cv"],
        "out_degree_cv": metrics["out_degree_cv"],
        "n_trees_total_enum": total_trees,
        "n_trees_total_enum_log": math.log10(max(1.0, float(total_trees))),
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


def select_diverse_rows(rows, k, features):
    if k <= 0 or k >= len(rows):
        return set(range(len(rows)))

    z = standardized_matrix(rows, features)
    selected = []

    def add_if_new_metric_point(idx):
        if idx in selected:
            return False
        if any(np.allclose(z[idx, :], z[existing, :], atol=1e-9) for existing in selected):
            return False
        selected.append(idx)
        return True

    for col in range(z.shape[1]):
        for idx in (int(np.argmin(z[:, col])), int(np.argmax(z[:, col]))):
            add_if_new_metric_point(idx)
            if len(selected) >= k:
                return set(selected)

    while len(selected) < k:
        selected_z = z[selected, :]
        distances = np.sqrt(((z[:, None, :] - selected_z[None, :, :]) ** 2).sum(axis=2))
        min_distances = distances.min(axis=1)
        min_distances[selected] = -1.0
        selected.append(int(np.argmax(min_distances)))

    return set(selected)


def build_rows(args):
    p = (args.N + 1) * args.D
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    seeds = parse_int_set(args.candidate_seeds)
    topology_dir = os.path.join(args.output_root, "topologies")
    os.makedirs(topology_dir, exist_ok=True)

    rows = []
    seen_edges = set()
    idx = 0
    for family in families:
        for seed in seeds:
            try:
                spec = graph_from_family(family, args.n_nodes, args.n_edges, seed=seed)
            except ValueError as exc:
                if args.verbose:
                    print(f"Skipping {family} seed={seed}: {exc}")
                continue
            edge_key = tuple(spec.edges)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            metrics = compute_topology_metrics(
                spec.n_nodes,
                spec.edges,
                p=p,
                n_context=args.N,
                z_dim=args.D,
            )
            if not metrics["strongly_connected"]:
                continue

            edge_json = os.path.abspath(
                os.path.join(topology_dir, f"g{idx:04d}_{family}_seed{seed}.json")
            )
            payload = {
                "name": f"{family}_n{args.n_nodes}_m{args.n_edges}_seed{seed}",
                "family": family,
                "seed": seed,
                "n_nodes": spec.n_nodes,
                "edges": [list(edge) for edge in spec.edges],
                "metrics": metrics,
            }
            with open(edge_json, "w") as f:
                json.dump(json_ready(payload), f, indent=2)

            rows.append(metric_row(idx, family, seed, spec, metrics, edge_json))
            idx += 1

    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--n_nodes", type=int, default=6)
    parser.add_argument("--n_edges", type=int, default=20)
    parser.add_argument("--N", type=int, default=4, help="Number of context items.")
    parser.add_argument("--D", type=int, default=4, help="Input dimension.")
    parser.add_argument(
        "--families",
        type=str,
        default=(
            "cycle_chords,random_sc,hub_spoke,two_module,"
            "degree_balanced,bottleneck_bridge,redundant_paths"
        ),
    )
    parser.add_argument("--candidate_seeds", type=str, default="1:80")
    parser.add_argument("--select_topologies", type=int, default=16)
    parser.add_argument(
        "--selection_features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rows = build_rows(args)
    if not rows:
        raise SystemExit("No strongly connected candidate topologies generated")

    features = [item.strip() for item in args.selection_features.split(",") if item.strip()]
    selected = select_diverse_rows(rows, args.select_topologies, features)
    for idx, row in enumerate(rows):
        row["selected"] = 1 if idx in selected else 0

    library_csv = os.path.join(args.output_root, "library.csv")
    selected_csv = os.path.join(args.output_root, "selected.csv")
    write_csv(library_csv, rows)
    write_csv(selected_csv, [row for idx, row in enumerate(rows) if idx in selected])

    print(f"Generated candidates: {len(rows)}")
    print(f"Selected topologies: {len(selected)}")
    print(f"Wrote {library_csv}")
    print(f"Wrote {selected_csv}")
    print("Selected summary:")
    for row in [row for idx, row in enumerate(rows) if idx in selected]:
        print(
            f"  {row['topology_id']} family={row['family']} "
            f"d_rel={row['d_rel']} eff_rank={float(row['effective_rank_D']):.3f} "
            f"root_gini={float(row['root_tree_count_gini']):.3f} "
            f"edge_gini={float(row['edge_participation_gini']):.3f}"
        )


if __name__ == "__main__":
    main()
