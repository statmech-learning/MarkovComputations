"""Create a multi-regime fixed-count topology sweep plan.

This is a planning layer over ``make_topology_library.py`` and
``submit_topology_library_sweep.py``.  It does not submit jobs.  It writes a
CSV of regimes and optional shell commands for generating selected topology
libraries and launching matched training arrays across ``N_n``, ``m``,
``N_c``, and ``D``.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
from typing import Iterable, List


FIELDS = [
    "regime_id",
    "n_nodes",
    "n_edges",
    "edge_regime",
    "N",
    "D",
    "p",
    "n_req",
    "input_coupled_count",
    "output_root",
    "library_csv",
    "selected_csv",
    "train_output_root",
    "make_library_command",
    "submit_command",
]


def parse_int_list(raw: str) -> List[int]:
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
                raise ValueError(f"Invalid integer range {part!r}")
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(part))
    return sorted(set(values))


def edge_count_for_regime(n_nodes: int, regime: str) -> int:
    max_edges = n_nodes * (n_nodes - 1)
    if regime == "minimal":
        return n_nodes
    if regime == "sparse":
        return min(max_edges, n_nodes + max(1, n_nodes // 2))
    if regime == "intermediate":
        return min(max_edges, max(n_nodes, 2 * n_nodes + n_nodes // 2))
    if regime == "dense":
        return min(max_edges, max(n_nodes, int(round(0.65 * max_edges))))
    raise ValueError(f"Unknown edge regime {regime!r}")


def input_count(total: int, raw: str) -> int:
    if "." in raw:
        fraction = float(raw)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("--input_coupled_count fraction must be in [0, 1]")
        return int(round(total * fraction))
    count = int(raw)
    if not 0 <= count <= total:
        raise ValueError(f"input-coupled count {count} outside [0, {total}]")
    return count


def build_rows(args) -> List[dict]:
    n_nodes_values = parse_int_list(args.n_nodes)
    n_context_values = parse_int_list(args.n_context)
    z_dim_values = parse_int_list(args.z_dims)
    edge_regimes = [item.strip() for item in args.edge_regimes.split(",") if item.strip()]
    rows = []
    for n_nodes in n_nodes_values:
        for edge_regime in edge_regimes:
            n_edges = edge_count_for_regime(n_nodes, edge_regime)
            for n_context in n_context_values:
                for z_dim in z_dim_values:
                    p = (n_context + 1) * z_dim
                    n_req = 2 * n_context * (n_context + 1) * z_dim
                    coupled_count = input_count(n_edges * p, args.input_coupled_count)
                    regime_id = (
                        f"n{n_nodes}_m{n_edges}_{edge_regime}_"
                        f"N{n_context}_D{z_dim}_c{coupled_count}"
                    )
                    output_root = os.path.abspath(os.path.join(args.output_root, regime_id))
                    library_csv = os.path.join(output_root, "selected.csv")
                    train_output_root = os.path.abspath(os.path.join(args.train_root, regime_id))
                    make_library_command = " ".join(
                        [
                            "python3",
                            "make_topology_library.py",
                            "--output_root",
                            shlex.quote(output_root),
                            "--n_nodes",
                            str(n_nodes),
                            "--n_edges",
                            str(n_edges),
                            "--N",
                            str(n_context),
                            "--D",
                            str(z_dim),
                            "--families",
                            shlex.quote(args.families),
                            "--candidate_seeds",
                            shlex.quote(args.candidate_seeds),
                            "--select_topologies",
                            str(args.select_topologies),
                        ]
                    )
                    submit_command = " ".join(
                        [
                            "python3",
                            "submit_topology_library_sweep.py",
                            "--library_csv",
                            shlex.quote(library_csv),
                            "--output_root",
                            shlex.quote(train_output_root),
                            "--seeds",
                            shlex.quote(args.train_seeds),
                            "--N",
                            str(n_context),
                            "--D",
                            str(z_dim),
                            "--dry-run",
                        ]
                    )
                    rows.append(
                        {
                            "regime_id": regime_id,
                            "n_nodes": n_nodes,
                            "n_edges": n_edges,
                            "edge_regime": edge_regime,
                            "N": n_context,
                            "D": z_dim,
                            "p": p,
                            "n_req": n_req,
                            "input_coupled_count": coupled_count,
                            "output_root": output_root,
                            "library_csv": os.path.join(output_root, "library.csv"),
                            "selected_csv": library_csv,
                            "train_output_root": train_output_root,
                            "make_library_command": make_library_command,
                            "submit_command": submit_command,
                        }
                    )
    return rows


def write_csv(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_commands(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for row in rows:
            handle.write(f"# {row['regime_id']}\n")
            handle.write(row["make_library_command"] + "\n")
            handle.write(row["submit_command"] + "\n\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--commands_sh", default=None)
    parser.add_argument("--output_root", default="results/expanded_topology_libraries")
    parser.add_argument("--train_root", default="results/expanded_topology_sweeps")
    parser.add_argument("--n_nodes", default="4:8")
    parser.add_argument("--edge_regimes", default="sparse,intermediate,dense")
    parser.add_argument("--n_context", default="2,3")
    parser.add_argument("--z_dims", default="1,2")
    parser.add_argument("--input_coupled_count", default="0.5")
    parser.add_argument(
        "--families",
        default=(
            "cycle_chords,random_sc,hub_spoke,two_module,"
            "degree_balanced,bottleneck_bridge,redundant_paths"
        ),
    )
    parser.add_argument("--candidate_seeds", default="1:80")
    parser.add_argument("--select_topologies", type=int, default=16)
    parser.add_argument("--train_seeds", default="1,2,3,4,5")
    args = parser.parse_args()

    rows = build_rows(args)
    write_csv(args.output_csv, rows)
    if args.commands_sh:
        write_commands(args.commands_sh, rows)
    print(f"Wrote {len(rows)} topology sweep regimes to {args.output_csv}")
    if args.commands_sh:
        print(f"Wrote commands to {args.commands_sh}")


if __name__ == "__main__":
    main()
