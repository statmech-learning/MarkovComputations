"""Build mechanism-isolating fixed-count topology sweep plans.

The goal is not to add more broad sweeps.  This utility scans topology or
capacity CSVs and proposes controlled contrasts that isolate one topological
mechanism at a time, for example: same ``d_rel`` but different bottleneck
participation, or same tree count but different root balance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Iterable, List, Optional, Sequence, Tuple


CONTRASTS = [
    {
        "name": "same_drel_different_bottleneck_participation",
        "group_cols": ["regime", "n_nodes", "n_edges", "d_rel"],
        "contrast_cols": ["bottleneck_edge_fraction_095", "edge_participation_gini"],
        "rationale": "Hold relative tree rank fixed while separating bottleneck/edge-participation heterogeneity.",
    },
    {
        "name": "same_tree_count_different_root_balance",
        "group_cols": ["regime", "n_nodes", "n_edges", "n_trees_total_enum"],
        "contrast_cols": ["root_tree_count_gini"],
        "rationale": "Hold total rooted-tree count fixed while separating tree-count imbalance across roots.",
    },
    {
        "name": "same_degree_sequence_different_normal_fan",
        "group_cols": ["regime", "n_nodes", "n_edges", "in_degree_cv", "out_degree_cv", "d_rel"],
        "contrast_cols": ["normal_fan_branch_tree_nmi_mean", "capacity_normal_fan_branch_tree_nmi_mean"],
        "rationale": "Approximately hold degree sequence/rank fixed while separating sampled tree-polytope normal-fan organization.",
    },
    {
        "name": "same_physical_graph_permuted_input_masks",
        "group_cols": ["regime", "physical_topology_name", "edge_json"],
        "contrast_cols": ["input_coord_load_gini", "input_edge_load_gini"],
        "rationale": "Hold the physical graph fixed while separating input-coordinate load heterogeneity across masks.",
    },
    {
        "name": "same_mask_count_different_coordinate_load",
        "group_cols": ["regime", "n_nodes", "n_edges", "input_coupled_parameter_count"],
        "contrast_cols": ["input_coord_load_gini", "input_edge_load_gini"],
        "rationale": "Hold mask count fixed while separating coordinate-load and edge-load heterogeneity.",
    },
]


def parse_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_csv(path: str, regime: Optional[str] = None) -> List[dict]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    label = regime or os.path.basename(os.path.dirname(os.path.abspath(path)))
    for row in rows:
        row.setdefault("source_csv", path)
        row.setdefault("regime", label)
    return rows


def rounded_key(row: dict, columns: Sequence[str], ndigits: int = 6) -> Optional[Tuple[str, ...]]:
    values = []
    for column in columns:
        value = row.get(column)
        numeric = parse_float(value)
        if numeric is not None:
            values.append(str(round(numeric, ndigits)))
        elif value not in (None, ""):
            values.append(str(value))
        else:
            return None
    return tuple(values)


def first_present(row: dict, columns: Sequence[str]) -> Optional[str]:
    for column in columns:
        if parse_float(row.get(column)) is not None:
            return column
    return None


def row_label(row: dict) -> str:
    for key in ["topology_name", "topology_id", "mask_name", "input_mask_name", "label"]:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def select_extreme_pair(rows: Sequence[dict], group_cols: Sequence[str], contrast_col: str) -> Optional[dict]:
    groups = defaultdict(list)
    for row in rows:
        value = parse_float(row.get(contrast_col))
        key = rounded_key(row, group_cols)
        if value is None or key is None:
            continue
        groups[key].append(row)

    best = None
    for key, items in groups.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda row: parse_float(row.get(contrast_col)))
        low = items[0]
        high = items[-1]
        low_value = parse_float(low.get(contrast_col))
        high_value = parse_float(high.get(contrast_col))
        if low_value is None or high_value is None:
            continue
        delta = high_value - low_value
        if delta <= 0:
            continue
        candidate = {
            "group_key": key,
            "contrast_metric": contrast_col,
            "contrast_delta": delta,
            "low": low,
            "high": high,
        }
        if best is None or candidate["contrast_delta"] > best["contrast_delta"]:
            best = candidate
    return best


def build_plan(rows: Sequence[dict], contrasts: Sequence[dict] = CONTRASTS) -> List[dict]:
    plan = []
    for contrast in contrasts:
        selected_metric = None
        selected_pair = None
        for metric in contrast["contrast_cols"]:
            if any(parse_float(row.get(metric)) is not None for row in rows):
                pair = select_extreme_pair(rows, contrast["group_cols"], metric)
                if pair is not None:
                    selected_metric = metric
                    selected_pair = pair
                    break
        if selected_pair is None:
            plan.append(
                {
                    "name": contrast["name"],
                    "status": "unavailable",
                    "rationale": contrast["rationale"],
                    "reason": "No group contained at least two rows with the required fixed columns and contrast metric.",
                    "group_cols": contrast["group_cols"],
                    "candidate_metrics": contrast["contrast_cols"],
                }
            )
            continue
        low = selected_pair["low"]
        high = selected_pair["high"]
        plan.append(
            {
                "name": contrast["name"],
                "status": "ready",
                "rationale": contrast["rationale"],
                "group_cols": contrast["group_cols"],
                "group_key": list(selected_pair["group_key"]),
                "contrast_metric": selected_metric,
                "contrast_delta": selected_pair["contrast_delta"],
                "low_item": row_label(low),
                "high_item": row_label(high),
                "low_family": low.get("family", low.get("mask_family", "")),
                "high_family": high.get("family", high.get("mask_family", "")),
                "low_value": parse_float(low.get(selected_metric)),
                "high_value": parse_float(high.get(selected_metric)),
                "low_source_csv": low.get("source_csv", ""),
                "high_source_csv": high.get("source_csv", ""),
            }
        )
    return plan


def markdown_report(plan: Sequence[dict]) -> str:
    lines = [
        "# Mechanism-Isolating Topology Sweep Plan",
        "",
        "This plan selects fixed-count contrasts designed to isolate mechanisms rather than merely add more data.",
        "",
        "| contrast | status | fixed fields | varied metric | delta | low item | high item |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for item in plan:
        fixed = ", ".join(item.get("group_cols", []))
        lines.append(
            f"| {item['name']} | {item['status']} | {fixed} | "
            f"{item.get('contrast_metric', '')} | {item.get('contrast_delta', '')} | "
            f"{item.get('low_item', '')} | {item.get('high_item', '')} |"
        )
    lines.extend(["", "## Details", ""])
    for item in plan:
        lines.append(f"### {item['name']}")
        lines.append("")
        lines.append(item.get("rationale", ""))
        lines.append("")
        if item.get("status") != "ready":
            lines.append(f"Status: `{item.get('status')}`. {item.get('reason', '')}")
            lines.append("")
            continue
        lines.extend(
            [
                f"- Fixed key: `{item.get('group_key')}`",
                f"- Contrast metric: `{item.get('contrast_metric')}`",
                f"- Low item: `{item.get('low_item')}` ({item.get('low_family')}) = `{item.get('low_value')}`",
                f"- High item: `{item.get('high_item')}` ({item.get('high_family')}) = `{item.get('high_value')}`",
                f"- Delta: `{item.get('contrast_delta')}`",
                "",
            ]
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    rows = []
    for path in args.input_csv:
        rows.extend(load_csv(path))
    plan = build_plan(rows)
    payload = {"n_rows": len(rows), "input_csv": args.input_csv, "contrasts": plan}
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(args.output_md, "w") as handle:
        handle.write(markdown_report(plan))
    print(f"Wrote mechanism-isolating plan with {len(plan)} contrasts to {args.output_md}")


if __name__ == "__main__":
    main()
