"""Compare extracted physical motifs with matched control retrains.

``make_matched_motif_controls.py`` creates one or more matched control graphs
for each extracted physical essential motif. This script joins:

* the extracted source motifs,
* from-scratch source-motif retrain aggregates,
* matched-control metadata,
* from-scratch matched-control retrain aggregates,

and reports whether extracted motifs retrain better than matched random or
degree-rewired controls under the same training protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


FIELDS = [
    "control_topology_name",
    "control_topology_id",
    "control_kind",
    "source_topology_name",
    "source_topology_id",
    "source_original_target_mean",
    "source_original_target_max",
    "source_retrain_target_mean",
    "source_retrain_target_max",
    "control_retrain_target_mean",
    "control_retrain_target_max",
    "control_minus_source_retrain_mean",
    "control_minus_source_retrain_max",
    "control_beats_source_retrain_mean",
    "control_beats_source_retrain_max",
    "match_score",
    "source_n_edges",
    "control_n_edges",
    "source_d_rel",
    "control_d_rel",
    "source_effective_rank_D",
    "control_effective_rank_D",
    "source_root_tree_count_gini",
    "control_root_tree_count_gini",
    "source_edge_participation_gini",
    "control_edge_participation_gini",
]


def parse_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_or_empty(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def load_rows(path: str) -> List[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def index_by(rows: Sequence[dict], keys: Sequence[str]) -> Dict[str, dict]:
    out = {}
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value not in (None, "") and str(value) not in out:
                out[str(value)] = row
    return out


def source_lookup(row: dict, sources_by_key: Dict[str, dict]) -> Optional[dict]:
    for key in ("source_topology_id", "source_topology_name"):
        value = row.get(key)
        if value not in (None, "") and str(value) in sources_by_key:
            return sources_by_key[str(value)]
    return None


def retrain_lookup(row: dict, retrains_by_key: Dict[str, dict]) -> Optional[dict]:
    for key in ("topology_name", "topology_id", "group"):
        value = row.get(key)
        if value not in (None, "") and str(value) in retrains_by_key:
            return retrains_by_key[str(value)]
    return None


def bool_from_compare(lhs: Optional[float], rhs: Optional[float]) -> Optional[int]:
    if lhs is None or rhs is None:
        return None
    return int(lhs > rhs)


def joined_rows(
    source_rows: Sequence[dict],
    source_retrain_rows: Sequence[dict],
    control_rows: Sequence[dict],
    control_retrain_rows: Sequence[dict],
) -> List[dict]:
    sources_by_key = index_by(source_rows, ["topology_id", "topology_name"])
    source_retrains_by_key = index_by(source_retrain_rows, ["topology_name", "group"])
    control_retrains_by_key = index_by(control_retrain_rows, ["topology_name", "group"])

    rows = []
    for control in control_rows:
        control_retrain = retrain_lookup(control, control_retrains_by_key)
        source = source_lookup(control, sources_by_key)
        if control_retrain is None or source is None:
            continue
        source_retrain = source_retrains_by_key.get(source.get("topology_name", ""))
        source_mean = parse_float((source_retrain or {}).get("target_mean"))
        source_max = parse_float((source_retrain or {}).get("target_max"))
        control_mean = parse_float(control_retrain.get("target_mean"))
        control_max = parse_float(control_retrain.get("target_max"))
        rows.append(
            {
                "control_topology_name": control.get("topology_name", ""),
                "control_topology_id": control.get("topology_id", ""),
                "control_kind": control.get("control_kind", ""),
                "source_topology_name": source.get("topology_name", control.get("source_topology_name", "")),
                "source_topology_id": source.get("topology_id", control.get("source_topology_id", "")),
                "source_original_target_mean": parse_float(source.get("source_test_novel_classes_mean")),
                "source_original_target_max": parse_float(source.get("source_test_novel_classes_max")),
                "source_retrain_target_mean": source_mean,
                "source_retrain_target_max": source_max,
                "control_retrain_target_mean": control_mean,
                "control_retrain_target_max": control_max,
                "control_minus_source_retrain_mean": (
                    control_mean - source_mean
                    if control_mean is not None and source_mean is not None
                    else None
                ),
                "control_minus_source_retrain_max": (
                    control_max - source_max
                    if control_max is not None and source_max is not None
                    else None
                ),
                "control_beats_source_retrain_mean": bool_from_compare(control_mean, source_mean),
                "control_beats_source_retrain_max": bool_from_compare(control_max, source_max),
                "match_score": parse_float(control.get("match_score")),
                "source_n_edges": parse_float(source.get("n_edges")),
                "control_n_edges": parse_float(control_retrain.get("n_edges") or control.get("n_edges")),
                "source_d_rel": parse_float(source.get("d_rel")),
                "control_d_rel": parse_float(control_retrain.get("d_rel") or control.get("d_rel")),
                "source_effective_rank_D": parse_float(source.get("effective_rank_D")),
                "control_effective_rank_D": parse_float(
                    control_retrain.get("effective_rank_D") or control.get("effective_rank_D")
                ),
                "source_root_tree_count_gini": parse_float(source.get("root_tree_count_gini")),
                "control_root_tree_count_gini": parse_float(
                    control_retrain.get("root_tree_count_gini") or control.get("root_tree_count_gini")
                ),
                "source_edge_participation_gini": parse_float(source.get("edge_participation_gini")),
                "control_edge_participation_gini": parse_float(
                    control_retrain.get("edge_participation_gini") or control.get("edge_participation_gini")
                ),
            }
        )
    return rows


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    arr = [value for value in values if value is not None]
    return float(np.mean(arr)) if arr else None


def max_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    arr = [value for value in values if value is not None]
    return float(np.max(arr)) if arr else None


def summarize_group(rows: Sequence[dict]) -> dict:
    return {
        "n": len(rows),
        "n_sources": len({row.get("source_topology_name") for row in rows}),
        "control_target_mean_mean": mean(row["control_retrain_target_mean"] for row in rows),
        "control_target_max_mean": mean(row["control_retrain_target_max"] for row in rows),
        "control_target_max_best": max_or_none(row["control_retrain_target_max"] for row in rows),
        "source_retrain_target_mean_mean": mean(row["source_retrain_target_mean"] for row in rows),
        "source_retrain_target_max_mean": mean(row["source_retrain_target_max"] for row in rows),
        "control_minus_source_retrain_mean_mean": mean(
            row["control_minus_source_retrain_mean"] for row in rows
        ),
        "control_minus_source_retrain_max_mean": mean(
            row["control_minus_source_retrain_max"] for row in rows
        ),
        "control_win_rate_mean": mean(row["control_beats_source_retrain_mean"] for row in rows),
        "control_win_rate_max": mean(row["control_beats_source_retrain_max"] for row in rows),
        "match_score_mean": mean(row["match_score"] for row in rows),
        "source_d_rel_mean": mean(row["source_d_rel"] for row in rows),
        "control_d_rel_mean": mean(row["control_d_rel"] for row in rows),
    }


def summary(rows: Sequence[dict]) -> dict:
    by_kind = defaultdict(list)
    for row in rows:
        by_kind[row.get("control_kind", "")].append(row)
    return {
        "n_joined": len(rows),
        "overall": summarize_group(rows),
        "by_control_kind": {
            kind: summarize_group(items)
            for kind, items in sorted(by_kind.items())
        },
    }


def write_csv(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: finite_or_empty(row.get(field)) for field in FIELDS} for row in rows)


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def markdown_table(report: dict) -> str:
    rows = []
    for kind, item in report.get("by_control_kind", {}).items():
        rows.append(
            "| "
            + " | ".join(
                [
                    kind,
                    str(item["n"]),
                    str(item["n_sources"]),
                    fmt(item["control_target_mean_mean"]),
                    fmt(item["source_retrain_target_mean_mean"]),
                    fmt(item["control_minus_source_retrain_mean_mean"]),
                    fmt(item["control_win_rate_mean"], 3),
                    fmt(item["match_score_mean"], 3),
                ]
            )
            + " |"
        )
    header = [
        "| control kind | controls | sources | control mean ICL | source motif mean ICL | delta mean | win rate | match score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    return "\n".join(header + rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_csv", required=True)
    parser.add_argument("--source_retrain_csv", required=True)
    parser.add_argument("--control_csv", required=True)
    parser.add_argument("--control_retrain_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    rows = joined_rows(
        load_rows(args.source_csv),
        load_rows(args.source_retrain_csv),
        load_rows(args.control_csv),
        load_rows(args.control_retrain_csv),
    )
    report = summary(rows)
    write_csv(args.output_csv, rows)
    with open(args.output_json, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if args.output_md:
        with open(args.output_md, "w") as handle:
            handle.write("# Matched Motif Control Comparison\n\n")
            handle.write(f"Joined controls: `{report['n_joined']}`.\n\n")
            handle.write(markdown_table(report))
            handle.write("\n")

    print(f"Joined matched controls: {report['n_joined']}")
    for kind, item in report["by_control_kind"].items():
        print(
            f"{kind}: n={item['n']} "
            f"control_mean={fmt(item['control_target_mean_mean'])} "
            f"source_mean={fmt(item['source_retrain_target_mean_mean'])} "
            f"delta={fmt(item['control_minus_source_retrain_mean_mean'])} "
            f"win_rate={fmt(item['control_win_rate_mean'], 3)}"
        )


if __name__ == "__main__":
    main()
