"""Compare extracted essential motifs with their from-scratch retrains."""

import argparse
import csv
import json
import math
import os

import numpy as np


DEFAULT_SELECTED = "essential_input50/selected.csv"
DEFAULT_RETRAIN_AGG = "essential_input50_retrain/topology_seed_aggregates.csv"
DEFAULT_INPUTMASK_SELECTED = "essential_inputmask50/selected.csv"
DEFAULT_INPUTMASK_RETRAIN_AGG = "essential_inputmask50_retrain/topology_seed_aggregates.csv"
BRANCH_METRIC_FALLBACKS = {
    "comparison_branch_common_d_rel_min": "comparison_branch_d_rel_min",
    "comparison_branch_common_d_rel_gini": "comparison_branch_d_rel_gini",
    "comparison_branch_input_overlap_min": "comparison_branch_input_count_min",
    "comparison_branch_input_overlap_gini": "comparison_branch_input_count_gini",
}


JOIN_FIELDS = [
    "topology_name",
    "topology_id",
    "source_labels",
    "n_edges",
    "d_rel",
    "comparison_branch_common_d_rel_min",
    "comparison_branch_common_d_rel_gini",
    "comparison_branch_common_d_rel_source",
    "comparison_branch_d_rel_min",
    "comparison_branch_d_rel_gini",
    "comparison_branch_input_overlap_min",
    "comparison_branch_input_overlap_gini",
    "comparison_branch_input_overlap_source",
    "effective_rank_D",
    "source_test_novel_classes_max",
    "source_test_novel_classes_mean",
    "source_target_accuracy_max",
    "source_target_accuracy_mean",
    "source_input_coupled_parameter_count_mean",
    "retrain_input_coupled_parameter_count",
    "retrain_target_max",
    "retrain_target_mean",
    "retrain_target_std",
    "retrain_retention_max",
    "retrain_retention_mean",
    "effective_rank_D_masked",
]


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def finite_or_empty(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for target, fallback in BRANCH_METRIC_FALLBACKS.items():
            if row.get(target) in (None, "") and row.get(fallback) not in (None, ""):
                row[target] = row[fallback]
                if target.startswith("comparison_branch_common_d_rel_"):
                    row["comparison_branch_common_d_rel_source"] = "legacy_branch_d_rel_fallback"
                if target.startswith("comparison_branch_input_overlap_"):
                    row["comparison_branch_input_overlap_source"] = "legacy_input_count_fallback"
        if row.get("comparison_branch_common_d_rel_min") not in (None, ""):
            if row.get("comparison_branch_common_d_rel_source") in (None, ""):
                row["comparison_branch_common_d_rel_source"] = "artifact"
        if row.get("comparison_branch_input_overlap_min") not in (None, ""):
            if row.get("comparison_branch_input_overlap_source") in (None, ""):
                row["comparison_branch_input_overlap_source"] = "artifact"
    return rows


def mean(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.mean(values))


def max_or_none(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.max(values))


def min_or_none(values):
    values = [value for value in values if value is not None]
    return None if not values else float(np.min(values))


def joined_rows(selected_rows, retrain_rows):
    selected_by_name = {row["topology_name"]: row for row in selected_rows}
    rows = []
    for retrain in retrain_rows:
        source = selected_by_name.get(retrain.get("topology_name"))
        if source is None:
            continue
        source_max = parse_float(source.get("source_test_novel_classes_max"))
        source_mean = parse_float(source.get("source_test_novel_classes_mean"))
        retrain_max = parse_float(retrain.get("target_max"))
        retrain_mean = parse_float(retrain.get("target_mean"))
        rows.append(
            {
                "topology_name": retrain.get("topology_name"),
                "topology_id": retrain.get("group"),
                "source_labels": source.get("source_labels"),
                "n_edges": parse_float(retrain.get("n_edges")),
                "d_rel": parse_float(retrain.get("d_rel")),
                "comparison_branch_common_d_rel_min": parse_float(
                    retrain.get("comparison_branch_common_d_rel_min")
                ),
                "comparison_branch_common_d_rel_gini": parse_float(
                    retrain.get("comparison_branch_common_d_rel_gini")
                ),
                "comparison_branch_common_d_rel_source": retrain.get(
                    "comparison_branch_common_d_rel_source",
                    "",
                ),
                "comparison_branch_d_rel_min": parse_float(
                    retrain.get("comparison_branch_d_rel_min")
                ),
                "comparison_branch_d_rel_gini": parse_float(
                    retrain.get("comparison_branch_d_rel_gini")
                ),
                "comparison_branch_input_overlap_min": parse_float(
                    retrain.get("comparison_branch_input_overlap_min")
                ),
                "comparison_branch_input_overlap_gini": parse_float(
                    retrain.get("comparison_branch_input_overlap_gini")
                ),
                "comparison_branch_input_overlap_source": retrain.get(
                    "comparison_branch_input_overlap_source",
                    "",
                ),
                "effective_rank_D": parse_float(retrain.get("effective_rank_D")),
                "source_test_novel_classes_max": source_max,
                "source_test_novel_classes_mean": source_mean,
                "source_target_accuracy_max": parse_float(source.get("source_target_accuracy_max")),
                "source_target_accuracy_mean": parse_float(source.get("source_target_accuracy_mean")),
                "source_input_coupled_parameter_count_mean": parse_float(
                    source.get("source_input_coupled_parameter_count_mean")
                ),
                "retrain_input_coupled_parameter_count": parse_float(
                    retrain.get("input_coupled_parameter_count")
                ),
                "retrain_target_max": retrain_max,
                "retrain_target_mean": retrain_mean,
                "retrain_target_std": parse_float(retrain.get("target_std")),
                "retrain_retention_max": (
                    retrain_max / source_max if retrain_max is not None and source_max else None
                ),
                "retrain_retention_mean": (
                    retrain_mean / source_mean if retrain_mean is not None and source_mean else None
                ),
                "effective_rank_D_masked": parse_float(retrain.get("effective_rank_D_masked")),
            }
        )
    return rows


def name_examples(rows, key):
    names = [row.get(key) for row in rows if row.get(key)]
    return ", ".join(names[:5]) if names else "none"


def summary(rows):
    return {
        "n_joined": len(rows),
        "source_max_mean": mean([row["source_test_novel_classes_max"] for row in rows]),
        "source_mean_mean": mean([row["source_test_novel_classes_mean"] for row in rows]),
        "retrain_max_mean": mean([row["retrain_target_max"] for row in rows]),
        "retrain_mean_mean": mean([row["retrain_target_mean"] for row in rows]),
        "retrain_max_best": max_or_none([row["retrain_target_max"] for row in rows]),
        "retrain_mean_best": max_or_none([row["retrain_target_mean"] for row in rows]),
        "retention_max_mean": mean([row["retrain_retention_max"] for row in rows]),
        "retention_mean_mean": mean([row["retrain_retention_mean"] for row in rows]),
        "n_edges_mean": mean([row["n_edges"] for row in rows]),
        "n_edges_min": min_or_none([row["n_edges"] for row in rows]),
        "n_edges_max": max_or_none([row["n_edges"] for row in rows]),
        "d_rel_mean": mean([row["d_rel"] for row in rows]),
        "comparison_branch_common_d_rel_min_mean": mean(
            [row["comparison_branch_common_d_rel_min"] for row in rows]
        ),
        "comparison_branch_d_rel_min_mean": mean(
            [row["comparison_branch_d_rel_min"] for row in rows]
        ),
        "comparison_branch_input_overlap_min_mean": mean(
            [row["comparison_branch_input_overlap_min"] for row in rows]
        ),
        "effective_rank_D_mean": mean([row["effective_rank_D"] for row in rows]),
        "effective_rank_D_masked_mean": mean([row["effective_rank_D_masked"] for row in rows]),
        "retrain_input_coupled_parameter_count_mean": mean(
            [row["retrain_input_coupled_parameter_count"] for row in rows]
        ),
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOIN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: finite_or_empty(row.get(field)) for field in JOIN_FIELDS})


def print_summary(report):
    print(f"Joined motifs: {report['n_joined']}")
    print(
        "Source novel ICL: "
        f"mean-source-max={report['source_max_mean']:.2f}, "
        f"mean-source-mean={report['source_mean_mean']:.2f}"
    )
    print(
        "Retrained motif novel ICL: "
        f"mean-max={report['retrain_max_mean']:.2f}, "
        f"mean-mean={report['retrain_mean_mean']:.2f}, "
        f"best-max={report['retrain_max_best']:.2f}"
    )
    print(
        "Retention: "
        f"max={report['retention_max_mean']:.3f}, "
        f"mean={report['retention_mean_mean']:.3f}"
    )
    print(
        "Motif size: "
        f"mean={report['n_edges_mean']:.2f}, "
        f"min={report['n_edges_min']:.0f}, "
        f"max={report['n_edges_max']:.0f}"
    )
    if report.get("comparison_branch_common_d_rel_min_mean") is not None:
        print(
            "Weakest common branch d_rel: "
            f"mean={report['comparison_branch_common_d_rel_min_mean']:.2f}"
        )
    elif report.get("comparison_branch_d_rel_min_mean") is not None:
        print(
            "Weakest branch d_rel: "
            f"mean={report['comparison_branch_d_rel_min_mean']:.2f}"
        )
    if report.get("retrain_input_coupled_parameter_count_mean") is not None:
        print(
            "Retrain input couplings: "
            f"mean={report['retrain_input_coupled_parameter_count_mean']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_root", type=str, default=None)
    parser.add_argument(
        "--layout",
        choices=["physical", "inputmask"],
        default="physical",
        help="Default selected/retrain paths to use with --base_root.",
    )
    parser.add_argument("--selected_csv", type=str, default=None)
    parser.add_argument("--retrain_aggregate_csv", type=str, default=None)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    if args.base_root:
        if args.layout == "inputmask":
            default_selected = DEFAULT_INPUTMASK_SELECTED
            default_retrain = DEFAULT_INPUTMASK_RETRAIN_AGG
        else:
            default_selected = DEFAULT_SELECTED
            default_retrain = DEFAULT_RETRAIN_AGG
        selected_csv = args.selected_csv or os.path.join(args.base_root, default_selected)
        retrain_csv = args.retrain_aggregate_csv or os.path.join(args.base_root, default_retrain)
    else:
        if not args.selected_csv or not args.retrain_aggregate_csv:
            raise SystemExit("Use --base_root or provide both input CSV paths")
        selected_csv = args.selected_csv
        retrain_csv = args.retrain_aggregate_csv

    selected_rows = load_rows(selected_csv)
    retrain_rows = load_rows(retrain_csv)
    rows = joined_rows(selected_rows, retrain_rows)
    if not rows:
        raise SystemExit(
            "No retrained motifs joined by topology_name. "
            f"selected examples: {name_examples(selected_rows, 'topology_name')}; "
            f"retrain examples: {name_examples(retrain_rows, 'topology_name')}"
        )
    rows.sort(
        key=lambda row: (
            row["retrain_target_max"] if row["retrain_target_max"] is not None else -1.0,
            row["retrain_target_mean"] if row["retrain_target_mean"] is not None else -1.0,
        ),
        reverse=True,
    )
    report = summary(rows)
    report.update(
        {
            "selected_csv": os.path.abspath(selected_csv),
            "retrain_aggregate_csv": os.path.abspath(retrain_csv),
            "output_csv": os.path.abspath(args.output_csv),
        }
    )
    write_csv(args.output_csv, rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print_summary(report)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
