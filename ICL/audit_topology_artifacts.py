"""Audit topology-sweep artifacts without mutating experiment outputs.

This script is deliberately read-only. It is meant for long-running cluster
sweeps where training, mechanism analysis, essential-mask extraction, and
retraining may finish at different times. The audit reports concrete artifact
counts so recovery can use ``--missing_only`` or guarded finalizers instead of
guessing which stage completed.
"""

import argparse
import csv
import json
import os
import sys


SOURCE_FILES = [
    "topology_results.csv",
    "topology_regression.json",
    "topology_seed_aggregates.csv",
    "topology_seed_aggregates.json",
    "mechanism_results.csv",
    "mechanism_summary.json",
]

RETRAIN_FILES = [
    "topology_results.csv",
    "topology_regression.json",
    "topology_seed_aggregates.csv",
    "topology_seed_aggregates.json",
]


def parse_experiment(raw):
    if "=" in raw:
        name, root = raw.split("=", 1)
    else:
        root = raw
        name = os.path.basename(os.path.abspath(root.rstrip(os.sep)))
    return name, os.path.abspath(root)


def parse_seeds(raw):
    return [int(seed) for seed in raw.split(",") if seed.strip()]


def exists(path):
    return os.path.exists(path)


def count_files(root, filename, exclude_roots=None):
    total = 0
    if not os.path.isdir(root):
        return total
    exclude_roots = [
        os.path.abspath(path)
        for path in (exclude_roots or [])
    ]
    for current_dir, _, files in os.walk(root):
        current = os.path.abspath(current_dir)
        if any(current == excluded or current.startswith(excluded + os.sep) for excluded in exclude_roots):
            continue
        if filename in files:
            total += 1
    return total


def count_csv_rows(path, selected_only=False):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if selected_only:
        rows = [
            row
            for row in rows
            if str(row.get("selected", "1")) in {"1", "True", "true"}
        ]
    return len(rows)


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def count_lines(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def file_status(root, filenames):
    return {name: exists(os.path.join(root, name)) for name in filenames}


def source_status(root, exclude_roots=None):
    return {
        "results_pkl": count_files(root, "results.pkl", exclude_roots=exclude_roots),
        "topology_metrics_json": count_files(root, "topology_metrics.json", exclude_roots=exclude_roots),
        "mechanism_metrics_json": count_files(root, "mechanism_metrics.json", exclude_roots=exclude_roots),
        "files": file_status(root, SOURCE_FILES),
    }


def referenced_file_status(rows, field):
    missing = []
    present = 0
    for row in rows:
        path = row.get(field)
        if not path:
            missing.append("")
        elif os.path.exists(path):
            present += 1
        else:
            missing.append(path)
    return {
        "present": present,
        "missing": len(missing),
        "missing_examples": missing[:5],
    }


def manifest_status(path):
    rows = read_csv_rows(path)
    if not rows:
        return {
            "exists": os.path.exists(path),
            "rows": 0,
            "completed_at_manifest_time": 0,
            "missing_at_manifest_time": 0,
        }
    completed = sum(1 for row in rows if str(row.get("completed")) == "True")
    return {
        "exists": True,
        "rows": len(rows),
        "completed_at_manifest_time": completed,
        "missing_at_manifest_time": len(rows) - completed,
    }


def essential_inputmask_status(root, directory):
    essential_root = os.path.join(root, directory)
    library_csv = os.path.join(essential_root, "library.csv")
    selected_csv = os.path.join(essential_root, "selected.csv")
    summary_json = os.path.join(essential_root, "summary.json")
    selected_rows = read_csv_rows(selected_csv)
    return {
        "root": essential_root,
        "library_csv": exists(library_csv),
        "selected_csv": exists(selected_csv),
        "summary_json": exists(summary_json),
        "library_rows": count_csv_rows(library_csv),
        "selected_rows": count_csv_rows(selected_csv, selected_only=True),
        "summary": read_json(summary_json),
        "edge_json": referenced_file_status(selected_rows, "edge_json"),
        "input_mask_json": referenced_file_status(selected_rows, "input_mask_json"),
    }


def retrain_status(root, directory, selected_rows, seeds):
    retrain_root = os.path.join(root, directory)
    expected = None if selected_rows is None else selected_rows * len(seeds)
    completed = count_files(retrain_root, "results.pkl")
    manifest = os.path.join(retrain_root, "task_manifest.csv")
    commands = os.path.join(retrain_root, "_array_meta", "commands.txt")
    return {
        "root": retrain_root,
        "exists": os.path.isdir(retrain_root),
        "expected_results": expected,
        "completed_results": completed,
        "missing_results": None if expected is None else max(expected - completed, 0),
        "files": file_status(retrain_root, RETRAIN_FILES),
        "manifest": manifest_status(manifest),
        "array_commands": count_lines(commands),
    }


def comparison_status(root, essential_directory):
    essential_root = os.path.join(root, essential_directory)
    comparison_csv = os.path.join(essential_root, "retrain_comparison.csv")
    comparison_json = os.path.join(essential_root, "retrain_comparison.json")
    return {
        "comparison_csv": exists(comparison_csv),
        "comparison_json": exists(comparison_json),
        "comparison_rows": count_csv_rows(comparison_csv),
        "summary": read_json(comparison_json),
    }


def audit_experiment(name, root, args):
    source = source_status(
        root,
        exclude_roots=[
            os.path.join(root, args.essential_directory),
            os.path.join(root, args.retrain_directory),
        ],
    )
    essential = essential_inputmask_status(root, args.essential_directory)
    retrain = retrain_status(
        root,
        args.retrain_directory,
        essential["selected_rows"],
        args.seeds,
    )
    comparison = comparison_status(root, args.essential_directory)
    failures = []

    if args.require_source_results and source["results_pkl"] == 0:
        failures.append("source has no results.pkl files")
    if args.require_mechanisms:
        if source["mechanism_metrics_json"] == 0:
            failures.append("source has no mechanism_metrics.json files")
        elif source["mechanism_metrics_json"] < source["results_pkl"]:
            failures.append("source mechanism_metrics.json count is lower than results.pkl count")
    if args.require_essential_inputmask:
        if not essential["library_csv"]:
            failures.append(f"missing {args.essential_directory}/library.csv")
        if not essential["selected_csv"]:
            failures.append(f"missing {args.essential_directory}/selected.csv")
        if not essential["summary_json"]:
            failures.append(f"missing {args.essential_directory}/summary.json")
        if not essential["selected_rows"]:
            failures.append("no selected essential input masks")
        if essential["edge_json"]["missing"]:
            failures.append("selected essential masks reference missing edge_json files")
        if essential["input_mask_json"]["missing"]:
            failures.append("selected essential masks reference missing input_mask_json files")
    if args.require_essential_retrains:
        if retrain["expected_results"] is None:
            failures.append("cannot infer expected retrain count without selected.csv")
        elif retrain["completed_results"] != retrain["expected_results"]:
            failures.append(
                "essential retrains incomplete: "
                f"{retrain['completed_results']}/{retrain['expected_results']}"
            )
        missing_retrain_files = [
            filename
            for filename, present in retrain["files"].items()
            if not present
        ]
        if missing_retrain_files:
            failures.append("missing retrain summary files: " + ", ".join(missing_retrain_files))
        if not comparison["comparison_csv"] or not comparison["comparison_json"]:
            failures.append("missing essential retrain comparison files")

    return {
        "name": name,
        "root": root,
        "source": source,
        "essential_inputmask": essential,
        "essential_inputmask_retrain": retrain,
        "comparison": comparison,
        "failures": failures,
    }


def format_bool(value):
    return "yes" if value else "no"


def print_experiment(report):
    source = report["source"]
    essential = report["essential_inputmask"]
    retrain = report["essential_inputmask_retrain"]
    comparison = report["comparison"]
    print(f"{report['name']}: {report['root']}")
    print(
        "  source: "
        f"results={source['results_pkl']} "
        f"metrics={source['topology_metrics_json']} "
        f"mechanisms={source['mechanism_metrics_json']} "
        f"aggregates={format_bool(source['files']['topology_seed_aggregates.csv'])}"
    )
    print(
        "  essential_inputmask: "
        f"library={essential['library_rows']} "
        f"selected={essential['selected_rows']} "
        f"edge_json_missing={essential['edge_json']['missing']} "
        f"input_mask_json_missing={essential['input_mask_json']['missing']}"
    )
    print(
        "  essential_retrain: "
        f"completed={retrain['completed_results']} "
        f"expected={retrain['expected_results']} "
        f"missing={retrain['missing_results']} "
        f"manifest_rows={retrain['manifest']['rows']} "
        f"array_commands={retrain['array_commands']}"
    )
    print(
        "  comparison: "
        f"csv={format_bool(comparison['comparison_csv'])} "
        f"json={format_bool(comparison['comparison_json'])} "
        f"rows={comparison['comparison_rows']}"
    )
    if report["failures"]:
        for failure in report["failures"]:
            print(f"  FAIL: {failure}")
    else:
        print("  status: ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Experiment root as NAME=PATH or PATH. May be repeated.",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--essential_directory", type=str, default="essential_inputmask50")
    parser.add_argument("--retrain_directory", type=str, default="essential_inputmask50_retrain")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require_source_results", action="store_true")
    parser.add_argument("--require_mechanisms", action="store_true")
    parser.add_argument("--require_essential_inputmask", action="store_true")
    parser.add_argument("--require_essential_retrains", action="store_true")
    args = parser.parse_args()
    args.seeds = parse_seeds(args.seeds)

    reports = [
        audit_experiment(name, root, args)
        for name, root in [parse_experiment(raw) for raw in args.experiment]
    ]
    for report in reports:
        print_experiment(report)

    payload = {"experiments": reports}
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {os.path.abspath(args.output_json)}")

    failed = any(report["failures"] for report in reports)
    if failed and args.strict:
        raise SystemExit("Topology artifact audit failed")


if __name__ == "__main__":
    main()
