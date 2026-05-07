"""Finalize essential input-mask retrain sweeps and refresh reports.

This wrapper is intentionally conservative. It first checks that each
``essential_inputmask50_retrain`` directory contains the expected number of
``results.pkl`` files, then runs the standard collection/aggregation pipeline,
joins retrained masks back to their source-mask metadata, and optionally
regenerates the focused input-mask report.
"""

import argparse
import csv
import os
import subprocess
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_experiment(raw):
    if "=" in raw:
        name, root = raw.split("=", 1)
    else:
        root = raw
        name = os.path.basename(os.path.abspath(root.rstrip(os.sep)))
    return name, os.path.abspath(root)


def parse_seeds(raw):
    return [int(seed) for seed in raw.split(",") if seed.strip()]


def count_selected(path):
    with open(path, newline="") as f:
        return sum(
            1
            for row in csv.DictReader(f)
            if str(row.get("selected", "1")) in {"1", "True", "true"}
        )


def selected_topology_ids(path):
    rows = []
    missing = []
    with open(path, newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            if str(row.get("selected", "1")) not in {"1", "True", "true"}:
                continue
            topology_id = row.get("topology_id")
            if not topology_id:
                missing.append(str(idx))
                continue
            rows.append(topology_id)
    if missing:
        raise ValueError(
            f"{path}: selected rows missing topology_id at CSV row indexes "
            + ", ".join(missing[:5])
        )
    return rows


def count_files(root, filename):
    total = 0
    for _, _, files in os.walk(root):
        if filename in files:
            total += 1
    return total


def find_files(root, filename):
    paths = []
    for current, _, files in os.walk(root):
        if filename in files:
            paths.append(os.path.join(current, filename))
    return sorted(paths)


def exact_retrain_status(root, topology_ids, seeds):
    expected_paths = [
        os.path.join(root, f"{topology_id}_trainseed{seed}", "results.pkl")
        for topology_id in topology_ids
        for seed in seeds
    ]
    missing = [path for path in expected_paths if not os.path.exists(path)]
    unexpected = sorted(set(find_files(root, "results.pkl")) - set(expected_paths))
    return {
        "expected_paths": expected_paths,
        "completed": len(expected_paths) - len(missing),
        "missing": missing,
        "unexpected": unexpected,
    }


def run_command(parts, dry_run=False):
    print(" ".join(parts))
    if dry_run:
        return
    subprocess.run(parts, check=True, cwd=THIS_DIR)


def python_script(script, *args):
    return [sys.executable, script, *args]


def experiment_paths(root):
    return {
        "selected_csv": os.path.join(root, "essential_inputmask50", "selected.csv"),
        "comparison_csv": os.path.join(root, "essential_inputmask50", "retrain_comparison.csv"),
        "comparison_json": os.path.join(root, "essential_inputmask50", "retrain_comparison.json"),
        "retrain_root": os.path.join(root, "essential_inputmask50_retrain"),
        "retrain_aggregate_csv": os.path.join(
            root,
            "essential_inputmask50_retrain",
            "topology_seed_aggregates.csv",
        ),
    }


def check_completion(name, root, seeds):
    paths = experiment_paths(root)
    selected_csv = paths["selected_csv"]
    if not os.path.exists(selected_csv):
        raise FileNotFoundError(f"{name}: missing {selected_csv}")
    topology_ids = selected_topology_ids(selected_csv)
    selected = len(topology_ids)
    expected = selected * len(seeds)
    exact = exact_retrain_status(paths["retrain_root"], topology_ids, seeds)
    completed = exact["completed"]
    total_results = completed + len(exact["unexpected"])
    print(
        f"{name}: selected_masks={selected} seeds={len(seeds)} "
        f"expected={expected} completed={completed} total_results={total_results}"
    )
    if exact["missing"]:
        print("  missing examples:")
        for path in exact["missing"][:5]:
            print(f"    {path}")
    if exact["unexpected"]:
        print("  unexpected result examples:")
        for path in exact["unexpected"][:5]:
            print(f"    {path}")
    return {
        "name": name,
        "root": root,
        "selected": selected,
        "expected": expected,
        "completed": completed,
        "unexpected": len(exact["unexpected"]),
        "paths": paths,
    }


def finalize_experiment(status, args):
    paths = status["paths"]
    run_command(
        python_script(
            "finalize_topology_sweep.py",
            "--input_root",
            paths["retrain_root"],
        ),
        dry_run=args.dry_run,
    )
    run_command(
        python_script(
            "compare_essential_retrains.py",
            "--selected_csv",
            paths["selected_csv"],
            "--retrain_aggregate_csv",
            paths["retrain_aggregate_csv"],
            "--output_csv",
            paths["comparison_csv"],
            "--output_json",
            paths["comparison_json"],
        ),
        dry_run=args.dry_run,
    )


def refresh_report(experiments, args):
    if not args.output_md and not args.output_json:
        return
    if not args.output_md or not args.output_json:
        raise SystemExit("Provide both --output_md and --output_json, or neither")
    parts = python_script("make_input_mask_report.py")
    for name, root in experiments:
        parts.extend(["--experiment", f"{name}={root}"])
    parts.extend(["--output_md", args.output_md, "--output_json", args.output_json])
    run_command(parts, dry_run=args.dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Source experiment root as NAME=PATH or PATH. May be repeated.",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--allow_extra", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    experiments = [parse_experiment(raw) for raw in args.experiment]
    seeds = parse_seeds(args.seeds)
    statuses = [check_completion(name, root, seeds) for name, root in experiments]
    incomplete = [
        status
        for status in statuses
        if status["completed"] != status["expected"]
    ]
    unexpected = [
        status
        for status in statuses
        if status["unexpected"]
    ]
    if incomplete and not args.allow_partial:
        for status in incomplete:
            print(
                f"Incomplete: {status['name']} has "
                f"{status['completed']}/{status['expected']} results.pkl files"
            )
        raise SystemExit("Retrain outputs are incomplete; use --allow_partial only for diagnostics")
    if unexpected and not args.allow_extra:
        for status in unexpected:
            print(
                f"Unexpected retrain outputs: {status['name']} has "
                f"{status['unexpected']} extra results.pkl files"
            )
        raise SystemExit("Retrain output root contains unexpected results; remove them or use --allow_extra")

    for status in statuses:
        finalize_experiment(status, args)
    refresh_report(experiments, args)


if __name__ == "__main__":
    main()
