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


def count_files(root, filename):
    total = 0
    for _, _, files in os.walk(root):
        if filename in files:
            total += 1
    return total


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
    selected = count_selected(selected_csv)
    expected = selected * len(seeds)
    completed = count_files(paths["retrain_root"], "results.pkl")
    print(f"{name}: selected_masks={selected} seeds={len(seeds)} expected={expected} completed={completed}")
    return {
        "name": name,
        "root": root,
        "selected": selected,
        "expected": expected,
        "completed": completed,
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
    if incomplete and not args.allow_partial:
        for status in incomplete:
            print(
                f"Incomplete: {status['name']} has "
                f"{status['completed']}/{status['expected']} results.pkl files"
            )
        raise SystemExit("Retrain outputs are incomplete; use --allow_partial only for diagnostics")

    for status in statuses:
        finalize_experiment(status, args)
    refresh_report(experiments, args)


if __name__ == "__main__":
    main()
