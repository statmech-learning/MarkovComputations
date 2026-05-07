"""Recover interrupted essential input-mask retrain sweeps.

This is a conservative orchestration wrapper around the existing read-only
auditor, missing-only library submitter, and guarded finalizer. It is intended
for cluster recovery after SLURM arrays are interrupted or left partially
pending: inspect source artifacts, emit status manifests, resubmit only missing
retrain runs when requested, and finalize only through the completion-checked
``finalize_essential_inputmask_retrains.py`` path.
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


def python_script(script, *args):
    return [sys.executable, script, *args]


def run_command(parts, dry_run=False):
    print(" ".join(parts))
    if dry_run:
        return
    subprocess.run(parts, check=True, cwd=THIS_DIR)


def experiment_args(experiments):
    parts = []
    for name, root in experiments:
        parts.extend(["--experiment", f"{name}={root}"])
    return parts


def audit_command(experiments, seeds, require_retrains=False, strict=False):
    parts = python_script(
        "audit_topology_artifacts.py",
        *experiment_args(experiments),
        "--seeds",
        seeds,
        "--require_source_results",
        "--require_mechanisms",
        "--require_essential_inputmask",
    )
    if require_retrains:
        parts.append("--require_essential_retrains")
    if strict:
        parts.append("--strict")
    return parts


def status_command(root, seeds):
    return python_script(
        "submit_topology_library_sweep.py",
        "--library_csv",
        os.path.join(root, "essential_inputmask50", "selected.csv"),
        "--output_root",
        os.path.join(root, "essential_inputmask50_retrain"),
        "--seeds",
        seeds,
        "--status_only",
        "--manifest_csv",
        status_manifest_path(root),
    )


def status_manifest_path(root):
    return os.path.join(root, "essential_inputmask50_retrain", "task_manifest.csv")


def count_missing_from_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    missing = sum(1 for row in rows if str(row.get("completed")) != "True")
    return missing, len(rows)


def submit_missing_command(root, seeds, max_concurrent, clean=True, dry_run=False):
    parts = python_script(
        "submit_topology_library_sweep.py",
        "--library_csv",
        os.path.join(root, "essential_inputmask50", "selected.csv"),
        "--output_root",
        os.path.join(root, "essential_inputmask50_retrain"),
        "--seeds",
        seeds,
        "--missing_only",
        "--array",
        "--max-concurrent",
        str(max_concurrent),
    )
    if clean:
        parts.append("--clean")
    if dry_run:
        parts.append("--dry-run")
    return parts


def finalizer_command(experiments, seeds, output_md, output_json):
    return python_script(
        "finalize_essential_inputmask_retrains.py",
        *experiment_args(experiments),
        "--seeds",
        seeds,
        "--output_md",
        output_md,
        "--output_json",
        output_json,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Source experiment root as NAME=PATH or PATH. May be repeated.",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--submit_missing", action="store_true")
    parser.add_argument("--no-clean", dest="clean", action="store_false")
    parser.add_argument("--finalize_if_complete", action="store_true")
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiments = [parse_experiment(raw) for raw in args.experiment]
    run_command(audit_command(experiments, args.seeds), dry_run=args.dry_run)

    submitted_missing = False
    for _, root in experiments:
        run_command(status_command(root, args.seeds), dry_run=args.dry_run)
        missing = None
        total = None
        if not args.dry_run:
            missing, total = count_missing_from_manifest(status_manifest_path(root))
            print(f"Missing retrain tasks for {root}: {missing}/{total}")
        if args.submit_missing:
            if missing == 0:
                print(f"No missing retrain tasks for {root}; skipping missing-only submit")
                continue
            run_command(
                submit_missing_command(
                    root,
                    args.seeds,
                    args.max_concurrent,
                    clean=args.clean,
                    dry_run=args.dry_run,
                ),
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                submitted_missing = True

    if args.finalize_if_complete:
        if not args.output_md or not args.output_json:
            raise SystemExit("Provide both --output_md and --output_json with --finalize_if_complete")
        if submitted_missing:
            print(
                "Submitted missing retrain jobs; skipping finalization in this run. "
                "Rerun with --finalize_if_complete after jobs finish."
            )
            return
        run_command(
            finalizer_command(
                experiments,
                args.seeds,
                args.output_md,
                args.output_json,
            ),
            dry_run=args.dry_run,
        )
        run_command(
            audit_command(
                experiments,
                args.seeds,
                require_retrains=True,
                strict=True,
            ),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
