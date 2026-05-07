"""Finalize essential physical-subgraph retrain sweeps and refresh reports.

This wrapper is the physical-edge counterpart of
``finalize_essential_inputmask_retrains.py``. It conservatively checks that
each ``essential_input50_retrain`` directory contains exactly the expected
``<topology_id>_trainseed<seed>`` outputs, runs the standard
collection/aggregation pipeline, and joins retrained physical motifs back to
their source metadata. Regenerate consolidated reports separately with
``make_topology_research_report.py`` after this script succeeds.
"""

import argparse
import os

from finalize_essential_inputmask_retrains import (
    count_csv_rows,
    exact_retrain_status,
    parse_experiment,
    parse_seeds,
    python_script,
    read_json,
    run_command,
    selected_topology_ids,
)


def experiment_paths(root):
    return {
        "selected_csv": os.path.join(root, "essential_input50", "selected.csv"),
        "comparison_csv": os.path.join(root, "essential_input50", "retrain_comparison.csv"),
        "comparison_json": os.path.join(root, "essential_input50", "retrain_comparison.json"),
        "retrain_root": os.path.join(root, "essential_input50_retrain"),
        "retrain_aggregate_csv": os.path.join(
            root,
            "essential_input50_retrain",
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
        f"{name}: selected_subgraphs={selected} seeds={len(seeds)} "
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
    if exact["missing_required"]:
        print("  missing required run-file examples:")
        for path in exact["missing_required"][:5]:
            print(f"    {path}")
    return {
        "name": name,
        "root": root,
        "selected": selected,
        "expected": expected,
        "completed": completed,
        "unexpected": len(exact["unexpected"]),
        "missing_required": len(exact["missing_required"]),
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
            "--base_root",
            status["root"],
            "--layout",
            "physical",
            "--output_csv",
            paths["comparison_csv"],
            "--output_json",
            paths["comparison_json"],
        ),
        dry_run=args.dry_run,
    )


def validate_finalized_outputs(status, args):
    if args.dry_run:
        return
    paths = status["paths"]
    selected = status["selected"]
    aggregate_csv = paths["retrain_aggregate_csv"]
    comparison_csv = paths["comparison_csv"]
    comparison_json = paths["comparison_json"]
    failures = []

    aggregate_rows = count_csv_rows(aggregate_csv)
    if aggregate_rows is None:
        failures.append(f"missing {aggregate_csv}")
    elif aggregate_rows != selected:
        failures.append(f"{aggregate_csv}: row count {aggregate_rows}/{selected}")

    comparison_rows = count_csv_rows(comparison_csv)
    if comparison_rows is None:
        failures.append(f"missing {comparison_csv}")
    elif comparison_rows != selected:
        failures.append(f"{comparison_csv}: row count {comparison_rows}/{selected}")

    comparison = read_json(comparison_json)
    if comparison is None:
        failures.append(f"missing {comparison_json}")
    else:
        joined = comparison.get("n_joined")
        if joined != selected:
            failures.append(f"{comparison_json}: n_joined {joined}/{selected}")

    if failures:
        raise SystemExit(
            "Finalized physical retrain artifacts are incomplete:\n"
            + "\n".join(f"  {failure}" for failure in failures)
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
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--allow_extra", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiments = [parse_experiment(raw) for raw in args.experiment]
    seeds = parse_seeds(args.seeds)
    statuses = [check_completion(name, root, seeds) for name, root in experiments]
    incomplete = [
        status
        for status in statuses
        if status["completed"] != status["expected"]
    ]
    missing_required = [
        status
        for status in statuses
        if status["missing_required"]
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
    if missing_required and not args.allow_partial:
        for status in missing_required:
            print(
                f"Incomplete run sidecars: {status['name']} has "
                f"{status['missing_required']} missing required run files"
            )
        raise SystemExit(
            "Retrain run sidecars are incomplete; use --allow_partial only for diagnostics"
        )
    if unexpected and not args.allow_extra:
        for status in unexpected:
            print(
                f"Unexpected retrain outputs: {status['name']} has "
                f"{status['unexpected']} extra results.pkl files"
            )
        raise SystemExit("Retrain output root contains unexpected results; remove them or use --allow_extra")

    for status in statuses:
        finalize_experiment(status, args)
        validate_finalized_outputs(status, args)


if __name__ == "__main__":
    main()
