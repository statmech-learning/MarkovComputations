"""Build, verify, and interpret the consolidated topology-ICL report.

Run this after source sweeps, mechanism summaries, physical essential retrains,
and input-mask essential retrains have already been recovered and finalized.
The wrapper is deliberately report-scoped: it does not submit jobs or collect
retrain directories. It regenerates ``make_topology_research_report.py``, runs
the strict ``verify_topology_completion.py --report_kind research`` gate, and
then writes the conservative H0/H1 interpretation.
"""

import argparse
import os
import subprocess
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TARGET = "test_novel_classes"


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


def report_command(experiments, target, output_md, output_json):
    return python_script(
        "make_topology_research_report.py",
        *experiment_args(experiments),
        "--target",
        target,
        "--output_md",
        output_md,
        "--output_json",
        output_json,
    )


def verifier_command(
    experiments,
    seeds,
    target,
    output_md,
    output_json,
    allow_unknown_provenance=False,
):
    parts = python_script(
        "verify_topology_completion.py",
        *experiment_args(experiments),
        "--seeds",
        seeds,
        "--report_kind",
        "research",
        "--target",
        target,
        "--report_md",
        output_md,
        "--report_json",
        output_json,
    )
    if allow_unknown_provenance:
        parts.append("--allow_unknown_provenance")
    return parts


def default_interpretation_path(path):
    root, ext = os.path.splitext(path)
    return f"{root}_interpretation{ext or '.txt'}"


def interpretation_command(report_json, output_md, output_json, min_n=None, delta=None):
    parts = python_script(
        "interpret_topology_report.py",
        "--report_json",
        report_json,
        "--report_kind",
        "research",
        "--output_md",
        output_md,
        "--output_json",
        output_json,
    )
    if min_n is not None:
        parts.extend(["--min_n", str(min_n)])
    if delta is not None:
        parts.extend(["--delta", str(delta)])
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Experiment root as NAME=PATH or PATH. May be repeated.",
    )
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--allow_unknown_provenance", action="store_true")
    parser.add_argument("--skip_interpretation", action="store_true")
    parser.add_argument("--interpret_output_md", type=str, default=None)
    parser.add_argument("--interpret_output_json", type=str, default=None)
    parser.add_argument(
        "--interpret_min_n",
        type=int,
        default=None,
        help="Optional min_n forwarded to interpret_topology_report.py.",
    )
    parser.add_argument(
        "--interpret_delta",
        type=float,
        default=None,
        help="Optional delta threshold forwarded to interpret_topology_report.py.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiments = [parse_experiment(raw) for raw in args.experiment]
    run_command(
        report_command(
            experiments,
            args.target,
            args.output_md,
            args.output_json,
        ),
        dry_run=args.dry_run,
    )
    run_command(
        verifier_command(
            experiments,
            args.seeds,
            args.target,
            args.output_md,
            args.output_json,
            allow_unknown_provenance=args.allow_unknown_provenance,
        ),
        dry_run=args.dry_run,
    )
    if not args.skip_interpretation:
        interpret_output_md = args.interpret_output_md or default_interpretation_path(args.output_md)
        interpret_output_json = args.interpret_output_json or default_interpretation_path(args.output_json)
        run_command(
            interpretation_command(
                args.output_json,
                interpret_output_md,
                interpret_output_json,
                min_n=args.interpret_min_n,
                delta=args.interpret_delta,
            ),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
