"""Collect and summarize a completed topology sweep.

This is a thin orchestration wrapper around the smaller analysis scripts. It
keeps the fixed-edge workflow reproducible:

1. collect run directories into ``topology_results.csv``;
2. run raw-count/tree-geometry regressions;
3. optionally submit or collect post-training mechanism analyses;
4. aggregate repeated train seeds at the topology level.

The wrapper does not wait for SLURM jobs. If mechanisms are submitted, rerun it
after those jobs finish with ``--collect_mechanisms``.
"""

import argparse
import os
import subprocess
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def path(root, name):
    return os.path.join(root, name)


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


def collect_topology(args):
    run_command(
        python_script(
            "collect_topology_results.py",
            "--input_root",
            args.input_root,
            "--output_csv",
            path(args.input_root, "topology_results.csv"),
        ),
        dry_run=args.dry_run,
    )
    run_command(
        python_script(
            "regress_topology_results.py",
            "--input_csv",
            path(args.input_root, "topology_results.csv"),
            "--output_json",
            path(args.input_root, "topology_regression.json"),
        ),
        dry_run=args.dry_run,
    )


def submit_mechanisms(args):
    parts = python_script(
        "submit_topology_mechanisms.py",
        "--input_root",
        args.input_root,
        "--n_samples",
        str(args.n_samples),
        "--device",
        args.device,
        "--python",
        args.job_python,
        "--array",
        "--max-concurrent",
        str(args.max_concurrent),
    )
    if args.ablate_input:
        parts.append("--ablate_input")
    if args.ablate_physical:
        parts.append("--ablate_physical")
    if args.force_mechanisms:
        parts.append("--force")
    if args.clean_mechanism_meta:
        parts.append("--clean")
    if args.skip_torch_check:
        parts.append("--skip_torch_check")
    if args.dry_run:
        parts.append("--dry-run")
    run_command(parts, dry_run=False)


def submit_causal(args):
    parts = python_script(
        "submit_causal_interventions.py",
        "--input_root",
        args.input_root,
        "--n_samples",
        str(args.causal_n_samples),
        "--n_repeats",
        str(args.causal_n_repeats),
        "--seed",
        str(args.causal_seed),
        "--device",
        args.device,
        "--interventions",
        args.causal_interventions,
        "--python",
        args.job_python,
        "--array",
        "--max-concurrent",
        str(args.max_concurrent),
    )
    if args.force_causal:
        parts.append("--force")
    if args.clean_causal_meta:
        parts.append("--clean")
    if args.skip_torch_check:
        parts.append("--skip_torch_check")
    if args.dry_run:
        parts.append("--dry-run")
    run_command(parts, dry_run=False)


def collect_mechanisms(args):
    mechanism_count = count_files(args.input_root, "mechanism_metrics.json")
    if mechanism_count == 0:
        print(f"No mechanism_metrics.json files found under {args.input_root}; skipping mechanism summaries")
        return False
    print(f"Found {mechanism_count} mechanism_metrics.json files")
    mechanism_csv = path(args.input_root, "mechanism_results.csv")
    run_command(
        python_script(
            "collect_mechanism_results.py",
            "--input_root",
            args.input_root,
            "--output_csv",
            mechanism_csv,
        ),
        dry_run=args.dry_run,
    )
    run_command(
        python_script(
            "summarize_topology_mechanisms.py",
            "--topology_csv",
            path(args.input_root, "topology_results.csv"),
            "--mechanism_csv",
            mechanism_csv,
            "--output_json",
            path(args.input_root, "mechanism_summary.json"),
        ),
        dry_run=args.dry_run,
    )
    return True


def collect_causal(args):
    causal_count = count_files(args.input_root, "causal_interventions.json")
    if causal_count == 0:
        print(f"No causal_interventions.json files found under {args.input_root}; skipping causal summaries")
        return False
    print(f"Found {causal_count} causal_interventions.json files")
    run_command(
        python_script(
            "collect_causal_interventions.py",
            "--input_root",
            args.input_root,
            "--output_csv",
            path(args.input_root, "causal_interventions.csv"),
            "--output_json",
            path(args.input_root, "causal_interventions_summary.json"),
        ),
        dry_run=args.dry_run,
    )
    return True


def aggregate_seeds(args, has_mechanisms):
    parts = python_script(
        "aggregate_topology_seeds.py",
        "--topology_csv",
        path(args.input_root, "topology_results.csv"),
        "--output_csv",
        path(args.input_root, "topology_seed_aggregates.csv"),
        "--output_json",
        path(args.input_root, "topology_seed_aggregates.json"),
    )
    if has_mechanisms:
        parts.extend(["--mechanism_csv", path(args.input_root, "mechanism_results.csv")])
    run_command(parts, dry_run=args.dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--max-concurrent", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--job_python",
        default=os.environ.get("TOPOLOGY_PYTHON", "python3"),
        help=(
            "Python command to use inside submitted SLURM mechanism jobs after any "
            "SLURM setup has run. Defaults to TOPOLOGY_PYTHON or python3."
        ),
    )
    parser.add_argument(
        "--skip_torch_check",
        action="store_true",
        help="Do not insert the Torch import preflight in submitted mechanism jobs.",
    )
    parser.add_argument("--submit_mechanisms", action="store_true")
    parser.add_argument("--collect_mechanisms", action="store_true")
    parser.add_argument("--submit_causal", action="store_true")
    parser.add_argument("--collect_causal", action="store_true")
    parser.add_argument("--causal_n_samples", type=int, default=500)
    parser.add_argument("--causal_n_repeats", type=int, default=5)
    parser.add_argument("--causal_seed", type=int, default=0)
    parser.add_argument(
        "--causal_interventions",
        default=(
            "context_block_shuffle,edge_projection_permutation,"
            "edge_rate_function_permutation,decoder_root_permutation,randomize_K_direction"
        ),
    )
    parser.add_argument("--ablate_input", action="store_true")
    parser.add_argument("--ablate_physical", action="store_true")
    parser.add_argument("--force_mechanisms", action="store_true")
    parser.add_argument("--force_causal", action="store_true")
    parser.add_argument("--clean_mechanism_meta", action="store_true")
    parser.add_argument("--clean_causal_meta", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.input_root = os.path.abspath(args.input_root)
    collect_topology(args)
    if args.submit_mechanisms:
        submit_mechanisms(args)
    if args.submit_causal:
        submit_causal(args)
    has_mechanisms = args.collect_mechanisms and collect_mechanisms(args)
    if args.collect_causal:
        collect_causal(args)
    if not has_mechanisms and os.path.exists(path(args.input_root, "mechanism_results.csv")):
        has_mechanisms = True
    aggregate_seeds(args, has_mechanisms=has_mechanisms)


if __name__ == "__main__":
    main()
