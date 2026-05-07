"""Guarded orchestration for expanded hard topology follow-ups.

The expanded hard pilots are the current strict-gate blocker for the
next-phase topology-ICL objective: their training summaries are tracked, but
their per-run mechanism and causal intervention artifacts must be produced on
the Engaging worktree where raw ``results.pkl`` and ``model.pt`` files exist.

This wrapper keeps that cluster workflow reproducible while protecting
source-light checkouts.  It refuses to submit or collect follow-up jobs for a
root with no raw ``results.pkl`` files unless explicitly overridden, because
``finalize_topology_sweep.py`` recollects topology results at startup and would
otherwise overwrite tracked summaries with empty tables.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class HardRegime:
    label: str
    root: str
    clustered_json: str | None = None
    capacity_json: str | None = None


DEFAULT_HARD_REGIMES = [
    HardRegime(
        "hard_n4_m6_N3_D2",
        "results/expanded_hard_sweeps/n4_m6_N3_D2",
        "results/expanded_hard_stats/n4_m6_N3_D2_branch_capacity_clustered_inference.json",
        "results/expanded_hard_libraries/n4_m6_N3_D2/branch_margin_capacity_summary.json",
    ),
    HardRegime(
        "hard_n5_m8_N3_D2",
        "results/expanded_hard_sweeps/n5_m8_N3_D2",
        "results/expanded_hard_stats/n5_m8_N3_D2_branch_capacity_clustered_inference.json",
        "results/expanded_hard_libraries/n5_m8_N3_D2/branch_margin_capacity_summary.json",
    ),
    HardRegime(
        "hard_n5_m12_N3_D2",
        "results/expanded_hard_sweeps/n5_m12_N3_D2",
        "results/expanded_hard_stats/n5_m12_N3_D2_branch_capacity_clustered_inference.json",
        "results/expanded_hard_libraries/n5_m12_N3_D2/branch_margin_capacity_summary.json",
    ),
]


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(THIS_DIR, path)


def count_files(root: str, filename: str) -> int:
    total = 0
    abs_root = resolve(root)
    for _, _, files in os.walk(abs_root):
        if filename in files:
            total += 1
    return total


def csv_data_rows(path: str) -> int:
    abs_path = resolve(path)
    if not os.path.exists(abs_path):
        return 0
    with open(abs_path, newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    return max(0, len(rows) - 1)


def run_command(parts: List[str], dry_run: bool = False) -> None:
    print(" ".join(parts))
    if not dry_run:
        subprocess.run(parts, check=True, cwd=THIS_DIR)


def parse_regime(raw: str) -> HardRegime:
    fields = raw.split("=")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("regime must be LABEL=ROOT")
    label, root = fields
    label = label.strip()
    root = root.strip()
    if not label or not root:
        raise argparse.ArgumentTypeError("regime label and root must be nonempty")
    return HardRegime(label, root)


def selected_regimes(items: Iterable[HardRegime] | None) -> List[HardRegime]:
    regimes = list(items or [])
    return regimes or list(DEFAULT_HARD_REGIMES)


def status_rows(regimes: Iterable[HardRegime]) -> List[dict]:
    rows = []
    for regime in regimes:
        rows.append(
            {
                "label": regime.label,
                "root": regime.root,
                "results_pkl": count_files(regime.root, "results.pkl"),
                "model_pt": count_files(regime.root, "model.pt"),
                "mechanisms": count_files(regime.root, "mechanism_metrics.json"),
                "causal": count_files(regime.root, "causal_interventions.json"),
                "topology_rows": csv_data_rows(os.path.join(regime.root, "topology_results.csv")),
            }
        )
    return rows


def print_status(regimes: Iterable[HardRegime]) -> None:
    print("label,root,results_pkl,model_pt,mechanisms,causal,topology_rows")
    for row in status_rows(regimes):
        print(
            "{label},{root},{results_pkl},{model_pt},{mechanisms},{causal},{topology_rows}".format(
                **row
            )
        )


def ensure_raw_sources(
    regimes: Iterable[HardRegime],
    allow_source_light: bool,
    require_models: bool = False,
) -> None:
    missing = [
        row
        for row in status_rows(regimes)
        if row["results_pkl"] == 0 or (require_models and row["model_pt"] == 0)
    ]
    if missing and not allow_source_light:
        details = ", ".join(
            f"{row['label']}:{row['root']} results_pkl={row['results_pkl']} model_pt={row['model_pt']}"
            for row in missing
        )
        raise SystemExit(
            "Refusing to run follow-up finalization without required raw files "
            f"for: {details}. Run this on the Engaging worktree with raw training "
            "outputs, or pass --allow_source_light only if you intentionally want "
            "to bypass this safety check."
        )


def git_summary() -> str:
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            cwd=THIS_DIR,
            text=True,
            capture_output=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            cwd=THIS_DIR,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "git=unavailable"
    return f"git={branch}@{commit}"


def check_torch_python(job_python: str) -> tuple[bool, str]:
    command = shlex.split(job_python) + [
        "-c",
        "import torch, sys; print(sys.executable); print(torch.__version__)",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        return False, str(exc)
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def preflight(args, regimes: Iterable[HardRegime]) -> None:
    rows = status_rows(regimes)
    print(git_summary())
    print_status(regimes)
    failures = []
    for row in rows:
        if row["results_pkl"] == 0:
            failures.append(f"{row['label']}: missing raw results.pkl files")
        if row["model_pt"] == 0:
            failures.append(f"{row['label']}: missing raw model.pt files")
        if row["topology_rows"] == 0:
            failures.append(f"{row['label']}: missing topology_results.csv rows")
    if args.skip_torch_check:
        print("torch_check=skipped")
    else:
        ok, output = check_torch_python(args.job_python)
        print(f"torch_check_python={args.job_python}")
        if output:
            print(output)
        if not ok:
            failures.append(f"job_python cannot import torch: {args.job_python}")
    if failures:
        for failure in failures:
            print(f"PREFLIGHT FAIL: {failure}", file=sys.stderr)
        raise SystemExit("Expanded hard follow-up preflight failed")
    print("Expanded hard follow-up preflight passed")


def finalize_parts(args, regime: HardRegime, mode: str) -> List[str]:
    parts = [
        sys.executable,
        "finalize_topology_sweep.py",
        "--input_root",
        regime.root,
        "--device",
        args.device,
        "--job_python",
        args.job_python,
        "--max-concurrent",
        str(args.max_concurrent),
    ]
    if mode == "submit":
        parts.extend(
            [
                "--submit_mechanisms",
                "--submit_causal",
                "--ablate_input",
                "--ablate_physical",
                "--causal_n_samples",
                str(args.causal_n_samples),
                "--causal_n_repeats",
                str(args.causal_n_repeats),
                "--causal_seed",
                str(args.causal_seed),
            ]
        )
    elif mode == "collect":
        parts.extend(["--collect_mechanisms", "--collect_causal"])
    else:
        raise ValueError(mode)
    if args.skip_torch_check:
        parts.append("--skip_torch_check")
    if args.dry_run:
        parts.append("--dry-run")
    return parts


def run_finalize(args, regimes: Iterable[HardRegime], mode: str) -> None:
    ensure_raw_sources(
        regimes,
        args.allow_source_light or args.dry_run,
        require_models=(mode == "submit"),
    )
    for regime in regimes:
        run_command(finalize_parts(args, regime, mode), dry_run=args.dry_run)


def refresh_report(args, regimes: Iterable[HardRegime]) -> None:
    parts = [
        sys.executable,
        "refresh_next_phase_report.py",
        "--report_json",
        args.report_json,
        "--output_json",
        args.report_json,
        "--output_md",
        args.report_md,
    ]
    for regime in regimes:
        if regime.clustered_json and os.path.exists(resolve(regime.clustered_json)):
            parts.extend(["--clustered_json", f"{regime.label}={regime.clustered_json}"])
        if regime.capacity_json and os.path.exists(resolve(regime.capacity_json)):
            parts.extend(["--branch_capacity_json", f"{regime.label}={regime.capacity_json}"])
        causal_summary = os.path.join(regime.root, "causal_interventions_summary.json")
        if os.path.exists(resolve(causal_summary)):
            parts.extend(["--causal_json", f"{regime.label}={causal_summary}"])
        parts.extend(["--expanded_root", f"{regime.label}={regime.root}"])
    run_command(parts, dry_run=args.dry_run)


def verify_report(args, strict: bool) -> None:
    parts = [
        sys.executable,
        "verify_topology_completion.py",
        "--experiment",
        "next=results/next_phase_stats",
        "--report_md",
        args.report_md,
        "--report_json",
        args.report_json,
        "--report_kind",
        "next_phase",
    ]
    if strict:
        parts.append("--require_expanded_followups")
    run_command(parts, dry_run=args.dry_run)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", type=parse_regime, action="append", default=[])
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--submit_followups", action="store_true")
    parser.add_argument("--collect_followups", action="store_true")
    parser.add_argument("--refresh_report", action="store_true")
    parser.add_argument("--verify_report", action="store_true")
    parser.add_argument("--strict_verify", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow_source_light", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--job_python",
        default=os.environ.get("TOPOLOGY_PYTHON", "python3"),
        help="Torch-enabled Python command to use inside submitted SLURM jobs.",
    )
    parser.add_argument("--skip_torch_check", action="store_true")
    parser.add_argument("--causal_n_samples", type=int, default=500)
    parser.add_argument("--causal_n_repeats", type=int, default=5)
    parser.add_argument("--causal_seed", type=int, default=0)
    parser.add_argument(
        "--report_json",
        default="results/next_phase_stats/next_phase_evidence_report.json",
    )
    parser.add_argument(
        "--report_md",
        default="results/next_phase_stats/next_phase_evidence_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regimes = selected_regimes(args.regime)
    if args.status:
        print_status(regimes)
    if args.preflight:
        preflight(args, regimes)
    if args.submit_followups:
        run_finalize(args, regimes, "submit")
    if args.collect_followups:
        run_finalize(args, regimes, "collect")
    if args.refresh_report:
        refresh_report(args, regimes)
    if args.verify_report or args.strict_verify:
        verify_report(args, strict=args.strict_verify)


if __name__ == "__main__":
    main()
