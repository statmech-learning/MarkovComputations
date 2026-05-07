"""SLURM array helper for post-training topology mechanism analysis."""

import argparse
import os
import shlex
import shutil


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARTITION = os.environ.get("SLURM_PARTITION", "mit_normal")
ACCOUNT = os.environ.get("SLURM_ACCOUNT", "")
TIME_LIMIT = os.environ.get("SLURM_TIME", "02:00:00")
MEM_PER_CPU = os.environ.get("SLURM_MEM_PER_CPU", "8G")
CPUS_PER_TASK = os.environ.get("SLURM_CPUS_PER_TASK", "1")
EXTRA_MODULES = os.environ.get("SLURM_MODULES", "")
EXTRA_SETUP = os.environ.get("SLURM_EXTRA_SETUP", "")
DEFAULT_PYTHON = os.environ.get("TOPOLOGY_PYTHON", "python3")


def modules_block():
    if not EXTRA_MODULES:
        return ""
    return "\n".join(f"module load {module}" for module in EXTRA_MODULES.split()) + "\n"


def setup_block():
    return EXTRA_SETUP + "\n" if EXTRA_SETUP else ""


def account_line():
    return f"#SBATCH --account={ACCOUNT}\n" if ACCOUNT else ""


def iter_run_dirs(root):
    for current, _, files in os.walk(root):
        if "results.pkl" in files and "config.json" in files and "topology.json" in files:
            yield os.path.abspath(current)


def command_for(run_dir, args):
    output_path = os.path.join(run_dir, args.output_name)
    parts = [
        args.python,
        "-u",
        "analyze_topology_model.py",
        "--run_dir",
        shlex.quote(run_dir),
        "--n_samples",
        str(args.n_samples),
        "--output",
        shlex.quote(output_path),
        "--device",
        args.device,
    ]
    if args.ablate_input:
        parts.append("--ablate_input")
    if args.ablate_physical:
        parts.append("--ablate_physical")
    if args.physical_epsilon is not None:
        parts.extend(["--physical_epsilon", str(args.physical_epsilon)])
    return " ".join(parts), output_path


def torch_check_block(args):
    if args.skip_torch_check:
        return ""
    return f"""{args.python} - <<'PY'
import sys
import torch
print("Using Python:", sys.executable)
print("Torch:", torch.__version__)
PY
"""


def write_array(run_dirs, args):
    meta_dir = os.path.join(os.path.abspath(args.input_root), "_mechanism_array_meta")
    if args.clean and os.path.exists(meta_dir):
        shutil.rmtree(meta_dir)
    os.makedirs(meta_dir, exist_ok=True)

    commands_path = os.path.join(meta_dir, "commands.txt")
    outputs_path = os.path.join(meta_dir, "outputs.txt")
    runs_path = os.path.join(meta_dir, "run_dirs.txt")
    with open(commands_path, "w") as commands, open(outputs_path, "w") as outputs, open(runs_path, "w") as runs:
        for run_dir in run_dirs:
            command, output_path = command_for(run_dir, args)
            commands.write(command + "\n")
            outputs.write(output_path + "\n")
            runs.write(run_dir + "\n")

    force_flag = "1" if args.force else "0"
    script_path = os.path.join(meta_dir, "run_mechanism_task.sh")
    with open(script_path, "w") as f:
        f.write(
            f"""#!/bin/bash
#SBATCH --job-name=topo_mech
#SBATCH --output={meta_dir}/task_%a.out
#SBATCH --error={meta_dir}/task_%a.err
#SBATCH --time={TIME_LIMIT}
#SBATCH --partition={PARTITION}
{account_line()}#SBATCH --nodes=1
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem-per-cpu={MEM_PER_CPU}
#SBATCH --array=0-{len(run_dirs) - 1}%{args.max_concurrent}

set -euo pipefail
cd {THIS_DIR}
{modules_block()}{setup_block()}{torch_check_block(args)}
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
CMD=$(sed -n "${{LINE_NUM}}p" {commands_path})
OUT=$(sed -n "${{LINE_NUM}}p" {outputs_path})
RUN_DIR=$(sed -n "${{LINE_NUM}}p" {runs_path})
echo "$RUN_DIR"
if [ -f "$OUT" ] && [ "{force_flag}" != "1" ]; then
    echo "Skipping $OUT (already exists)"
    exit 0
fi
echo "$CMD"
eval "$CMD"
"""
        )
    os.chmod(script_path, 0o755)
    return meta_dir, script_path


def submit_array(script_path, dry_run):
    if dry_run:
        print(f"Dry run. Submit with: sbatch {script_path}")
        return
    rc = os.system(f"sbatch {script_path}")
    if rc != 0:
        print(f"sbatch failed. Submit manually with: sbatch {script_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--python",
        type=str,
        default=DEFAULT_PYTHON,
        help="Python command to use inside the SLURM job after setup. Defaults to TOPOLOGY_PYTHON or python3.",
    )
    parser.add_argument(
        "--skip_torch_check",
        action="store_true",
        help="Do not insert a per-task Torch import preflight in the SLURM script.",
    )
    parser.add_argument("--output_name", type=str, default="mechanism_metrics.json")
    parser.add_argument("--ablate_input", action="store_true")
    parser.add_argument("--ablate_physical", action="store_true")
    parser.add_argument("--physical_epsilon", type=float, default=1e-6)
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--array", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    run_dirs = sorted(iter_run_dirs(args.input_root))
    if not run_dirs:
        raise SystemExit(f"No completed topology runs found under {args.input_root}")

    meta_dir, script_path = write_array(run_dirs, args)
    print(f"Input root: {os.path.abspath(args.input_root)}")
    print(f"Wrote array metadata: {meta_dir}")
    print(f"Wrote array script: {script_path}")
    print(f"Tasks: {len(run_dirs)}")
    if args.array:
        submit_array(script_path, args.dry_run)
    else:
        print(f"Submit with: sbatch {script_path}")


if __name__ == "__main__":
    main()
