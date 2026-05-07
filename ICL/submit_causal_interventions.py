"""SLURM array helper for causal topology intervention analysis."""

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


def modules_block():
    if not EXTRA_MODULES:
        return ""
    return "\n".join(f"module load {module}" for module in EXTRA_MODULES.split()) + "\n"


def setup_block():
    return EXTRA_SETUP + "\n" if EXTRA_SETUP else ""


def account_line():
    return f"#SBATCH --account={ACCOUNT}\n" if ACCOUNT else ""


def iter_run_dirs(root):
    excluded = {
        "_array_meta",
        "_mechanism_array_meta",
        "_causal_array_meta",
        "essential_input50",
        "essential_input50_retrain",
        "essential_inputmask50",
        "essential_inputmask50_retrain",
    }
    for current, dirs, files in os.walk(root):
        dirs[:] = [item for item in dirs if item not in excluded]
        if "results.pkl" in files and "config.json" in files and "topology.json" in files and "model.pt" in files:
            yield os.path.abspath(current)


def command_for(run_dir, args):
    output_path = os.path.join(run_dir, args.output_name)
    parts = [
        "python3",
        "-u",
        "causal_topology_interventions.py",
        "--run_dir",
        shlex.quote(run_dir),
        "--n_samples",
        str(args.n_samples),
        "--n_repeats",
        str(args.n_repeats),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--interventions",
        shlex.quote(args.interventions),
        "--output",
        shlex.quote(output_path),
    ]
    return " ".join(parts), output_path


def write_array(run_dirs, args):
    meta_dir = os.path.join(os.path.abspath(args.input_root), "_causal_array_meta")
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
    script_path = os.path.join(meta_dir, "run_causal_task.sh")
    with open(script_path, "w") as handle:
        handle.write(
            f"""#!/bin/bash
#SBATCH --job-name=topo_causal
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
{modules_block()}{setup_block()}
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
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--interventions",
        default=(
            "context_block_shuffle,edge_projection_permutation,"
            "edge_rate_function_permutation,decoder_root_permutation,randomize_K_direction"
        ),
    )
    parser.add_argument("--output_name", default="causal_interventions.json")
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
