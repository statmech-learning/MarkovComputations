"""Submit training runs for a selected topology library."""

import argparse
import csv
import os
import shlex
import shutil


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARTITION = os.environ.get("SLURM_PARTITION", "mit_normal")
ACCOUNT = os.environ.get("SLURM_ACCOUNT", "")
TIME_LIMIT = os.environ.get("SLURM_TIME", "04:00:00")
MEM_PER_CPU = os.environ.get("SLURM_MEM_PER_CPU", "8G")
CPUS_PER_TASK = os.environ.get("SLURM_CPUS_PER_TASK", "1")
EXTRA_MODULES = os.environ.get("SLURM_MODULES", "")
EXTRA_SETUP = os.environ.get("SLURM_EXTRA_SETUP", "")


BASE_ARGS = {
    "K": 128,
    "L": 128,
    "D": 4,
    "N": 4,
    "B": 1,
    "epsilon": 1e-3,
    "epochs": 100,
    "lr": 0.0025,
    "batch_size": 50,
    "train_samples": 5000,
    "val_samples": 1000,
    "eval_frequency": 10,
    "n_eval_samples": 200,
    "test_samples": 500,
    "method": "direct_solve",
    "temperature": 1.0,
    "device": "auto",
}


def modules_block():
    if not EXTRA_MODULES:
        return ""
    return "\n".join(f"module load {module}" for module in EXTRA_MODULES.split()) + "\n"


def setup_block():
    return EXTRA_SETUP + "\n" if EXTRA_SETUP else ""


def account_line():
    return f"#SBATCH --account={ACCOUNT}\n" if ACCOUNT else ""


def parse_seeds(raw):
    return [int(seed) for seed in raw.split(",") if seed.strip()]


def load_topologies(path, selected_only=True, limit=None):
    rows = []
    base_dir = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        for row in csv.DictReader(f):
            if selected_only and str(row.get("selected", "1")) not in {"1", "True", "true"}:
                continue
            row["_library_dir"] = base_dir
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"No selected topology rows found in {path}")
    return rows


def resolve_path(value, base_dir):
    if value in (None, ""):
        return None
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def command_for(topology_row, train_seed, args):
    params = BASE_ARGS.copy()
    for key in BASE_ARGS:
        value = getattr(args, key)
        if value is not None:
            params[key] = value

    label = f"{topology_row['topology_id']}_trainseed{train_seed}"
    output = os.path.abspath(os.path.join(args.output_root, label))
    base_dir = topology_row.get("_library_dir", os.getcwd())
    edge_json = resolve_path(topology_row["edge_json"], base_dir)
    parts = [
        "python3",
        "-u",
        "run_topology_icl.py",
        "--output",
        shlex.quote(output),
        "--edge_json",
        shlex.quote(edge_json),
        "--seed",
        str(train_seed),
        "--input_mask_seed",
        str(train_seed),
        "--no_progress",
    ]
    input_mask_json = resolve_path(topology_row.get("input_mask_json"), base_dir)
    if input_mask_json:
        parts.extend(["--input_mask_json", shlex.quote(input_mask_json)])
    for key in [
        "K",
        "L",
        "D",
        "N",
        "B",
        "epsilon",
        "epochs",
        "lr",
        "batch_size",
        "train_samples",
        "val_samples",
        "eval_frequency",
        "n_eval_samples",
        "test_samples",
        "method",
        "temperature",
        "device",
    ]:
        parts.extend([f"--{key}", str(params[key])])
    return " ".join(parts), output


def write_array(topology_rows, train_seeds, args):
    meta_dir = os.path.join(os.path.abspath(args.output_root), "_array_meta")
    if args.clean and os.path.exists(meta_dir):
        shutil.rmtree(meta_dir)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(args.output_root, exist_ok=True)

    commands_path = os.path.join(meta_dir, "commands.txt")
    outputs_path = os.path.join(meta_dir, "outputs.txt")
    written_tasks = 0
    skipped_existing = 0
    with open(commands_path, "w") as commands, open(outputs_path, "w") as outputs:
        for row in topology_rows:
            for seed in train_seeds:
                command, output = command_for(row, seed, args)
                if args.missing_only and not args.force and os.path.exists(os.path.join(output, "results.pkl")):
                    skipped_existing += 1
                    continue
                commands.write(command + "\n")
                outputs.write(output + "\n")
                written_tasks += 1

    n_tasks = written_tasks
    if n_tasks == 0:
        raise SystemExit(
            f"No tasks to write for {args.output_root}; "
            f"skipped {skipped_existing} completed outputs"
        )
    force_flag = "1" if args.force else "0"
    script_path = os.path.join(meta_dir, "run_task.sh")
    with open(script_path, "w") as f:
        f.write(
            f"""#!/bin/bash
#SBATCH --job-name=topo_lib
#SBATCH --output={meta_dir}/task_%a.out
#SBATCH --error={meta_dir}/task_%a.err
#SBATCH --time={TIME_LIMIT}
#SBATCH --partition={PARTITION}
{account_line()}#SBATCH --nodes=1
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem-per-cpu={MEM_PER_CPU}
#SBATCH --array=0-{n_tasks - 1}%{args.max_concurrent}

set -euo pipefail
cd {THIS_DIR}
{modules_block()}{setup_block()}
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
CMD=$(sed -n "${{LINE_NUM}}p" {commands_path})
OUT=$(sed -n "${{LINE_NUM}}p" {outputs_path})
mkdir -p "$OUT"
if [ -f "$OUT/results.pkl" ] && [ "{force_flag}" != "1" ]; then
    echo "Skipping $OUT (results.pkl exists)"
    exit 0
fi
echo "$CMD"
eval "$CMD"
"""
        )
    os.chmod(script_path, 0o755)
    return meta_dir, script_path, n_tasks, skipped_existing


def submit_array(script_path, dry_run):
    if dry_run:
        print(f"Dry run. Submit with: sbatch {script_path}")
        return
    rc = os.system(f"sbatch {script_path}")
    if rc != 0:
        print(f"sbatch failed. Submit manually with: sbatch {script_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library_csv", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="1,2")
    parser.add_argument("--include_unselected", action="store_true")
    parser.add_argument("--limit_topologies", type=int, default=None)
    parser.add_argument("--max-concurrent", type=int, default=24)
    parser.add_argument("--array", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--missing_only",
        action="store_true",
        help="Only write tasks whose output directory does not already contain results.pkl.",
    )

    for key, value in BASE_ARGS.items():
        value_type = int if isinstance(value, int) else float if isinstance(value, float) else str
        parser.add_argument(f"--{key}", type=value_type, default=None)

    args = parser.parse_args()

    rows = load_topologies(
        args.library_csv,
        selected_only=not args.include_unselected,
        limit=args.limit_topologies,
    )
    train_seeds = parse_seeds(args.seeds)
    meta_dir, script_path, n_tasks, skipped_existing = write_array(rows, train_seeds, args)
    print(f"Library: {os.path.abspath(args.library_csv)}")
    print(f"Output root: {os.path.abspath(args.output_root)}")
    print(f"Wrote array metadata: {meta_dir}")
    print(f"Wrote array script: {script_path}")
    print(f"Topologies: {len(rows)}")
    print(f"Train seeds: {len(train_seeds)}")
    print(f"Tasks: {n_tasks}")
    if skipped_existing:
        print(f"Skipped existing outputs: {skipped_existing}")
    if args.array:
        submit_array(script_path, args.dry_run)
    else:
        print(f"Submit with: sbatch {script_path}")


if __name__ == "__main__":
    main()
