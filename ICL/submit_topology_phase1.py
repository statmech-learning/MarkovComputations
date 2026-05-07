"""SLURM array helper for controlled first-order topology ICL sweeps.

The generated commands run ``run_topology_icl.py`` from this ``ICL``
directory. Cluster-specific values are provided through environment variables,
so the same file can be used on Engaging without hard-coding another agent's
workspace:

    export SLURM_OUTPUT_BASE=/pool/<group>/<user>/topology_phase1
    export SLURM_PARTITION=<partition>
    export SLURM_ACCOUNT=<account>
    export SLURM_TIME=08:00:00
    export SLURM_MEM_PER_CPU=8G
    export SLURM_EXTRA_SETUP='module load python/3.11; source ~/venvs/icl/bin/activate'
    python3 submit_topology_phase1.py --phase smoke --array --dry-run
"""

import argparse
import csv
import os
import shutil


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.environ.get(
    "SLURM_OUTPUT_BASE",
    os.path.join(THIS_DIR, "results", "topology_phase1"),
)
PARTITION = os.environ.get("SLURM_PARTITION", "mit_normal")
ACCOUNT = os.environ.get("SLURM_ACCOUNT", "")
TIME_LIMIT = os.environ.get("SLURM_TIME", "08:00:00")
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
    "n_nodes": 6,
    "epochs": 1000,
    "lr": 0.0025,
    "batch_size": 50,
    "train_samples": 25000,
    "val_samples": 5000,
    "eval_frequency": 25,
    "n_eval_samples": 500,
    "test_samples": 1000,
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


def phase_configs(phase, seeds):
    configs = []
    if phase == "smoke":
        base = {
            "epochs": 2,
            "train_samples": 80,
            "val_samples": 40,
            "eval_frequency": 1,
            "n_eval_samples": 20,
            "test_samples": 40,
            "n_nodes": 4,
            "n_edges": 8,
        }
        for family in ["complete", "cycle_chords", "random_sc"]:
            for seed in seeds[:2]:
                row = base.copy()
                row.update(
                    {
                        "label": f"{family}_smoke_seed{seed}",
                        "topology_family": family,
                        "seed": seed,
                        "topology_seed": seed,
                    }
                )
                if family == "complete":
                    row["n_edges"] = None
                configs.append(row)
        return configs

    if phase != "phase1":
        raise ValueError(f"Unknown phase: {phase}")

    n_nodes = 6
    edge_counts = [8, 10, 12, 16, 20]
    min_edges_by_family = {
        "cycle_chords": n_nodes,
        "random_sc": n_nodes,
        "hub_spoke": 2 * (n_nodes - 1),
        "two_module": 2 * ((n_nodes // 2) * (n_nodes // 2 - 1))
        + 2 * ((n_nodes - n_nodes // 2) * (n_nodes - n_nodes // 2 - 1))
        + 2,
    }
    for n_edges in edge_counts:
        for family in ["cycle_chords", "random_sc", "hub_spoke", "two_module"]:
            if n_edges < min_edges_by_family[family]:
                continue
            for seed in seeds:
                configs.append(
                    {
                        "label": f"{family}_n{n_nodes}_m{n_edges}_seed{seed}",
                        "topology_family": family,
                        "n_nodes": n_nodes,
                        "n_edges": n_edges,
                        "seed": seed,
                        "topology_seed": seed,
                    }
                )
    for seed in seeds:
        configs.append(
            {
                "label": f"complete_n{n_nodes}_seed{seed}",
                "topology_family": "complete",
                "n_nodes": n_nodes,
                "n_edges": None,
                "seed": seed,
                "topology_seed": seed,
            }
        )
    return configs


def command_for(row):
    args = BASE_ARGS.copy()
    args.update(row)
    output = os.path.join(OUTPUT_BASE, row["label"])
    parts = ["python3", "-u", "run_topology_icl.py", "--output", output, "--no_progress"]
    for key in [
        "seed",
        "K",
        "L",
        "D",
        "N",
        "B",
        "epsilon",
        "n_nodes",
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
        "topology_family",
        "topology_seed",
    ]:
        parts.extend([f"--{key}", str(args[key])])
    if args.get("n_edges") is not None:
        parts.extend(["--n_edges", str(args["n_edges"])])
    return " ".join(parts), output


def write_array(configs, max_concurrent):
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    meta_dir = os.path.join(OUTPUT_BASE, "_array_meta")
    os.makedirs(meta_dir, exist_ok=True)
    index_csv = os.path.join(meta_dir, "index.csv")
    with open(index_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "label",
                "topology_family",
                "n_nodes",
                "n_edges",
                "seed",
                "topology_seed",
                "output",
                "command",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(configs):
            command, output = command_for(row)
            writer.writerow(
                {
                    "idx": idx,
                    "label": row["label"],
                    "topology_family": row["topology_family"],
                    "n_nodes": row.get("n_nodes", BASE_ARGS["n_nodes"]),
                    "n_edges": row.get("n_edges"),
                    "seed": row["seed"],
                    "topology_seed": row["topology_seed"],
                    "output": output,
                    "command": command,
                }
            )

    script_path = os.path.join(meta_dir, "run_task.sh")
    with open(script_path, "w") as f:
        f.write(
            f"""#!/bin/bash
#SBATCH --job-name=topo_icl
#SBATCH --output={meta_dir}/task_%a.out
#SBATCH --error={meta_dir}/task_%a.err
#SBATCH --time={TIME_LIMIT}
#SBATCH --partition={PARTITION}
{account_line()}#SBATCH --nodes=1
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem-per-cpu={MEM_PER_CPU}
#SBATCH --array=0-{len(configs) - 1}%{max_concurrent}

set -euo pipefail
cd {THIS_DIR}
{modules_block()}{setup_block()}
LINE=$(awk -F, -v idx="$SLURM_ARRAY_TASK_ID" '$1==idx {{print; exit}}' {index_csv})
OUT=$(echo "$LINE" | cut -d, -f8)
CMD=$(echo "$LINE" | cut -d, -f9-)
mkdir -p "$OUT"
if [ -f "$OUT/results.pkl" ]; then
    echo "Skipping $OUT (results.pkl exists)"
    exit 0
fi
echo "$CMD"
eval "$CMD"
"""
        )
    os.chmod(script_path, 0o755)
    return index_csv, script_path


def submit_array(script_path, dry_run):
    if dry_run:
        print(f"Dry run. Submit with: sbatch {script_path}")
        return
    rc = os.system(f"sbatch {script_path}")
    if rc != 0:
        print(f"sbatch failed. Submit manually with: sbatch {script_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "phase1"], default="phase1")
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--array", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    configs = phase_configs(args.phase, seeds)
    if args.clean and os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)

    index_csv, script_path = write_array(configs, args.max_concurrent)
    print(f"Output base: {OUTPUT_BASE}")
    print(f"Wrote index: {index_csv}")
    print(f"Wrote array script: {script_path}")
    print(f"Tasks: {len(configs)}")
    if args.array:
        submit_array(script_path, args.dry_run)
    else:
        print(f"Submit with: sbatch {script_path}")


if __name__ == "__main__":
    main()
