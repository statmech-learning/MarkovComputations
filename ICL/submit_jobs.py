"""SLURM submission helper for ICL training sweeps.

Cluster-specific values (output base, partition, account, etc.) are read from
environment variables so this is portable across UChicago Midway, MIT
Engaging, and similar SLURM clusters. To use on Engaging:

    export SLURM_OUTPUT_BASE=/pool/<group>/icl_results
    export SLURM_PARTITION=mit_normal       # check `sinfo` on the cluster
    export SLURM_ACCOUNT=<your-pi-account>
    export SLURM_TIME=02:00:00              # tight; the runs are fast
    export SLURM_MEM_PER_CPU=2G             # tight; models are ~MB
    python3 submit_jobs.py

Or use the array-job mode for the full sweep, which is much friendlier on
the scheduler than thousands of individual sbatch calls:

    python3 submit_jobs.py --array

Defaults below preserve the original UChicago Midway behavior.
"""

import argparse
import os
import shutil

import numpy as np


# ---------------------------------------------------------------------------
# Cluster configuration (override via environment variables)
# ---------------------------------------------------------------------------
this_dir = os.path.dirname(os.path.abspath(__file__))

output_base = os.environ.get(
    "SLURM_OUTPUT_BASE",
    "/project/svaikunt/csfloyd/MarkovComputation/DirsICL/n_nodes_N_seed/",
)
run_script = os.environ.get(
    "ICL_RUN_SCRIPT",
    os.path.join(this_dir, "run_icl.py"),
)
partition = os.environ.get("SLURM_PARTITION", "caslake")
account = os.environ.get("SLURM_ACCOUNT", "pi-svaikunt")
mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU", "32000")
time_limit = os.environ.get("SLURM_TIME", "32:00:00")
cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "1")
extra_modules = os.environ.get("SLURM_MODULES", "")  # e.g. "anaconda3/2024.06"
extra_setup = os.environ.get(
    "SLURM_EXTRA_SETUP", ""
)  # e.g. "source activate icl_env"


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
n_params = 3

# n_nodes (N_n) sweep
param1_values = list(range(1, 16))
# context length (N_c) sweep — note: run_icl.py reads param2 as
# sparsity_rho_all (float). To use this script for the n_nodes x N_c sweep
# from the paper, edit run_icl.py to map param2 to N or use a thin wrapper.
param2_values = list(range(1, 16))
# random seed
param3_values = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# SLURM job templates
# ---------------------------------------------------------------------------
HEADER = """#!/bin/bash
#SBATCH --job-name=icl
#SBATCH --output={output}/training_batch.out
#SBATCH --error={output}/training_batch.err
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}

{extra_modules_block}{extra_setup_block}"""


def _modules_block():
    if not extra_modules:
        return ""
    return "\n".join(f"module load {m}" for m in extra_modules.split()) + "\n"


def _setup_block():
    if not extra_setup:
        return ""
    return extra_setup + "\n"


def render_header(output):
    return HEADER.format(
        output=output,
        time_limit=time_limit,
        partition=partition,
        account=account,
        cpus_per_task=cpus_per_task,
        mem_per_cpu=mem_per_cpu,
        extra_modules_block=_modules_block(),
        extra_setup_block=_setup_block(),
    )


# ---------------------------------------------------------------------------
# Submission modes
# ---------------------------------------------------------------------------
def submit_one_by_one():
    """Original behavior: one sbatch per (param1, param2, param3) cell."""
    job_template_1 = render_header("{output}") + (
        "\npython3 {run_script} --param1 {param1} --output {output}\n"
    )
    job_template_2 = render_header("{output}") + (
        "\npython3 {run_script} --param1 {param1} --param2 {param2} "
        "--output {output}\n"
    )
    job_template_3 = render_header("{output}") + (
        "\npython3 {run_script} --param1 {param1} --param2 {param2} "
        "--param3 {param3} --output {output}\n"
    )

    if n_params == 1:
        for param1 in param1_values:
            output = output_base + f"{param1}"
            if os.path.exists(output):
                shutil.rmtree(output)
            os.makedirs(output)
            print(f"Created directory: {output}")
            content = job_template_1.format(
                param1=param1, output=output, run_script=run_script
            )
            job_filename = os.path.join(output, f"job_{param1}.sh")
            with open(job_filename, "w") as f:
                f.write(content)
            os.system(f"sbatch {job_filename}")
            print(f"Submitted: param1={param1} output={output}")

    elif n_params == 2:
        for param1 in param1_values:
            for param2 in param2_values:
                output = os.path.join(output_base, f"{param1}_{param2}")
                if os.path.exists(output):
                    shutil.rmtree(output)
                os.makedirs(output)
                print(f"Created directory: {output}")
                content = job_template_2.format(
                    param1=param1,
                    param2=param2,
                    output=output,
                    run_script=run_script,
                )
                job_filename = os.path.join(output, f"job_{param1}_{param2}.sh")
                with open(job_filename, "w") as f:
                    f.write(content)
                os.system(f"sbatch {job_filename}")
                print(f"Submitted: param1={param1} param2={param2} output={output}")

    elif n_params == 3:
        for param1 in param1_values:
            for param2 in param2_values:
                for param3 in param3_values:
                    output = os.path.join(
                        output_base, f"{param1}_{param2}_{param3}"
                    )
                    results_file = os.path.join(output, "results.pkl")
                    if os.path.exists(results_file):
                        print(f"Skipping {output} - results.pkl already exists")
                        continue
                    if os.path.exists(output):
                        shutil.rmtree(output)
                    os.makedirs(output)
                    print(f"Created directory: {output}")
                    content = job_template_3.format(
                        param1=param1,
                        param2=param2,
                        param3=param3,
                        output=output,
                        run_script=run_script,
                    )
                    job_filename = os.path.join(
                        output, f"job_{param1}_{param2}_{param3}.sh"
                    )
                    with open(job_filename, "w") as f:
                        f.write(content)
                    os.system(f"sbatch {job_filename}")
                    print(
                        f"Submitted: param1={param1} param2={param2} "
                        f"param3={param3} output={output}"
                    )


def submit_array(max_concurrent=200):
    """Submit the full Cartesian-product sweep as a single SLURM job array.

    Much friendlier on the scheduler than thousands of one-off sbatch calls.
    Each task in the array decodes its task ID into (param1, param2, param3).
    """
    if n_params != 3:
        raise SystemExit(
            "Array mode currently expects n_params=3 (the full sweep). "
            "Edit submit_jobs.py if you need a smaller array."
        )

    n1, n2, n3 = len(param1_values), len(param2_values), len(param3_values)
    total = n1 * n2 * n3

    os.makedirs(output_base, exist_ok=True)
    array_dir = os.path.join(output_base, "_array_meta")
    os.makedirs(array_dir, exist_ok=True)

    # Write the index -> (param1, param2, param3) lookup as a CSV the array
    # task can read.
    index_csv = os.path.join(array_dir, "index.csv")
    with open(index_csv, "w") as f:
        f.write("idx,param1,param2,param3,output\n")
        for idx in range(total):
            i1 = idx // (n2 * n3)
            i2 = (idx // n3) % n2
            i3 = idx % n3
            p1, p2, p3 = param1_values[i1], param2_values[i2], param3_values[i3]
            output = os.path.join(output_base, f"{p1}_{p2}_{p3}")
            f.write(f"{idx},{p1},{p2},{p3},{output}\n")

    # Worker bash that each array task runs.
    array_script = os.path.join(array_dir, "run_task.sh")
    with open(array_script, "w") as f:
        f.write(
            f"""#!/bin/bash
#SBATCH --job-name=icl_array
#SBATCH --output={output_base}/_array_meta/task_%a.out
#SBATCH --error={output_base}/_array_meta/task_%a.err
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --array=0-{total - 1}%{max_concurrent}

{_modules_block()}{_setup_block()}
LINE=$(awk -F, -v idx="$SLURM_ARRAY_TASK_ID" '$1==idx {{print; exit}}' {index_csv})
P1=$(echo "$LINE" | cut -d, -f2)
P2=$(echo "$LINE" | cut -d, -f3)
P3=$(echo "$LINE" | cut -d, -f4)
OUT=$(echo "$LINE" | cut -d, -f5)
mkdir -p "$OUT"
if [ -f "$OUT/results.pkl" ]; then
    echo "Skipping $OUT (results.pkl exists)"
    exit 0
fi
python3 {run_script} --param1 "$P1" --param2 "$P2" --param3 "$P3" --output "$OUT"
"""
        )
    os.chmod(array_script, 0o755)
    print(f"Wrote array index: {index_csv}")
    print(f"Wrote array script: {array_script}")
    print(f"Total tasks: {total}, max concurrent: {max_concurrent}")
    rc = os.system(f"sbatch {array_script}")
    if rc != 0:
        print(
            "sbatch returned non-zero. If you're not on a SLURM head node, "
            f"submit manually with:\n  sbatch {array_script}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--array",
        action="store_true",
        help=(
            "Submit the full sweep as a single SLURM job array (recommended "
            "for big sweeps like the paper's Fig 2C heatmap)."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=200,
        help="Cap on simultaneously running array tasks. Default: 200.",
    )
    args = parser.parse_args()

    if args.array:
        submit_array(max_concurrent=args.max_concurrent)
    else:
        submit_one_by_one()


if __name__ == "__main__":
    main()
