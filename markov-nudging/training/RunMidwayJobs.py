"""SLURM submission helper for the markov-nudging training scripts.

Cluster-specific values (output base, partition, account, etc.) are read from
environment variables so this is portable across UChicago Midway, MIT
Engaging, and similar SLURM clusters. To use:

    export SLURM_OUTPUT_BASE=/path/to/where/you/want/runs
    export SLURM_PARTITION=mit_normal       # or caslake, sched_mit_hill, etc.
    export SLURM_ACCOUNT=<your-pi-account>
    python3 RunMidwayJobs.py

Defaults below preserve the original UChicago Midway behavior.
"""

import os
import shutil

# ---------------------------------------------------------------------------
# Cluster configuration (override via environment variables)
# ---------------------------------------------------------------------------
output_base = os.environ.get(
    "SLURM_OUTPUT_BASE",
    "/project/svaikunt/csfloyd/MarkovComputation/Dirs/M_p_full_long3/",
)
partition = os.environ.get("SLURM_PARTITION", "caslake")
account = os.environ.get("SLURM_ACCOUNT", "pi-svaikunt")
mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU", "32000")
time_limit = os.environ.get("SLURM_TIME", "32:00:00")

# Path to the Python training script. Resolved relative to this file by default
# so it works from a fresh checkout without editing.
this_dir = os.path.dirname(os.path.abspath(__file__))
default_script_perceptron = os.path.join(this_dir, "TrainingMidwayPerceptron.py")
default_script_stacked = os.path.join(this_dir, "TrainingMidwayStacked.py")
script_perceptron = os.environ.get("MARKOV_PERCEPTRON_SCRIPT", default_script_perceptron)
script_stacked = os.environ.get("MARKOV_STACKED_SCRIPT", default_script_stacked)

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
n_params = 1
param1_values = [1, 2, 4, 6]
# Other historical sweep choices kept as comments for reference:
#param1_values = [60, 70, 80, 90, 100]
#param1_values = [5, 10, 15, 20]
#param1_values = [10, 20, 30, 40, 50]
#param1_values = [50, 60, 70]
#param1_values = [5, 25, 50, 75]
#param1_values = [1, 5, 10, 15, 20, 15, 30]
#param1_values = [5, 10, 20, 30]
#param1_values = [1, 2, 3, 4, 5]  # random seed

# ---------------------------------------------------------------------------
# SLURM job templates
# ---------------------------------------------------------------------------
job_template = """#!/bin/bash
#SBATCH --job-name=computation
#SBATCH --output={output}/training_batch.out
#SBATCH --error={output}/training_batch.err
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --mem-per-cpu={mem_per_cpu}

# module load python3

python3 {script} --param1 {param1} --output {output}
"""

job_template_2 = """#!/bin/bash
#SBATCH --job-name=computation
#SBATCH --output={output}/training_batch.out
#SBATCH --error={output}/training_batch.err
#SBATCH --time=5:00:00
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --mem-per-cpu={mem_per_cpu}

# module load python3

python3 {script} --param1 {param1} --param2 {param2} --output {output}
"""

if n_params == 1:
    for param1 in param1_values:
        output = output_base + f"{param1}"

        if os.path.exists(output):
            shutil.rmtree(output)
        os.makedirs(output)

        print(f"Created directory: {output}")

        job_script_content = job_template.format(
            param1=param1,
            output=output,
            partition=partition,
            account=account,
            time_limit=time_limit,
            mem_per_cpu=mem_per_cpu,
            script=script_perceptron,
        )

        job_filename = os.path.join(output, f"job_{param1}.sh")
        with open(job_filename, "w") as job_file:
            job_file.write(job_script_content)

        os.system(f"sbatch {job_filename}")
        print(f"Submitted job with param1={param1} and output={output}")


if n_params == 2:
    # NOTE: param2_values is not defined above; left intentionally — historical
    # multi-parameter sweep harness preserved for reference. Define
    # param2_values before flipping n_params to 2.
    for param1 in param1_values:
        for param2 in param2_values:  # noqa: F821 - see note above
            output = os.path.join(output_base, f"{param1}_{param2}")

            if os.path.exists(output):
                shutil.rmtree(output)
            os.makedirs(output)

            print(f"Created directory: {output}")

            job_script_content = job_template_2.format(
                param1=param1,
                param2=param2,
                output=output,
                partition=partition,
                account=account,
                mem_per_cpu=mem_per_cpu,
                script=script_stacked,
            )

            job_filename = os.path.join(output, f"job_{param1}_{param2}.sh")
            with open(job_filename, "w") as job_file:
                job_file.write(job_script_content)

            os.system(f"sbatch {job_filename}")
            print(f"Submitted job with param1={param1}, param2={param2}, and output={output}")
