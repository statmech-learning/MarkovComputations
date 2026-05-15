"""
Submit WTA-ICL retraining jobs on the MIT Engaging cluster.

Writes one SLURM script per (n_nodes, rho_all, seed) triple and sbatch-es it.
By default it reproduces the two paper checkpoints (8_1.0_30 and 12_1.0_20);
edit JOBS for a wider sweep.

BEFORE RUNNING: fill in the CONFIG block below, then on Engaging:
    python submit_jobs_engaging.py            # writes scripts + submits
    python submit_jobs_engaging.py --dry-run  # writes scripts only
"""

import os
import shutil
import argparse
import subprocess

# ============================ CONFIG (EDIT ME) =============================
ACCOUNT   = "YOUR_ENGAGING_ACCOUNT"   # SLURM --account value
PARTITION = "mit_normal"

# Command that activates your Python env with torch (runs inside each job).
#   venv:  "source /home/<user>/envs/wta/bin/activate"
#   conda: "source activate wta"
ENV_ACTIVATE = "source /PATH/TO/YOUR/ENV/bin/activate"

# Path to the repo's ICL directory ON ENGAGING (where run_icl_wta.py lives).
REPO_ICL = "/PATH/ON/ENGAGING/MarkovComputations/ICL"

# Where per-job output directories are created ON ENGAGING.
OUTPUT_BASE = "/PATH/ON/ENGAGING/DirsICL/wta_n_nodes_rhoall_seed"

TIME = "04:00:00"
CPUS = 4
MEM  = "8G"

# (n_nodes, rho_all, seed) triples to run.
# Default = the two checkpoints shipped with the paper.
JOBS = [
    (8,  1.0, 30),
    (12, 1.0, 20),
]
# Full sweep example (uncomment):
# JOBS = [(n, 1.0, s) for n in range(1, 10) for s in (10, 20, 30, 40, 50)]
# ===========================================================================

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=wta_{n}_{r}_{s}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --output={output}/training_batch.out
#SBATCH --error={output}/training_batch.err
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}

set -euo pipefail
export OMP_NUM_THREADS={cpus}

{env_activate}

cd {repo_icl}
python run_icl_wta.py --param1 {n} --param2 {r} --param3 {s} --output {output}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true',
                    help='Write job scripts but do not sbatch them')
    args = ap.parse_args()

    for (n, r, s) in JOBS:
        output = os.path.join(OUTPUT_BASE, f"{n}_{r}_{s}")
        if os.path.exists(output):
            shutil.rmtree(output)
        os.makedirs(output)

        script = JOB_TEMPLATE.format(
            n=n, r=r, s=s, partition=PARTITION, account=ACCOUNT,
            output=output, time=TIME, cpus=CPUS, mem=MEM,
            env_activate=ENV_ACTIVATE, repo_icl=REPO_ICL,
        )
        job_path = os.path.join(output, f"job_{n}_{r}_{s}.sh")
        with open(job_path, "w") as f:
            f.write(script)
        print(f"Wrote {job_path}")

        if not args.dry_run:
            subprocess.run(["sbatch", job_path], check=True)
            print(f"  submitted: n_nodes={n} rho_all={r} seed={s}")

    print(f"\n{len(JOBS)} job(s) processed"
          f"{' (dry run)' if args.dry_run else ''}.")


if __name__ == '__main__':
    main()
