#!/bin/bash
# ---------------------------------------------------------------------------
# Standalone SLURM script for ONE WTA-ICL training run on MIT Engaging.
# Use this for a quick test job; use submit_jobs_engaging.py for a sweep.
#
# EDIT the three lines marked  <<< EDIT  before submitting, then:
#     sbatch job_wta_template.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=wta_icl
#SBATCH --partition=mit_normal
#SBATCH --account=YOUR_ENGAGING_ACCOUNT          # <<< EDIT
#SBATCH --output=training_batch.out
#SBATCH --error=training_batch.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Activate your Python env with torch (venv: source .../bin/activate;
# conda: source activate <env>).
source /PATH/TO/YOUR/ENV/bin/activate          # <<< EDIT

# Path to the repo's ICL directory on Engaging.
cd /PATH/ON/ENGAGING/MarkovComputations/ICL    # <<< EDIT

# Reproduces the paper's 8-node checkpoint (n_nodes=8, rho_all=1.0, seed=30).
python run_icl_wta.py --param1 8 --param2 1.0 --param3 30 \
    --output ./results/wta_8_1.0_30
