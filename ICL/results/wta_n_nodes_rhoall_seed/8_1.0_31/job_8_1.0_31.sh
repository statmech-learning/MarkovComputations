#!/bin/bash
#SBATCH --job-name=wta_8_1.0_31
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --output=/orcd/home/002/aadarwal/MarkovComputations/ICL/results/wta_n_nodes_rhoall_seed/8_1.0_31/training_batch.out
#SBATCH --error=/orcd/home/002/aadarwal/MarkovComputations/ICL/results/wta_n_nodes_rhoall_seed/8_1.0_31/training_batch.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail
export OMP_NUM_THREADS=4
source /home/aadarwal/wta-icl-env/bin/activate
cd /orcd/home/002/aadarwal/MarkovComputations/ICL
python run_icl_wta.py --param1 8 --param2 1.0 --param3 31 --output /orcd/home/002/aadarwal/MarkovComputations/ICL/results/wta_n_nodes_rhoall_seed/8_1.0_31
