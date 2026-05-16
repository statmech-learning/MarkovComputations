#!/bin/bash
#SBATCH --job-name=computation
#SBATCH --output=/project/svaikunt/csfloyd/MarkovComputation/DirsICL/wta_n_nodes_rhoall_seed/8_1.0_30/training_batch.out   # Redirect stdout to the output directory
#SBATCH --error=/project/svaikunt/csfloyd/MarkovComputation/DirsICL/wta_n_nodes_rhoall_seed/8_1.0_30/training_batch.err    # Redirect stderr to the output directory
#SBATCH --time=32:00:00
#SBATCH --partition=caslake
##SBATCH --partition=svaikunt 
#SBATCH --account=pi-svaikunt
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32000

# module load python3

python3 /project/svaikunt/csfloyd/MarkovComputation/Python/ICL/run_icl_wta.py --param1 8 --param2 1.0 --param3 30 --output /project/svaikunt/csfloyd/MarkovComputation/DirsICL/wta_n_nodes_rhoall_seed/8_1.0_30
