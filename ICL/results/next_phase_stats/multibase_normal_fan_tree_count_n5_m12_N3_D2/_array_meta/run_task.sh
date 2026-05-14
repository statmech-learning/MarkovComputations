#!/bin/bash
#SBATCH --job-name=mb_nf_tree
#SBATCH --output=/home/aadarwal/repos/topology/ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/_array_meta/task_%a.out
#SBATCH --error=/home/aadarwal/repos/topology/ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/_array_meta/task_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-184%24

set -euo pipefail
cd /home/aadarwal/repos/topology/ICL
module load miniforge/25.11.0-0
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
CMD=$(sed -n "${LINE_NUM}p" /home/aadarwal/repos/topology/ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/_array_meta/commands.txt)
OUT=$(sed -n "${LINE_NUM}p" /home/aadarwal/repos/topology/ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/_array_meta/outputs.txt)
mkdir -p "$OUT"
if [ -f "$OUT/results.pkl" ]; then
    echo "Skipping $OUT (results.pkl exists)"
    exit 0
fi
echo "$CMD"
eval "$CMD"
