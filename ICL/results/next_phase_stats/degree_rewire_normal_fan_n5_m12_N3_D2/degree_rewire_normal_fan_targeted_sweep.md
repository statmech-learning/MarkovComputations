# Degree-Rewired Normal-Fan Targeted Sweep

This is the targeted follow-up for the previously missing contrast: same exact in/out degree sequence and same `d_rel`, but different tree-polytope normal-fan diagnostics.

## Library

- Base topology: `g0354_degree_balanced_seed9`
- Variants: `32`
- `N_n=5`, `m=12`, `N_c=3`, `D=2`
- Exact in-degree sequence: `[3, 2, 2, 3, 2]`
- Exact out-degree sequence: `[3, 2, 2, 3, 2]`
- `d_rel` values: `[88.0]`
- Root tree-count Gini range: `0.0` to `0.0`
- Edge participation Gini range: `0.10000000000000009` to `0.16153846153846163`
- Total rooted-tree count range: `55.0` to `85.0`

All variants were produced by directed double-edge swaps that preserve each node's exact in-degree and out-degree.

## Capacity Extremes

| metric | low topology | low value | high topology | high value | delta |
| --- | --- | ---: | --- | ---: | ---: |
| `normal_fan_branch_tree_nmi_mean` | `degrewire0027_baseg0354_degree_balanced_seed9_seed34` | 0.135706 | `degrewire0003_baseg0354_degree_balanced_seed9_seed3` | 0.160333 | 0.024627 |
| `normal_fan_active_tree_count_mean` | `degrewire0002_baseg0354_degree_balanced_seed9_seed2` | 39.250000 | `degrewire0024_baseg0354_degree_balanced_seed9_seed31` | 52.750000 | 13.500000 |

The branch-tree NMI contrast is modest, but active-tree count changes substantially under exact degree and `d_rel` control. These four extremes were selected for training.

## Training Submission

- Selected training CSV: `selected_training.csv`
- Training topologies: `degrewire0027_baseg0354_degree_balanced_seed9_seed34, degrewire0003_baseg0354_degree_balanced_seed9_seed3, degrewire0002_baseg0354_degree_balanced_seed9_seed2, degrewire0024_baseg0354_degree_balanced_seed9_seed31`
- Seeds: `1,2,3,4,5`
- Expected runs: `20`
- Slurm job: `13538460`
- Completed `results.pkl` files at report generation: `0`
- Current `squeue` line: `13538460_[0-19%8] mit_norma topo_lib aadarwal PD       0:00      1 (Priority)`

This sweep is intentionally small: it isolates the exact-degree/normal-fan mechanism rather than adding another broad graph-family sweep.
