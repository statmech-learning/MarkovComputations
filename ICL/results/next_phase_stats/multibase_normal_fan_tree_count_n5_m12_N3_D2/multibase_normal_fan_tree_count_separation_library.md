# Multi-Base Normal-Fan / Tree-Count Separation Library

## Status

Library and training manifest are ready. No training jobs were submitted by this generator.

## Controls

| control | value |
| --- | --- |
| n_nodes | 5 |
| n_edges | 12 |
| n_context | 3 |
| z_dim | 2 |
| p | 8 |
| target_d_rel | 88 |
| input_mask | full coupling; exact multiplicity M_alpha=m for every input coordinate |
| degree_control | exact in/out degree sequence within each base_id |

## Counts

| quantity | value |
| --- | --- |
| candidate topologies | 560 |
| selected topologies | 37 |
| matched pairs | 20 |
| training tasks | 185 |

## Base Summary

| base | candidates | selected | tree count range | normal-fan score range |
| --- | --- | --- | --- | --- |
| base00_degree_balanced_seed9 | 80 | 5 | [55.0, 85.0] | [-4.72748171899873, 4.685491951404499] |
| base01_degree_balanced_seed60 | 80 | 7 | [63.0, 96.0] | [-2.0905633794853338, 6.044497179528069] |
| base02_random_sc_seed2 | 80 | 2 | [64.0, 66.0] | [-4.64696533079197, 1.090372498897572] |
| base03_random_sc_seed92 | 80 | 4 | [54.0, 84.0] | [-5.038009100985015, 3.798505855360493] |
| base04_cycle_chords_seed41 | 80 | 4 | [72.0, 78.0] | [-1.4684483833378281, 4.61324931941978] |
| base05_redundant_paths_seed1 | 80 | 9 | [63.0, 94.0] | [-2.7846464103274435, 6.879477559985576] |
| base06_two_module_seed2 | 80 | 6 | [42.0, 69.0] | [-8.915552953432755, -0.02204936397792126] |

## Arm Quality

| arm | pairs | mean abs tree delta | mean abs normal-fan delta |
| --- | --- | --- | --- |
| arm_A_fixed_tree_count_variable_normal_fan | 11 | 0.909 | 4.353 |
| arm_B_variable_tree_count_matched_normal_fan | 9 | 14.667 | 0.060 |

## Files

- `ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/candidate_library.csv`
- `ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/selected.csv`
- `ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/pair_manifest.csv`
- `ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/training_manifest.csv`
- `ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/_array_meta/run_task.sh`

## Training Command

Submit on Engaging with: `sbatch /home/aadarwal/repos/topology/ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/_array_meta/run_task.sh`
