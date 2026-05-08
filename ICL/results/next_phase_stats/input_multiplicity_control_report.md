# Input Multiplicity Control Report

This report uses existing fixed-count input-mask data. It is an existing-data control, not a new training sweep.

## Mask-Family Summary

| input_mask_family | n_groups | mean_group_mean_icl | mean_group_best_icl | mean_seed_std | mean_M_mean | mean_M_gini | mean_branch_input_overlap_min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 7 | 74.046 | 85.029 | 7.476 | 10.000 | 0.001 | 17.143 |
| coord_block | 3 | 63.387 | 73.200 | 9.076 | 10.000 | 0.500 | 0.000 |
| edge_block | 23 | 74.847 | 85.670 | 8.543 | 10.000 | 0.000 | 40.000 |
| entry_random | 11 | 75.585 | 85.273 | 7.506 | 10.000 | 0.130 | 17.636 |
| high_participation_edges | 2 | 76.400 | 84.800 | 7.497 | 10.000 | 0.000 | 40.000 |
| low_participation_edges | 2 | 76.960 | 85.900 | 6.052 | 10.000 | 0.000 | 40.000 |

## Interpretation

Input multiplicity is not a scalar law in these data. The useful signal is branch-aware and mask-aware: zero or imbalanced context/query overlap is a clearer risk than average input count alone.
