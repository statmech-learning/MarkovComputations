# Prospective Tree-Difference Multiplicity Causal Report

## Status

- Training rows: `80` of expected `80`
- Groups with results: `16` of expected `16`
- Mechanism CSV present: `True`

## Group Summary

| load stratum | contrast | groups | mean min diff overlap | mean ICL | best ICL | seed std | branch failures | trained margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_load | high | 4 | 0.936 | 75.360 | 84.900 | 7.031 | 53.994 | 0.204 |
| balanced_load | low | 4 | 0.858 | 79.330 | 89.700 | 7.553 | 42.391 | 0.839 |
| imbalanced_coord_load | high | 4 | 0.658 | 68.830 | 74.450 | 4.666 | 67.475 | -0.316 |
| imbalanced_coord_load | low | 4 | 0.309 | 69.900 | 76.250 | 3.980 | 64.300 | -0.216 |

## High-Low Contrasts

| load stratum | outcome | n high | n low | high mean | low mean | delta | CI95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| balanced_load | mean_seed_novel_icl | 4 | 4 | 75.360 | 79.330 | -3.970 | [-7.990499999999995, 0.6699999999999875] |
| balanced_load | best_seed_novel_icl | 4 | 4 | 84.900 | 89.700 | -4.800 | [-9.0, -0.14999999999997726] |
| imbalanced_coord_load | mean_seed_novel_icl | 4 | 4 | 68.830 | 69.900 | -1.070 | [-4.539999999999992, 2.549999999999997] |
| imbalanced_coord_load | best_seed_novel_icl | 4 | 4 | 74.450 | 76.250 | -1.800 | [-7.150000000000006, 3.0999999999999943] |

## Grouped LOO Models

| outcome | model | groups | LOO R2 | reason |
| --- | --- | --- | --- | --- |
| mean_seed_novel_icl | controls_only | 16 | 0.488 | NA |
| mean_seed_novel_icl | edge_level_multiplicity_plus_controls | 16 | 0.429 | NA |
| mean_seed_novel_icl | tree_level_multiplicity_plus_controls | 16 | 0.488 | NA |
| mean_seed_novel_icl | tree_difference_multiplicity_plus_controls | 16 | 0.447 | NA |
| mean_seed_novel_icl | gamma_no_bias_plus_controls | 16 | 0.320 | NA |
| mean_seed_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 16 | 0.266 | NA |
| best_seed_novel_icl | controls_only | 16 | 0.601 | NA |
| best_seed_novel_icl | edge_level_multiplicity_plus_controls | 16 | 0.591 | NA |
| best_seed_novel_icl | tree_level_multiplicity_plus_controls | 16 | 0.631 | NA |
| best_seed_novel_icl | tree_difference_multiplicity_plus_controls | 16 | 0.545 | NA |
| best_seed_novel_icl | gamma_no_bias_plus_controls | 16 | 0.571 | NA |
| best_seed_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 16 | 0.495 | NA |
| seed_std_novel_icl | controls_only | 16 | 0.294 | NA |
| seed_std_novel_icl | edge_level_multiplicity_plus_controls | 16 | 0.296 | NA |
| seed_std_novel_icl | tree_level_multiplicity_plus_controls | 16 | 0.282 | NA |
| seed_std_novel_icl | tree_difference_multiplicity_plus_controls | 16 | 0.239 | NA |
| seed_std_novel_icl | gamma_no_bias_plus_controls | 16 | 0.232 | NA |
| seed_std_novel_icl | gamma_no_bias_plus_tree_difference_multiplicity | 16 | 0.292 | NA |
| branch_failures | controls_only | 16 | 0.443 | NA |
| branch_failures | edge_level_multiplicity_plus_controls | 16 | 0.449 | NA |
| branch_failures | tree_level_multiplicity_plus_controls | 16 | 0.421 | NA |
| branch_failures | tree_difference_multiplicity_plus_controls | 16 | 0.425 | NA |
| branch_failures | gamma_no_bias_plus_controls | 16 | 0.317 | NA |
| branch_failures | gamma_no_bias_plus_tree_difference_multiplicity | 16 | 0.274 | NA |
| trained_branch_margin | controls_only | 16 | 0.426 | NA |
| trained_branch_margin | edge_level_multiplicity_plus_controls | 16 | 0.464 | NA |
| trained_branch_margin | tree_level_multiplicity_plus_controls | 16 | 0.438 | NA |
| trained_branch_margin | tree_difference_multiplicity_plus_controls | 16 | 0.432 | NA |
| trained_branch_margin | gamma_no_bias_plus_controls | 16 | 0.354 | NA |
| trained_branch_margin | gamma_no_bias_plus_tree_difference_multiplicity | 16 | 0.338 | NA |

## Interpretation

- Prospective causal signal: `not_decisive`
- Primary balanced mean-ICL delta: `-3.970`
- Primary balanced mean-ICL CI95: `[-7.990499999999995, 0.6699999999999875]`
- This report is prospective because masks were selected before these training outcomes were collected.
