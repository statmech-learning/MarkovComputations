# Normal-Fan / Tree-Count Training Report

## Status

existing_training_reanalyzed_no_new_training_launched

No new broad training sweep was launched. The existing 32-group exact-degree exact-d_rel full-coupling experiment was reanalyzed because the goal explicitly forbids broad sweeps without a diagnostic reason.

## LOO Models

| outcome | model | groups | predictors | LOO R2 | reason |
| --- | --- | --- | --- | --- | --- |
| outcome_mean | tree_count_only | 32 | 2 | 0.112 | NA |
| outcome_mean | normal_fan_only | 32 | 3 | 0.034 | NA |
| outcome_mean | tree_count_plus_normal_fan | 32 | 5 | -0.029 | NA |
| outcome_mean | normal_fan_plus_cross_root | 32 | 5 | -0.037 | NA |
| outcome_best | tree_count_only | 32 | 2 | 0.092 | NA |
| outcome_best | normal_fan_only | 32 | 3 | 0.169 | NA |
| outcome_best | tree_count_plus_normal_fan | 32 | 5 | 0.137 | NA |
| outcome_best | normal_fan_plus_cross_root | 32 | 5 | 0.102 | NA |
| outcome_seed_std | tree_count_only | 32 | 2 | -0.070 | NA |
| outcome_seed_std | normal_fan_only | 32 | 3 | -0.201 | NA |
| outcome_seed_std | tree_count_plus_normal_fan | 32 | 5 | -0.318 | NA |
| outcome_seed_std | normal_fan_plus_cross_root | 32 | 5 | -0.307 | NA |

## Interpretation

Current one-base exact-degree data cannot separate total rooted-tree abundance from active-tree/normal-fan geometry. log rooted-tree count and active-tree count are strongly correlated, so their weak positive predictive signals should be treated as a combined geometry/abundance direction.

Build multi-base degree-preserving libraries with arms that hold tree count approximately fixed while varying normal-fan coverage and vice versa.
