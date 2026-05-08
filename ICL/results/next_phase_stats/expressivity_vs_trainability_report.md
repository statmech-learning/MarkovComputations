# Expressivity vs Trainability Report

Best seed is treated as an expressivity-envelope proxy, mean seed as trainability/reliability, and seed standard deviation as optimization instability.

Hard-regime groups analyzed: 36

## best_seed_novel_icl

| predictor | n_groups | pearson_r |
| --- | --- | --- |
| comparison_branch_common_d_rel_min | 36 | 0.860 |
| effective_rank_D_masked | 36 | 0.895 |
| condition_number_D_masked_log10 | 36 | 0.432 |
| capacity_linear_test_margin_p10 | 36 |  |
| mechanism_branch_active_tree_nmi_mean | 36 | 0.792 |
| mechanism_target_logprob_margin_branch_mean_min | 36 | 0.875 |
| mechanism_tree_entropy_mean | 36 | 0.882 |

## mean_novel_icl

| predictor | n_groups | pearson_r |
| --- | --- | --- |
| comparison_branch_common_d_rel_min | 36 | 0.849 |
| effective_rank_D_masked | 36 | 0.897 |
| condition_number_D_masked_log10 | 36 | 0.429 |
| capacity_linear_test_margin_p10 | 36 |  |
| mechanism_branch_active_tree_nmi_mean | 36 | 0.849 |
| mechanism_target_logprob_margin_branch_mean_min | 36 | 0.942 |
| mechanism_tree_entropy_mean | 36 | 0.883 |

## seed_std_novel_icl

| predictor | n_groups | pearson_r |
| --- | --- | --- |
| comparison_branch_common_d_rel_min | 36 | 0.352 |
| effective_rank_D_masked | 36 | 0.345 |
| condition_number_D_masked_log10 | 36 | 0.132 |
| capacity_linear_test_margin_p10 | 36 |  |
| mechanism_branch_active_tree_nmi_mean | 36 | 0.102 |
| mechanism_target_logprob_margin_branch_mean_min | 36 | 0.178 |
| mechanism_tree_entropy_mean | 36 | 0.304 |
