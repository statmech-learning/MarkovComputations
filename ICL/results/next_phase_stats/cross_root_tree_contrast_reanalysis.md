# Cross-Root Tree-Contrast Reanalysis

## Scope

This report implements decoder-aware cross-root tree-difference comparison metrics. It compares trees rooted at different species because the Markov steady state normalizes all rooted tree numerators jointly and the decoder is learned.

## Datasets

| dataset | groups | diff min median | cross min median | cross best-root min median |
| --- | --- | --- | --- | --- |
| fixed_m20_retrospective | 48 | 0.921 | 0.959 | 0.998 |
| prospective_tree_diff_exact_control | 16 | 0.758 | 0.848 | 0.938 |
| exact_degree_normal_fan | 32 | NA | 1.000 | 1.000 |

## fixed_m20_retrospective LOO Models

| outcome | model | groups | LOO R2 | reason |
| --- | --- | --- | --- | --- |
| outcome_mean | same_root_tree_diff | 48 | 0.412 | NA |
| outcome_mean | cross_root_overlap | 48 | 0.412 | NA |
| outcome_mean | cross_root_oriented | 48 | 0.358 | NA |
| outcome_mean | same_plus_cross_minimal | 48 | 0.290 | NA |
| outcome_best | same_root_tree_diff | 48 | 0.405 | NA |
| outcome_best | cross_root_overlap | 48 | 0.311 | NA |
| outcome_best | cross_root_oriented | 48 | 0.192 | NA |
| outcome_best | same_plus_cross_minimal | 48 | 0.199 | NA |

## fixed_m20_retrospective Correlations

| outcome | feature | n | r |
| --- | --- | --- | --- |
| outcome_mean | diff_overlap_norm_min | 48 | 0.483 |
| outcome_mean | cross_overlap_norm_min | 48 | 0.484 |
| outcome_mean | cross_best_root_pair_overlap_norm_min | 48 | 0.464 |
| outcome_mean | cross_separation_norm_mean | 48 | -0.478 |
| outcome_mean | cross_contrast_effective_rank_mean | 48 | -0.154 |
| outcome_best | diff_overlap_norm_min | 48 | 0.456 |
| outcome_best | cross_overlap_norm_min | 48 | 0.474 |
| outcome_best | cross_best_root_pair_overlap_norm_min | 48 | 0.479 |
| outcome_best | cross_separation_norm_mean | 48 | -0.492 |
| outcome_best | cross_contrast_effective_rank_mean | 48 | -0.231 |

## prospective_tree_diff_exact_control LOO Models

| outcome | model | groups | LOO R2 | reason |
| --- | --- | --- | --- | --- |
| outcome_mean | same_root_tree_diff | 16 | 0.420 | NA |
| outcome_mean | cross_root_overlap | 16 | 0.483 | NA |
| outcome_mean | cross_root_oriented | 16 | 0.185 | NA |
| outcome_mean | same_plus_cross_minimal | 16 | 0.312 | NA |
| outcome_best | same_root_tree_diff | 16 | 0.558 | NA |
| outcome_best | cross_root_overlap | 16 | 0.773 | NA |
| outcome_best | cross_root_oriented | 16 | 0.618 | NA |
| outcome_best | same_plus_cross_minimal | 16 | 0.518 | NA |
| branch_failures | same_root_tree_diff | 16 | 0.300 | NA |
| branch_failures | cross_root_overlap | 16 | 0.185 | NA |
| branch_failures | cross_root_oriented | 16 | 0.332 | NA |
| branch_failures | same_plus_cross_minimal | 16 | 0.133 | NA |
| trained_branch_margin | same_root_tree_diff | 16 | 0.307 | NA |
| trained_branch_margin | cross_root_overlap | 16 | 0.061 | NA |
| trained_branch_margin | cross_root_oriented | 16 | 0.415 | NA |
| trained_branch_margin | same_plus_cross_minimal | 16 | 0.369 | NA |

## prospective_tree_diff_exact_control Correlations

| outcome | feature | n | r |
| --- | --- | --- | --- |
| outcome_mean | diff_overlap_norm_min | 16 | 0.593 |
| outcome_mean | cross_overlap_norm_min | 16 | 0.729 |
| outcome_mean | cross_best_root_pair_overlap_norm_min | 16 | 0.764 |
| outcome_mean | cross_separation_norm_mean | 16 | -0.777 |
| outcome_mean | cross_contrast_effective_rank_mean | 16 | 0.755 |
| outcome_best | diff_overlap_norm_min | 16 | 0.634 |
| outcome_best | cross_overlap_norm_min | 16 | 0.767 |
| outcome_best | cross_best_root_pair_overlap_norm_min | 16 | 0.835 |
| outcome_best | cross_separation_norm_mean | 16 | -0.833 |
| outcome_best | cross_contrast_effective_rank_mean | 16 | 0.780 |

## exact_degree_normal_fan LOO Models

| outcome | model | groups | LOO R2 | reason |
| --- | --- | --- | --- | --- |
| outcome_mean | same_root_tree_diff | 0 | NA | too_few_groups_or_complete_cases |
| outcome_mean | cross_root_overlap | 32 | -0.066 | NA |
| outcome_mean | cross_root_oriented | 32 | 0.087 | NA |
| outcome_mean | same_plus_cross_minimal | 0 | NA | too_few_groups_or_complete_cases |
| outcome_best | same_root_tree_diff | 0 | NA | too_few_groups_or_complete_cases |
| outcome_best | cross_root_overlap | 32 | -0.066 | NA |
| outcome_best | cross_root_oriented | 32 | 0.090 | NA |
| outcome_best | same_plus_cross_minimal | 0 | NA | too_few_groups_or_complete_cases |

## exact_degree_normal_fan Correlations

| outcome | feature | n | r |
| --- | --- | --- | --- |
| outcome_mean | diff_overlap_norm_min | 0 | NA |
| outcome_mean | cross_overlap_norm_min | 32 | NA |
| outcome_mean | cross_best_root_pair_overlap_norm_min | 32 | NA |
| outcome_mean | cross_separation_norm_mean | 32 | NA |
| outcome_mean | cross_contrast_effective_rank_mean | 32 | 0.473 |
| outcome_best | diff_overlap_norm_min | 0 | NA |
| outcome_best | cross_overlap_norm_min | 32 | NA |
| outcome_best | cross_best_root_pair_overlap_norm_min | 32 | NA |
| outcome_best | cross_separation_norm_mean | 32 | NA |
| outcome_best | cross_contrast_effective_rank_mean | 32 | 0.452 |

## Interpretation

Cross-root metrics are implemented and evaluated as diagnostics.  They should not be treated as selectors unless they survive exact-control held-out tests.  In the current data, cross-root co-participation improves some small-sample fits after load-stratum control, but it does not rescue the prospective high/low causal contrast by itself.  In the full-coupling normal-fan experiment, binary cross-root overlap saturates at one and only controllability/rank-style cross-root summaries vary.
