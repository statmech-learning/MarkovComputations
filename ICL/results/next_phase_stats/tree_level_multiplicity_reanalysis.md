# Tree-Level Multiplicity Reanalysis

## Gate Result

Phase 2 implements edge-, tree-, and tree-difference-level multiplicity metrics and evaluates them with grouped leave-one-out regressions.  These are pre-training structural metrics; post-training learned-weight APIs are implemented but not used as predictors in aggregate CSV reports because learned K tensors are not stored there.

## Metric Definitions

- Edge-level multiplicity uses `M_alpha = sum_e Omega[e, alpha]` and edge-wise comparison overlap.
- Tree-level multiplicity uses `A[T, alpha] = sum_{e in T} Omega[e, alpha]` and root-conditioned comparison overlap.
- Tree-difference multiplicity uses same-root pairs `(T,T')` and `sum_e |s_T(e)-s_T'(e)| Omega[e, alpha]`.
- Every raw overlap is reported with a normalized overlap because raw counts scale with the number of trees or tree pairs.
- Learned/post-training variants are implemented in `learned_weighted_tree_multiplicity_summary` and `posterior_weighted_tree_overlap`; aggregate reports treat trained margin as an outcome because aggregate CSVs do not store learned `K` tensors.

## Datasets

| dataset | seed rows | groups | with tree metrics | N_c | D | topology sources |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_m20_masks_cluster_topology | 240 | 48 | 48 | 4 | 4 | {'ssh:engaging': 48} |
| hard_full_mask_local | 180 | 36 | 36 | parsed_per_regime | parsed_per_regime | {'local': 36} |

## fixed_m20_masks_cluster_topology: Mean Novel-Class ICL

| model | groups | LOO R2 | reason |
| --- | --- | --- | --- |
| edge_level_multiplicity | 48 | -0.002 | NA |
| tree_level_multiplicity | 48 | 0.403 | NA |
| tree_difference_multiplicity | 48 | 0.435 | NA |
| tree_and_difference_multiplicity | 48 | 0.037 | NA |

## fixed_m20_masks_cluster_topology: Best Seed ICL

| model | groups | LOO R2 | reason |
| --- | --- | --- | --- |
| edge_level_multiplicity | 48 | 0.109 | NA |
| tree_level_multiplicity | 48 | 0.245 | NA |
| tree_difference_multiplicity | 48 | 0.419 | NA |
| tree_and_difference_multiplicity | 48 | -0.057 | NA |

## fixed_m20_masks_cluster_topology: Seed Standard Deviation

| model | groups | LOO R2 | reason |
| --- | --- | --- | --- |
| edge_level_multiplicity | 48 | -0.142 | NA |
| tree_level_multiplicity | 48 | -0.169 | NA |
| tree_difference_multiplicity | 48 | -0.047 | NA |
| tree_and_difference_multiplicity | 48 | -0.057 | NA |

## fixed_m20_masks_cluster_topology: Branch Failures

| model | groups | LOO R2 | reason |
| --- | --- | --- | --- |
| edge_level_multiplicity | 0 | NA | too_few_groups_or_complete_cases |
| tree_level_multiplicity | 0 | NA | too_few_groups_or_complete_cases |
| tree_difference_multiplicity | 0 | NA | too_few_groups_or_complete_cases |
| tree_and_difference_multiplicity | 0 | NA | too_few_groups_or_complete_cases |

## fixed_m20_masks_cluster_topology: Trained Branch Margin

| model | groups | LOO R2 | reason |
| --- | --- | --- | --- |
| edge_level_multiplicity | 0 | NA | too_few_groups_or_complete_cases |
| tree_level_multiplicity | 0 | NA | too_few_groups_or_complete_cases |
| tree_difference_multiplicity | 0 | NA | too_few_groups_or_complete_cases |
| tree_and_difference_multiplicity | 0 | NA | too_few_groups_or_complete_cases |

## hard_full_mask_local: Mean Novel-Class ICL

| model | groups | LOO R2 | reason |
| --- | --- | --- | --- |
| edge_level_multiplicity | 36 | 0.689 | NA |
| tree_level_multiplicity | 36 | 0.758 | NA |
| tree_difference_multiplicity | 33 | 0.703 | NA |
| tree_and_difference_multiplicity | 33 | -0.063 | NA |

## hard_full_mask_local: Best Seed ICL

| model | groups | LOO R2 | reason |
| --- | --- | --- | --- |
| edge_level_multiplicity | 36 | 0.696 | NA |
| tree_level_multiplicity | 36 | 0.753 | NA |
| tree_difference_multiplicity | 33 | 0.661 | NA |
| tree_and_difference_multiplicity | 33 | -0.063 | NA |

## hard_full_mask_local: Seed Standard Deviation

| model | groups | LOO R2 | reason |
| --- | --- | --- | --- |
| edge_level_multiplicity | 36 | -0.024 | NA |
| tree_level_multiplicity | 36 | -0.040 | NA |
| tree_difference_multiplicity | 33 | -0.099 | NA |
| tree_and_difference_multiplicity | 33 | -0.063 | NA |

## hard_full_mask_local: Branch Failures

| model | groups | LOO R2 | reason |
| --- | --- | --- | --- |
| edge_level_multiplicity | 36 | 0.763 | NA |
| tree_level_multiplicity | 36 | 0.844 | NA |
| tree_difference_multiplicity | 33 | 0.793 | NA |
| tree_and_difference_multiplicity | 33 | -0.063 | NA |

## hard_full_mask_local: Trained Branch Margin

| model | groups | LOO R2 | reason |
| --- | --- | --- | --- |
| edge_level_multiplicity | 36 | 0.544 | NA |
| tree_level_multiplicity | 36 | 0.655 | NA |
| tree_difference_multiplicity | 33 | 0.586 | NA |
| tree_and_difference_multiplicity | 33 | -0.063 | NA |

## Interpretation

Use the LOO tables as a screening diagnostic, not as a causal claim.  Fixed-m20 masks are the right local test bed for input multiplicity because physical edge count is fixed and raw masks are read from cluster topology JSONs.  Hard full-mask sweeps add trained branch-failure and branch-margin outcomes, but their full input masks make edge-level multiplicity less diagnostic within each exact regime.
