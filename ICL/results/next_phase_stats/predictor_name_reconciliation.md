# Predictor Name Reconciliation

## Gate Result

Phase 1 passes only after the ambiguous bare names are replaced by the recommended names. The fixed-m20 0.409 and 0.158 LOO R2 values are not contradictory because they are different regressions.

## Numerical Collision

- Collision: `fixed m20 "tree_geometry"`.
- Structural rename: `tree_geometry_structural_full`; LOO R2 `0.409`.
- Markov rename: `tree_geometry_markov_reanalysis_subset`; LOO R2 `0.158`.
- Difference: `0.251`.
- Reason: The 0.409 structural number and 0.158 Markov-reanalysis number come from different feature columns, different standardization/regularization schemes, and differently named target fields.  They should not share the bare name 'tree_geometry'.

## Fixed-m20 Predictor Ledger

| source | analysis | old_name | recommended_name | target | unit | groups | loo_r2 | feature_columns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| markov_expressivity_reanalysis | fixed_m20 | capacity_proxy | capacity_proxy_markov_reanalysis | best_seed_novel_icl | group-best | 48 | 0.180 | `capacity_support_fraction`, `capacity_linear_test_margin_p10`, `capacity_linear_test_accuracy` |
| markov_expressivity_reanalysis | fixed_m20 | capacity_proxy | capacity_proxy_markov_reanalysis | mean_novel_icl | group-mean | 48 | 0.145 | `capacity_support_fraction`, `capacity_linear_test_margin_p10`, `capacity_linear_test_accuracy` |
| markov_expressivity_reanalysis | fixed_m20 | capacity_proxy | capacity_proxy_markov_reanalysis | seed_std_novel_icl | seed-std | 48 | -0.043 | `capacity_support_fraction`, `capacity_linear_test_margin_p10`, `capacity_linear_test_accuracy` |
| markov_expressivity_reanalysis | fixed_m20 | comparison_multiplicity | comparison_edge_multiplicity_markov_reanalysis | best_seed_novel_icl | group-best | 48 | 0.107 | `comparison_branch_input_count_min`, `comparison_branch_input_overlap_min`, `comparison_branch_input_overlap_gini` |
| markov_expressivity_reanalysis | fixed_m20 | comparison_multiplicity | comparison_edge_multiplicity_markov_reanalysis | mean_novel_icl | group-mean | 48 | 0.046 | `comparison_branch_input_count_min`, `comparison_branch_input_overlap_min`, `comparison_branch_input_overlap_gini` |
| markov_expressivity_reanalysis | fixed_m20 | comparison_multiplicity | comparison_edge_multiplicity_markov_reanalysis | seed_std_novel_icl | seed-std | 48 | -0.138 | `comparison_branch_input_count_min`, `comparison_branch_input_overlap_min`, `comparison_branch_input_overlap_gini` |
| markov_expressivity_reanalysis | fixed_m20 | multiplicity | edge_multiplicity_markov_reanalysis | best_seed_novel_icl | group-best | 48 | 0.129 | `M_mean_aggregate`, `M_zero_fraction_aggregate`, `M_gini_aggregate` |
| markov_expressivity_reanalysis | fixed_m20 | multiplicity | edge_multiplicity_markov_reanalysis | mean_novel_icl | group-mean | 48 | 0.095 | `M_mean_aggregate`, `M_zero_fraction_aggregate`, `M_gini_aggregate` |
| markov_expressivity_reanalysis | fixed_m20 | multiplicity | edge_multiplicity_markov_reanalysis | seed_std_novel_icl | seed-std | 48 | -0.080 | `M_mean_aggregate`, `M_zero_fraction_aggregate`, `M_gini_aggregate` |
| markov_expressivity_reanalysis | fixed_m20 | tree_geometry | tree_geometry_markov_reanalysis_subset | best_seed_novel_icl | group-best | 48 | 0.080 | `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_mean`, `effective_rank_D_masked`, `condition_number_D_masked_log10` |
| markov_expressivity_reanalysis | fixed_m20 | tree_geometry | tree_geometry_markov_reanalysis_subset | mean_novel_icl | group-mean | 48 | 0.158 | `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_mean`, `effective_rank_D_masked`, `condition_number_D_masked_log10` |
| markov_expressivity_reanalysis | fixed_m20 | tree_geometry | tree_geometry_markov_reanalysis_subset | seed_std_novel_icl | seed-std | 48 | 0.061 | `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_mean`, `effective_rank_D_masked`, `condition_number_D_masked_log10` |
| structural_clustered_inference | fixed_m20 | branch_margin_capacity | branch_margin_capacity_structural | target_max | group-best | 48 | 0.180 | `input_coupled_parameter_count`, `capacity_support_fraction`, `capacity_support_min`, `capacity_linear_test_accuracy`, `capacity_linear_test_margin_p10` |
| structural_clustered_inference | fixed_m20 | branch_margin_capacity | branch_margin_capacity_structural | target_mean | group-mean | 48 | 0.145 | `input_coupled_parameter_count`, `capacity_support_fraction`, `capacity_support_min`, `capacity_linear_test_accuracy`, `capacity_linear_test_margin_p10` |
| structural_clustered_inference | fixed_m20 | branch_margin_capacity | branch_margin_capacity_structural | target_std | seed-std | 48 | -0.043 | `input_coupled_parameter_count`, `capacity_support_fraction`, `capacity_support_min`, `capacity_linear_test_accuracy`, `capacity_linear_test_margin_p10` |
| structural_clustered_inference | fixed_m20 | branch_margin_capacity_plus_drel | branch_margin_capacity_plus_drel_structural | target_max | group-best | 48 | 0.144 | `input_coupled_parameter_count`, `d_rel`, `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_gini`, `capacity_support_fraction`, `capacity_support_min`, `capacity_linear_test_accuracy`, `capacity_linear_test_margin_p10` |
| structural_clustered_inference | fixed_m20 | branch_margin_capacity_plus_drel | branch_margin_capacity_plus_drel_structural | target_mean | group-mean | 48 | 0.044 | `input_coupled_parameter_count`, `d_rel`, `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_gini`, `capacity_support_fraction`, `capacity_support_min`, `capacity_linear_test_accuracy`, `capacity_linear_test_margin_p10` |
| structural_clustered_inference | fixed_m20 | branch_margin_capacity_plus_drel | branch_margin_capacity_plus_drel_structural | target_std | seed-std | 48 | -0.137 | `input_coupled_parameter_count`, `d_rel`, `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_gini`, `capacity_support_fraction`, `capacity_support_min`, `capacity_linear_test_accuracy`, `capacity_linear_test_margin_p10` |
| structural_clustered_inference | fixed_m20 | input_count | input_count_structural | target_max | group-best | 48 | -0.043 | `input_coupled_parameter_count` |
| structural_clustered_inference | fixed_m20 | input_count | input_count_structural | target_mean | group-mean | 48 | -0.043 | `input_coupled_parameter_count` |
| structural_clustered_inference | fixed_m20 | input_count | input_count_structural | target_std | seed-std | 48 | -0.043 | `input_coupled_parameter_count` |
| structural_clustered_inference | fixed_m20 | input_count_plus_branch_drel | input_count_plus_branch_drel_structural | target_max | group-best | 48 | -0.097 | `input_coupled_parameter_count`, `d_rel`, `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_gini`, `comparison_branch_d_rel_min`, `comparison_branch_d_rel_gini` |
| structural_clustered_inference | fixed_m20 | input_count_plus_branch_drel | input_count_plus_branch_drel_structural | target_mean | group-mean | 48 | 0.034 | `input_coupled_parameter_count`, `d_rel`, `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_gini`, `comparison_branch_d_rel_min`, `comparison_branch_d_rel_gini` |
| structural_clustered_inference | fixed_m20 | input_count_plus_branch_drel | input_count_plus_branch_drel_structural | target_std | seed-std | 48 | -0.239 | `input_coupled_parameter_count`, `d_rel`, `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_gini`, `comparison_branch_d_rel_min`, `comparison_branch_d_rel_gini` |
| structural_clustered_inference | fixed_m20 | input_count_plus_drel | input_count_plus_drel_structural | target_max | group-best | 48 | 0.180 | `input_coupled_parameter_count`, `d_rel` |
| structural_clustered_inference | fixed_m20 | input_count_plus_drel | input_count_plus_drel_structural | target_mean | group-mean | 48 | 0.145 | `input_coupled_parameter_count`, `d_rel` |
| structural_clustered_inference | fixed_m20 | input_count_plus_drel | input_count_plus_drel_structural | target_std | seed-std | 48 | -0.043 | `input_coupled_parameter_count`, `d_rel` |
| structural_clustered_inference | fixed_m20 | masked_tree_geometry | masked_tree_geometry_structural | target_max | group-best | 48 | -0.027 | `input_coupled_parameter_count`, `d_rel`, `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_gini`, `comparison_branch_d_rel_min`, `comparison_branch_d_rel_gini`, `effective_rank_D_masked`, `condition_number_D_masked`, `input_edge_load_gini`, `input_coord_load_gini` |
| structural_clustered_inference | fixed_m20 | masked_tree_geometry | masked_tree_geometry_structural | target_mean | group-mean | 48 | 0.189 | `input_coupled_parameter_count`, `d_rel`, `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_gini`, `comparison_branch_d_rel_min`, `comparison_branch_d_rel_gini`, `effective_rank_D_masked`, `condition_number_D_masked`, `input_edge_load_gini`, `input_coord_load_gini` |
| structural_clustered_inference | fixed_m20 | masked_tree_geometry | masked_tree_geometry_structural | target_std | seed-std | 48 | -0.639 | `input_coupled_parameter_count`, `d_rel`, `comparison_branch_common_d_rel_min`, `comparison_branch_common_d_rel_gini`, `comparison_branch_d_rel_min`, `comparison_branch_d_rel_gini`, `effective_rank_D_masked`, `condition_number_D_masked`, `input_edge_load_gini`, `input_coord_load_gini` |
| structural_clustered_inference | fixed_m20 | raw_count | raw_count_structural | target_max | group-best | 48 | -0.043 | `raw_physical_parameter_count` |
| structural_clustered_inference | fixed_m20 | raw_count | raw_count_structural | target_mean | group-mean | 48 | -0.043 | `raw_physical_parameter_count` |
| structural_clustered_inference | fixed_m20 | raw_count | raw_count_structural | target_std | seed-std | 48 | -0.043 | `raw_physical_parameter_count` |
| structural_clustered_inference | fixed_m20 | raw_plus_drel | raw_plus_drel_structural | target_max | group-best | 48 | 0.180 | `raw_physical_parameter_count`, `d_rel` |
| structural_clustered_inference | fixed_m20 | raw_plus_drel | raw_plus_drel_structural | target_mean | group-mean | 48 | 0.145 | `raw_physical_parameter_count`, `d_rel` |
| structural_clustered_inference | fixed_m20 | raw_plus_drel | raw_plus_drel_structural | target_std | seed-std | 48 | -0.043 | `raw_physical_parameter_count`, `d_rel` |
| structural_clustered_inference | fixed_m20 | tree_geometry | tree_geometry_structural_full | target_max | group-best | 48 | 0.280 | `raw_physical_parameter_count`, `d_rel`, `effective_rank_D`, `root_tree_count_gini`, `edge_participation_gini`, `bottleneck_edge_fraction_095` |
| structural_clustered_inference | fixed_m20 | tree_geometry | tree_geometry_structural_full | target_mean | group-mean | 48 | 0.409 | `raw_physical_parameter_count`, `d_rel`, `effective_rank_D`, `root_tree_count_gini`, `edge_participation_gini`, `bottleneck_edge_fraction_095` |
| structural_clustered_inference | fixed_m20 | tree_geometry | tree_geometry_structural_full | target_std | seed-std | 48 | -0.085 | `raw_physical_parameter_count`, `d_rel`, `effective_rank_D`, `root_tree_count_gini`, `edge_participation_gini`, `bottleneck_edge_fraction_095` |

## Regression Definitions

- Structural clustered inference: group rows are topology/mask aggregates, OLS is fit after full-design standardization, and one row is left out at a time.
- Markov reanalysis: group rows are topology/mask aggregates, predictors are standardized inside each training fold, and a ridge of `1e-6` stabilizes the fold-wise linear solve.
- Run-level seed rows remain grouped by topology/mask; seed rows should not be treated as independent topologies.

## Required Rename Map

| structural old name | recommended name |
| --- | --- |
| branch_margin_capacity | branch_margin_capacity_structural |
| branch_margin_capacity_plus_drel | branch_margin_capacity_plus_drel_structural |
| branch_rank_weighted_capacity | branch_rank_weighted_capacity_structural |
| branch_rank_weighted_capacity_plus_drel | branch_rank_weighted_capacity_plus_drel_structural |
| input_count | input_count_structural |
| input_count_plus_branch_drel | input_count_plus_branch_drel_structural |
| input_count_plus_drel | input_count_plus_drel_structural |
| masked_tree_geometry | masked_tree_geometry_structural |
| normal_fan_capacity | normal_fan_capacity_structural |
| normal_fan_capacity_plus_drel | normal_fan_capacity_plus_drel_structural |
| raw_count | raw_count_structural |
| raw_plus_drel | raw_plus_drel_structural |
| rooted_tree_polytope_capacity | rooted_tree_polytope_capacity_structural |
| rooted_tree_polytope_capacity_plus_drel | rooted_tree_polytope_capacity_plus_drel_structural |
| tree_geometry | tree_geometry_structural_full |
| tropical_tree_capacity | tropical_tree_capacity_structural |
| tropical_tree_capacity_plus_drel | tropical_tree_capacity_plus_drel_structural |

| markov old name | recommended name |
| --- | --- |
| capacity_proxy | capacity_proxy_markov_reanalysis |
| comparison_multiplicity | comparison_edge_multiplicity_markov_reanalysis |
| multiplicity | edge_multiplicity_markov_reanalysis |
| tree_geometry | tree_geometry_markov_reanalysis_subset |

## No New Claim

This artifact only resolves naming and regression definitions. It does not add a new scientific claim about which predictor family is better.
