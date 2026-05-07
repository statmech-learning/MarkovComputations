# Topology-ICL Progress Report

Generated from `/home/aadarwal/repos/topology/ICL`.

## Experiment Coverage

| experiment | runs | groups | m values | mean ICL | best ICL | mean seed std |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| random | 80 | 16 | 20 | 76.76 | 94.60 | 7.08 |
| cycle | 80 | 16 | 20 | 76.38 | 94.20 | 8.37 |
| hub | 80 | 16 | 20 | 69.87 | 91.40 | 8.66 |

## Topology Library Provenance

| experiment | library selected/candidates | families | d_rel values | mean effective rank | mean edge gini |
| --- | ---: | --- | --- | ---: | ---: |
| random | 16/322 | balanced, coord_block, edge_block, entry_random | 190, 200 | 180.084 | 0.105 |
| cycle | 16/322 | balanced, coord_block, edge_block, entry_random, high_participation_edges, low_participation_edges | 190, 200 | 179.991 | 0.095 |
| hub | 16/322 | balanced, coord_block, edge_block, entry_random, high_participation_edges, low_participation_edges | 190, 200 | 172.855 | 0.217 |

## Pooled Fixed-Edge Regime Analysis

Rows pooled across supplied regimes: run-level `240`, topology groups `48`, retrained motif groups `96`.

These models test whether tree-geometry and post-training mechanism features explain accuracy beyond edge count when `m` varies across regimes.

Common branch-rank source counts: run rows `{'recomputed': 240}`, topology groups `{'recomputed': 48}`, retrained groups `{'artifact': 48, 'recomputed': 48}`. Legacy fallback means `comparison_branch_common_d_rel_*` was approximated from the older loose `comparison_branch_d_rel_*` upper-bound metric; regenerate collection artifacts to get exact common-subspace ranks.

Input-overlap source counts: run rows `{'recomputed': 240}`, topology groups `{'recomputed': 48}`, retrained groups `{'artifact': 48, 'recomputed': 48}`. Legacy fallback means `comparison_branch_input_overlap_*` was approximated from the older per-branch input-count metric; regenerate collection artifacts to get exact context/query input-overlap counts.

### Run-Level Novel-Class ICL

| model | n | R2 | LOO_R2 | RMSE | predictors |
| --- | ---: | ---: | ---: | ---: | --- |
| edge_count | 240 | 0.000 | -0.008 | 10.469 | n_edges |
| edge_plus_drel | 240 | 0.073 | 0.057 | 10.081 | n_edges, d_rel |
| input_count | 240 | 0.000 | -0.008 | 10.469 | input_coupled_parameter_count |
| input_plus_drel | 240 | 0.073 | 0.057 | 10.081 | input_coupled_parameter_count, d_rel |
| input_plus_branch_drel | 240 | 0.101 | 0.058 | 9.925 | input_coupled_parameter_count, d_rel, comparison_branch_common_d_rel_min, comparison_branch_common_d_rel_gini, comparison_branch_d_rel_min, comparison_branch_d_rel_gini |
| edge_plus_tree_geometry | 240 | 0.164 | 0.135 | 9.572 | n_edges, d_rel, effective_rank_D, condition_number_D, root_tree_count_gini, edge_participation_gini, mean_shortest_path |
| input_plus_masked_geometry | 240 | 0.183 | 0.116 | 9.466 | input_coupled_parameter_count, d_rel, comparison_branch_common_d_rel_min, comparison_branch_common_d_rel_gini, comparison_branch_d_rel_min, comparison_branch_d_rel_gini, effective_rank_D_masked, condition_number_D_masked, input_edge_load_gini, input_coord_load_gini |
| edge_plus_mechanism | 0 | NA | NA | NA | n_edges, target_logprob_margin_mean, target_logprob_margin_branch_mean_min, branch_active_tree_mi, input_ablation_max_loss |
| edge_plus_projection | 240 | 0.750 | 0.740 | 5.231 | n_edges, posterior_matched_comparison_gap_mean, tree_comparison_energy_fraction_mean, active_tree_matched_comparison_gap_mean |

### Topology Mean Across Seeds

| model | n | R2 | LOO_R2 | RMSE | predictors |
| --- | ---: | ---: | ---: | ---: | --- |
| edge_count | 48 | 0.000 | -0.043 | 6.059 | n_edges |
| edge_plus_drel | 48 | 0.218 | 0.145 | 5.359 | n_edges, d_rel |
| input_count | 48 | 0.000 | -0.043 | 6.059 | input_coupled_parameter_count |
| input_plus_drel | 48 | 0.218 | 0.145 | 5.359 | input_coupled_parameter_count, d_rel |
| input_plus_branch_drel | 48 | 0.302 | 0.034 | 5.061 | input_coupled_parameter_count, d_rel, comparison_branch_common_d_rel_min, comparison_branch_common_d_rel_gini, comparison_branch_d_rel_min, comparison_branch_d_rel_gini |
| edge_plus_tree_geometry | 48 | 0.490 | 0.409 | 4.329 | n_edges, d_rel, effective_rank_D, condition_number_D, root_tree_count_gini, edge_participation_gini, mean_shortest_path |
| input_plus_masked_geometry | 48 | 0.545 | 0.189 | 4.087 | input_coupled_parameter_count, d_rel, comparison_branch_common_d_rel_min, comparison_branch_common_d_rel_gini, comparison_branch_d_rel_min, comparison_branch_d_rel_gini, effective_rank_D_masked, condition_number_D_masked, input_edge_load_gini, input_coord_load_gini |
| edge_plus_mechanism | 0 | NA | NA | NA | n_edges, target_logprob_margin_mean_mean, target_logprob_margin_branch_mean_min_mean, branch_active_tree_mi_mean, input_ablation_max_loss_mean |
| edge_plus_projection | 48 | 0.839 | 0.809 | 2.433 | n_edges, posterior_matched_comparison_gap_mean_mean, tree_comparison_energy_fraction_mean_mean, active_tree_matched_comparison_gap_mean_mean |

### Topology Best Seed

| model | n | R2 | LOO_R2 | RMSE | predictors |
| --- | ---: | ---: | ---: | ---: | --- |
| edge_count | 48 | 0.000 | -0.043 | 6.150 | n_edges |
| edge_plus_drel | 48 | 0.232 | 0.180 | 5.389 | n_edges, d_rel |
| input_count | 48 | 0.000 | -0.043 | 6.150 | input_coupled_parameter_count |
| input_plus_drel | 48 | 0.232 | 0.180 | 5.389 | input_coupled_parameter_count, d_rel |
| input_plus_branch_drel | 48 | 0.286 | -0.097 | 5.198 | input_coupled_parameter_count, d_rel, comparison_branch_common_d_rel_min, comparison_branch_common_d_rel_gini, comparison_branch_d_rel_min, comparison_branch_d_rel_gini |
| edge_plus_tree_geometry | 48 | 0.373 | 0.280 | 4.868 | n_edges, d_rel, effective_rank_D, condition_number_D, root_tree_count_gini, edge_participation_gini, mean_shortest_path |
| input_plus_masked_geometry | 48 | 0.430 | -0.027 | 4.641 | input_coupled_parameter_count, d_rel, comparison_branch_common_d_rel_min, comparison_branch_common_d_rel_gini, comparison_branch_d_rel_min, comparison_branch_d_rel_gini, effective_rank_D_masked, condition_number_D_masked, input_edge_load_gini, input_coord_load_gini |
| edge_plus_mechanism | 0 | NA | NA | NA | n_edges, target_logprob_margin_mean_mean, target_logprob_margin_branch_mean_min_mean, branch_active_tree_mi_mean, input_ablation_max_loss_mean |
| edge_plus_projection | 48 | 0.654 | 0.599 | 3.618 | n_edges, posterior_matched_comparison_gap_mean_mean, tree_comparison_energy_fraction_mean_mean, active_tree_matched_comparison_gap_mean_mean |

## Fixed-Topology Seed Aggregates

Values are `R2/LOO_R2` for topology-level regressions. `target_mean` tracks trainability/reliability across seeds; `target_max` tracks best-seed expressivity.

### Novel-Class ICL Mean Across Seeds

| experiment | rank_only R2/LOO | input_count R2/LOO | input_count_plus_drel R2/LOO | tree_geometry R2/LOO | masked_tree_geometry R2/LOO | mechanism R2/LOO | projection_alignment R2/LOO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0.299/-7.122 | 0.000/-0.138 | 0.299/-7.122 | 0.299/-7.122 | 0.714/-113.490 | NA/NA | 0.855/0.693 |
| cycle | 0.275/-17.407 | 0.000/-0.138 | 0.275/-17.407 | 0.275/-17.407 | 0.634/-14.613 | NA/NA | 0.738/0.420 |
| hub | 0.339/-7.101 | 0.000/-0.138 | 0.339/-7.101 | 0.339/-7.101 | 0.866/0.275 | NA/NA | 0.824/0.647 |

### Novel-Class ICL Best Seed

| experiment | rank_only R2/LOO | input_count R2/LOO | input_count_plus_drel R2/LOO | tree_geometry R2/LOO | masked_tree_geometry R2/LOO | mechanism R2/LOO | projection_alignment R2/LOO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0.330/-9.098 | 0.000/-0.138 | 0.330/-9.098 | 0.330/-9.098 | 0.700/-2517.772 | NA/NA | 0.655/0.448 |
| cycle | 0.235/-14.155 | 0.000/-0.138 | 0.235/-14.155 | 0.235/-14.155 | 0.851/-9.710 | NA/NA | 0.303/-0.729 |
| hub | 0.250/-8.320 | 0.000/-0.138 | 0.250/-8.320 | 0.250/-8.320 | 0.875/-3.275 | NA/NA | 0.793/0.580 |

## Run-Level Raw Count Control

Values are `R2/LOO_R2` for run-level regressions. In fixed-edge libraries, raw count is intentionally constant and should not explain residual topology variation.

| experiment | raw_count R2/LOO | raw_plus_drel R2/LOO | input_count R2/LOO | input_count_plus_drel R2/LOO | input_count_plus_branch_drel R2/LOO | tree_geometry R2/LOO | masked_tree_geometry R2/LOO | trainability_geometry R2/LOO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0.000/-0.025 | 0.112/0.053 | 0.000/-0.025 | 0.112/0.053 | 0.161/0.020 | 0.112/0.053 | 0.268/0.069 | 0.112/0.053 |
| cycle | 0.000/-0.025 | 0.047/-0.000 | 0.000/-0.025 | 0.047/-0.000 | 0.105/-0.040 | 0.047/-0.000 | 0.109/-0.180 | 0.047/-0.000 |
| hub | 0.000/-0.025 | 0.088/0.036 | 0.000/-0.025 | 0.088/0.036 | 0.171/0.021 | 0.088/0.036 | 0.226/-0.016 | 0.088/0.036 |

## Mechanism Correlations

Pearson correlations with novel-class ICL at the run level, using completed mechanism analyses.

| metric | random | cycle | hub |
| --- | --- | --- | --- |
| relative tree dimension | 0.335 | 0.218 | 0.297 |
| weakest context/query common tree-contrast rank | 0.120 | 0.179 | 0.255 |
| weakest comparison-branch paired rank upper bound | 0.301 | 0.224 | 0.280 |
| comparison-branch rank imbalance | -0.313 | -0.234 | -0.284 |
| tree spectrum effective rank | NA | NA | NA |
| masked relative tree effective rank | 0.198 | 0.193 | 0.230 |
| bottleneck/participation heterogeneity | NA | NA | NA |
| input mask edge-load heterogeneity | 0.019 | 0.121 | 0.192 |
| input mask coordinate-load heterogeneity | -0.248 | -0.229 | -0.267 |
| trained branch margin | 0.877 | 0.832 | 0.744 |
| worst branch mean margin | NA | NA | NA |
| branch-active-tree MI | 0.821 | 0.776 | 0.752 |
| tree-sum comparison alignment | 0.340 | 0.388 | 0.245 |
| posterior matched comparison gap | 0.640 | 0.566 | 0.488 |
| input-coupling ablation max loss | 0.747 | 0.761 | 0.844 |
| physical ablation max loss | 0.348 | 0.426 | 0.899 |

## Essential Motif Retraining

Input-ablation 50%-coverage essential physical subgraphs or input-encoding masks were extracted and retrained from scratch.

| source experiment | type | joined motifs | source mean ICL | retrain mean ICL | retrain best ICL | retention mean/max | motif size |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| random | physical subgraph | 16 | 89.57 | 65.69 | 90.00 | 0.733/0.822 | edges 12.69/9/17 |
| random | input mask | 16 | 88.69 | 55.37 | 72.20 | 0.625/0.684 | edges 20.00/20/20 |
| cycle | physical subgraph | 16 | 88.93 | 67.40 | 88.60 | 0.758/0.834 | edges 13.25/10/16 |
| cycle | input mask | 16 | 88.43 | 54.59 | 73.20 | 0.618/0.707 | edges 20.00/20/20 |
| hub | physical subgraph | 16 | 84.22 | 72.09 | 92.40 | 0.856/0.949 | edges 15.25/12/19 |
| hub | input mask | 16 | 82.46 | 52.40 | 65.20 | 0.637/0.702 | edges 20.00/20/20 |

### Top Retrained Motifs

| source experiment | type | motif | size | d_rel | source ICL | retrain mean | retrain max | retention mean/max |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | physical subgraph | essential_input_ablation_loss_n6_m16_0012 | 16 | 300 | 90.20 | 79.84 | 90.00 | 0.885/0.998 |
| random | physical subgraph | essential_input_ablation_loss_n6_m15_0017 | 15 | 280 | 89.20 | 72.60 | 88.80 | 0.814/0.996 |
| random | physical subgraph | essential_input_ablation_loss_n6_m17_0072 | 17 | 320 | 94.60 | 81.68 | 88.40 | 0.863/0.934 |
| random | input mask | random_sc_n6_m20_seed3__mask_essential_input_ablation_loss_c80_0022 | 20 | 80 | 92.20 | 67.76 | 72.20 | 0.735/0.773 |
| random | input mask | random_sc_n6_m20_seed3__mask_essential_input_ablation_loss_c80_0043 | 20 | 80 | 88.00 | 62.64 | 67.60 | 0.712/0.768 |
| random | input mask | random_sc_n6_m20_seed3__mask_essential_input_ablation_loss_c80_0056 | 20 | 80 | 88.00 | 63.32 | 67.40 | 0.720/0.766 |
| cycle | physical subgraph | essential_input_ablation_loss_n6_m15_0043 | 15 | 280 | 93.40 | 80.40 | 88.60 | 0.861/0.949 |
| cycle | physical subgraph | essential_input_ablation_loss_n6_m15_0005 | 15 | 280 | 85.40 | 77.00 | 84.40 | 0.902/0.988 |
| cycle | physical subgraph | essential_input_ablation_loss_n6_m15_0007 | 15 | 280 | 88.60 | 68.88 | 83.80 | 0.777/0.946 |
| cycle | input mask | cycle_chords_n6_m20_seed3__mask_essential_input_ablation_loss_c52_0005 | 20 | 52 | 85.40 | 55.44 | 73.20 | 0.649/0.857 |
| cycle | input mask | cycle_chords_n6_m20_seed3__mask_essential_input_ablation_loss_c80_0069 | 20 | 80 | 87.40 | 60.16 | 69.80 | 0.688/0.799 |
| cycle | input mask | cycle_chords_n6_m20_seed3__mask_essential_input_ablation_loss_c57_0012 | 20 | 57 | 86.60 | 55.72 | 69.60 | 0.643/0.804 |
| hub | physical subgraph | essential_input_ablation_loss_n6_m16_0067 | 16 | 300 | 89.60 | 86.00 | 92.40 | 0.960/1.031 |
| hub | physical subgraph | essential_input_ablation_loss_n6_m18_0016 | 18 | 340 | 81.80 | 83.68 | 92.20 | 1.023/1.127 |
| hub | physical subgraph | essential_input_ablation_loss_n6_m18_0020 | 18 | 340 | 87.20 | 82.84 | 91.00 | 0.950/1.044 |
| hub | input mask | hub_spoke_n6_m20_seed63__mask_essential_input_ablation_loss_c80_0026 | 20 | 80 | 84.20 | 52.60 | 65.20 | 0.625/0.774 |
| hub | input mask | hub_spoke_n6_m20_seed63__mask_essential_input_ablation_loss_c40_0000 | 20 | 40 | 91.40 | 54.84 | 64.00 | 0.600/0.700 |
| hub | input mask | hub_spoke_n6_m20_seed63__mask_essential_input_ablation_loss_c46_0002 | 20 | 46 | 82.60 | 54.24 | 63.60 | 0.657/0.770 |

## Essential Motif Retrain Regressions

Values are `R2/LOO_R2` on retrained essential motif topology groups.

### Retrain Mean Across Seeds

| experiment | rank_only R2/LOO | input_count R2/LOO | input_count_plus_drel R2/LOO | tree_geometry R2/LOO | masked_tree_geometry R2/LOO | mechanism R2/LOO | projection_alignment R2/LOO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random (physical subgraph) | 0.789/0.733 | 0.789/0.733 | 0.789/0.733 | 0.928/0.779 | 0.856/0.776 | NA/NA | NA/NA |
| random (input mask) | 0.555/0.380 | 0.555/0.380 | 0.555/0.380 | 0.555/0.380 | 0.904/-7.846 | NA/NA | NA/NA |
| cycle (physical subgraph) | 0.681/0.587 | 0.613/0.515 | 0.754/0.580 | 0.837/0.537 | 0.853/0.678 | NA/NA | NA/NA |
| cycle (input mask) | 0.311/0.057 | 0.311/0.057 | 0.311/0.057 | 0.311/0.057 | 0.839/-387.846 | NA/NA | NA/NA |
| hub (physical subgraph) | 0.645/0.550 | 0.634/0.536 | 0.646/0.541 | 0.869/0.674 | 0.827/0.708 | NA/NA | NA/NA |
| hub (input mask) | 0.105/-0.260 | 0.105/-0.260 | 0.105/-0.260 | 0.105/-0.260 | 0.781/-4.455 | NA/NA | NA/NA |

### Retrain Best Seed

| experiment | rank_only R2/LOO | input_count R2/LOO | input_count_plus_drel R2/LOO | tree_geometry R2/LOO | masked_tree_geometry R2/LOO | mechanism R2/LOO | projection_alignment R2/LOO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random (physical subgraph) | 0.815/0.758 | 0.815/0.758 | 0.815/0.758 | 0.901/0.639 | 0.850/0.746 | NA/NA | NA/NA |
| random (input mask) | 0.283/0.037 | 0.283/0.037 | 0.283/0.037 | 0.283/0.037 | 0.914/-0.569 | NA/NA | NA/NA |
| cycle (physical subgraph) | 0.721/0.627 | 0.690/0.595 | 0.728/0.620 | 0.797/0.471 | 0.741/0.502 | NA/NA | NA/NA |
| cycle (input mask) | 0.099/-0.171 | 0.099/-0.171 | 0.099/-0.171 | 0.099/-0.171 | 0.949/-6.322 | NA/NA | NA/NA |
| hub (physical subgraph) | 0.663/0.577 | 0.664/0.575 | 0.665/0.572 | 0.830/0.371 | 0.805/0.655 | NA/NA | NA/NA |
| hub (input mask) | 0.008/-0.420 | 0.008/-0.420 | 0.008/-0.420 | 0.008/-0.420 | 0.819/-4.287 | NA/NA | NA/NA |

### Pooled Retrained Motifs

Pooled retrain models include an `essential_layout_is_input_mask` covariate so physical-edge subgraph retrains and input-encoding mask retrains are not conflated silently.

#### Retrain Mean Across Seeds

| model | n | R2 | LOO_R2 | RMSE | predictors |
| --- | ---: | ---: | ---: | ---: | --- |
| layout_type | 96 | 0.530 | 0.510 | 6.722 | essential_layout_is_input_mask |
| layout_plus_edge_count | 96 | 0.805 | 0.793 | 4.328 | essential_layout_is_input_mask, n_edges |
| layout_plus_edge_plus_drel | 96 | 0.831 | 0.816 | 4.030 | essential_layout_is_input_mask, n_edges, d_rel |
| layout_plus_input_count | 96 | 0.824 | 0.813 | 4.111 | essential_layout_is_input_mask, input_coupled_parameter_count |
| layout_plus_input_plus_drel | 96 | 0.835 | 0.821 | 3.987 | essential_layout_is_input_mask, input_coupled_parameter_count, d_rel |
| layout_plus_input_plus_branch_drel | 96 | 0.840 | 0.817 | 3.920 | essential_layout_is_input_mask, input_coupled_parameter_count, d_rel, comparison_branch_common_d_rel_min, comparison_branch_common_d_rel_gini, comparison_branch_d_rel_min, comparison_branch_d_rel_gini |
| layout_plus_edge_plus_tree_geometry | 96 | 0.862 | 0.832 | 3.637 | essential_layout_is_input_mask, n_edges, d_rel, effective_rank_D, condition_number_D, root_tree_count_gini, edge_participation_gini, mean_shortest_path |
| layout_plus_input_plus_masked_geometry | 96 | 0.878 | 0.849 | 3.428 | essential_layout_is_input_mask, input_coupled_parameter_count, d_rel, comparison_branch_common_d_rel_min, comparison_branch_common_d_rel_gini, comparison_branch_d_rel_min, comparison_branch_d_rel_gini, effective_rank_D_masked, condition_number_D_masked, input_edge_load_gini, input_coord_load_gini |
| layout_plus_edge_plus_mechanism | 0 | NA | NA | NA | essential_layout_is_input_mask, n_edges, target_logprob_margin_mean_mean, target_logprob_margin_branch_mean_min_mean, branch_active_tree_mi_mean, input_ablation_max_loss_mean |
| layout_plus_edge_plus_projection | 0 | NA | NA | NA | essential_layout_is_input_mask, n_edges, posterior_matched_comparison_gap_mean_mean, tree_comparison_energy_fraction_mean_mean, active_tree_matched_comparison_gap_mean_mean |

#### Retrain Best Seed

| model | n | R2 | LOO_R2 | RMSE | predictors |
| --- | ---: | ---: | ---: | ---: | --- |
| layout_type | 96 | 0.487 | 0.465 | 7.758 | essential_layout_is_input_mask |
| layout_plus_edge_count | 96 | 0.770 | 0.757 | 5.196 | essential_layout_is_input_mask, n_edges |
| layout_plus_edge_plus_drel | 96 | 0.789 | 0.772 | 4.981 | essential_layout_is_input_mask, n_edges, d_rel |
| layout_plus_input_count | 96 | 0.785 | 0.773 | 5.017 | essential_layout_is_input_mask, input_coupled_parameter_count |
| layout_plus_input_plus_drel | 96 | 0.788 | 0.774 | 4.983 | essential_layout_is_input_mask, input_coupled_parameter_count, d_rel |
| layout_plus_input_plus_branch_drel | 96 | 0.820 | 0.772 | 4.600 | essential_layout_is_input_mask, input_coupled_parameter_count, d_rel, comparison_branch_common_d_rel_min, comparison_branch_common_d_rel_gini, comparison_branch_d_rel_min, comparison_branch_d_rel_gini |
| layout_plus_edge_plus_tree_geometry | 96 | 0.810 | 0.773 | 4.718 | essential_layout_is_input_mask, n_edges, d_rel, effective_rank_D, condition_number_D, root_tree_count_gini, edge_participation_gini, mean_shortest_path |
| layout_plus_input_plus_masked_geometry | 96 | 0.861 | 0.799 | 4.044 | essential_layout_is_input_mask, input_coupled_parameter_count, d_rel, comparison_branch_common_d_rel_min, comparison_branch_common_d_rel_gini, comparison_branch_d_rel_min, comparison_branch_d_rel_gini, effective_rank_D_masked, condition_number_D_masked, input_edge_load_gini, input_coord_load_gini |
| layout_plus_edge_plus_mechanism | 0 | NA | NA | NA | essential_layout_is_input_mask, n_edges, target_logprob_margin_mean_mean, target_logprob_margin_branch_mean_min_mean, branch_active_tree_mi_mean, input_ablation_max_loss_mean |
| layout_plus_edge_plus_projection | 0 | NA | NA | NA | essential_layout_is_input_mask, n_edges, posterior_matched_comparison_gap_mean_mean, tree_comparison_energy_fraction_mean_mean, active_tree_matched_comparison_gap_mean_mean |

## Current Interpretation

- Fixed-edge sweeps directly test topology beyond raw degree count because `n_edges` and raw parameter count are matched within each library.
- Topology-level tree-geometry regressions are the cleanest pre-training test of the matrix-tree hypothesis; mechanism and projection-alignment regressions are post-training explanatory diagnostics.
- Branch-active-tree mutual information, logit margin, comparison-alignment, and ablation losses are functional evidence about what trained models actually used.
- Essential-subgraph retraining separates expressive minimal motifs from dense-graph trainability; retention below 1.0 means dense graphs still helped optimization or supplied redundant pathways.
