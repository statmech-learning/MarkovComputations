# Input-Mask Topology-ICL Report

Generated from `/home/aadarwal/repos/topology/ICL`.

## Controlled Regimes

These sweeps hold the physical edge count and the number of input-coupled parameters fixed inside each experiment. The rows below are mask-level seed aggregates unless noted otherwise.

| experiment | physical backbone | runs | masks | mask families | input params | d_rel range | mean ICL | best ICL |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| random | random_sc_n6_m20_seed3 | 80 | 16 | 4 | 200 | 190-200 | 76.76 | 94.60 |
| cycle | cycle_chords_n6_m20_seed3 | 80 | 16 | 6 | 200 | 190-200 | 76.38 | 94.20 |
| hub | hub_spoke_n6_m20_seed63 | 80 | 16 | 6 | 200 | 190-200 | 69.87 | 91.40 |

## Pooled Fixed-Input-Count Regressions

Run rows: `240`. Mask groups: `48`. Target: `test_novel_classes`.

Raw count controls should be weak here because `n_edges` and `input_coupled_parameter_count` were fixed by construction. Physical-backbone and mask-family terms test controlled topology effects; mechanism terms test what trained models actually used.

Common branch-rank source counts: run rows `{'recomputed': 240}`, mask groups `{'recomputed': 48}`. Legacy fallback means `comparison_branch_common_d_rel_*` was approximated from the older loose `comparison_branch_d_rel_*` upper-bound metric; regenerate collection artifacts to get exact common-subspace ranks.

Input-overlap source counts: run rows `{'recomputed': 240}`, mask groups `{'recomputed': 48}`. Legacy fallback means `comparison_branch_input_overlap_*` was approximated from the older per-branch input-count metric; regenerate collection artifacts to get exact context/query input-overlap counts.

### Run-Level Novel-Class ICL

| model | n | predictors | R2 | LOO_R2 | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_counts | 240 | 2 | 0.000 | -0.008 | 10.469 |
| physical_backbone | 240 | 2 | 0.091 | 0.068 | 9.981 |
| mask_family | 240 | 5 | 0.077 | 0.030 | 10.058 |
| physical_plus_family | 240 | 7 | 0.171 | 0.118 | 9.530 |
| d_rel | 240 | 1 | 0.073 | 0.057 | 10.081 |
| masked_geometry | 240 | 9 | 0.183 | 0.116 | 9.466 |
| physical_plus_masked_geometry | 240 | 11 | 0.218 | 0.138 | 9.260 |
| physical_family_masked_geometry | 240 | 16 | 0.224 | 0.119 | 9.220 |
| mechanism | 0 | NA | NA | NA | NA |
| physical_plus_mechanism | 0 | NA | NA | NA | NA |

### Mask Mean Across Seeds

| model | n | predictors | R2 | LOO_R2 | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_counts | 48 | 2 | 0.000 | -0.043 | 6.059 |
| physical_backbone | 48 | 2 | 0.272 | 0.172 | 5.170 |
| mask_family | 48 | 5 | 0.230 | -0.166 | 5.316 |
| physical_plus_family | 48 | 7 | 0.512 | 0.309 | 4.233 |
| d_rel | 48 | 1 | 0.218 | 0.145 | 5.359 |
| masked_geometry | 48 | 9 | 0.545 | 0.189 | 4.087 |
| physical_plus_masked_geometry | 48 | 11 | 0.650 | 0.327 | 3.584 |
| physical_family_masked_geometry | 48 | 16 | 0.670 | -0.912 | 3.480 |
| mechanism | 0 | NA | NA | NA | NA |
| physical_plus_mechanism | 0 | NA | NA | NA | NA |

### Mask Best Seed

| model | n | predictors | R2 | LOO_R2 | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_counts | 48 | 2 | 0.000 | -0.043 | 6.150 |
| physical_backbone | 48 | 2 | 0.141 | 0.023 | 5.700 |
| mask_family | 48 | 5 | 0.234 | -0.047 | 5.381 |
| physical_plus_family | 48 | 7 | 0.376 | 0.148 | 4.857 |
| d_rel | 48 | 1 | 0.232 | 0.180 | 5.389 |
| masked_geometry | 48 | 9 | 0.430 | -0.027 | 4.641 |
| physical_plus_masked_geometry | 48 | 11 | 0.534 | -0.003 | 4.199 |
| physical_family_masked_geometry | 48 | 16 | 0.553 | -2.274 | 4.110 |
| mechanism | 0 | NA | NA | NA | NA |
| physical_plus_mechanism | 0 | NA | NA | NA | NA |

### Mask Seed Variability

| model | n | predictors | R2 | LOO_R2 | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_counts | 48 | 2 | 0.000 | -0.043 | 2.886 |
| physical_backbone | 48 | 2 | 0.057 | -0.073 | 2.803 |
| mask_family | 48 | 5 | 0.057 | -0.138 | 2.802 |
| physical_plus_family | 48 | 7 | 0.136 | -0.153 | 2.682 |
| d_rel | 48 | 1 | 0.009 | -0.043 | 2.873 |
| masked_geometry | 48 | 9 | 0.248 | -0.639 | 2.502 |
| physical_plus_masked_geometry | 48 | 11 | 0.275 | -0.695 | 2.457 |
| physical_family_masked_geometry | 48 | 16 | 0.368 | -1.386 | 2.294 |
| mechanism | 0 | NA | NA | NA | NA |
| physical_plus_mechanism | 0 | NA | NA | NA | NA |

## Backbone And Mask-Family Summary

| experiment | physical backbone | mask family | masks | mean ICL | best ICL | mean seed std | d_rel | common branch rank | worst branch margin | tree MI | tree NMI | tree purity | input abl. loss | physical abl. loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cycle | cycle_chords_n6_m20_seed3 | balanced | 2 | 78.24 | 88.60 | 6.64 | 200 | 18 | NA | 1.042 | 0.753 | 0.079 | 15.720 | 23.240 |
| cycle | cycle_chords_n6_m20_seed3 | coord_block | 1 | 68.12 | 77.00 | 7.92 | 190 | 0 | NA | 0.786 | 0.568 | 0.165 | 13.720 | 23.520 |
| cycle | cycle_chords_n6_m20_seed3 | edge_block | 7 | 76.04 | 93.40 | 9.40 | 200 | 40 | NA | 0.867 | 0.627 | 0.181 | 20.057 | 31.634 |
| cycle | cycle_chords_n6_m20_seed3 | entry_random | 4 | 75.12 | 94.20 | 8.96 | 200 | 18 | NA | 0.982 | 0.710 | 0.100 | 15.010 | 22.760 |
| cycle | cycle_chords_n6_m20_seed3 | high_participation_edges | 1 | 81.24 | 89.80 | 6.57 | 200 | 40 | NA | 0.938 | 0.678 | 0.174 | 21.000 | 28.680 |
| cycle | cycle_chords_n6_m20_seed3 | low_participation_edges | 1 | 83.44 | 89.00 | 4.56 | 200 | 40 | NA | 0.982 | 0.710 | 0.192 | 19.640 | 35.840 |
| hub | hub_spoke_n6_m20_seed63 | balanced | 3 | 68.32 | 89.60 | 8.86 | 200 | 16 | NA | 0.845 | 0.611 | 0.111 | 13.080 | 41.653 |
| hub | hub_spoke_n6_m20_seed63 | coord_block | 1 | 57.84 | 69.80 | 9.46 | 190 | 0 | NA | 0.600 | 0.434 | 0.263 | 6.960 | 31.560 |
| hub | hub_spoke_n6_m20_seed63 | edge_block | 7 | 71.63 | 88.80 | 8.90 | 200 | 40 | NA | 0.803 | 0.580 | 0.210 | 15.943 | 43.794 |
| hub | hub_spoke_n6_m20_seed63 | entry_random | 3 | 70.57 | 91.40 | 8.06 | 200 | 16 | NA | 0.856 | 0.619 | 0.118 | 13.547 | 45.040 |
| hub | hub_spoke_n6_m20_seed63 | high_participation_edges | 1 | 71.56 | 79.80 | 8.42 | 200 | 40 | NA | 0.831 | 0.601 | 0.169 | 14.920 | 45.560 |
| hub | hub_spoke_n6_m20_seed63 | low_participation_edges | 1 | 70.48 | 82.80 | 7.54 | 200 | 40 | NA | 0.597 | 0.432 | 0.187 | 16.040 | 42.320 |
| random | random_sc_n6_m20_seed3 | balanced | 2 | 78.44 | 94.60 | 6.24 | 200 | 17 | NA | 0.998 | 0.721 | 0.093 | 16.460 | 22.660 |
| random | random_sc_n6_m20_seed3 | coord_block | 1 | 64.20 | 72.80 | 9.85 | 190 | 0 | NA | 0.732 | 0.529 | 0.140 | 9.880 | 25.360 |
| random | random_sc_n6_m20_seed3 | edge_block | 9 | 76.42 | 93.40 | 7.60 | 200 | 40 | NA | 0.893 | 0.646 | 0.175 | 19.578 | 29.556 |
| random | random_sc_n6_m20_seed3 | entry_random | 4 | 79.81 | 94.60 | 5.63 | 200 | 19 | NA | 1.060 | 0.766 | 0.088 | 15.930 | 24.360 |

## Run-Level Mechanism Correlations

Pearson correlations with novel-class ICL. Structural masked-geometry rows are pre-training controls; margin, active-tree MI, and ablation losses are post-training functional diagnostics.

| metric | random | cycle | hub |
| --- | --- | --- | --- |
| d_rel | 0.335 | 0.218 | 0.297 |
| comparison_branch_common_d_rel_min | 0.120 | 0.179 | 0.255 |
| comparison_branch_common_d_rel_gini | -0.300 | -0.214 | -0.320 |
| comparison_branch_d_rel_min | 0.301 | 0.224 | 0.280 |
| comparison_branch_d_rel_gini | -0.313 | -0.234 | -0.284 |
| effective_rank_D_masked | 0.198 | 0.193 | 0.230 |
| condition_number_D_masked | 0.180 | -0.057 | -0.169 |
| input_edge_load_gini | 0.019 | 0.121 | 0.192 |
| input_coord_load_gini | -0.248 | -0.229 | -0.267 |
| target_logprob_margin_mean | 0.877 | 0.832 | 0.744 |
| target_logprob_margin_branch_mean_min | NA | NA | NA |
| target_logprob_margin_branch_mean_gini | NA | NA | NA |
| target_accuracy_branch_mean_min | NA | NA | NA |
| branch_active_root_mi | 0.784 | 0.757 | 0.722 |
| branch_active_tree_mi | 0.821 | 0.776 | 0.752 |
| branch_active_root_nmi | 0.784 | 0.757 | 0.723 |
| branch_active_tree_nmi | 0.820 | 0.776 | 0.752 |
| branch_active_root_purity_mean | 0.546 | 0.640 | 0.154 |
| branch_active_tree_purity_mean | -0.360 | -0.239 | -0.223 |
| posterior_matched_comparison_gap_mean | 0.640 | 0.566 | 0.488 |
| input_ablation_max_loss | 0.747 | 0.761 | 0.844 |
| physical_ablation_max_loss | 0.348 | 0.426 | 0.899 |

## Best Mask Groups

| experiment | mask | family | mean ICL | best ICL | seed std | d_rel | common branch rank | worst branch margin | eff rank masked | edge gini | tree MI | tree NMI | tree purity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | random_sc_n6_m20_seed3__mask0034_entry_random_c200_seed35 | entry_random | 89.32 | 94.60 | 3.06 | 200 | 25 | NA | 179.346 | 0.161 | 1.153 | 0.833 | 0.084 |
| random | random_sc_n6_m20_seed3__mask0080_edge_block_c200_seed1 | edge_block | 86.88 | 93.40 | 6.39 | 200 | 40 | NA | 181.649 | 0.500 | 1.039 | 0.751 | 0.150 |
| cycle | cycle_chords_n6_m20_seed3__mask0321_low_participation_edges_c200_seed1 | low_participation_edges | 83.44 | 89.00 | 4.56 | 200 | 40 | NA | 177.732 | 0.500 | 0.982 | 0.710 | 0.192 |
| cycle | cycle_chords_n6_m20_seed3__mask0320_high_participation_edges_c200_seed1 | high_participation_edges | 81.24 | 89.80 | 6.57 | 200 | 40 | NA | 176.940 | 0.500 | 0.938 | 0.678 | 0.174 |
| random | random_sc_n6_m20_seed3__mask0043_entry_random_c200_seed44 | entry_random | 80.80 | 89.20 | 6.26 | 200 | 18 | NA | 180.358 | 0.117 | 1.043 | 0.754 | 0.098 |
| cycle | cycle_chords_n6_m20_seed3__mask0000_entry_random_c200_seed1 | entry_random | 80.60 | 94.20 | 11.46 | 200 | 17 | NA | 180.069 | 0.091 | 1.057 | 0.764 | 0.090 |
| cycle | cycle_chords_n6_m20_seed3__mask0242_balanced_c200_seed3 | balanced | 80.44 | 88.60 | 4.98 | 200 | 19 | NA | 184.137 | 0.000 | 1.044 | 0.754 | 0.083 |
| random | random_sc_n6_m20_seed3__mask0291_balanced_c200_seed52 | balanced | 79.92 | 94.60 | 9.17 | 200 | 16 | NA | 180.807 | 0.000 | 1.023 | 0.739 | 0.096 |

## Weakest Mask Groups

| experiment | mask | family | mean ICL | best ICL | seed std | d_rel | common branch rank | worst branch margin | eff rank masked | edge gini | tree MI | tree NMI | tree purity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hub | hub_spoke_n6_m20_seed63__mask0160_coord_block_c200_seed1 | coord_block | 57.84 | 69.80 | 9.46 | 190 | 0 | NA | 151.206 | 0.000 | 0.600 | 0.434 | 0.263 |
| hub | hub_spoke_n6_m20_seed63__mask0306_balanced_c200_seed67 | balanced | 63.32 | 72.20 | 6.30 | 200 | 16 | NA | 178.986 | 0.010 | 0.776 | 0.561 | 0.101 |
| hub | hub_spoke_n6_m20_seed63__mask0044_entry_random_c200_seed45 | entry_random | 63.76 | 72.00 | 6.55 | 200 | 14 | NA | 171.540 | 0.103 | 0.780 | 0.564 | 0.113 |
| random | random_sc_n6_m20_seed3__mask0160_coord_block_c200_seed1 | coord_block | 64.20 | 72.80 | 9.85 | 190 | 0 | NA | 159.798 | 0.000 | 0.732 | 0.529 | 0.140 |
| hub | hub_spoke_n6_m20_seed63__mask0310_balanced_c200_seed71 | balanced | 66.72 | 82.60 | 10.77 | 200 | 15 | NA | 175.982 | 0.000 | 0.851 | 0.615 | 0.109 |
| hub | hub_spoke_n6_m20_seed63__mask0150_edge_block_c200_seed71 | edge_block | 67.28 | 79.60 | 8.00 | 200 | 40 | NA | 176.699 | 0.500 | 0.791 | 0.572 | 0.219 |
| cycle | cycle_chords_n6_m20_seed3__mask0160_coord_block_c200_seed1 | coord_block | 68.12 | 77.00 | 7.92 | 190 | 0 | NA | 158.986 | 0.000 | 0.786 | 0.568 | 0.165 |
| hub | hub_spoke_n6_m20_seed63__mask0020_entry_random_c200_seed21 | entry_random | 68.64 | 79.80 | 8.20 | 200 | 17 | NA | 173.787 | 0.121 | 0.841 | 0.608 | 0.109 |

## Essential Input-Mask Retraining

Input-ablation 50%-coverage essential masks keep the physical graph fixed and prune only input-coupling rows, then retrain those masks from scratch.

### Extracted Essential Input Masks

| source experiment | selected/candidates | source input couplings | essential input couplings min/mean/max | raw edge rows mean/max | d_rel mean | common branch rank mean | source best ICL mean/best |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| random | 16/75 | 200.0 | 40/61.1/80 | 3.9/5 | 61.1 | NA | 89.22/94.60 |
| cycle | 16/75 | 200.0 | 39/60.2/80 | 3.9/5 | 60.2 | NA | 88.71/94.20 |
| hub | 16/74 | 200.0 | 40/56.2/80 | 3.6/5 | 56.2 | NA | 83.76/91.40 |

### Top Extracted Essential Input Masks

| source experiment | mask | source best ICL | input couplings | source couplings | raw edge rows | d_rel | common branch rank | eff rank masked |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | random_sc_n6_m20_seed3__mask_essential_input_ablation_loss_c54_0010 | 94.60 | 54 | 200 | 4 | 54 | NA | 52.610 |
| random | random_sc_n6_m20_seed3__mask_essential_input_ablation_loss_c56_0012 | 90.20 | 56 | 200 | 5 | 56 | NA | 53.211 |
| random | random_sc_n6_m20_seed3__mask_essential_input_ablation_loss_c45_0013 | 88.60 | 45 | 200 | 4 | 45 | NA | 42.963 |
| cycle | cycle_chords_n6_m20_seed3__mask_essential_input_ablation_loss_c49_0000 | 94.20 | 49 | 200 | 4 | 49 | NA | 47.226 |
| cycle | cycle_chords_n6_m20_seed3__mask_essential_input_ablation_loss_c39_0002 | 91.80 | 39 | 200 | 4 | 39 | NA | 38.109 |
| cycle | cycle_chords_n6_m20_seed3__mask_essential_input_ablation_loss_c52_0005 | 85.40 | 52 | 200 | 5 | 52 | NA | 50.234 |
| hub | hub_spoke_n6_m20_seed63__mask_essential_input_ablation_loss_c40_0000 | 91.40 | 40 | 200 | 4 | 40 | NA | 38.759 |
| hub | hub_spoke_n6_m20_seed63__mask_essential_input_ablation_loss_c46_0002 | 82.60 | 46 | 200 | 4 | 46 | NA | 44.899 |
| hub | hub_spoke_n6_m20_seed63__mask_essential_input_ablation_loss_c41_0004 | 86.00 | 41 | 200 | 4 | 41 | NA | 39.890 |

### Retrain Retention

| source experiment | joined masks | source mean ICL | retrain mean ICL | retrain best ICL | retention mean/max | retrain input couplings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 16 | 88.69 | 55.37 | 72.20 | 0.625/0.684 | 61.1 |
| cycle | 16 | 88.43 | 54.59 | 73.20 | 0.618/0.707 | 60.2 |
| hub | 16 | 82.46 | 52.40 | 65.20 | 0.637/0.702 | 56.2 |

### Top Retrained Essential Input Masks

| source experiment | mask | source ICL | retrain mean | retrain max | retention mean/max | input couplings | d_rel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | random_sc_n6_m20_seed3__mask_essential_input_ablation_loss_c80_0022 | 92.20 | 67.76 | 72.20 | 0.735/0.773 | 80 | 80 |
| random | random_sc_n6_m20_seed3__mask_essential_input_ablation_loss_c80_0043 | 88.00 | 62.64 | 67.60 | 0.712/0.768 | 80 | 80 |
| random | random_sc_n6_m20_seed3__mask_essential_input_ablation_loss_c80_0056 | 88.00 | 63.32 | 67.40 | 0.720/0.766 | 80 | 80 |
| cycle | cycle_chords_n6_m20_seed3__mask_essential_input_ablation_loss_c52_0005 | 85.40 | 55.44 | 73.20 | 0.649/0.857 | 52 | 52 |
| cycle | cycle_chords_n6_m20_seed3__mask_essential_input_ablation_loss_c80_0069 | 87.40 | 60.16 | 69.80 | 0.688/0.799 | 80 | 80 |
| cycle | cycle_chords_n6_m20_seed3__mask_essential_input_ablation_loss_c57_0012 | 86.60 | 55.72 | 69.60 | 0.643/0.804 | 57 | 57 |
| hub | hub_spoke_n6_m20_seed63__mask_essential_input_ablation_loss_c80_0026 | 84.20 | 52.60 | 65.20 | 0.625/0.774 | 80 | 80 |
| hub | hub_spoke_n6_m20_seed63__mask_essential_input_ablation_loss_c40_0000 | 91.40 | 54.84 | 64.00 | 0.600/0.700 | 40 | 40 |
| hub | hub_spoke_n6_m20_seed63__mask_essential_input_ablation_loss_c46_0002 | 82.60 | 54.24 | 63.60 | 0.657/0.770 | 46 | 46 |

## Current Interpretation

- The fixed-count input-mask design removes raw trainable degree count as an explanation for within-regime variation.
- Physical-backbone and mask-family regressions quantify topology effects before using trained-model diagnostics.
- Masked relative tree geometry is a coarse pre-training proxy; weak LOO performance means it should not be overinterpreted as a complete capacity theory.
- Branch-active-tree mutual information, logit margin, and edge ablation losses test whether successful trained models organize ICL through functional tree/edge structure.
