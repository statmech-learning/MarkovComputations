# Next-Phase Topology-ICL Evidence Report

Generated: `2026-05-07T21:59:04.092689+00:00`.

Scope: first-order CRNs with exponential input-dependent rates. These results do not claim a topology theory for autocatalytic or WTA CRNs.

Conservative headline: in the tested fixed-count regimes, topology-associated structural and functional variables explain residual novel-class ICL variation that raw count does not explain.

## Clustered And Group-Aware Inference

### pooled_original

Rows: `240`. Groups: `48`. Families: `3`.

| model | n | group LOO R2 | boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| input_count_plus_branch_drel | 48 | 0.034 | 0.125 | [0.032, 0.234] | 1.000 | -0.257 |
| input_count_plus_drel | 48 | 0.145 | 0.077 | [0.000, 0.177] | 0.967 | -0.131 |
| masked_tree_geometry | 48 | 0.189 | 0.204 | [0.092, 0.327] | 1.000 | -0.185 |
| raw_count | 48 | -0.043 | NA | [NA, NA] | NA | -0.340 |
| raw_plus_drel | 48 | 0.145 | 0.072 | [0.000, 0.170] | 0.950 | -0.131 |
| tree_geometry | 48 | 0.409 | 0.169 | [0.070, 0.281] | 1.000 | 0.131 |

### pooled_branch_capacity

Rows: `240`. Groups: `48`. Families: `3`.

| model | n | group LOO R2 | boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| branch_margin_capacity | 48 | 0.145 | 0.075 | [0.000, 0.169] | 0.963 | -0.131 |
| branch_margin_capacity_plus_drel | 48 | 0.044 | 0.086 | [0.012, 0.180] | 1.000 | -0.195 |
| input_count_plus_branch_drel | 48 | 0.034 | 0.125 | [0.035, 0.230] | 1.000 | -0.257 |
| input_count_plus_drel | 48 | 0.145 | 0.074 | [0.000, 0.187] | 0.960 | -0.131 |
| masked_tree_geometry | 48 | 0.189 | 0.201 | [0.087, 0.322] | 1.000 | -0.185 |
| raw_count | 48 | -0.043 | NA | [NA, NA] | NA | -0.340 |
| raw_plus_drel | 48 | 0.145 | 0.070 | [0.000, 0.162] | 0.940 | -0.131 |
| tree_geometry | 48 | 0.409 | 0.165 | [0.072, 0.271] | 1.000 | 0.131 |

### n5_m7

Rows: `60`. Groups: `12`. Families: `12`.

| model | n | group LOO R2 | boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| branch_margin_capacity | 12 | -0.190 | 0.000 | [-0.000, 0.000] | 0.197 | -0.190 |
| branch_margin_capacity_plus_drel | 12 | -1394.386 | 0.002 | [-0.000, 0.007] | 0.677 | -0.209 |
| input_count_plus_branch_drel | 12 | -2701.327 | 0.002 | [0.000, 0.009] | 0.647 | -0.209 |
| input_count_plus_drel | 12 | -410.923 | 0.002 | [0.000, 0.010] | 0.620 | -0.209 |
| masked_tree_geometry | 12 | -2739.092 | 0.058 | [0.010, 0.168] | 1.000 | -0.789 |
| raw_count | 12 | -0.190 | NA | [NA, NA] | NA | -0.190 |
| raw_plus_drel | 12 | -410.923 | 0.002 | [0.000, 0.009] | 0.630 | -0.209 |
| tree_geometry | 12 | -460.075 | 0.076 | [0.016, 0.193] | 1.000 | -2.822 |

### n5_m12

Rows: `60`. Groups: `12`. Families: `12`.

| model | n | group LOO R2 | boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| branch_margin_capacity | 12 | -0.190 | 0.000 | [-0.000, 0.000] | 0.177 | -0.190 |
| branch_margin_capacity_plus_drel | 12 | -0.190 | -0.000 | [-0.000, 0.000] | 0.187 | -0.190 |
| input_count_plus_branch_drel | 12 | -0.190 | 0.000 | [0.000, 0.000] | 0.000 | -0.190 |
| input_count_plus_drel | 12 | -0.190 | 0.000 | [0.000, 0.000] | 0.000 | -0.190 |
| masked_tree_geometry | 12 | -0.427 | 0.036 | [0.006, 0.102] | 1.000 | -0.427 |
| raw_count | 12 | -0.190 | NA | [NA, NA] | NA | -0.190 |
| raw_plus_drel | 12 | -0.190 | 0.000 | [0.000, 0.000] | 0.000 | -0.190 |
| tree_geometry | 12 | -0.710 | 0.070 | [0.021, 0.154] | 1.000 | -0.710 |

### hard_n4_m6_N3_D2

Rows: `60`. Groups: `12`. Families: `12`.

| model | n | group LOO R2 | boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| raw_count | 12 | -0.190 | NA | [NA, NA] | NA | -0.190 |
| raw_plus_drel | 12 | 0.203 | 0.068 | [0.000, 0.189] | 0.883 | 0.203 |
| input_count_plus_drel | 12 | 0.203 | 0.062 | [0.000, 0.155] | 0.883 | 0.203 |
| input_count_plus_branch_drel | 12 | 0.203 | 0.067 | [0.000, 0.157] | 0.913 | 0.203 |
| tree_geometry | 12 | -0.112 | 0.136 | [0.055, 0.267] | 1.000 | -0.112 |
| masked_tree_geometry | 12 | -0.151 | 0.096 | [0.010, 0.213] | 1.000 | -0.151 |
| branch_rank_weighted_capacity | 12 | 0.203 | 0.066 | [-0.000, 0.180] | 0.907 | 0.203 |
| branch_rank_weighted_capacity_plus_drel | 12 | 0.203 | 0.069 | [-0.000, 0.175] | 0.933 | 0.203 |
| tropical_tree_capacity | 12 | -0.266 | 0.168 | [0.062, 0.294] | 1.000 | -0.266 |
| tropical_tree_capacity_plus_drel | 12 | -0.017 | 0.178 | [0.064, 0.304] | 1.000 | -0.017 |

### hard_n5_m8_N3_D2

Rows: `60`. Groups: `12`. Families: `12`.

| model | n | group LOO R2 | boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| raw_count | 12 | -0.190 | NA | [NA, NA] | NA | -0.190 |
| raw_plus_drel | 12 | -0.260 | 0.020 | [0.000, 0.075] | 0.867 | -0.260 |
| input_count_plus_drel | 12 | -0.260 | 0.019 | [0.000, 0.087] | 0.893 | -0.260 |
| input_count_plus_branch_drel | 12 | -0.260 | 0.021 | [0.000, 0.076] | 0.907 | -0.260 |
| tree_geometry | 12 | 0.041 | 0.209 | [0.056, 0.352] | 1.000 | 0.041 |
| masked_tree_geometry | 12 | -0.110 | 0.179 | [0.041, 0.314] | 1.000 | -0.110 |
| branch_rank_weighted_capacity | 12 | -0.260 | 0.019 | [-0.000, 0.077] | 0.927 | -0.260 |
| branch_rank_weighted_capacity_plus_drel | 12 | -0.260 | 0.019 | [0.000, 0.067] | 0.957 | -0.260 |
| tropical_tree_capacity | 12 | -1.620 | 0.192 | [0.048, 0.344] | 1.000 | -1.620 |
| tropical_tree_capacity_plus_drel | 12 | -1.506 | 0.213 | [0.065, 0.355] | 1.000 | -1.506 |

### hard_n5_m12_N3_D2

Rows: `60`. Groups: `12`. Families: `12`.

| model | n | group LOO R2 | boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| raw_count | 12 | -0.190 | NA | [NA, NA] | NA | -0.190 |
| raw_plus_drel | 12 | -0.190 | 0.000 | [0.000, 0.000] | 0.000 | -0.190 |
| input_count_plus_drel | 12 | -0.190 | 0.000 | [0.000, 0.000] | 0.000 | -0.190 |
| input_count_plus_branch_drel | 12 | -0.190 | 0.000 | [0.000, 0.000] | 0.000 | -0.190 |
| tree_geometry | 12 | 0.324 | 0.355 | [0.124, 0.592] | 1.000 | 0.324 |
| masked_tree_geometry | 12 | 0.447 | 0.297 | [0.093, 0.532] | 1.000 | 0.447 |
| branch_rank_weighted_capacity | 12 | -0.190 | -0.000 | [-0.000, 0.000] | 0.310 | -0.190 |
| branch_rank_weighted_capacity_plus_drel | 12 | -0.190 | -0.000 | [-0.000, 0.000] | 0.300 | -0.190 |
| tropical_tree_capacity | 12 | -0.224 | 0.361 | [0.127, 0.630] | 1.000 | -0.224 |
| tropical_tree_capacity_plus_drel | 12 | -0.224 | 0.344 | [0.107, 0.591] | 1.000 | -0.224 |

## Causal Alignment Interventions

### random

Rows: `1200`. Runs: `80`.

| intervention | n | mean delta | min | max |
| --- | --- | --- | --- | --- |
| context_block_shuffle | 240 | -61.95 | -92.00 | -6.50 |
| decoder_root_permutation | 240 | -55.75 | -91.00 | -5.00 |
| edge_projection_permutation | 240 | -50.65 | -77.00 | -11.50 |
| edge_rate_function_permutation | 240 | -50.61 | -78.00 | -16.50 |
| randomize_K_direction | 240 | -50.82 | -73.00 | -20.00 |

### cycle

Rows: `1200`. Runs: `80`.

| intervention | n | mean delta | min | max |
| --- | --- | --- | --- | --- |
| context_block_shuffle | 240 | -61.34 | -94.50 | -1.00 |
| decoder_root_permutation | 240 | -56.77 | -89.50 | -14.00 |
| edge_projection_permutation | 240 | -51.31 | -75.00 | -17.00 |
| edge_rate_function_permutation | 240 | -51.53 | -79.50 | -17.50 |
| randomize_K_direction | 240 | -51.14 | -71.50 | -20.00 |

### hub

Rows: `1200`. Runs: `80`.

| intervention | n | mean delta | min | max |
| --- | --- | --- | --- | --- |
| context_block_shuffle | 240 | -52.74 | -92.00 | -3.00 |
| decoder_root_permutation | 240 | -49.13 | -84.50 | -2.50 |
| edge_projection_permutation | 240 | -41.91 | -71.50 | -14.00 |
| edge_rate_function_permutation | 240 | -43.94 | -76.50 | -13.50 |
| randomize_K_direction | 240 | -44.44 | -71.50 | -18.00 |

## Branch-Margin Capacity Probes

### random

Rows: `16`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 2 | 0.975 | 0.975 | NA | NA | NA | NA | NA |
| coord_block | 1 | 0.561 | 0.561 | NA | NA | NA | NA | NA |
| edge_block | 9 | 0.975 | 0.975 | NA | NA | NA | NA | NA |
| entry_random | 4 | 0.975 | 0.975 | NA | NA | NA | NA | NA |

### cycle

Rows: `16`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 2 | 0.975 | 0.975 | NA | NA | NA | NA | NA |
| coord_block | 1 | 0.561 | 0.561 | NA | NA | NA | NA | NA |
| edge_block | 7 | 0.975 | 0.975 | NA | NA | NA | NA | NA |
| entry_random | 4 | 0.975 | 0.975 | NA | NA | NA | NA | NA |
| high_participation_edges | 1 | 0.975 | 0.975 | NA | NA | NA | NA | NA |
| low_participation_edges | 1 | 0.975 | 0.975 | NA | NA | NA | NA | NA |

### hub

Rows: `16`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 3 | 0.975 | 0.975 | NA | NA | NA | NA | NA |
| coord_block | 1 | 0.561 | 0.561 | NA | NA | NA | NA | NA |
| edge_block | 7 | 0.975 | 0.975 | NA | NA | NA | NA | NA |
| entry_random | 3 | 0.975 | 0.975 | NA | NA | NA | NA | NA |
| high_participation_edges | 1 | 0.975 | 0.975 | NA | NA | NA | NA | NA |
| low_participation_edges | 1 | 0.975 | 0.975 | NA | NA | NA | NA | NA |

### n5_m7

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottleneck_bridge | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA |
| cycle_chords | 9 | 0.866 | 0.866 | NA | NA | NA | NA | NA |
| degree_balanced | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA |
| random_sc | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA |

### n5_m12

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottleneck_bridge | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA |
| cycle_chords | 4 | 0.866 | 0.866 | NA | NA | NA | NA | NA |
| degree_balanced | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA |
| hub_spoke | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA |
| random_sc | 2 | 0.866 | 0.866 | NA | NA | NA | NA | NA |
| redundant_paths | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA |
| two_module | 2 | 0.866 | 0.866 | NA | NA | NA | NA | NA |

### hard_n4_m6_N3_D2

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cycle_chords | 9 | 0.880 | 0.880 | 0.880 | 0.880 | 0.354 | 0.414 | 2.681 |
| hub_spoke | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.353 | 0.415 | 2.430 |
| random_sc | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.349 | 0.393 | 2.662 |
| two_module | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.352 | 0.404 | 2.429 |

### hard_n5_m8_N3_D2

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottleneck_bridge | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.367 | 0.417 | 2.692 |
| cycle_chords | 7 | 0.880 | 0.880 | 0.880 | 0.880 | 0.367 | 0.427 | 3.069 |
| degree_balanced | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.358 | 0.412 | 3.122 |
| hub_spoke | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.361 | 0.399 | 3.066 |
| random_sc | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.369 | 0.415 | 2.899 |
| redundant_paths | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.366 | 0.401 | 2.951 |

### hard_n5_m12_N3_D2

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottleneck_bridge | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.381 | 0.430 | 2.834 |
| cycle_chords | 4 | 0.880 | 0.880 | 0.880 | 0.880 | 0.377 | 0.461 | 3.096 |
| degree_balanced | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.408 | 0.471 | 3.695 |
| hub_spoke | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.365 | 0.391 | 3.093 |
| random_sc | 2 | 0.880 | 0.880 | 0.880 | 0.880 | 0.379 | 0.424 | 3.141 |
| redundant_paths | 1 | 0.880 | 0.880 | 0.880 | 0.880 | 0.384 | 0.412 | 3.577 |
| two_module | 2 | 0.880 | 0.880 | 0.880 | 0.880 | 0.377 | 0.454 | 3.034 |

## Matched Essential-Motif Controls

### random

Joined controls: `32`. Source motifs represented: `16`.

Matched controls retrain above the extracted motifs here, so this backbone does not support a unique extracted-motif superiority claim.

| control kind | controls | sources | control mean ICL | source motif mean ICL | control-source delta | control win rate | match score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| degree_rewire | 16 | 16 | 69.83 | 65.69 | 4.15 | 0.875 | 0.003 |
| random_sc | 16 | 16 | 68.25 | 65.69 | 2.57 | 0.688 | 0.007 |

### cycle

Joined controls: `32`. Source motifs represented: `16`.

Matched controls are mixed or comparable to the extracted motifs here; interpret motif retraining as evidence that small matched physical subgraphs can support ICL, not that the extracted edge set is uniquely superior.

| control kind | controls | sources | control mean ICL | source motif mean ICL | control-source delta | control win rate | match score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| degree_rewire | 16 | 16 | 66.37 | 67.40 | -1.03 | 0.375 | 0.004 |
| random_sc | 16 | 16 | 67.94 | 67.40 | 0.54 | 0.562 | 0.011 |

### hub

Joined controls: `32`. Source motifs represented: `16`.

Matched controls are mixed or comparable to the extracted motifs here; interpret motif retraining as evidence that small matched physical subgraphs can support ICL, not that the extracted edge set is uniquely superior.

| control kind | controls | sources | control mean ICL | source motif mean ICL | control-source delta | control win rate | match score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| degree_rewire | 16 | 16 | 71.20 | 72.09 | -0.89 | 0.500 | 0.001 |
| random_sc | 16 | 16 | 75.42 | 72.09 | 3.33 | 0.625 | 0.011 |

## Expanded Pilot Status

| regime | root | results.pkl | mechanisms | causal |
| --- | --- | --- | --- | --- |
| n5_m7 | `results/expanded_pilot_sweeps/n5_m7_N2_D1` | 60 | 0 | 0 |
| n5_m12 | `results/expanded_pilot_sweeps/n5_m12_N2_D1` | 60 | 0 | 0 |
| hard_n4_m6_N3_D2 | `results/expanded_hard_sweeps/n4_m6_N3_D2` | 60 | 0 | 0 |
| hard_n5_m8_N3_D2 | `results/expanded_hard_sweeps/n5_m8_N3_D2` | 60 | 0 | 0 |
| hard_n5_m12_N3_D2 | `results/expanded_hard_sweeps/n5_m12_N3_D2` | 60 | 0 | 0 |

## Interpretation Guardrails

- Treat run rows as seeds nested inside topology/mask groups; group-level and clustered summaries are the safer evidence.
- Use `test_novel_classes` as the headline ICL metric.
- Interpret causal scrambling as evidence for branch/projection alignment only when baseline accuracy is high enough to make a collapse meaningful.
- Treat branch-margin capacity as a proxy for tree-polytope branch coverage, not as the final capacity theory.
