# Next-Phase Topology-ICL Evidence Report

Generated: `2026-05-07T22:36:16.243143+00:00`.

Scope: first-order CRNs with exponential input-dependent rates. These results do not claim a topology theory for autocatalytic or WTA CRNs.

Conservative headline: in the tested fixed-count regimes, topology-associated structural and functional variables explain residual novel-class ICL variation that raw count does not explain.

## Clustered And Group-Aware Inference

### pooled_original

Rows: `240`. Groups: `48`. Families: `3`.

| model | n | group LOO R2 | boot delta R2 | family boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| input_count_plus_branch_drel | 48 | 0.034 | 0.125 | NA | [0.032, 0.234] | 1.000 | -0.257 |
| input_count_plus_drel | 48 | 0.145 | 0.077 | NA | [0.000, 0.177] | 0.967 | -0.131 |
| masked_tree_geometry | 48 | 0.189 | 0.204 | NA | [0.092, 0.327] | 1.000 | -0.185 |
| raw_count | 48 | -0.043 | NA | NA | [NA, NA] | NA | -0.340 |
| raw_plus_drel | 48 | 0.145 | 0.072 | NA | [0.000, 0.170] | 0.950 | -0.131 |
| tree_geometry | 48 | 0.409 | 0.169 | NA | [0.070, 0.281] | 1.000 | 0.131 |

### pooled_branch_capacity

Rows: `240`. Groups: `48`. Families: `3`.

| model | n | group LOO R2 | boot delta R2 | family boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| branch_margin_capacity | 48 | 0.145 | 0.075 | NA | [0.000, 0.169] | 0.963 | -0.131 |
| branch_margin_capacity_plus_drel | 48 | 0.044 | 0.086 | NA | [0.012, 0.180] | 1.000 | -0.195 |
| input_count_plus_branch_drel | 48 | 0.034 | 0.125 | NA | [0.035, 0.230] | 1.000 | -0.257 |
| input_count_plus_drel | 48 | 0.145 | 0.074 | NA | [0.000, 0.187] | 0.960 | -0.131 |
| masked_tree_geometry | 48 | 0.189 | 0.201 | NA | [0.087, 0.322] | 1.000 | -0.185 |
| raw_count | 48 | -0.043 | NA | NA | [NA, NA] | NA | -0.340 |
| raw_plus_drel | 48 | 0.145 | 0.070 | NA | [0.000, 0.162] | 0.940 | -0.131 |
| tree_geometry | 48 | 0.409 | 0.165 | NA | [0.072, 0.271] | 1.000 | 0.131 |

### n5_m7

Rows: `60`. Groups: `12`. Families: `12`.

| model | n | group LOO R2 | boot delta R2 | family boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| branch_margin_capacity | 12 | -0.190 | 0.000 | NA | [-0.000, 0.000] | 0.197 | -0.190 |
| branch_margin_capacity_plus_drel | 12 | -1394.386 | 0.002 | NA | [-0.000, 0.007] | 0.677 | -0.209 |
| input_count_plus_branch_drel | 12 | -2701.327 | 0.002 | NA | [0.000, 0.009] | 0.647 | -0.209 |
| input_count_plus_drel | 12 | -410.923 | 0.002 | NA | [0.000, 0.010] | 0.620 | -0.209 |
| masked_tree_geometry | 12 | -2739.092 | 0.058 | NA | [0.010, 0.168] | 1.000 | -0.789 |
| raw_count | 12 | -0.190 | NA | NA | [NA, NA] | NA | -0.190 |
| raw_plus_drel | 12 | -410.923 | 0.002 | NA | [0.000, 0.009] | 0.630 | -0.209 |
| tree_geometry | 12 | -460.075 | 0.076 | NA | [0.016, 0.193] | 1.000 | -2.822 |

### n5_m12

Rows: `60`. Groups: `12`. Families: `12`.

| model | n | group LOO R2 | boot delta R2 | family boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| branch_margin_capacity | 12 | -0.190 | 0.000 | NA | [-0.000, 0.000] | 0.177 | -0.190 |
| branch_margin_capacity_plus_drel | 12 | -0.190 | -0.000 | NA | [-0.000, 0.000] | 0.187 | -0.190 |
| input_count_plus_branch_drel | 12 | -0.190 | 0.000 | NA | [0.000, 0.000] | 0.000 | -0.190 |
| input_count_plus_drel | 12 | -0.190 | 0.000 | NA | [0.000, 0.000] | 0.000 | -0.190 |
| masked_tree_geometry | 12 | -0.427 | 0.036 | NA | [0.006, 0.102] | 1.000 | -0.427 |
| raw_count | 12 | -0.190 | NA | NA | [NA, NA] | NA | -0.190 |
| raw_plus_drel | 12 | -0.190 | 0.000 | NA | [0.000, 0.000] | 0.000 | -0.190 |
| tree_geometry | 12 | -0.710 | 0.070 | NA | [0.021, 0.154] | 1.000 | -0.710 |

### hard_n4_m6_N3_D2

Rows: `60`. Groups: `12`. Families: `4` via `derived_graph_family`.

| model | n | group LOO R2 | boot delta R2 | family boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_count | 12 | -0.190 | NA | NA | [NA, NA] | NA | -1.034 |
| raw_plus_drel | 12 | 0.203 | 0.066 | 0.077 | [0.000, 0.163] | 0.882 | 0.295 |
| input_count_plus_drel | 12 | 0.203 | 0.068 | 0.077 | [0.000, 0.168] | 0.904 | 0.295 |
| input_count_plus_branch_drel | 12 | 0.203 | 0.067 | 0.081 | [0.000, 0.182] | 0.880 | 0.295 |
| tree_geometry | 12 | -0.112 | 0.135 | 0.124 | [0.042, 0.255] | 1.000 | -0.111 |
| masked_tree_geometry | 12 | -0.151 | 0.096 | 0.090 | [0.011, 0.199] | 1.000 | 0.224 |
| branch_rank_weighted_capacity | 12 | 0.203 | 0.067 | 0.074 | [-0.000, 0.173] | 0.914 | 0.295 |
| branch_rank_weighted_capacity_plus_drel | 12 | 0.203 | 0.068 | 0.077 | [0.000, 0.165] | 0.934 | 0.295 |
| tropical_tree_capacity | 12 | -0.339 | 0.156 | 0.135 | [0.055, 0.312] | 1.000 | 0.055 |
| tropical_tree_capacity_plus_drel | 12 | -0.645 | 0.172 | 0.145 | [0.076, 0.315] | 1.000 | 0.153 |
| rooted_tree_polytope_capacity | 12 | -1.585 | 0.155 | 0.143 | [0.055, 0.289] | 1.000 | 0.126 |
| rooted_tree_polytope_capacity_plus_drel | 12 | -1.585 | 0.153 | 0.137 | [0.054, 0.292] | 1.000 | 0.124 |

### hard_n5_m8_N3_D2

Rows: `60`. Groups: `12`. Families: `6` via `derived_graph_family`.

| model | n | group LOO R2 | boot delta R2 | family boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_count | 12 | -0.190 | NA | NA | [NA, NA] | NA | -0.497 |
| raw_plus_drel | 12 | -0.260 | 0.020 | 0.017 | [0.000, 0.080] | 0.914 | -0.610 |
| input_count_plus_drel | 12 | -0.260 | 0.019 | 0.018 | [0.000, 0.075] | 0.898 | -0.610 |
| input_count_plus_branch_drel | 12 | -0.260 | 0.018 | 0.018 | [0.000, 0.077] | 0.866 | -0.610 |
| tree_geometry | 12 | 0.041 | 0.206 | 0.172 | [0.052, 0.344] | 1.000 | -0.683 |
| masked_tree_geometry | 12 | -0.110 | 0.187 | 0.154 | [0.033, 0.332] | 1.000 | -0.117 |
| branch_rank_weighted_capacity | 12 | -0.260 | 0.017 | 0.018 | [-0.000, 0.076] | 0.892 | -0.610 |
| branch_rank_weighted_capacity_plus_drel | 12 | -0.260 | 0.017 | 0.019 | [-0.000, 0.064] | 0.916 | -0.610 |
| tropical_tree_capacity | 12 | -0.713 | 0.200 | 0.164 | [0.054, 0.361] | 1.000 | -2.073 |
| tropical_tree_capacity_plus_drel | 12 | -0.960 | 0.216 | 0.173 | [0.063, 0.362] | 1.000 | -1.979 |
| rooted_tree_polytope_capacity | 12 | -36.428 | 0.130 | 0.110 | [0.023, 0.287] | 1.000 | -1.185 |
| rooted_tree_polytope_capacity_plus_drel | 12 | -41.290 | 0.147 | 0.118 | [0.034, 0.317] | 1.000 | -0.602 |

### hard_n5_m12_N3_D2

Rows: `60`. Groups: `12`. Families: `7` via `derived_graph_family`.

| model | n | group LOO R2 | boot delta R2 | family boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_count | 12 | -0.190 | NA | NA | [NA, NA] | NA | -0.233 |
| raw_plus_drel | 12 | -0.190 | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | -0.233 |
| input_count_plus_drel | 12 | -0.190 | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | -0.233 |
| input_count_plus_branch_drel | 12 | -0.190 | 0.000 | 0.000 | [0.000, 0.000] | 0.000 | -0.233 |
| tree_geometry | 12 | 0.324 | 0.356 | 0.381 | [0.124, 0.592] | 1.000 | 0.102 |
| masked_tree_geometry | 12 | 0.447 | 0.297 | 0.316 | [0.096, 0.508] | 1.000 | 0.399 |
| branch_rank_weighted_capacity | 12 | -0.190 | -0.000 | -0.000 | [-0.000, 0.000] | 0.274 | -0.233 |
| branch_rank_weighted_capacity_plus_drel | 12 | -0.190 | -0.000 | 0.000 | [-0.000, 0.000] | 0.278 | -0.233 |
| tropical_tree_capacity | 12 | -0.251 | 0.404 | 0.398 | [0.167, 0.620] | 1.000 | -1.499 |
| tropical_tree_capacity_plus_drel | 12 | -0.251 | 0.393 | 0.417 | [0.155, 0.606] | 1.000 | -1.499 |
| rooted_tree_polytope_capacity | 12 | 0.081 | 0.307 | 0.337 | [0.091, 0.537] | 1.000 | -0.235 |
| rooted_tree_polytope_capacity_plus_drel | 12 | 0.081 | 0.311 | 0.337 | [0.080, 0.546] | 1.000 | -0.235 |

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

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank | rooted support frac | rooted branch best-rank min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 2 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |
| coord_block | 1 | 0.561 | 0.561 | NA | NA | NA | NA | NA | NA | NA |
| edge_block | 9 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |
| entry_random | 4 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |

### cycle

Rows: `16`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank | rooted support frac | rooted branch best-rank min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 2 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |
| coord_block | 1 | 0.561 | 0.561 | NA | NA | NA | NA | NA | NA | NA |
| edge_block | 7 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |
| entry_random | 4 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |
| high_participation_edges | 1 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |
| low_participation_edges | 1 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |

### hub

Rows: `16`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank | rooted support frac | rooted branch best-rank min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 3 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |
| coord_block | 1 | 0.561 | 0.561 | NA | NA | NA | NA | NA | NA | NA |
| edge_block | 7 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |
| entry_random | 3 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |
| high_participation_edges | 1 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |
| low_participation_edges | 1 | 0.975 | 0.975 | NA | NA | NA | NA | NA | NA | NA |

### n5_m7

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank | rooted support frac | rooted branch best-rank min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottleneck_bridge | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA | NA | NA |
| cycle_chords | 9 | 0.866 | 0.866 | NA | NA | NA | NA | NA | NA | NA |
| degree_balanced | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA | NA | NA |
| random_sc | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA | NA | NA |

### n5_m12

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank | rooted support frac | rooted branch best-rank min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottleneck_bridge | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA | NA | NA |
| cycle_chords | 4 | 0.866 | 0.866 | NA | NA | NA | NA | NA | NA | NA |
| degree_balanced | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA | NA | NA |
| hub_spoke | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA | NA | NA |
| random_sc | 2 | 0.866 | 0.866 | NA | NA | NA | NA | NA | NA | NA |
| redundant_paths | 1 | 0.866 | 0.866 | NA | NA | NA | NA | NA | NA | NA |
| two_module | 2 | 0.866 | 0.866 | NA | NA | NA | NA | NA | NA | NA |

### hard_n4_m6_N3_D2

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank | rooted support frac | rooted branch best-rank min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cycle_chords | 9 | 0.877 | 0.877 | 0.877 | 0.877 | 0.383 | 0.440 | 2.677 | 1.000 | 3.556 |
| hub_spoke | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.374 | 0.444 | 2.428 | 0.000 | 0.000 |
| random_sc | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.382 | 0.426 | 2.664 | 1.000 | 4.000 |
| two_module | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.375 | 0.452 | 2.425 | 0.000 | 0.000 |

### hard_n5_m8_N3_D2

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank | rooted support frac | rooted branch best-rank min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottleneck_bridge | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.391 | 0.432 | 2.692 | 1.000 | 2.000 |
| cycle_chords | 7 | 0.877 | 0.877 | 0.877 | 0.877 | 0.391 | 0.447 | 3.070 | 1.000 | 5.429 |
| degree_balanced | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.392 | 0.436 | 3.114 | 1.000 | 4.000 |
| hub_spoke | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.384 | 0.417 | 3.062 | 0.000 | 0.000 |
| random_sc | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.399 | 0.440 | 2.899 | 1.000 | 6.000 |
| redundant_paths | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.396 | 0.456 | 2.950 | 1.000 | 6.000 |

### hard_n5_m12_N3_D2

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max | rank-weighted linear mean | rank-weighted linear max | tropical accuracy mean | tropical accuracy max | tropical root eff-rank | rooted support frac | rooted branch best-rank min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottleneck_bridge | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.401 | 0.455 | 2.844 | 1.000 | 14.000 |
| cycle_chords | 4 | 0.877 | 0.877 | 0.877 | 0.877 | 0.395 | 0.457 | 3.079 | 1.000 | 13.000 |
| degree_balanced | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.425 | 0.500 | 3.681 | 1.000 | 12.000 |
| hub_spoke | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.382 | 0.425 | 3.099 | 1.000 | 12.000 |
| random_sc | 2 | 0.877 | 0.877 | 0.877 | 0.877 | 0.389 | 0.433 | 3.144 | 1.000 | 14.000 |
| redundant_paths | 1 | 0.877 | 0.877 | 0.877 | 0.877 | 0.417 | 0.468 | 3.586 | 1.000 | 12.000 |
| two_module | 2 | 0.877 | 0.877 | 0.877 | 0.877 | 0.394 | 0.455 | 3.050 | 1.000 | 14.000 |

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
