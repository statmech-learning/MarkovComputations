# Next-Phase Topology-ICL Evidence Report

Generated: `2026-05-07T19:48:13.626592+00:00`.

Scope: first-order CRNs with exponential input-dependent rates. These results do not claim a topology theory for autocatalytic or WTA CRNs.

Conservative headline: in the tested fixed-count regimes, topology-associated structural and functional variables explain residual novel-class ICL variation that raw count does not explain.

## Clustered And Group-Aware Inference

### pooled_original

Rows: `240`. Groups: `48`. Families: `3`.

| model | n | group LOO R2 | boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| raw_count | 48 | -0.043 | NA | [NA, NA] | NA | -0.340 |
| raw_plus_drel | 48 | 0.145 | 0.072 | [0.000, 0.170] | 0.950 | -0.131 |
| input_count_plus_drel | 48 | 0.145 | 0.077 | [0.000, 0.177] | 0.967 | -0.131 |
| input_count_plus_branch_drel | 48 | 0.034 | 0.125 | [0.032, 0.234] | 1.000 | -0.257 |
| tree_geometry | 48 | 0.409 | 0.169 | [0.070, 0.281] | 1.000 | 0.131 |
| masked_tree_geometry | 48 | 0.189 | 0.204 | [0.092, 0.327] | 1.000 | -0.185 |

### pooled_branch_capacity

Rows: `240`. Groups: `48`. Families: `3`.

| model | n | group LOO R2 | boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| raw_count | 48 | -0.043 | NA | [NA, NA] | NA | -0.340 |
| raw_plus_drel | 48 | 0.145 | 0.070 | [0.000, 0.162] | 0.940 | -0.131 |
| input_count_plus_drel | 48 | 0.145 | 0.074 | [0.000, 0.187] | 0.960 | -0.131 |
| input_count_plus_branch_drel | 48 | 0.034 | 0.125 | [0.035, 0.230] | 1.000 | -0.257 |
| tree_geometry | 48 | 0.409 | 0.165 | [0.072, 0.271] | 1.000 | 0.131 |
| masked_tree_geometry | 48 | 0.189 | 0.201 | [0.087, 0.322] | 1.000 | -0.185 |
| branch_margin_capacity | 48 | 0.145 | 0.075 | [0.000, 0.169] | 0.963 | -0.131 |
| branch_margin_capacity_plus_drel | 48 | 0.044 | 0.086 | [0.012, 0.180] | 1.000 | -0.195 |

### n5_m7

Rows: `60`. Groups: `12`. Families: `12`.

| model | n | group LOO R2 | boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| raw_count | 12 | -0.190 | NA | [NA, NA] | NA | -0.190 |
| raw_plus_drel | 12 | -410.923 | 0.002 | [0.000, 0.009] | 0.630 | -0.209 |
| input_count_plus_drel | 12 | -410.923 | 0.002 | [0.000, 0.010] | 0.620 | -0.209 |
| input_count_plus_branch_drel | 12 | -2701.327 | 0.002 | [0.000, 0.009] | 0.647 | -0.209 |
| tree_geometry | 12 | -460.075 | 0.076 | [0.016, 0.193] | 1.000 | -2.822 |
| masked_tree_geometry | 12 | -2739.092 | 0.058 | [0.010, 0.168] | 1.000 | -0.789 |
| branch_margin_capacity | 12 | -0.190 | 0.000 | [-0.000, 0.000] | 0.197 | -0.190 |
| branch_margin_capacity_plus_drel | 12 | -1394.386 | 0.002 | [-0.000, 0.007] | 0.677 | -0.209 |

### n5_m12

Rows: `60`. Groups: `12`. Families: `12`.

| model | n | group LOO R2 | boot delta R2 | CI95 | P(delta>0) | heldout R2 |
| --- | --- | --- | --- | --- | --- | --- |
| raw_count | 12 | -0.190 | NA | [NA, NA] | NA | -0.190 |
| raw_plus_drel | 12 | -0.190 | 0.000 | [0.000, 0.000] | 0.000 | -0.190 |
| input_count_plus_drel | 12 | -0.190 | 0.000 | [0.000, 0.000] | 0.000 | -0.190 |
| input_count_plus_branch_drel | 12 | -0.190 | 0.000 | [0.000, 0.000] | 0.000 | -0.190 |
| tree_geometry | 12 | -0.710 | 0.070 | [0.021, 0.154] | 1.000 | -0.710 |
| masked_tree_geometry | 12 | -0.427 | 0.036 | [0.006, 0.102] | 1.000 | -0.427 |
| branch_margin_capacity | 12 | -0.190 | 0.000 | [-0.000, 0.000] | 0.177 | -0.190 |
| branch_margin_capacity_plus_drel | 12 | -0.190 | -0.000 | [-0.000, 0.000] | 0.187 | -0.190 |

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

| family | n | linear accuracy mean | linear accuracy max |
| --- | --- | --- | --- |
| balanced | 2 | 0.975 | 0.975 |
| coord_block | 1 | 0.561 | 0.561 |
| edge_block | 9 | 0.975 | 0.975 |
| entry_random | 4 | 0.975 | 0.975 |

### cycle

Rows: `16`.

| family | n | linear accuracy mean | linear accuracy max |
| --- | --- | --- | --- |
| balanced | 2 | 0.975 | 0.975 |
| coord_block | 1 | 0.561 | 0.561 |
| edge_block | 7 | 0.975 | 0.975 |
| entry_random | 4 | 0.975 | 0.975 |
| high_participation_edges | 1 | 0.975 | 0.975 |
| low_participation_edges | 1 | 0.975 | 0.975 |

### hub

Rows: `16`.

| family | n | linear accuracy mean | linear accuracy max |
| --- | --- | --- | --- |
| balanced | 3 | 0.975 | 0.975 |
| coord_block | 1 | 0.561 | 0.561 |
| edge_block | 7 | 0.975 | 0.975 |
| entry_random | 3 | 0.975 | 0.975 |
| high_participation_edges | 1 | 0.975 | 0.975 |
| low_participation_edges | 1 | 0.975 | 0.975 |

### n5_m7

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max |
| --- | --- | --- | --- |
| bottleneck_bridge | 1 | 0.866 | 0.866 |
| cycle_chords | 9 | 0.866 | 0.866 |
| degree_balanced | 1 | 0.866 | 0.866 |
| random_sc | 1 | 0.866 | 0.866 |

### n5_m12

Rows: `12`.

| family | n | linear accuracy mean | linear accuracy max |
| --- | --- | --- | --- |
| bottleneck_bridge | 1 | 0.866 | 0.866 |
| cycle_chords | 4 | 0.866 | 0.866 |
| degree_balanced | 1 | 0.866 | 0.866 |
| hub_spoke | 1 | 0.866 | 0.866 |
| random_sc | 2 | 0.866 | 0.866 |
| redundant_paths | 1 | 0.866 | 0.866 |
| two_module | 2 | 0.866 | 0.866 |

## Expanded Pilot Status

| regime | root | results.pkl | mechanisms | causal |
| --- | --- | --- | --- | --- |
| n5_m7 | `results/expanded_pilot_sweeps/n5_m7_N2_D1` | 60 | 0 | 0 |
| n5_m12 | `results/expanded_pilot_sweeps/n5_m12_N2_D1` | 60 | 0 | 0 |

## Interpretation Guardrails

- Treat run rows as seeds nested inside topology/mask groups; group-level and clustered summaries are the safer evidence.
- Use `test_novel_classes` as the headline ICL metric.
- Interpret causal scrambling as evidence for branch/projection alignment only when baseline accuracy is high enough to make a collapse meaningful.
- Treat branch-margin capacity as a proxy for tree-polytope branch coverage, not as the final capacity theory.
