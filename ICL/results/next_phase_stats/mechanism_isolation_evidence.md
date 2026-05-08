# Mechanism-Isolation Evidence

This report connects the mechanism-isolating contrast plan to actual trained outcomes where the checked-in data supports the requested fixed-count comparisons.

It is intentionally conservative: unavailable contrasts are reported as missing rather than inferred from weaker proxies.

## Summary

- Generated at commit `e2435210e8ec90e339b5bd308a8c81f484ee8b30`.
- Evidence rows: `9`.
- Missing requested contrasts: `same_degree_sequence_different_normal_fan`.

| contrast | status | regime | varied metric | delta metric | delta mean ICL | note |
| --- | --- | --- | --- | ---: | ---: | --- |
| same_drel_different_bottleneck_participation | observed_with_trained_outcomes | n5_m12_N3_D2 | edge_participation_gini | 0.237 | -0.240 | Largest available same-d_rel edge-participation contrast in the checked-in hard library; trained means are nearly identical, so this contrast does not support edge participation alone as a strong driver. |
| same_tree_count_different_root_balance | observed_with_trained_outcomes | n4_m6_N3_D2 | root_tree_count_gini | 0.312 | 3.760 | Largest available same-total-tree-count root-balance contrast. Higher root imbalance has higher mean here, so root balance is not a monotone penalty in this tiny contrast. |
| same_physical_graph_permuted_input_masks | observed_with_trained_outcomes | fixed_m20_input_masks | input_edge_load_gini | 0.500 | -6.320 | Fixed physical graph/count/d_rel; contrasts edge-load heterogeneity among available masks. |
| same_mask_count_different_coordinate_load | observed_with_trained_outcomes | fixed_m20_input_masks | input_coord_load_gini | 0.161 | 2.120 | Fixed physical graph/count/d_rel; contrasts coordinate-load heterogeneity among available masks. |
| same_physical_graph_permuted_input_masks | observed_with_trained_outcomes | fixed_m20_input_masks | input_edge_load_gini | 0.500 | 3.440 | Fixed physical graph/count/d_rel; contrasts edge-load heterogeneity among available masks. |
| same_mask_count_different_coordinate_load | observed_with_trained_outcomes | fixed_m20_input_masks | input_coord_load_gini | 0.161 | -1.520 | Fixed physical graph/count/d_rel; contrasts coordinate-load heterogeneity among available masks. |
| same_physical_graph_permuted_input_masks | observed_with_trained_outcomes | fixed_m20_input_masks | input_edge_load_gini | 0.500 | 6.960 | Fixed physical graph/count/d_rel; contrasts edge-load heterogeneity among available masks. |
| same_mask_count_different_coordinate_load | observed_with_trained_outcomes | fixed_m20_input_masks | input_coord_load_gini | 0.135 | -16.480 | Fixed physical graph/count/d_rel; contrasts coordinate-load heterogeneity among available masks. |
| same_degree_sequence_different_normal_fan | missing_in_checked_in_data | n5_m12_N3_D2 | capacity_normal_fan_branch_tree_nmi_mean | NA | NA | Exact degree-sequence search found 0 groups with at least two topologies in hard_n5_m12_with_gamma_star_opt_capacity.csv; a degree-rewired library is still needed for this requested contrast. |

## Detailed Contrasts

### same_drel_different_bottleneck_participation

- Status: `observed_with_trained_outcomes`
- Regime: `n5_m12_N3_D2`
- Fixed fields: `n_nodes=5, n_edges=12, d_rel=88`
- Varied metric: `edge_participation_gini`

| side | id | family | varied metric | mean ICL | best ICL | extra |
| --- | --- | --- | ---: | ---: | ---: | --- |
| low | `g0040_cycle_chords_seed41` | `cycle_chords` | 0.080 | 83.240 | 91.200 | trees=78; root_gini=0.5076923076923077; group=cycle_chords_n5_m12_seed41 |
| high | `g0082_cycle_chords_seed83` | `cycle_chords` | 0.317 | 83.000 | 90.400 | trees=40; root_gini=0.27; group=cycle_chords_n5_m12_seed83 |

Delta mean ICL, high minus low: `-0.240` percentage points.

Interpretation: Largest available same-d_rel edge-participation contrast in the checked-in hard library; trained means are nearly identical, so this contrast does not support edge participation alone as a strong driver.

### same_tree_count_different_root_balance

- Status: `observed_with_trained_outcomes`
- Regime: `n4_m6_N3_D2`
- Fixed fields: `n_nodes=4, n_edges=6, n_trees_total_enum=8`
- Varied metric: `root_tree_count_gini`

| side | id | family | varied metric | mean ICL | best ICL | extra |
| --- | --- | --- | ---: | ---: | ---: | --- |
| low | `g0004_cycle_chords_seed5` | `cycle_chords` | 0.000 | 67.600 | 78.600 | d_rel=40; edge_gini=0.2222222222222221; group=cycle_chords_n4_m6_seed5 |
| high | `g0010_cycle_chords_seed12` | `cycle_chords` | 0.312 | 71.360 | 80.000 | d_rel=40; edge_gini=0.19444444444444442; group=cycle_chords_n4_m6_seed12 |

Delta mean ICL, high minus low: `3.760` percentage points.

Interpretation: Largest available same-total-tree-count root-balance contrast. Higher root imbalance has higher mean here, so root balance is not a monotone penalty in this tiny contrast.

### same_physical_graph_permuted_input_masks

- Status: `observed_with_trained_outcomes`
- Regime: `fixed_m20_input_masks`
- Fixed fields: `physical_topology=cycle_chords_n6_m20_seed3, n_nodes=6.0, n_edges=20.0, input_coupled_count=200.0, d_rel=200.0`
- Varied metric: `input_edge_load_gini`

| side | id | family | varied metric | mean ICL | best ICL | extra |
| --- | --- | --- | ---: | ---: | ---: | --- |
| low | `cycle_chords_n6_m20_seed3__mask0242_balanced_c200_seed3` | `balanced` | 0.000 | 80.440 | 88.600 | edge_load_gini=0.0; coord_load_gini=0.0; mask_family=balanced |
| high | `cycle_chords_n6_m20_seed3__mask0080_edge_block_c200_seed1` | `edge_block` | 0.500 | 74.120 | 80.400 | edge_load_gini=0.5; coord_load_gini=0.0; mask_family=edge_block |

Delta mean ICL, high minus low: `-6.320` percentage points.

Interpretation: Fixed physical graph/count/d_rel; contrasts edge-load heterogeneity among available masks.

### same_mask_count_different_coordinate_load

- Status: `observed_with_trained_outcomes`
- Regime: `fixed_m20_input_masks`
- Fixed fields: `physical_topology=cycle_chords_n6_m20_seed3, n_nodes=6.0, n_edges=20.0, input_coupled_count=200.0, d_rel=200.0`
- Varied metric: `input_coord_load_gini`

| side | id | family | varied metric | mean ICL | best ICL | extra |
| --- | --- | --- | ---: | ---: | ---: | --- |
| low | `cycle_chords_n6_m20_seed3__mask0080_edge_block_c200_seed1` | `edge_block` | 0.000 | 74.120 | 80.400 | edge_load_gini=0.5; coord_load_gini=0.0; mask_family=edge_block |
| high | `cycle_chords_n6_m20_seed3__mask0020_entry_random_c200_seed21` | `entry_random` | 0.161 | 76.240 | 86.600 | edge_load_gini=0.121; coord_load_gini=0.16149999999999998; mask_family=entry_random |

Delta mean ICL, high minus low: `2.120` percentage points.

Interpretation: Fixed physical graph/count/d_rel; contrasts coordinate-load heterogeneity among available masks.

### same_physical_graph_permuted_input_masks

- Status: `observed_with_trained_outcomes`
- Regime: `fixed_m20_input_masks`
- Fixed fields: `physical_topology=hub_spoke_n6_m20_seed63, n_nodes=6.0, n_edges=20.0, input_coupled_count=200.0, d_rel=200.0`
- Varied metric: `input_edge_load_gini`

| side | id | family | varied metric | mean ICL | best ICL | extra |
| --- | --- | --- | ---: | ---: | ---: | --- |
| low | `hub_spoke_n6_m20_seed63__mask0310_balanced_c200_seed71` | `balanced` | 0.000 | 66.720 | 82.600 | edge_load_gini=0.0; coord_load_gini=0.0; mask_family=balanced |
| high | `hub_spoke_n6_m20_seed63__mask0080_edge_block_c200_seed1` | `edge_block` | 0.500 | 70.160 | 81.800 | edge_load_gini=0.5; coord_load_gini=0.0; mask_family=edge_block |

Delta mean ICL, high minus low: `3.440` percentage points.

Interpretation: Fixed physical graph/count/d_rel; contrasts edge-load heterogeneity among available masks.

### same_mask_count_different_coordinate_load

- Status: `observed_with_trained_outcomes`
- Regime: `fixed_m20_input_masks`
- Fixed fields: `physical_topology=hub_spoke_n6_m20_seed63, n_nodes=6.0, n_edges=20.0, input_coupled_count=200.0, d_rel=200.0`
- Varied metric: `input_coord_load_gini`

| side | id | family | varied metric | mean ICL | best ICL | extra |
| --- | --- | --- | ---: | ---: | ---: | --- |
| low | `hub_spoke_n6_m20_seed63__mask0080_edge_block_c200_seed1` | `edge_block` | 0.000 | 70.160 | 81.800 | edge_load_gini=0.5; coord_load_gini=0.0; mask_family=edge_block |
| high | `hub_spoke_n6_m20_seed63__mask0020_entry_random_c200_seed21` | `entry_random` | 0.161 | 68.640 | 79.800 | edge_load_gini=0.121; coord_load_gini=0.16149999999999998; mask_family=entry_random |

Delta mean ICL, high minus low: `-1.520` percentage points.

Interpretation: Fixed physical graph/count/d_rel; contrasts coordinate-load heterogeneity among available masks.

### same_physical_graph_permuted_input_masks

- Status: `observed_with_trained_outcomes`
- Regime: `fixed_m20_input_masks`
- Fixed fields: `physical_topology=random_sc_n6_m20_seed3, n_nodes=6.0, n_edges=20.0, input_coupled_count=200.0, d_rel=200.0`
- Varied metric: `input_edge_load_gini`

| side | id | family | varied metric | mean ICL | best ICL | extra |
| --- | --- | --- | ---: | ---: | ---: | --- |
| low | `random_sc_n6_m20_seed3__mask0291_balanced_c200_seed52` | `balanced` | 0.000 | 79.920 | 94.600 | edge_load_gini=0.0; coord_load_gini=0.0; mask_family=balanced |
| high | `random_sc_n6_m20_seed3__mask0080_edge_block_c200_seed1` | `edge_block` | 0.500 | 86.880 | 93.400 | edge_load_gini=0.5; coord_load_gini=0.0; mask_family=edge_block |

Delta mean ICL, high minus low: `6.960` percentage points.

Interpretation: Fixed physical graph/count/d_rel; contrasts edge-load heterogeneity among available masks.

### same_mask_count_different_coordinate_load

- Status: `observed_with_trained_outcomes`
- Regime: `fixed_m20_input_masks`
- Fixed fields: `physical_topology=random_sc_n6_m20_seed3, n_nodes=6.0, n_edges=20.0, input_coupled_count=200.0, d_rel=200.0`
- Varied metric: `input_coord_load_gini`

| side | id | family | varied metric | mean ICL | best ICL | extra |
| --- | --- | --- | ---: | ---: | ---: | --- |
| low | `random_sc_n6_m20_seed3__mask0080_edge_block_c200_seed1` | `edge_block` | 0.000 | 86.880 | 93.400 | edge_load_gini=0.5; coord_load_gini=0.0; mask_family=edge_block |
| high | `random_sc_n6_m20_seed3__mask0000_entry_random_c200_seed1` | `entry_random` | 0.135 | 70.400 | 85.200 | edge_load_gini=0.09050000000000002; coord_load_gini=0.135; mask_family=entry_random |

Delta mean ICL, high minus low: `-16.480` percentage points.

Interpretation: Fixed physical graph/count/d_rel; contrasts coordinate-load heterogeneity among available masks.

### same_degree_sequence_different_normal_fan

- Status: `missing_in_checked_in_data`
- Regime: `n5_m12_N3_D2`
- Fixed fields: `n_nodes,n_edges,d_rel,exact in/out degree sequence`
- Varied metric: `capacity_normal_fan_branch_tree_nmi_mean`

Interpretation: Exact degree-sequence search found 0 groups with at least two topologies in hard_n5_m12_with_gamma_star_opt_capacity.csv; a degree-rewired library is still needed for this requested contrast.

## Takeaway

- The existing data now supports trained-outcome checks for same-`d_rel`/different-participation, same-tree-count/different-root-balance, same-physical-graph mask variation, and same-mask-count coordinate-load variation.
- The exact same-degree-sequence/different-normal-fan contrast remains uncovered in checked-in data; it needs a degree-rewired graph library or targeted sweep.
- The contrast effects are not monotone. This supports the broader conclusion that no single coarse topology scalar is enough; branch/tree/projection capacity and trained alignment remain the sharper targets.
