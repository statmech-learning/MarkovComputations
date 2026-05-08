# Tree-Multiplicity Causal Mask Library

This library is built from the already-trained fixed-m20 mask groups. It keeps physical graph identity explicit and uses normalized tree/difference overlap metrics; raw overlap counts are not used as standalone selectors.

## Controls

- Mask/topology groups: `48`
- Seed runs summarized inside groups: `240`
- Physical graphs: `cycle_chords_n6_m20_seed3, hub_spoke_n6_m20_seed63, random_sc_n6_m20_seed3`
- Input-coupled parameter count: `[200]`
- Aggregate edge-level `M_mean`: `[10.0]`
- `d_rel` counts: `{'190.0': 3, '200.0': 45}`

## Category Summary

High/low labels are assigned within physical graph and coordinate-load stratum, so the imbalanced high-overlap stratum should be read as high among imbalanced masks rather than globally as high as edge-block masks.

| category | groups | physical graphs | mean min diff overlap | mean min tree overlap | mean coord gini | mean novel ICL |
| --- | --- | --- | --- | --- | --- | --- |
| high_tree_diff_overlap_balanced_coordinate_load | 18 | cycle_chords_n6_m20_seed3, hub_spoke_n6_m20_seed63, random_sc_n6_m20_seed3 | 0.963 | 0.955 | 0.000 | 75.773 |
| high_tree_diff_overlap_imbalanced_coordinate_load | 8 | cycle_chords_n6_m20_seed3, hub_spoke_n6_m20_seed63, random_sc_n6_m20_seed3 | 0.812 | 0.696 | 0.130 | 72.775 |
| low_tree_diff_overlap_balanced_aggregate_multiplicity | 16 | cycle_chords_n6_m20_seed3, hub_spoke_n6_m20_seed63, random_sc_n6_m20_seed3 | 0.898 | 0.906 | 0.001 | 73.913 |
| low_tree_diff_overlap_high_coordinate_load_imbalance | 6 | cycle_chords_n6_m20_seed3, hub_spoke_n6_m20_seed63, random_sc_n6_m20_seed3 | 0.350 | 0.281 | 0.316 | 73.233 |

## Selected Mask Groups

| category | physical graph | mask family | mask group | min diff overlap | coord gini | mean novel ICL |
| --- | --- | --- | --- | --- | --- | --- |
| high_tree_diff_overlap_balanced_coordinate_load | cycle_chords_n6_m20_seed3 | edge_block | cycle_chords_n6_m20_seed3__mask0107_edge_block_c200_seed28 | 0.983 | 0.000 | 79.080 |
| high_tree_diff_overlap_balanced_coordinate_load | cycle_chords_n6_m20_seed3 | edge_block | cycle_chords_n6_m20_seed3__mask0080_edge_block_c200_seed1 | 0.977 | 0.000 | 74.120 |
| high_tree_diff_overlap_balanced_coordinate_load | hub_spoke_n6_m20_seed63 | edge_block | hub_spoke_n6_m20_seed63__mask0080_edge_block_c200_seed1 | 0.975 | 0.000 | 70.160 |
| high_tree_diff_overlap_balanced_coordinate_load | hub_spoke_n6_m20_seed63 | edge_block | hub_spoke_n6_m20_seed63__mask0097_edge_block_c200_seed18 | 0.959 | 0.000 | 77.080 |
| high_tree_diff_overlap_balanced_coordinate_load | random_sc_n6_m20_seed3 | edge_block | random_sc_n6_m20_seed3__mask_random_sc_n6_m20_seed3__mask0098_edge_block_c200_seed19 | 0.981 | 0.000 | 79.240 |
| high_tree_diff_overlap_balanced_coordinate_load | random_sc_n6_m20_seed3 | edge_block | random_sc_n6_m20_seed3__mask_random_sc_n6_m20_seed3__mask0096_edge_block_c200_seed17 | 0.980 | 0.000 | 73.760 |
| high_tree_diff_overlap_imbalanced_coordinate_load | cycle_chords_n6_m20_seed3 | entry_random | cycle_chords_n6_m20_seed3__mask0047_entry_random_c200_seed48 | 0.884 | 0.105 | 69.440 |
| high_tree_diff_overlap_imbalanced_coordinate_load | cycle_chords_n6_m20_seed3 | entry_random | cycle_chords_n6_m20_seed3__mask0004_entry_random_c200_seed5 | 0.877 | 0.093 | 74.200 |
| high_tree_diff_overlap_imbalanced_coordinate_load | hub_spoke_n6_m20_seed63 | entry_random | hub_spoke_n6_m20_seed63__mask0044_entry_random_c200_seed45 | 0.766 | 0.122 | 63.760 |
| high_tree_diff_overlap_imbalanced_coordinate_load | hub_spoke_n6_m20_seed63 | entry_random | hub_spoke_n6_m20_seed63__mask0020_entry_random_c200_seed21 | 0.642 | 0.161 | 68.640 |
| high_tree_diff_overlap_imbalanced_coordinate_load | random_sc_n6_m20_seed3 | entry_random | random_sc_n6_m20_seed3__mask_random_sc_n6_m20_seed3__mask0031_entry_random_c200_seed32 | 0.870 | 0.132 | 78.720 |
| high_tree_diff_overlap_imbalanced_coordinate_load | random_sc_n6_m20_seed3 | entry_random | random_sc_n6_m20_seed3__mask_random_sc_n6_m20_seed3__mask0043_entry_random_c200_seed44 | 0.829 | 0.128 | 80.800 |
| low_tree_diff_overlap_balanced_aggregate_multiplicity | cycle_chords_n6_m20_seed3 | balanced | cycle_chords_n6_m20_seed3__mask0294_balanced_c200_seed55 | 0.916 | 0.000 | 76.040 |
| low_tree_diff_overlap_balanced_aggregate_multiplicity | cycle_chords_n6_m20_seed3 | balanced | cycle_chords_n6_m20_seed3__mask0242_balanced_c200_seed3 | 0.918 | 0.000 | 80.440 |
| low_tree_diff_overlap_balanced_aggregate_multiplicity | hub_spoke_n6_m20_seed63 | balanced | hub_spoke_n6_m20_seed63__mask0306_balanced_c200_seed67 | 0.805 | 0.000 | 63.320 |
| low_tree_diff_overlap_balanced_aggregate_multiplicity | hub_spoke_n6_m20_seed63 | balanced | hub_spoke_n6_m20_seed63__mask0310_balanced_c200_seed71 | 0.818 | 0.000 | 66.720 |
| low_tree_diff_overlap_balanced_aggregate_multiplicity | random_sc_n6_m20_seed3 | balanced | random_sc_n6_m20_seed3__mask_random_sc_n6_m20_seed3__mask0291_balanced_c200_seed52 | 0.912 | 0.000 | 79.920 |
| low_tree_diff_overlap_balanced_aggregate_multiplicity | random_sc_n6_m20_seed3 | balanced | random_sc_n6_m20_seed3__mask_random_sc_n6_m20_seed3__mask0298_balanced_c200_seed59 | 0.932 | 0.000 | 76.960 |
| low_tree_diff_overlap_high_coordinate_load_imbalance | cycle_chords_n6_m20_seed3 | coord_block | cycle_chords_n6_m20_seed3__mask0160_coord_block_c200_seed1 | 0.000 | 0.500 | 68.120 |
| low_tree_diff_overlap_high_coordinate_load_imbalance | cycle_chords_n6_m20_seed3 | entry_random | cycle_chords_n6_m20_seed3__mask0000_entry_random_c200_seed1 | 0.683 | 0.135 | 80.600 |
| low_tree_diff_overlap_high_coordinate_load_imbalance | hub_spoke_n6_m20_seed63 | coord_block | hub_spoke_n6_m20_seed63__mask0160_coord_block_c200_seed1 | 0.000 | 0.500 | 57.840 |
| low_tree_diff_overlap_high_coordinate_load_imbalance | hub_spoke_n6_m20_seed63 | entry_random | hub_spoke_n6_m20_seed63__mask0000_entry_random_c200_seed1 | 0.641 | 0.135 | 79.320 |
| low_tree_diff_overlap_high_coordinate_load_imbalance | random_sc_n6_m20_seed3 | coord_block | random_sc_n6_m20_seed3__mask_random_sc_n6_m20_seed3__mask0160_coord_block_c200_seed1 | 0.000 | 0.500 | 64.200 |
| low_tree_diff_overlap_high_coordinate_load_imbalance | random_sc_n6_m20_seed3 | entry_random | random_sc_n6_m20_seed3__mask_random_sc_n6_m20_seed3__mask0034_entry_random_c200_seed35 | 0.775 | 0.123 | 89.320 |
