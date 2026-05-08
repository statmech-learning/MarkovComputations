# Mechanism-Isolating Topology Sweep Plan

This plan selects fixed-count contrasts designed to isolate mechanisms rather than merely add more data.

| contrast | status | fixed fields | varied metric | delta | low item | high item |
| --- | --- | --- | --- | ---: | --- | --- |
| same_drel_different_bottleneck_participation | ready | regime, n_nodes, n_edges, d_rel | edge_participation_gini | 0.2365384615384618 | cycle_chords | cycle_chords |
| same_tree_count_different_root_balance | ready | regime, n_nodes, n_edges, n_trees_total_enum | root_tree_count_gini | 0.3125 | cycle_chords | cycle_chords |
| same_degree_sequence_different_normal_fan | unavailable | regime, n_nodes, n_edges, in_degree_cv, out_degree_cv, d_rel |  |  |  |  |
| same_physical_graph_permuted_input_masks | unavailable | regime, physical_topology_name, edge_json |  |  |  |  |
| same_mask_count_different_coordinate_load | unavailable | regime, n_nodes, n_edges, input_coupled_parameter_count |  |  |  |  |

## Details

### same_drel_different_bottleneck_participation

Hold relative tree rank fixed while separating bottleneck/edge-participation heterogeneity.

- Fixed key: `['n5_m12_N3_D2', '5.0', '12.0', '88.0']`
- Contrast metric: `edge_participation_gini`
- Low item: `cycle_chords` (cycle_chords) = `0.08012820512820507`
- High item: `cycle_chords` (cycle_chords) = `0.3166666666666669`
- Delta: `0.2365384615384618`

### same_tree_count_different_root_balance

Hold total rooted-tree count fixed while separating tree-count imbalance across roots.

- Fixed key: `['n4_m6_N3_D2', '4.0', '6.0', '8.0']`
- Contrast metric: `root_tree_count_gini`
- Low item: `cycle_chords` (cycle_chords) = `0.0`
- High item: `cycle_chords` (cycle_chords) = `0.3125`
- Delta: `0.3125`

### same_degree_sequence_different_normal_fan

Approximately hold degree sequence/rank fixed while separating sampled tree-polytope normal-fan organization.

Status: `unavailable`. No group contained at least two rows with the required fixed columns and contrast metric.

### same_physical_graph_permuted_input_masks

Hold the physical graph fixed while separating input-coordinate load heterogeneity across masks.

Status: `unavailable`. No group contained at least two rows with the required fixed columns and contrast metric.

### same_mask_count_different_coordinate_load

Hold mask count fixed while separating coordinate-load and edge-load heterogeneity.

Status: `unavailable`. No group contained at least two rows with the required fixed columns and contrast metric.
