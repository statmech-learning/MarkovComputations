# Normal-Fan / Tree-Count Separation Library

## Status

The existing one-base exact-degree normal-fan library was reanalyzed. It is useful as a diagnostic library but does not cleanly instantiate the requested fixed-tree-count and matched-normal-fan arms.

## Controls

- N_n=5
- m=12
- N_c=3
- D=2
- exact in-degree sequence
- exact out-degree sequence
- full input coupling
- d_rel=88

## Arm Availability

| arm | status |
| --- | --- |
| arm_A_fixed_tree_count_variable_normal_fan | not available cleanly in current one-base library because active-tree count and log rooted-tree count are highly correlated |
| arm_B_variable_tree_count_matched_normal_fan | not available cleanly in current one-base library for the same collinearity reason |
| arm_C_multi_base_rewire_libraries | required next; current library uses one base degree sequence |

## Feature Collinearity

| feature x | feature y | n | r |
| --- | --- | --- | --- |
| library_n_trees_total_enum_log | library_root_tree_count_gini | 32 | NA |
| capacity_normal_fan_active_tree_count_mean | library_n_trees_total_enum_log | 32 | 0.974 |
| capacity_normal_fan_active_tree_count_mean | library_root_tree_count_gini | 32 | NA |
| capacity_normal_fan_active_tree_count_mean | capacity_normal_fan_branch_tree_nmi_mean | 32 | 0.792 |
| capacity_normal_fan_active_tree_count_mean | capacity_normal_fan_branch_active_tree_count_min_mean | 32 | 0.947 |
| capacity_normal_fan_active_tree_count_mean | cross_overlap_norm_min | 32 | NA |
| capacity_normal_fan_active_tree_count_mean | cross_contrast_effective_rank_mean | 32 | 0.950 |
| capacity_normal_fan_branch_tree_nmi_mean | library_n_trees_total_enum_log | 32 | 0.789 |
| capacity_normal_fan_branch_active_tree_count_min_mean | library_n_trees_total_enum_log | 32 | 0.939 |
| cross_overlap_norm_min | library_n_trees_total_enum_log | 32 | NA |
| cross_contrast_effective_rank_mean | library_n_trees_total_enum_log | 32 | 0.971 |
