# Multibase Exact-Control Results

Run rows: `185`. Topology groups: `37`.

## Grouped LOO R2

| outcome | model | n | LOO R2 |
| --- | --- | --- | --- |
| mean_novel_icl | controls_base_only | 37 | 0.431 |
| mean_novel_icl | tree_count_plus_base | 37 | 0.561 |
| mean_novel_icl | normal_fan_plus_base | 37 | 0.518 |
| mean_novel_icl | tree_count_plus_normal_fan_plus_base | 37 | 0.499 |
| mean_novel_icl | cross_root_plus_tree_count_normal_fan_plus_base | 37 | 0.446 |
| mean_novel_icl | gamma_exact_plus_base | 37 | 0.375 |
| mean_novel_icl | gamma_plus_tree_count_normal_fan_plus_base | 37 | 0.470 |
| best_seed_novel_icl | controls_base_only | 37 | 0.793 |
| best_seed_novel_icl | tree_count_plus_base | 37 | 0.834 |
| best_seed_novel_icl | normal_fan_plus_base | 37 | 0.796 |
| best_seed_novel_icl | tree_count_plus_normal_fan_plus_base | 37 | 0.788 |
| best_seed_novel_icl | cross_root_plus_tree_count_normal_fan_plus_base | 37 | 0.787 |
| best_seed_novel_icl | gamma_exact_plus_base | 37 | 0.797 |
| best_seed_novel_icl | gamma_plus_tree_count_normal_fan_plus_base | 37 | 0.806 |
| seed_std_novel_icl | controls_base_only | 37 | -0.148 |
| seed_std_novel_icl | tree_count_plus_base | 37 | -0.168 |
| seed_std_novel_icl | normal_fan_plus_base | 37 | -0.102 |
| seed_std_novel_icl | tree_count_plus_normal_fan_plus_base | 37 | -0.152 |
| seed_std_novel_icl | cross_root_plus_tree_count_normal_fan_plus_base | 37 | -0.248 |
| seed_std_novel_icl | gamma_exact_plus_base | 37 | -0.273 |
| seed_std_novel_icl | gamma_plus_tree_count_normal_fan_plus_base | 37 | -0.359 |

## Paired Arm Contrasts

### arm_A_fixed_tree_count_variable_normal_fan

| contrast | n | mean | 95% CI |
| --- | --- | --- | --- |
| mean novel ICL high-low | 11 | 1.193 | [-1.2910909090909106, 4.0074545454545385] |
| best seed ICL high-low | 11 | 0.764 | [-0.0909090909090922, 1.5636363636363626] |
| seed std high-low | 11 | 0.073 | [-3.314914918132876, 3.327512237764908] |

### arm_B_variable_tree_count_matched_normal_fan

| contrast | n | mean | 95% CI |
| --- | --- | --- | --- |
| mean novel ICL high-low | 9 | 4.067 | [0.8755555555555566, 7.364444444444435] |
| best seed ICL high-low | 9 | 0.556 | [-0.533333333333333, 1.4888888888888863] |
| seed std high-low | 9 | -1.737 | [-5.028699919746365, 1.716500486486013] |

Interpretation should follow the paired arms: Arm A isolates normal-fan variation at nearly fixed tree count; Arm B isolates tree-count variation at matched normal-fan score.
