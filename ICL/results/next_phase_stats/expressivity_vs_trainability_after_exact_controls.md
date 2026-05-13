# Expressivity vs Trainability After Exact Controls

## Best Models By Outcome

| dataset | target | best model | LOO R2 |
| --- | --- | --- | --- |
| fixed_m20_retrospective | mean | same_root_tree_diff | 0.412 |
| fixed_m20_retrospective | best | same_root_tree_diff | 0.405 |
| fixed_m20_retrospective | seed_std | cross_root_oriented | -0.069 |
| prospective_tree_diff_exact_control | mean | cross_root_overlap | 0.483 |
| prospective_tree_diff_exact_control | best | cross_root_overlap | 0.773 |
| prospective_tree_diff_exact_control | seed_std | same_plus_cross_minimal | 0.394 |
| exact_degree_normal_fan | mean | cross_root_oriented | 0.087 |
| exact_degree_normal_fan | best | cross_root_oriented | 0.090 |
| exact_degree_normal_fan | seed_std | cross_root_overlap | -0.066 |

## Interpretation

- best_seed: best-seed ICL remains the closest available expressivity envelope
- mean_seed: mean-seed ICL mixes expressivity and training reliability
- seed_std: current structural metrics generally do not explain seed variance well
