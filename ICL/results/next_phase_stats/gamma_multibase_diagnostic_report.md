# Gamma Multibase Diagnostic Report

Status: `computed`.

Gamma remains a diagnostic unless it improves held-out exact-control prediction beyond tree count, normal fan, and base controls.

| outcome | model | n | LOO R2 |
| --- | --- | --- | --- |
| mean_novel_icl | gamma_exact_plus_base | 37 | 0.375 |
| mean_novel_icl | gamma_plus_tree_count_normal_fan_plus_base | 37 | 0.470 |
| best_seed_novel_icl | gamma_exact_plus_base | 37 | 0.797 |
| best_seed_novel_icl | gamma_plus_tree_count_normal_fan_plus_base | 37 | 0.806 |
| seed_std_novel_icl | gamma_exact_plus_base | 37 | -0.273 |
| seed_std_novel_icl | gamma_plus_tree_count_normal_fan_plus_base | 37 | -0.359 |
