# Cross-Root Decoder-Aware Contrast Reanalysis

Full input coupling makes binary cross-root overlap mostly saturated; rank/effective-rank and decoder-agnostic root-pair summaries are the varying cross-root diagnostics.

## Cross-Root Model Rows

| outcome | model | n | LOO R2 |
| --- | --- | --- | --- |
| mean_novel_icl | cross_root_plus_tree_count_normal_fan_plus_base | 37 | 0.446 |
| best_seed_novel_icl | cross_root_plus_tree_count_normal_fan_plus_base | 37 | 0.787 |
| seed_std_novel_icl | cross_root_plus_tree_count_normal_fan_plus_base | 37 | -0.248 |

These metrics are pre-training diagnostics unless learned decoder or posterior weights are explicitly used.
