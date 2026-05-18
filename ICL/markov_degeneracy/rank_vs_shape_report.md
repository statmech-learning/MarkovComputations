# Direction C — rank/coverage vs load-shape

**The two are not separable predictors — they are one quantity.**

Rank/coverage and load-shape are not separable predictors -- on density-0.5 masks they are one quantity (the decorrelation search found zero overlap). That single coverage quantity sets the accuracy ceiling: effective_rank_D_masked vs ceiling Spearman r=+0.555. The prior program's regression horse-race between 'rank', 'tree-diff' and 'load' metrics was comparing one quantity with itself. Graph-shape (vs coverage) is untestable on this grid because the graph is fixed; that needs a multi-graph follow-up.

- 16 input masks, one fixed graph, retrained to convergence (5 seeds each).
- Decorrelation search: **INSEPARABLE** — 8000 density-0.5 masks, coord-load gini and effective rank could not be decorrelated.

## Part 1 — the candidate predictors are collinear

Spearman |r| among the varying mask metrics: min 0.701, median 0.868, max 1.000. They are all proxies of one underlying property — how evenly the mask spreads input coupling — so a regression that pits them against each other has no identifiability.

| metric pair | Spearman r |
|---|---|
| effective_rank_D_masked ~ condition_number_D_masked | -0.806 |
| effective_rank_D_masked ~ comparison_branch_common_d_rel_mean | +0.711 |
| effective_rank_D_masked ~ comparison_branch_d_rel_mean | +0.868 |
| effective_rank_D_masked ~ input_coord_load_gini | -0.868 |
| condition_number_D_masked ~ comparison_branch_common_d_rel_mean | -0.701 |
| condition_number_D_masked ~ comparison_branch_d_rel_mean | -0.868 |
| condition_number_D_masked ~ input_coord_load_gini | +0.868 |
| comparison_branch_common_d_rel_mean ~ comparison_branch_d_rel_mean | +0.871 |
| comparison_branch_common_d_rel_mean ~ input_coord_load_gini | -0.871 |
| comparison_branch_d_rel_mean ~ input_coord_load_gini | -1.000 |

## Part 2 — accuracy ceiling vs the coverage quantity

| mask metric | vs ceiling (Spearman) | vs mean acc |
|---|---|---|
| effective_rank_D_masked | +0.555 (p=0.026) | +0.609 |
| condition_number_D_masked | -0.686 (p=0.003) | -0.659 |
| comparison_branch_common_d_rel_mean | +0.695 (p=0.003) | +0.701 |
| comparison_branch_d_rel_mean | +0.801 (p=0.000) | +0.759 |
| input_coord_load_gini | -0.801 (p=0.000) | -0.759 |

## Within-stratum check

Is there an effective-rank signal *beyond* the binary balanced/imbalanced split?

| stratum | n | eff_rank vs ceiling (Spearman) | ceiling range |
|---|---|---|---|
| balanced | 8 | -0.455 (p=0.257) | 93.9–98.5% |
| imbalanced | 8 | -0.500 (p=0.207) | 85.6–94.7% |

## Per-mask

| mask | stratum | acc mean | ceiling | eff_rank_D_masked | coord_gini |
|---|---|---|---|---|---|
| prospective_balanced_load_0114_c200_seed60732 | balanced | 93.8 | 98.5 | 181.56 | 0.000 |
| prospective_balanced_load_0024_c200_seed60642 | balanced | 95.1 | 98.2 | 181.49 | 0.000 |
| prospective_balanced_load_0135_c200_seed60753 | balanced | 92.0 | 98.1 | 182.42 | 0.000 |
| prospective_balanced_load_0104_c200_seed60722 | balanced | 82.4 | 97.5 | 181.43 | 0.000 |
| prospective_balanced_load_0047_c200_seed60665 | balanced | 93.9 | 96.6 | 182.56 | 0.000 |
| prospective_balanced_load_0066_c200_seed60684 | balanced | 92.5 | 96.6 | 181.06 | 0.000 |
| prospective_balanced_load_0008_c200_seed60626 | balanced | 90.1 | 96.2 | 182.75 | 0.000 |
| prospective_imbalanced_coord_load_0252_c200_seed60710 | imbalanced | 83.5 | 94.7 | 175.14 | 0.250 |
| prospective_imbalanced_coord_load_0206_c200_seed60664 | imbalanced | 83.8 | 94.0 | 175.49 | 0.250 |
| prospective_balanced_load_0035_c200_seed60653 | balanced | 88.2 | 93.9 | 182.51 | 0.000 |
| prospective_imbalanced_coord_load_0261_c200_seed60719 | imbalanced | 84.3 | 93.9 | 175.91 | 0.250 |
| prospective_imbalanced_coord_load_0218_c200_seed60676 | imbalanced | 82.9 | 93.4 | 175.23 | 0.250 |
| prospective_imbalanced_coord_load_0193_c200_seed60651 | imbalanced | 72.5 | 91.0 | 177.10 | 0.250 |
| prospective_imbalanced_coord_load_0279_c200_seed60737 | imbalanced | 79.5 | 90.3 | 176.29 | 0.250 |
| prospective_imbalanced_coord_load_0207_c200_seed60665 | imbalanced | 78.7 | 89.0 | 175.37 | 0.250 |
| prospective_imbalanced_coord_load_0220_c200_seed60678 | imbalanced | 79.5 | 85.6 | 176.03 | 0.250 |
