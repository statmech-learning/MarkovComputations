# Tree-Difference Failure Diagnosis

## Direct Answer

The prospective contrast did not reproduce the fixed-m20 signal because same-root co-participation is too coarse once graph, count, d_rel, and load structure are fixed.  The prospective high/low masks were genuinely separated in same-root overlap, and high masks also usually had higher cross-root minimum overlap, but they did not improve best seed, mean seed, branch failures, or trained margins.  Balanced low masks were not in the zero-overlap regime that helped drive the retrospective family signal, and imbalanced masks tested lower overlap only under a trainability/load confound.  The failure is therefore not explained by a single saturation story; it points to missing orientation, controllability, root-pair choice, and optimization variables.

## Prospective High-Low Contrasts

| load | outcome | high mean | low mean | high-low |
| --- | --- | --- | --- | --- |
| balanced_load | outcome_mean | 75.360 | 79.330 | -3.970 |
| balanced_load | outcome_best | 84.900 | 89.700 | -4.800 |
| balanced_load | branch_failures | 53.994 | 42.391 | 11.604 |
| balanced_load | trained_branch_margin | 0.204 | 0.839 | -0.635 |
| imbalanced_coord_load | outcome_mean | 68.830 | 69.900 | -1.070 |
| imbalanced_coord_load | outcome_best | 74.450 | 76.250 | -1.800 |
| imbalanced_coord_load | branch_failures | 67.475 | 64.300 | 3.175 |
| imbalanced_coord_load | trained_branch_margin | -0.316 | -0.216 | -0.099 |

## Range / Saturation Diagnostic

| quantity | value |
| --- | --- |
| fixed median diff overlap | 0.921 |
| fixed p75 diff overlap | 0.961 |
| balanced low prospective mean | 0.858 |
| imbalanced low prospective mean | 0.309 |
| balanced low above fixed median | 0.000 |
| balanced low above fixed p75 | 0.000 |

## Fixed-m20 Within-Graph Correlations

| physical graph | n | r(diff min, mean ICL) |
| --- | --- | --- |
| cycle_chords_n6_m20_seed3 | 16 | 0.485 |
| hub_spoke_n6_m20_seed63 | 16 | 0.555 |
| random_sc_n6_m20_seed3 | 16 | 0.445 |

## Viable Explanations

- the balanced prospective library did not test the low/zero-overlap regime present in retrospective coord-block masks
- co-participation without sign/orientation or controllability is too coarse
- one-graph prospective evidence is insufficient even though fixed-m20 within-graph correlations were positive
- trainability and post-training organization remain important because branch/projection scrambles cause large drops

## Weakened Explanations

- same-root tree-difference overlap is a standalone causal knob
- repaired gamma can be used as a topology selector
