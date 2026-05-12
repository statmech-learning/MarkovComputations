# Expressivity vs Trainability Exact-Control Report

## Scope

This report separates best-seed ICL as an expressivity upper envelope from mean-seed ICL and seed-standard deviation as trainability and reliability diagnostics. It uses only the repaired-gamma existing-data reanalysis, the prospective tree-difference exact-control experiment, and the exact-degree normal-fan expansion.

## Repaired Gamma Existing-Data Result

In the fixed-m20 mask library, the best repaired no-bias gamma model for mean ICL was `gamma_no_bias_plus_tree_difference_multiplicity` with grouped LOO `R2 = 0.078`. The tree-difference multiplicity model remained stronger with grouped LOO `R2 = 0.435`. The selector gate is `not_cleared_for_large_sweeps`.

## Prospective Tree-Difference Control

| outcome | controls | tree-diff + controls | gamma + controls |
| --- | --- | --- | --- |
| mean seed ICL | 0.488 | 0.447 | 0.320 |
| best seed ICL | 0.601 | 0.545 | 0.571 |

The prospective exact-control result does not show a clean expressivity/trainability split in favor of tree-difference overlap or repaired gamma. Controls-only already explain more than the added tree-difference and gamma models for mean ICL; tree-level overlap, not tree-difference overlap, was the only small best-seed improvement in that report.

## Exact-Degree Normal-Fan Expansion

| diagnostic | model | grouped LOO R2 |
| --- | --- | --- |
| mean seed ICL best model | tree_count | 0.114 |
| best seed ICL best model | gamma_plus_normal_fan | 0.117 |
| seed std best model | tree_count | -0.069 |
| gamma exact, mean ICL | gamma_no_bias_exact | -0.127 |
| gamma exact, best ICL | gamma_no_bias_exact | -0.072 |
| gamma + normal fan, best ICL | gamma_plus_normal_fan | 0.117 |

The exact-degree normal-fan expansion gives a weak signal for active-tree/tree-count geometry on both mean and best seed ICL. Repaired gamma alone is not predictive here, and adding gamma to normal-fan features only reaches the weak normal-fan range.

## Interpretation

- Expressivity: weak evidence that normal-fan active-tree/tree-count variables predict the best-seed upper envelope under exact controls.
- Trainability: no clean metric yet explains seed variance; the tested seed-std LOO models are negative or small.
- Gamma: repaired `gamma_no_bias` passed analytic toys but is not yet useful as a selector in these existing-data or exact-control analyses.
