# Post-Multibase Exact-Control Synthesis

## Exact Theory

First-order CRNs compute through rooted tree-sum projections by the matrix-tree theorem; this remains exact.

## Pre-Training Prediction

Best grouped-LOO mean-ICL model: `tree_count_plus_base` with R2 `0.561`.
Best grouped-LOO best-seed model: `tree_count_plus_base` with R2 `0.834`.

## Expressivity

Best-seed ICL is reported separately as the expressivity envelope; do not collapse it with mean-seed trainability.

## Trainability

Seed standard deviation is reported separately; weak seed-std prediction means trainability is not reduced to the same scalar as expressivity.

## Post-Training Mechanism

Mechanism status: `computed`.

## Causal Interventions

Same-root tree-difference overlap failed as a standalone prospective causal knob. Multibase mechanism scrambles are only supported if the collected scramble rows are nonempty.

## Cross-Root Contrast

Full input coupling makes binary cross-root overlap mostly saturated; rank/effective-rank and decoder-agnostic root-pair summaries are the varying cross-root diagnostics.

## Gamma

Gamma remains a diagnostic unless it improves held-out exact-control prediction beyond tree count, normal fan, and base controls.

## Tree-Difference Failure Diagnosis

Same-root tree-difference overlap should not be used as a standalone selector. It remains a secondary diagnostic and should be modified toward cross-root/decoder-aware contrast geometry where feasible.

## Thermodynamics

Thermodynamics remains untested; no F_max or entropy-production claim is supported.

## Bottom Line

The multibase library tests whether rooted-tree abundance, normal-fan geometry, or cross-root contrast rank improves pre-training prediction under exact controls. Claims should be based on grouped LOO, paired arms, and held-out-base behavior rather than seed-level rows.
