# Post Exact-Control Failure Diagnosis Synthesis

## Direct Answers

| question | answer |
| --- | --- |
| why_tree_difference_failed | The prospective contrast did not reproduce the fixed-m20 signal because same-root co-participation is too coarse once graph, count, d_rel, and load structure are fixed.  The prospective high/low masks were genuinely separated in same-root overlap, and high masks also usually had higher cross-root minimum overlap, but they did not improve best seed, mean seed, branch failures, or trained margins.  Balanced low masks were not in the zero-overlap regime that helped drive the retrospective family signal, and imbalanced masks tested lower overlap only under a trainability/load confound.  The failure is therefore not explained by a single saturation story; it points to missing orientation, controllability, root-pair choice, and optimization variables. |
| same_root_saturated_confounded_or_wrong_object | Not a pure saturation story.  The balanced prospective arm missed the zero-overlap regime but still had separated high/low masks, and the imbalanced arm reached lower overlap only under a load/trainability confound.  Same-root co-participation is therefore too coarse as a standalone knob. |
| cross_root_improvement | Cross-root overlap improved the prospective load-stratum-controlled mean-ICL LOO R2 from 0.420 for same-root tree-difference to 0.483, but high-overlap masks still lost the direct high-low causal contrast.  Treat cross-root metrics as diagnostics, not selectors. |
| tree_count_vs_normal_fan | Current one-base exact-degree data cannot separate total rooted-tree abundance from active-tree/normal-fan geometry. log rooted-tree count and active-tree count are strongly correlated, so their weak positive predictive signals should be treated as a combined geometry/abundance direction. |
| best_pretraining_metric | current weak direction: tree_count_only for mean ICL, but only within the one-base exact-degree data |
| gamma_status | Gamma remains a sanity-checked diagnostic. It passed analytic toys, but exact-control trained data do not support using it as a topology selector. |
| next_experiment | Build multi-base degree-preserving libraries with arms that hold tree count approximately fixed while varying normal-fan coverage and vice versa. |
| thermodynamics | No thermodynamic Fmax claim was tested in this phase; it remains untested. |

## Claim Separation

| claim type | status |
| --- | --- |
| expressivity | First-order tree-sum expressivity remains exact; repaired gamma is analytic-toy valid but not yet predictive in trained data. |
| trainability | Mean-vs-best and seed-std analyses still show no strong scalar pre-training trainability law. |
| mechanism | successful trained models remain branch/projection/tree dependent |
| causal_evidence | Prospective same-root tree-difference overlap failed as a standalone causal knob; normal-fan/tree-count remains weak and entangled. |
| thermodynamic_physics | untested; no Fmax conclusions. |

## Supported

- matrix-tree rooted tree-sum basis is the correct first-order computational basis
- post-training branch/projection/tree dependence is strong in selected trained models
- same-root tree-difference overlap is not sufficient as a standalone prospective causal control
- gamma is repaired on analytic toys but remains diagnostic, not a selector

## Weakened

- retrospective fixed-m20 tree-difference multiplicity as a general causal knob
- single scalar pre-training topology law
- gamma as an immediate trained-performance predictor

## Open

- whether cross-root contrast geometry predicts under multi-base exact controls
- whether total rooted-tree count can be separated from task-aligned normal-fan coverage
- which variables explain seed variance and trainability
- thermodynamic force-budget effects in a reversible-support Markov parameterization
