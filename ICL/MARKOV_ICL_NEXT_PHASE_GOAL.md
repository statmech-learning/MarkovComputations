# Long-Running Agent Goal: Markov-ICL Post Exact-Control Handoff

## Purpose

This file is the current handoff for the next long-running agent. It replaces
the previous prospective exact-control goal because that phase has now been
completed.

Use this file as the high-level `@` reference:

```text
@/Users/aadarwal/code/statmech/topology/ICL/MARKOV_ICL_NEXT_PHASE_GOAL.md
```

The concrete completed reports live under:

```text
@/Users/aadarwal/code/statmech/topology/ICL/results/next_phase_stats/
```

The most important completed synthesis is:

```text
@/Users/aadarwal/code/statmech/topology/ICL/results/next_phase_stats/post_gamma_repair_exact_control_synthesis.md
```

Exact-control results commit:

```text
2c4cba6dfd729e8c43d52351288665940fd1e80e
```

Exact-control GitHub commit:

```text
https://github.com/statmech-learning/MarkovComputations/commit/2c4cba6dfd729e8c43d52351288665940fd1e80e
```

---

## What Was Just Completed

The completed phase tested whether repaired no-bias `gamma*_ICL` and normalized
same-root tree-difference comparison overlap survive exact controls as
pre-training predictors of first-order Markov-ICL.

Completed tracks:

```text
Track 0: orientation audit
Track 1: repaired-gamma existing-data reanalysis
Track 2: prospective tree-difference multiplicity causal control
Track 3: exact-degree / exact-drel / exact-multiplicity normal-fan expansion
Track 4: expressivity vs trainability split
Track 5: mechanism and causal scramble follow-up
Track 6: thermodynamic Fmax delayed and not tested
```

Verification:

```text
prospective tree-diff training: 80 / 80 runs completed
exact-degree normal-fan training: 160 / 160 runs completed
mechanism metrics: 80 prospective-control rows
selected causal scrambles: 48 intervention rows
selected edge ablations: 4 trained models
```

Thermodynamic claims were not tested. Do not infer thermodynamic results from
these arbitrary directed first-order exponential-rate CRNs.

---

## Current Scientific State

### 1. First-order tree-sum theory remains the scoped theory

For a strongly connected directed first-order CRN with exponential
input-dependent rates,

```math
k_e(z)=\exp(b_e+K_e^\top z),
```

the matrix-tree theorem gives

```math
\bar C_r(z)
=
\frac{
\sum_{T\in\mathcal T_r(G)}
\exp(\beta_T+\Theta_T^\top z)
}{
\sum_s
\sum_{T\in\mathcal T_s(G)}
\exp(\beta_T+\Theta_T^\top z)
},
\qquad
\Theta_T=\sum_{e\in T}K_e.
```

The computational basis is the rooted tree-sum basis `{Theta_T}`, not the
isolated edge-vector basis `{K_e}`.

This theory remains scoped to first-order CRNs / Markov jump processes with
exponential input-dependent rates. Do not apply it directly to autocatalytic or
winner-take-all CRNs.

### 2. Predictor naming is resolved

Do not use the bare name `tree_geometry`.

Use:

```text
tree_geometry_structural_full
```

for the fixed-m20 structural model with group-mean LOO `R2 = 0.409`, and:

```text
tree_geometry_markov_reanalysis_subset
```

for the Markov-reanalysis subset model with group-mean LOO `R2 = 0.158`.

### 3. Tree-level multiplicity still matters, but the causal story changed

The fixed-m20 reanalysis showed a strong existing-data signal:

```text
mean novel-class ICL:
  edge-level multiplicity          R2 = -0.002
  tree-level multiplicity          R2 =  0.403
  tree-difference multiplicity     R2 =  0.435

best-seed novel-class ICL:
  edge-level multiplicity          R2 =  0.109
  tree-level multiplicity          R2 =  0.245
  tree-difference multiplicity     R2 =  0.419
```

The previous existing-data causal control was supportive but not decisive:

```text
controls-only mean ICL LOO R2                       = 0.376
tree-difference + controls mean ICL LOO R2          = 0.452
strict d_rel=200 controls-only R2                   = 0.187
strict d_rel=200 tree-difference + controls R2      = 0.293
matched high-low mean ICL contrast                  = +2.195 points
bootstrap 95% CI                                    = [-0.882, 5.122]
```

The new prospective exact-control test did not reproduce a positive causal
effect:

```text
balanced-load high tree-diff mean ICL   = 75.360
balanced-load low tree-diff mean ICL    = 79.330
high-minus-low mean ICL delta           = -3.970
bootstrap 95% CI                        = [-7.9905, 0.6700]

balanced-load high tree-diff best ICL   = 84.900
balanced-load low tree-diff best ICL    = 89.700
high-minus-low best ICL delta           = -4.800
bootstrap 95% CI                        = [-9.000, -0.150]
```

Grouped LOO in the prospective test:

```text
mean ICL:
  controls only                         R2 = 0.488
  tree-level + controls                 R2 = 0.488
  tree-difference + controls            R2 = 0.447
  gamma + controls                      R2 = 0.320

best ICL:
  controls only                         R2 = 0.601
  tree-level + controls                 R2 = 0.631
  tree-difference + controls            R2 = 0.545
  gamma + controls                      R2 = 0.571
```

Interpretation:

```text
Tree-difference overlap remains a useful mask-design covariate, but the first
prospective balanced-load exact-control test weakens the claim that it is a
standalone causal knob.
```

### 4. Gamma is repaired on toys but is not a predictor yet

The repaired toy gate passed before this exact-control phase:

```text
Toy A: two species, both branches, no bias       -> fails as expected
Toy B: two species, max branch, no bias          -> passes
Toy C: three species, both branches, no bias     -> passes
```

No-bias analytic summary:

```text
Toy A accuracy = 0.500, LCVaR = -11.039
Toy B accuracy = 1.000, LCVaR =   3.513
Toy C accuracy = 1.000, LCVaR =   1.516
```

However, repaired gamma did not predict existing or exact-control outcomes
better than current tree/mask metrics:

```text
fixed-m20 mean ICL:
  tree-difference multiplicity          R2 =  0.435
  best repaired gamma model             R2 =  0.078
  repaired gamma exact                  R2 = -0.075
  repaired gamma tropical               R2 = -0.115
  repaired gamma hard-root              R2 = -0.150

exact-degree normal-fan expansion:
  gamma exact, mean ICL                 R2 = -0.127
  gamma exact, best ICL                 R2 = -0.072
```

Interpretation:

```text
gamma_no_bias is a candidate diagnostic, not a topology selector.
Do not launch a gamma-selected sweep unless the experiment is explicitly
designed to test gamma as a hypothesis.
```

### 5. Normal-fan / tree-count variables are now the best next pre-training signal

The exact-degree / exact-drel / exact-multiplicity normal-fan expansion fixed:

```text
Nn, m, Nc, D
exact in-degree sequence
exact out-degree sequence
full input-coupled count
d_rel
full-coupling multiplicity distribution
```

It trained:

```text
32 topology groups
5 seeds per group
160 total runs
```

Grouped LOO results:

```text
mean seed ICL:
  active_tree_count                     R2 =  0.099
  normal_fan_pair                       R2 =  0.112
  tree_count                            R2 =  0.114
  gamma_no_bias_exact                   R2 = -0.127
  gamma_plus_normal_fan                 R2 =  0.059

best seed ICL:
  active_tree_count                     R2 =  0.091
  normal_fan_pair                       R2 =  0.110
  tree_count                            R2 =  0.093
  gamma_no_bias_exact                   R2 = -0.072
  gamma_plus_normal_fan                 R2 =  0.117

seed std:
  all tested models were negative or very small
```

Interpretation:

```text
Normal-fan active-tree/tree-count variables give the strongest new
pre-training signal, but the signal is weak. This is not a universal scalar law.
```

### 6. Mechanism evidence is strong after training

Mechanism diagnostics were collected for all 80 prospective-control trained
runs. Selected causal scrambles were run on four high-performing trained models.

Selected scramble accuracy drops:

```text
context_block_shuffle                         mean delta = -71.458 points
stat_preserving_projection_scramble           mean delta = -59.688 points
stat_preserving_branch_alignment_scramble     mean delta = -57.778 points
decoder_root_permutation                      mean delta = -53.125 points
```

Selected edge ablations:

```text
input_ablation_max_loss       mean = 21.146 points
input_ablation_mean_loss      mean =  6.125 points
physical_ablation_max_loss    mean = 26.563 points
physical_ablation_mean_loss   mean = 11.010 points
```

Interpretation:

```text
Trained models that work depend on branch/projection organization and
ablation-sensitive edges. This is mechanism evidence after training, not
pre-training causal proof of tree-difference overlap.
```

---

## Required Reports To Read First

Read these before doing any new work:

```text
ICL/results/next_phase_stats/post_gamma_repair_exact_control_synthesis.md
ICL/results/next_phase_stats/repaired_gamma_existing_data_reanalysis.md
ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_causal_report.md
ICL/results/next_phase_stats/exact_degree_exact_drel_exact_multiplicity_training_report.md
ICL/results/next_phase_stats/expressivity_vs_trainability_exact_control_report.md
ICL/results/next_phase_stats/mechanism_and_causal_scramble_followup_report.md
ICL/results/next_phase_stats/next_phase_orientation_audit.md
ICL/results/next_phase_stats/post_phase3_markov_icl_synthesis.md
ICL/results/next_phase_stats/gamma_toy_repair_final_report.md
ICL/results/next_phase_stats/input_multiplicity_causal_control_report.md
ICL/results/next_phase_stats/tree_multiplicity_causal_mask_library.md
ICL/results/next_phase_stats/predictor_name_reconciliation.md
ICL/results/next_phase_stats/tree_level_multiplicity_reanalysis.md
```

Also preserve the conclusions of:

```text
the original CRN-ICL paper, especially Fig. 3 and Appendix B.2-B.3
the Markov expressivity paper, especially input multiplicity, monotonicity,
coefficient constraints, and sharpness
the topology-ICL first-order report / synthesis
```

---

## Local Artifacts And Data

Committed report artifacts:

```text
ICL/results/next_phase_stats/post_gamma_repair_exact_control_synthesis.md
ICL/results/next_phase_stats/post_gamma_repair_exact_control_synthesis.json
ICL/results/next_phase_stats/repaired_gamma_existing_data_reanalysis.md
ICL/results/next_phase_stats/repaired_gamma_existing_data_reanalysis.json
ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_mask_library.md
ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_mask_library.json
ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_training_plan.md
ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_causal_report.md
ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_causal_report.json
ICL/results/next_phase_stats/exact_degree_exact_drel_exact_multiplicity_normal_fan_library.md
ICL/results/next_phase_stats/exact_degree_exact_drel_exact_multiplicity_normal_fan_library.json
ICL/results/next_phase_stats/exact_degree_exact_drel_exact_multiplicity_training_report.md
ICL/results/next_phase_stats/exact_degree_exact_drel_exact_multiplicity_training_report.json
ICL/results/next_phase_stats/expressivity_vs_trainability_exact_control_report.md
ICL/results/next_phase_stats/expressivity_vs_trainability_exact_control_report.json
ICL/results/next_phase_stats/mechanism_and_causal_scramble_followup_report.md
ICL/results/next_phase_stats/mechanism_and_causal_scramble_followup_report.json
ICL/results/next_phase_stats/next_phase_orientation_audit.md
ICL/results/next_phase_stats/next_phase_orientation_audit.json
```

Committed CSV summaries:

```text
ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_training_results.csv
ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_training_manifest.csv
ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_mechanism_results.csv
ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_selected_ablation_mechanism_results.csv
ICL/results/next_phase_stats/prospective_tree_diff_multiplicity_causal_interventions.csv
ICL/results/next_phase_stats/exact_degree_exact_drel_exact_multiplicity_normal_fan_training_results.csv
ICL/results/next_phase_stats/exact_degree_exact_drel_exact_multiplicity_normal_fan_training_manifest.csv
ICL/results/next_phase_stats/repaired_gamma_existing_data_gamma_rows.csv
```

Committed prospective mask library:

```text
ICL/results/prospective_tree_diff_multiplicity_n6_m20_c200/
```

Local untracked raw trained-model directories, if still present in this
workspace:

```text
ICL/results/prospective_tree_diff_multiplicity_training/
ICL/results/exact_degree_exact_drel_exact_multiplicity_normal_fan_training/
```

These raw directories contain model checkpoints and per-run files. They were
not pushed to GitHub; the committed CSV summaries and reports are the stable
shared artifacts.

---

## Next Recommended Long-Running Task

The next task should not rerun the just-completed phase. It should use the
completed negative/inconclusive result to sharpen the theory.

Immediate objective:

```text
Determine whether normal-fan active-tree/tree-count geometry remains
predictive after separating it from raw rooted-tree count, and diagnose why
the fixed-m20 tree-difference signal failed under the first prospective
exact-control mask design.
```

### Track A: Diagnose the tree-difference prospective failure

Questions:

```text
Was the prospective tree-difference range too narrow in the balanced stratum?
Did the generated masks introduce hidden branch difficulty or load correlations?
Does fixed-m20 tree-difference signal depend on mask family or graph family?
Does tree-difference overlap only help when paired with normal-fan support?
```

Required analyses:

```text
Compare prospective masks against fixed-m20 masks in tree-level, tree-difference,
edge overlap, coordinate-load, branch-common d_rel, condition number, root tree
count, and normal-fan summaries.

Residualize fixed-m20 outcomes by physical graph / mask family where possible.

Construct a wider matched-pair mask library on at least two physical graphs,
but do not train until the library proves the intended contrast is clean.
```

Deliverables:

```text
tree_difference_failure_diagnostic_report.md
tree_difference_failure_diagnostic_report.json
tree_difference_wider_contrast_mask_library.md
tree_difference_wider_contrast_mask_library.json
```

### Track B: Expand normal-fan exact controls across base graphs

Questions:

```text
Does active-tree/tree-count geometry survive multiple base graphs or degree
sequences?
Is the weak normal-fan signal really geometry, or mostly total rooted-tree count?
Can branch-tree NMI or active-tree count predict held-out base graph families?
```

Design:

```text
Use multiple base degree sequences.
Preserve exact in-degree and out-degree within each family.
Fix or stratify d_rel and input-coupled count.
Fix or stratify multiplicity distribution.
Track rooted-tree count separately from normal-fan active-tree support.
Use grouped LOO, held-out base graph, and clustered bootstrap.
```

Do not claim a universal scalar law. The current signal is weak.

Deliverables:

```text
normal_fan_multibase_exact_control_library.md
normal_fan_multibase_exact_control_library.json
normal_fan_multibase_exact_control_report.md
normal_fan_multibase_exact_control_report.json
```

### Track C: Keep gamma as a diagnostic, not a selector

Questions:

```text
Why does repaired gamma pass analytic toys but fail prediction?
Is the issue lower-tail margin definition, root assignment, branch sampling,
or mismatch between expressivity and trainability?
Does gamma predict any subset once normal-fan/tree-count variables are controlled?
```

Required rule:

```text
Do not select new topologies solely by gamma_no_bias.
```

Deliverables:

```text
gamma_predictor_failure_audit.md
gamma_predictor_failure_audit.json
```

### Track D: Mechanism checks on normal-fan extremes

Use selected high/mid/low normal-fan trained models. Compute:

```text
branch-active-tree MI
branch-to-root MI
tree posterior entropy
trained branch margin
projection alignment
posterior matched comparison gap
input-coupling ablation loss
physical edge ablation loss
functional edge importance
stat-preserving branch-alignment scrambles
stat-preserving projection scrambles
context-block shuffles
decoder-root permutations
```

Deliverables:

```text
normal_fan_mechanism_scramble_report.md
normal_fan_mechanism_scramble_report.json
```

### Track E: Thermodynamics remains delayed

Do not make thermodynamic claims until a reversible-support Markov
parameterization is implemented and validated:

```math
W_{ij}=\exp(E_j-B_{ij}+F_{ij}/2+\text{input drive}),
\qquad
B_{ij}=B_{ji},
\qquad
F_{ij}=-F_{ji},
\qquad
|F_{ij}|\le F_{\max}.
```

First verify:

```text
reversible edge support
detailed balance at Fmax=0
correct generator convention
stable steady-state solve
valid comparison to the first-order CRN-ICL task
```

---

## Non-Negotiable Rules For The Next Agent

1. Stay within first-order CRNs / Markov jump processes unless explicitly
   deriving a separate nonlinear theory.
2. Do not apply first-order tree-sum claims to autocatalytic or WTA CRNs.
3. Use novel-class ICL accuracy as the primary metric.
4. Keep physical topology `G`, input mask `Omega`, trained functional topology,
   and post-training mechanism diagnostics separate.
5. Use grouped or hierarchical inference; seeds are nested inside topology/mask
   groups.
6. Do not use the bare predictor name `tree_geometry`.
7. Do not treat edge-level multiplicity as sufficient.
8. Do not treat the completed prospective tree-difference result as positive
   causal proof; it was negative/inconclusive.
9. Do not use `gamma_no_bias` as a topology selector yet.
10. Do not launch a broad sweep and call it causal.
11. Do not make thermodynamic claims from arbitrary directed graphs.
12. Do not claim motif uniqueness or a universal scalar law.

---

## Concrete Bottom Line

The project state is now:

```text
Gamma is repaired on analytic toys but not predictive in current data.
Tree-difference overlap was strong in fixed-m20 reanalysis but failed the
first prospective exact-control causal test.
Normal-fan active-tree/tree-count variables are the best current pre-training
signal, but only weakly.
Trained successful models causally depend on branch/projection organization.
Thermodynamic physics remains untested.
```

The next model should diagnose the failed tree-difference causal contrast and
run a multi-base exact-control normal-fan expansion, while keeping gamma as a
diagnostic and thermodynamics delayed.
