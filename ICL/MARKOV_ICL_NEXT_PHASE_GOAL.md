# Long-Running Agent Goal: Prospective Markov-ICL Exact-Control Phase

## Executive Objective

Continue the first-order Markov-ICL / topology-ICL project **after the repaired gamma toy gate and the existing-data tree-multiplicity control**.

The project is now past the question "does topology matter at all?" in the tested first-order regimes. The next task is to determine whether the best current structural variables -- especially **same-root tree-difference comparison overlap** and repaired **lower-tail `gamma*_ICL`** -- survive prospective exact controls and predict novel-class ICL before training.

The immediate scientific question is:

```text
At fixed physical graph controls, input-coupled count, d_rel, aggregate multiplicity, and preferably degree / load distributions, does comparison-coordinate co-participation in rooted-tree and tree-difference features causally control first-order CRN-ICL?
```

A second, coupled question is:

```text
Does repaired no-bias gamma*_ICL predict best-seed expressivity or mean-seed trainability beyond tree-difference overlap and existing masked-tree geometry?
```

This should be a **targeted exact-control program**, not a broad topology sweep.

---

## Current State To Preserve

### 1. First-order tree-sum theory is established

For a strongly connected directed first-order CRN with exponential input-dependent rates,

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

Thus the computational basis is the rooted tree-sum basis `{Theta_T}`, not the isolated edge-vector basis `{K_e}`.

### 2. Phase 1 predictor naming is resolved

The previous `tree_geometry` name collision is resolved. Do not use the bare name `tree_geometry`.

Use:

```text
tree_geometry_structural_full
```

for the fixed-m20 structural model with group-mean LOO `R2 = 0.409`, and

```text
tree_geometry_markov_reanalysis_subset
```

for the Markov-reanalysis subset model with group-mean LOO `R2 = 0.158`.

### 3. Phase 2 tree-level multiplicity succeeded

Edge-level multiplicity is not sufficient. In the fixed-m20 mask data, grouped LOO results showed:

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

This supports the refined statement:

```text
For first-order CRN-ICL, useful input multiplicity lives in rooted-tree and tree-difference space, not merely edge space.
```

### 4. The existing-data causal control is supportive but not final

In the fixed-m20 existing trained mask library:

```text
controls-only mean ICL LOO R2                         = 0.376
tree-difference multiplicity + controls mean ICL R2   = 0.452
strict d_rel=200 controls-only R2                     = 0.187
strict d_rel=200 tree-difference + controls R2        = 0.293
```

The matched high-low tree-difference contrast was positive but not decisive:

```text
mean novel-class ICL high-low contrast = +2.195 points
bootstrap 95% CI                       = [-0.882, 5.122]
```

Therefore tree-difference overlap is the right metric to carry forward, but causal proof is still open.

### 5. Gamma toy gate is now repaired

The current repaired `gamma*_ICL` gate passed:

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

Optimizer warm starts preserved or improved the analytic constructions:

```text
Toy B final LCVaR = 4.929
Toy C final LCVaR = 3.035
```

This means `gamma*_ICL` is no longer gated off. It is now a **candidate diagnostic**, not yet a proven predictor.

---

## Scope And Non-Negotiable Rules

1. Stay within **first-order CRNs / Markov jump processes with exponential input-dependent rates** unless explicitly deriving a separate nonlinear theory.
2. Do not apply first-order tree-sum claims to autocatalytic or WTA CRNs.
3. Use **novel-class ICL accuracy** as the primary metric.
4. Keep physical topology `G`, input-encoding mask `Omega`, trained functional topology, and post-training mechanism diagnostics separate.
5. Use grouped / hierarchical inference. Training seeds are nested inside topology/mask groups and are not independent topology samples.
6. Do not use the bare predictor name `tree_geometry`.
7. Do not treat edge-level multiplicity as sufficient.
8. Do not claim causal control from the existing fixed-m20 library alone; its high-low contrast is supportive but not decisive.
9. Do not claim `gamma*_ICL` is a capacity law just because it passes analytic toys.
10. Do not launch broad sweeps before the exact-control mask and gamma reanalysis phases below.
11. Do not make thermodynamic `Fmax` claims from arbitrary directed exponential-rate models.
12. Do not claim motif uniqueness.

---

## Required Source Reports To Read First

Read and preserve the conclusions of:

```text
post_phase3_markov_icl_synthesis.md
gamma_toy_repair_final_report.md
input_multiplicity_causal_control_report.md
tree_multiplicity_causal_mask_library.md
predictor_name_reconciliation.md
tree_level_multiplicity_reanalysis.md
prior topology-ICL first-order report / synthesis
original CRN-ICL paper, especially Fig. 3 and Appendix B.2-B.3
Markov expressivity paper, especially input multiplicity, monotonicity, coefficient constraints, and sharpness
```

---

# Track 0: Audit Current State And Code Paths

## Goal

Create a short reproducibility / orientation artifact before doing new analyses.

## Tasks

1. Confirm the current branch and commit.
2. Confirm that the repaired gamma report and tree multiplicity reports are present.
3. Locate the exact scripts used for:
   - tree-level and tree-difference multiplicity;
   - repaired gamma toy validation;
   - existing-data causal control;
   - fixed-m20 mask library generation;
   - grouped LOO inference.
4. Confirm whether learned `K` tensors are available for the fixed-m20 runs. If not, mark post-training weighted tree-overlap analyses as unavailable for that dataset.

## Deliverable

```text
next_phase_orientation_audit.md
next_phase_orientation_audit.json
```

---

# Track 1: Existing-Data Predictive Reanalysis With Repaired Gamma

## Why

Now that `gamma*_ICL` passes the analytic toy gate, the cheapest next test is whether repaired no-bias gamma predicts existing trained outcomes better than current structural metrics.

This phase should not launch new training.

## Metrics To Compare

For each available existing topology/mask group, compute or collect:

```text
raw_count_structural
raw_plus_drel_structural
masked_tree_geometry_structural
tree_geometry_structural_full
tree_geometry_markov_reanalysis_subset
edge_multiplicity_markov_reanalysis
tree_level_multiplicity
tree_difference_multiplicity
repaired_gamma_no_bias_exact
repaired_gamma_no_bias_tropical
repaired_gamma_no_bias_hard_root
gamma_no_bias + tree_difference_multiplicity summaries
```

If `gamma_with_bias` is computed, report it separately and do not use it to claim capacity for the original no-bias first-order setting.

## Outcomes

Evaluate against:

```text
mean novel-class ICL
best-seed novel-class ICL
seed standard deviation
branch failures, where available
trained branch margin, where available
post-training branch/tree/projection diagnostics, where available
```

## Statistical Protocol

Use:

- grouped LOO `R2`;
- clustered bootstrap delta `R2`;
- strict `d_rel` subset sensitivity where applicable;
- held-out physical graph / backbone tests where possible;
- regime-residualized correlations for pooled hard-regime results.

## Key Questions

1. Does repaired `gamma_no_bias` improve prediction over `d_rel`, masked tree geometry, and tree-difference multiplicity?
2. Does `gamma_no_bias` predict **best seed** better than **mean seed**, suggesting it captures expressivity more than trainability?
3. Does adding tree-difference multiplicity to `gamma_no_bias` improve prediction?
4. Does gamma fail in regimes where multiplicity succeeds, or vice versa?

## Deliverables

```text
repaired_gamma_existing_data_reanalysis.md
repaired_gamma_existing_data_reanalysis.json
```

## Gate

Do not use gamma as a selector for new large sweeps unless this reanalysis shows at least one meaningful improvement over existing structural metrics, or unless the new sweep is explicitly designed to test gamma as a hypothesis.

---

# Track 2: Prospective Tree-Difference Multiplicity Causal Control

## Goal

Run the first true prospective causal test of the Phase 2 tree-difference multiplicity signal.

The question is:

```text
At fixed G, input-coupled count, aggregate multiplicity, and d_rel, does normalized same-root tree-difference comparison overlap causally improve novel-class ICL?
```

## Preferred Design

Start with one or more fixed physical graphs `G`. For each graph, generate masks prospectively rather than reusing the old fixed-m20 library.

Match or control:

```text
N_n, m, N_c, D
physical graph G
input-coupled parameter count
aggregate M_mean
preferably full M_alpha distribution
input edge count
input coordinate count
edge-load Gini
coordinate-load Gini, if not the target contrast
d_rel(G, Omega)
```

Vary deliberately:

```text
min normalized same-root tree-difference comparison overlap
mean normalized same-root tree-difference comparison overlap
gini of tree-difference comparison overlap
normalized tree-level comparison overlap as secondary variable
```

## Core Metrics

For input mask `Omega`, define edge-level multiplicity:

```math
M_\alpha=\sum_e \Omega_{e\alpha}.
```

Tree-level participation:

```math
A_{T,\alpha}=\sum_{e\in T}\Omega_{e\alpha}.
```

Same-root tree-difference participation:

```math
A^{\mathrm{diff}}_{T,T',\alpha}
=
\sum_e |s_T(e)-s_{T'}(e)|\Omega_{e\alpha}.
```

Root-conditioned tree overlap:

```math
\bar O^{\mathrm{tree}}_{r,i,q,d}
=
\frac{1}{|\mathcal T_r|}
\sum_{T\in\mathcal T_r}
\mathbf 1[A_{T,i,d}>0]\mathbf 1[A_{T,q,d}>0].
```

Same-root tree-difference overlap:

```math
\bar O^{\mathrm{diff}}_{r,i,q,d}
=
\frac{1}{|\mathcal P_r|}
\sum_{(T,T')\in\mathcal P_r}
\mathbf 1[A^{\mathrm{diff}}_{T,T',i,d}>0]
\mathbf 1[A^{\mathrm{diff}}_{T,T',q,d}>0].
```

Use normalized scores. Raw counts must not be used alone.

## Mask Categories

Construct matched masks in categories such as:

1. high tree-diff comparison overlap, balanced coordinate load;
2. low tree-diff comparison overlap, balanced coordinate load;
3. high tree-diff comparison overlap, imbalanced coordinate load;
4. low tree-diff comparison overlap, imbalanced coordinate load.

The cleanest first causal contrast is category 1 vs category 2 with coordinate/load summaries matched as tightly as possible.

## Training

For each mask group:

```text
same training protocol as prior first-order topology runs
at least 5 seeds per group; more if feasible
novel-class ICL primary outcome
```

Report:

```text
mean seed novel-class ICL
best seed novel-class ICL
seed standard deviation
branch failures
trained branch margin
post-training branch/tree/projection diagnostics where available
```

## Statistical Tests

Compare:

```text
controls only
edge-level multiplicity + controls
tree-level multiplicity + controls
tree-difference multiplicity + controls
gamma_no_bias + controls
gamma_no_bias + tree-difference multiplicity + controls
```

Use grouped LOO, matched-pair contrasts, and clustered bootstrap.

## Success Criteria

A strong positive result would show:

```text
high tree-difference comparison overlap > low tree-difference comparison overlap
```

under matched `G`, input count, aggregate multiplicity, and `d_rel`, with confidence intervals not crossing zero for mean novel-class ICL or best-seed ICL.

A useful negative result would show that the Phase 2 signal was mostly due to confounding by mask family, graph family, or load imbalance.

## Deliverables

```text
prospective_tree_diff_multiplicity_mask_library.md
prospective_tree_diff_multiplicity_mask_library.json
prospective_tree_diff_multiplicity_training_plan.md
prospective_tree_diff_multiplicity_causal_report.md
prospective_tree_diff_multiplicity_causal_report.json
```

---

# Track 3: Exact-Degree / Exact-d_rel / Exact-Multiplicity Normal-Fan Expansion

## Start Condition

Begin this only after Track 1 and the first prospective Track 2 causal library are complete or explicitly waived.

## Goal

Scale the exact-degree / normal-fan pilot into a real statistical test.

The question is:

```text
After controlling count, degree sequence, d_rel, and tree-difference multiplicity, does normal-fan / tree-polytope branch geometry still predict ICL?
```

## Controls

Fix:

```text
N_n, m, N_c, D
exact in-degree sequence
exact out-degree sequence
input-coupled count
d_rel
aggregate M_mean
preferably M_alpha distribution
normalized tree-difference comparison overlap stratum
```

Vary:

```text
active-tree count
branch-tree NMI
normal-fan support / active tree coverage
tree-polytope support geometry
branch sharpness R_{r,b}
repaired gamma_no_bias metrics
rooted tree counts / log tree counts if not fixed
edge participation heterogeneity if not fixed
```

## Sampling Strategy

Generate many degree-preserving directed rewires per base graph. Stratify by normal-fan/tree-polytope variables while holding the controls above fixed.

Do not select only extremes unless the goal is an initial contrast. For regression, sample across the full range.

## Training

Train enough topology/mask groups for grouped inference.

Minimum recommended structure:

```text
>= 30 topology/mask groups per regime if feasible
>= 5 seeds per group
multiple base graphs or base degree sequences if feasible
```

## Outcomes

Same as Track 2:

```text
mean novel-class ICL
best-seed novel-class ICL
seed std
branch failures
trained branch margin
post-training active tree / branch-tree diagnostics
causal scrambles for selected high/mid/low trained models
```

## Deliverables

```text
exact_degree_exact_drel_exact_multiplicity_normal_fan_library.md
exact_degree_exact_drel_exact_multiplicity_training_report.md
exact_degree_exact_drel_exact_multiplicity_training_report.json
```

---

# Track 4: Expressivity vs Trainability Split

## Goal

Separate what a topology/mask **can express** from how reliably gradient training finds it.

Use:

```text
best-seed ICL       ~= expressivity upper envelope
mean-seed ICL       ~= trainability / reliability
seed std            ~= optimization instability
```

## Experiment

Construct matched pairs or groups with similar:

```text
gamma_no_bias
tree-difference comparison overlap
d_rel
input count
```

but different:

```text
conditioning of masked tree-difference matrix
edge participation bottlenecks
tree posterior entropy / active-tree entropy
redundancy / number of alternative trees
normal-fan concentration
```

## Hypotheses

If best seed is similar but mean seed differs, the structural capacity is similar but trainability differs.

If both best and mean differ, the capacity metric is missing an expressivity-relevant variable.

## Deliverable

```text
expressivity_vs_trainability_exact_control_report.md
expressivity_vs_trainability_exact_control_report.json
```

---

# Track 5: Mechanistic Follow-Up And Causal Scrambles

## Goal

For any high-performing or surprising groups from Tracks 2-4, determine whether trained models use the predicted mechanism.

## Required Analyses

For selected trained models, compute:

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
```

Then run statistic-preserving scrambles:

```text
branch-alignment scramble
projection scramble
context-block shuffle
decoder-root permutation
```

Preserve as many coarse statistics as possible, including graph, mask support, `d_rel`, root tree counts, and projection row norms.

## Deliverable

```text
mechanism_and_causal_scramble_followup_report.md
mechanism_and_causal_scramble_followup_report.json
```

---

# Track 6: Thermodynamic Fmax Experiment, Delayed

Do not start this until the first-order exact-control tracks are stable or the user explicitly requests it.

Existing arbitrary directed exponential-rate CRNs are not valid for thermodynamic force-budget claims.

A valid thermodynamic Markov experiment requires reversible support and a parameterization such as:

```math
W_{ij}=\exp(E_j-B_{ij}+F_{ij}/2+\text{input drive}),
\qquad
B_{ij}=B_{ji},
\qquad
F_{ij}=-F_{ji},
\qquad
|F_{ij}|\le F_{\max}.
```

Before training, verify:

```text
reversible edge support
detailed-balance behavior at Fmax=0
correct generator convention
stable steady-state solve
valid comparison to the first-order CRN-ICL task
```

Only then sweep `Fmax` and ask whether novel-class ICL and lower-tail branch margin increase with allowed non-equilibrium driving.

Deliverable, if run:

```text
thermodynamic_fmax_markov_icl_report.md
thermodynamic_fmax_markov_icl_report.json
```

---

# Required Final Synthesis

At the end of this task, produce:

```text
post_gamma_repair_exact_control_synthesis.md
post_gamma_repair_exact_control_synthesis.json
```

It must separate claims into:

```text
expressivity
trainability
mechanism
causal evidence
thermodynamic physics
```

It must answer:

1. Does repaired `gamma_no_bias` predict existing-data ICL better than current tree/mask metrics?
2. Does prospective tree-difference comparison overlap causally affect ICL under exact controls?
3. Does the tree-difference signal survive beyond mask family and physical graph identity?
4. Does exact-degree / exact-drel / exact-multiplicity normal-fan geometry predict ICL?
5. Does gamma predict best-seed expressivity, mean-seed trainability, both, or neither?
6. Which metric should be used in the next larger experiment?
7. Which claims are supported, weakened, or still open?
8. Was any thermodynamic claim tested? If not, say explicitly that it remains untested.

---

# Acceptance Criteria

## Strong positive outcome

At least one prospective exact-control experiment shows that normalized same-root tree-difference comparison overlap, repaired `gamma_no_bias`, or normal-fan/tree-polytope geometry improves prediction of novel-class ICL beyond `d_rel`, raw count, physical graph controls, and aggregate multiplicity.

## Strong expressivity/trainability outcome

`gamma_no_bias` or tree-difference overlap predicts best-seed ICL, while separate conditioning/redundancy metrics explain mean-seed reliability or seed variance.

## Useful negative outcome

The prospective exact-control experiment fails to reproduce the existing-data tree-difference signal. Then the Phase 2 signal was likely confounded by mask family, physical graph identity, or load imbalance.

## Invalid outcome

A broad sweep is run without exact controls and then interpreted as a causal topology result. Do not do this.

---

# Bottom Line For The Agent

The current project state is:

```text
Gamma is repaired on analytic toys.
Tree-difference comparison overlap is the best current input-mask structural metric.
Existing-data causal evidence is supportive but not final.
```

The immediate next goal is:

```text
Use repaired gamma and normalized same-root tree-difference comparison overlap in prospective exact-control tests.
```

Answer this before launching another broad topology sweep or a thermodynamic extension.
