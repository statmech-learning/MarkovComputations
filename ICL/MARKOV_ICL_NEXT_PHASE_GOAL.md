# Long-Running Agent Goal: Diagnose the Prospective Exact-Control Failure and Build the Next Markov-ICL Topology Test

## Executive Objective

Continue the first-order Markov-ICL / topology-ICL project **after the May 12 exact-control update**.

The project is no longer asking whether topology matters broadly. The current evidence says:

```text
First-order topology and input masks shape the rooted tree-sum basis used for ICL.
Existing-data tree-difference multiplicity was predictive, but the first prospective exact-control contrast did not support it as a standalone causal knob.
The best current weak pre-training signal is normal-fan / active-tree / tree-count geometry.
The strongest mechanism evidence remains post-training branch/projection/tree dependence.
```

The immediate scientific task is therefore:

```text
Diagnose why the fixed-m20 retrospective tree-difference signal failed under prospective exact control, then test whether normal-fan / active-tree / tree-count geometry survives stronger multi-base exact controls after separating it from total rooted-tree count.
```

This should be a **diagnostic exact-control program**, not a broad sweep.

---

## Current Status To Preserve

### 1. First-order tree-sum theory is exact

For first-order CRNs with exponential input-dependent rates,

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
\sum_s\sum_{T\in\mathcal T_s(G)}
\exp(\beta_T+\Theta_T^\top z)
},
\qquad
\Theta_T=\sum_{e\in T}K_e.
```

So the computational basis is not the isolated edge-vector set `{K_e}`. It is the rooted tree-sum basis `{Theta_T}`.

This part is not open. The open question is which structural properties of `(G, Omega)` predict trained ICL.

### 2. `gamma*_ICL` is repaired on analytic toys but is not predictive yet

The repaired no-bias gamma gate passed the original analytic toy checks:

```text
Toy A: two species, both branches       -> fails as expected
Toy B: two species, max branch          -> passes
Toy C: three species, both branches     -> passes
```

This means repaired `gamma*_ICL` is a legitimate diagnostic candidate.

But in fixed-m20 existing data, repaired gamma did not predict trained outcomes well:

```text
tree-difference multiplicity mean-ICL LOO R2   = 0.435
best repaired gamma model mean-ICL LOO R2      = 0.078
exact / tropical / hard-root gamma alone       = negative
```

Therefore:

```text
Use repaired gamma as a diagnostic, not as a topology selector.
```

### 3. Tree-difference multiplicity was retrospectively strong but prospectively weak

The retrospective fixed-m20 reanalysis showed:

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

This supported the idea that useful input multiplicity lives in rooted-tree and tree-difference space.

However, the first prospective exact-control tree-difference experiment fixed one physical graph, input-coupled count, `d_rel`, aggregate multiplicity, edge load, and coordinate-load stratum. It trained 16 mask groups with five seeds each. The high-overlap masks did **not** improve ICL:

```text
balanced load:
  high overlap mean ICL = 75.360
  low overlap mean ICL  = 79.330
  high-low              = -3.970

balanced load best seed:
  high overlap best ICL = 84.900
  low overlap best ICL  = 89.700
  high-low              = -4.800

imbalanced load:
  high overlap mean ICL = 68.830
  low overlap mean ICL  = 69.900
  high-low              = -1.070
```

Grouped LOO told the same story:

```text
mean ICL:
  controls only                 R2 = 0.488
  tree-difference + controls    R2 = 0.447
  gamma + controls              R2 = 0.320

best ICL:
  controls only                 R2 = 0.601
  tree-level + controls         R2 = 0.631
  tree-difference + controls    R2 = 0.545
  gamma + controls              R2 = 0.571
```

Therefore:

```text
Tree-difference overlap is useful, but it is not a standalone causal knob.
```

### 4. Normal-fan / active-tree / tree-count geometry is the current best weak pre-training signal

The scaled exact-degree normal-fan experiment fixed:

```text
N_n = 5
m = 12
N_c = 3
D = 2
exact in-degree sequence
exact out-degree sequence
full input coupling
d_rel = 88
full-coupling multiplicity distribution
```

It trained 32 topology groups with 5 seeds per group.

Grouped LOO values were weak but positive for normal-fan and tree-count variables:

```text
mean ICL:
  active-tree count       R2 = 0.099
  normal-fan pair         R2 = 0.112
  tree count              R2 = 0.114
  gamma exact             R2 = -0.127
  gamma + normal fan      R2 = 0.059

best ICL:
  active-tree count       R2 = 0.091
  normal-fan pair         R2 = 0.110
  tree count              R2 = 0.093
  gamma exact             R2 = -0.072
  gamma + normal fan      R2 = 0.117
```

Correlations were clearer than LOO prediction:

```text
active-tree count vs mean ICL      r ≈ 0.446
active-tree count vs best ICL      r ≈ 0.441
log rooted-tree count vs mean ICL  r ≈ 0.460
log rooted-tree count vs best ICL  r ≈ 0.439
```

Therefore:

```text
The next experiment must separate total rooted-tree abundance from genuine task-aligned normal-fan / branch geometry.
```

### 5. Post-training mechanism remains strong

The prospective pre-training tree-difference control was negative or inconclusive, but trained successful models still depended on branch/projection organization.

Selected high-performing trained models showed large drops under statistic-preserving interventions:

```text
context-block shuffle                         mean drop ≈ -71.458
stat-preserving projection scramble           mean drop ≈ -59.688
stat-preserving branch-alignment scramble     mean drop ≈ -57.778
decoder-root permutation                      mean drop ≈ -53.125
```

Selected edge ablations also mattered:

```text
mean max input-edge ablation loss      ≈ 21.146 points
mean max physical-edge ablation loss   ≈ 26.563 points
```

Therefore:

```text
The mechanism claim is stronger than the current pre-training selector claim.
```

---

## Scope And Non-Negotiable Rules

1. Stay within **first-order CRNs / Markov jump processes with exponential input-dependent rates** unless explicitly deriving a separate nonlinear theory.
2. Do not apply first-order matrix-tree claims to autocatalytic or WTA CRNs.
3. Use **novel-class ICL accuracy** as the primary metric.
4. Keep physical topology `G`, input mask `Omega`, trained functional topology, and post-training mechanism diagnostics separate.
5. Use grouped or hierarchical inference. Seeds are nested inside topology/mask groups.
6. Do not use the bare predictor name `tree_geometry`; use reconciled predictor names.
7. Do not treat edge-level multiplicity as sufficient.
8. Do not treat same-root tree-difference overlap as a standalone causal knob after the failed prospective contrast.
9. Do not use `gamma*_ICL` as a selector. It is repaired on toys but not predictive in trained data.
10. Do not launch broad sweeps. Every new training run should target a specific diagnostic question.
11. Do not make thermodynamic `Fmax` claims from arbitrary directed exponential-rate models.
12. Do not claim motif uniqueness.
13. Do not collapse expressivity, trainability, and mechanism into one accuracy number.

---

## Required Source Reports To Read First

Read and preserve the conclusions of:

```text
topology_icl_first_order_report.pdf                # May 12 current synthesis
post_phase3_markov_icl_synthesis.md
gamma_toy_repair_final_report.md
input_multiplicity_causal_control_report.md
tree_multiplicity_causal_mask_library.md
tree_level_multiplicity_reanalysis.md
predictor_name_reconciliation.md
original CRN-ICL paper, especially Fig. 3 and Appendix B.2-B.3
Markov expressivity paper, especially input multiplicity, coefficient constraints, sharpness, and non-equilibrium driving
```

This task supersedes earlier goals that treated tree-difference overlap or `gamma*_ICL` as likely standalone predictors.

---

# Track 0: Orientation And Artifact Audit

## Goal

Confirm the current repository state and identify exactly where the May 12 results live.

## Tasks

1. Confirm branch, commit, and worktree status.
2. Confirm that the May 12 report artifacts are available.
3. Locate scripts and outputs for:
   - prospective tree-difference exact control;
   - fixed-m20 tree-difference reanalysis;
   - repaired gamma existing-data reanalysis;
   - scaled exact-degree normal-fan expansion;
   - mechanism scrambles after the prospective exact-control phase.
4. Confirm whether learned `K` tensors are available for fixed-m20 and prospective-control models.
5. Confirm whether branch failures and trained margins are available for the prospective-control models and exact-degree normal-fan models.

## Deliverables

```text
current_state_orientation_audit.md
current_state_orientation_audit.json
```

---

# Track 1: Diagnose Why Tree-Difference Overlap Failed Prospectively

## Scientific Question

The retrospective fixed-m20 tree-difference signal was strong. The prospective exact-control tree-difference contrast was negative or inconclusive.

The task is to diagnose why.

Possible explanations:

1. **Saturation:** both high and low prospective masks already had enough comparison overlap.
2. **Wrong contrast:** same-root tree-difference overlap is not the relevant object; cross-root contrasts matter more.
3. **Missing sign / geometry:** co-participation ignores whether coordinates enter with helpful relative orientation.
4. **Coefficient controllability:** useful tree differences exist structurally but are not independently controllable through edge parameters.
5. **Trainability:** the masks are expressive but optimization finds solutions differently.
6. **One-graph artifact:** the prospective contrast used one physical graph and may not generalize.
7. **Mask-family confounding:** the retrospective signal may have been partly family/backbone/load driven.

## Required Analyses

Compare the old fixed-m20 mask groups and the prospective masks in the same feature space.

For each group compute or collect:

```text
physical graph identity
mask family / construction type
input-coupled count
d_rel
edge-level multiplicity summaries
coordinate-load and edge-load Gini
same-root tree-level comparison overlap
same-root tree-difference comparison overlap
cross-root tree-difference comparison overlap, if implemented
total rooted-tree count
rooted tree count per root and Gini
active-tree count / normal-fan support metrics
branch-tree NMI / branch-root NMI proxies
masked effective rank and condition number
root-pair contrast metrics
coefficient-controllability / Jacobian conditioning proxies, if feasible
```

Then answer:

1. Are the prospective high/low masks actually separated in tree-difference space relative to fixed-m20 masks?
2. Are the prospective masks separated only in same-root overlap but not in cross-root or decoder-relevant contrast geometry?
3. Did the prospective masks live in a saturated overlap regime?
4. Did load balance, edge load, rooted-tree abundance, or normal-fan geometry differ in the opposite direction?
5. Is the negative prospective contrast explained by one physical graph?
6. Does tree-difference overlap predict branch failures or trained margins better than mean ICL?

## Statistical Protocol

Use:

```text
group-level rows
within-physical-graph residualization
matched-pair contrasts
clustered bootstrap by physical graph where multiple graphs exist
regime-residualized correlations
feature distribution overlays
```

Do not treat seed rows as independent topologies.

## Deliverables

```text
tree_difference_failure_diagnosis.md
tree_difference_failure_diagnosis.json
```

The report must end with a clear answer:

```text
Why did the fixed-m20 signal not survive the prospective exact-control test?
```

If no single explanation is identifiable, say so and list which hypotheses remain viable.

---

# Track 2: Build Decoder-Aware And Cross-Root Contrast Metrics

## Motivation

Same-root tree-difference overlap may be too narrow.

The steady state is normalized across roots:

```math
\bar C_r = \tau_r / \sum_s \tau_s.
```

The decoder then compares species/root concentrations.

Therefore the relevant pre-training contrast may be between trees rooted at different species:

```text
T in T_r(G), T' in T_s(G), r != s
```

rather than only same-root pairs.

## New Metrics To Implement

### 1. Cross-root tree-difference comparison overlap

For input coordinate alpha and tree pair `(T_r, T_s)`:

```math
A^{cross}_{T_r,T_s,\alpha}
=
\sum_e |s_{T_r}(e)-s_{T_s}(e)|\Omega_{e\alpha}.
```

For comparison pair `(i,q,d)`:

```math
\bar O^{cross}_{r,s,i,q,d}
=
\frac{1}{|\mathcal T_r||\mathcal T_s|}
\sum_{T\in\mathcal T_r}\sum_{T'\in\mathcal T_s}
\mathbf 1[A^{cross}_{T,T',i,d}>0]
\mathbf 1[A^{cross}_{T,T',q,d}>0].
```

Summaries:

```text
min_cross_overlap_comparison
mean_cross_overlap_comparison
gini_cross_overlap_comparison
root-pair min / mean / max versions
```

### 2. Decoder-agnostic root-pair contrast diversity

Because decoder `B` is learned, pre-training metrics should not assume a fixed output root.

Compute root-pair feature diversity across all root pairs:

```text
number of root pairs with usable comparison overlap
entropy over root-pair overlap scores
minimum over context/query comparisons of best root-pair overlap
mean over comparisons of top-k root-pair overlap
```

### 3. Signed / oriented comparison participation proxy

Current overlap metrics ignore sign/orientation.

Add structural proxies for whether `z_i,d` and `z_q,d` can enter tree contrasts differently, not merely jointly.

Possible proxies:

```text
coordinate separation support: tree contrasts where i,d participates and q,d participates in different edge subsets
imbalance support: |A_diff[i,d] - A_diff[q,d]|
pairwise contrast rank for the subspace vector e_{i,d} - e_{q,d}
```

### 4. Coefficient-controllability proxy

For small enumerated graphs, estimate how independently edge parameters can control tree-score contrasts.

Compute:

```text
rank of tree-contrast incidence restricted to comparison coordinates
effective rank of masked cross-root tree-difference matrix
condition number of masked cross-root tree-difference matrix
edge participation bottleneck score among contrast-relevant edges
```

Do not overbuild a huge feature battery. Start with low-dimensional summaries.

## Evaluate Against Existing Results

Compare old metrics and new metrics on:

```text
fixed-m20 existing data
prospective tree-difference control data
scaled exact-degree normal-fan data
hard n5_m12 data if compatible
```

Outcomes:

```text
mean novel-class ICL
best-seed novel-class ICL
seed std
branch failures
trained branch margin
post-training branch/projection diagnostics
```

## Deliverables

```text
cross_root_tree_contrast_metrics.py
cross_root_tree_contrast_reanalysis.md
cross_root_tree_contrast_reanalysis.json
```

## Success Criteria

A useful result would show that cross-root / decoder-aware metrics explain why the prospective same-root tree-difference contrast failed and improve prediction over same-root metrics.

A useful negative result would show that even decoder-aware structural metrics remain weak, supporting the idea that trainability or post-training organization dominates.

---

# Track 3: Separate Total Rooted-Tree Count From Normal-Fan Branch Geometry

## Motivation

The scaled exact-degree normal-fan experiment found weak positive signal from active-tree count, normal-fan pair metrics, and rooted-tree count. But active-tree count and total rooted-tree count may be entangled.

The central question is:

```text
Is ICL helped by many rooted trees, or by task-aligned normal-fan / branch geometry?
```

These are not equivalent.

## Required Design

Generate or identify graph sets that separate:

```text
total rooted-tree abundance
rooted tree count balance
active-tree count
normal-fan branch coverage
branch-tree NMI
tree-polytope support geometry
```

while controlling as much as possible:

```text
N_n, m, N_c, D
exact in-degree sequence
exact out-degree sequence
d_rel
input-coupled count
full input multiplicity or matched input mask summaries
```

## Experimental Arms

Design at least two types of matched libraries:

### Arm A: fixed tree count, variable normal-fan geometry

Hold approximate or exact total rooted-tree count fixed, but vary:

```text
active-tree count
branch-tree NMI
normal-fan pair score
branch sharpness proxies
```

### Arm B: variable tree count, matched normal-fan geometry

Hold normal-fan metrics approximately fixed, but vary total rooted-tree count.

### Arm C: multi-base exact-degree rewire libraries

Repeat the exact-degree rewire construction across multiple base graphs or degree sequences, not just one.

This is necessary to avoid one-base artifacts.

## Training Plan

For each selected topology group:

```text
train >= 5 seeds
use novel-class ICL as primary outcome
collect best seed, mean seed, and seed std
collect mechanism diagnostics where feasible
```

Minimum target:

```text
>= 30 topology groups per regime if feasible
multiple base graphs or degree sequences
```

Use grouped LOO and held-out-base-graph tests where possible.

## Deliverables

```text
normal_fan_tree_count_separation_library.md
normal_fan_tree_count_separation_library.json
normal_fan_tree_count_training_report.md
normal_fan_tree_count_training_report.json
```

## Success Criteria

A strong result would separate the two possibilities:

```text
1. total rooted-tree count predicts ICL even after normal-fan controls
2. normal-fan / branch geometry predicts ICL after tree-count controls
3. neither predicts well, implying training and post-training organization dominate
```

---

# Track 4: Use `gamma*_ICL` As A Diagnostic, Not A Selector

## Current Status

`gamma*_ICL` passes analytic toys but does not predict fixed-m20 trained ICL well.

Therefore, do not use gamma to choose broad new training libraries.

## Questions To Ask

1. Does gamma correlate with **best-seed ICL** more than mean-seed ICL?
2. Does gamma predict branch failures better than aggregate accuracy?
3. Does gamma correlate with trained branch margin?
4. Does gamma fail in cases where normal-fan/tree-count metrics succeed?
5. Does gamma improve after adding cross-root / decoder-aware contrast metrics?
6. Does gamma behave differently on fixed physical graphs versus varying physical graphs?

## Analyses

Compute repaired no-bias exact/tropical/hard-root gamma on any new libraries from Tracks 2 and 3, but label it as diagnostic.

Compare:

```text
gamma alone
normal-fan metrics alone
tree count alone
cross-root metrics alone
gamma + normal-fan
gamma + cross-root metrics
```

Outcomes:

```text
mean ICL
best-seed ICL
seed std
branch failures
trained branch margin
post-training projection diagnostics
```

## Deliverable

```text
gamma_diagnostic_reanalysis_after_exact_controls.md
gamma_diagnostic_reanalysis_after_exact_controls.json
```

---

# Track 5: Mechanism Follow-Up For Every New Exact-Control Result

## Motivation

The strongest evidence remains post-training mechanism and causal dependence.

Every new exact-control experiment should include mechanism follow-up, especially for surprising successes or failures.

## Required Diagnostics

For selected high-, mid-, and low-performing models compute:

```text
branch-active-tree MI
branch-to-root MI
active-tree entropy
tree posterior entropy
trained branch margin
projection alignment
posterior matched comparison gap
input-coupling ablation loss
physical edge ablation loss
functional edge importance
```

## Required Causal Scrambles

For selected trained models run:

```text
context-block shuffle
stat-preserving projection scramble
stat-preserving branch-alignment scramble
decoder-root permutation
```

Preserve as many coarse statistics as possible:

```text
physical graph
input mask support
d_rel
root tree counts
projection row norms
edge-load and coordinate-load summaries where possible
```

## Deliverable

```text
mechanism_followup_after_exact_controls.md
mechanism_followup_after_exact_controls.json
```

The report must explicitly separate:

```text
pre-training predictor claim
post-training mechanism claim
causal intervention claim
```

---

# Track 6: Expressivity vs Trainability Split

## Motivation

Best seed, mean seed, and seed variance are different targets:

```text
best seed     ~= expressivity upper envelope
mean seed     ~= trainability / reliability
seed std      ~= optimization instability
```

The May 12 report found that many structural variables track best and mean seed but do not explain seed variance well.

## Tasks

For every new exact-control library, report predictors separately for:

```text
mean novel-class ICL
best-seed novel-class ICL
seed std
```

Then ask:

1. Does a metric predict best seed but not mean seed?
2. Does a metric predict mean seed but not best seed?
3. Does any metric predict seed variance?
4. Are failures due to lack of expressivity or unreliable training?

## Candidate Trainability Metrics

Compute where feasible:

```text
condition number of masked tree-difference / cross-root contrast matrix
edge participation bottleneck scores
rooted-tree redundancy
posterior entropy after training
number of alternative high-probability active trees
loss landscape / gradient norm summaries if available
```

## Deliverable

```text
expressivity_vs_trainability_after_exact_controls.md
expressivity_vs_trainability_after_exact_controls.json
```

---

# Track 7: Thermodynamic `Fmax` Experiment Remains Delayed

Do not start this unless explicitly requested.

Existing arbitrary directed exponential-rate CRNs are not valid for thermodynamic force-budget claims.

A valid thermodynamic experiment requires reversible support and a parameterization such as:

```math
W_{ij}=\exp(E_j-B_{ij}+F_{ij}/2+\text{input drive}),
\qquad
B_{ij}=B_{ji},
\qquad
F_{ij}=-F_{ji},
\qquad
|F_{ij}|\le F_{\max}.
```

Before any `Fmax` sweep, verify:

```text
reversible edge support
detailed-balance behavior at Fmax=0
correct generator convention
stable steady-state solve
valid comparison to the first-order CRN-ICL task
```

Deliverable only if run:

```text
thermodynamic_fmax_markov_icl_report.md
thermodynamic_fmax_markov_icl_report.json
```

---

# Required Final Synthesis

At the end of this task, produce:

```text
post_exact_control_failure_diagnosis_synthesis.md
post_exact_control_failure_diagnosis_synthesis.json
```

It must answer:

1. Why did the fixed-m20 tree-difference signal fail under prospective exact control?
2. Was same-root tree-difference overlap saturated, confounded, or simply the wrong object?
3. Do cross-root / decoder-aware contrast metrics improve over same-root tree-difference metrics?
4. Can total rooted-tree count be separated from normal-fan / branch geometry?
5. Which pre-training metric best predicts mean ICL, best-seed ICL, and seed variance under exact controls?
6. Does repaired gamma become useful in any exact-control setting, or does it remain diagnostic only?
7. Which claims are supported, weakened, or still open?
8. What should be the next experiment after this phase?
9. Was any thermodynamic claim tested? If not, state explicitly that thermodynamics remains untested.

The synthesis must separate claims into:

```text
expressivity
trainability
mechanism
causal evidence
thermodynamic physics
```

---

# Acceptance Criteria

## Strong positive outcome

At least one exact-control experiment shows that normal-fan / active-tree / tree-count geometry or a cross-root contrast metric predicts novel-class ICL beyond raw count, `d_rel`, aggregate multiplicity, and physical-graph controls.

## Strong diagnostic outcome

The project clearly explains why the retrospective fixed-m20 tree-difference signal failed prospectively and identifies a better structural variable or shows that no simple pre-training variable is currently adequate.

## Strong mechanism outcome

New trained models continue to show large drops under statistic-preserving branch/projection scrambles, confirming that mechanism is branch/projection/tree dependent even when pre-training selectors are weak.

## Useful negative outcome

No pre-training structural metric survives exact controls. Then the current conclusion should be that first-order topology shapes the basis, but trained ICL depends primarily on optimization and post-training organization rather than a simple scalar graph metric.

## Invalid outcome

A broad sweep is run without exact controls and interpreted as a causal topology result. Do not do this.

---

# Bottom Line For The Agent

The project has moved past the search for one obvious scalar topology law.

The next phase should answer:

```text
Why did tree-difference overlap fail prospectively?
Is the missing object cross-root contrast geometry, rooted-tree abundance, normal-fan branch coverage, coefficient controllability, or trainability?
```

Treat `gamma*_ICL` as a sanity-checked diagnostic, not a selector.

Treat normal-fan / active-tree / tree-count geometry as the best current weak pre-training direction.

Keep mechanism scrambles central, because post-training branch/projection dependence remains the strongest evidence.
