# Long-Running Agent Task: Repair `gamma*_ICL` and Test Tree-Level Multiplicity Causally

## Executive Status

This task continues the first-order Markov-ICL / topology-ICL project after the latest reports.

The current state is:

1. **Phase 1 passed.** The predictor-name collision is resolved. The fixed-m20 `tree_geometry` value `0.409` and the Markov-reanalysis `tree_geometry` value `0.158` are different regressions and must be named differently:
   - `tree_geometry_structural_full`
   - `tree_geometry_markov_reanalysis_subset`

2. **Phase 2 passed and produced a real scientific signal.** Edge-level input multiplicity is much weaker than tree-level and tree-difference multiplicity. In the fixed-m20 mask dataset, grouped LOO `R^2` for mean novel-class ICL is:
   - edge-level multiplicity: `-0.002`
   - tree-level multiplicity: `0.403`
   - tree-difference multiplicity: `0.435`

   For best-seed ICL in the same dataset:
   - edge-level multiplicity: `0.109`
   - tree-level multiplicity: `0.245`
   - tree-difference multiplicity: `0.419`

   This strongly supports the updated statement:

   > For first-order CRN-ICL, useful input multiplicity lives in rooted-tree and tree-difference space, not merely edge space.

3. **Phase 3 failed, and that failure is informative.** The current lower-tail `gamma*_ICL` probe failed the analytic toy gate. Do **not** use the current `gamma*_ICL` implementation for large topology selection. The failure should be treated as a probe/definition/debugging failure, not as a failure of the first-order tree-sum theory.

The next phase has two parallel tracks:

- **Track A:** repair and validate `gamma*_ICL` using analytic two- and three-species cases from the original CRN-ICL paper.
- **Track B:** run a clean input-multiplicity causal control using the successful tree-level and tree-difference multiplicity metrics from Phase 2.

Do **not** launch broad new topology sweeps. Do **not** use the current `gamma*_ICL` probe as a topology selector until Track A passes.

## Scope

Stay within **first-order CRNs / Markov jump processes with exponential input-dependent rates** unless explicitly deriving a separate nonlinear theory.

Do not apply first-order matrix-tree/tree-sum claims to autocatalytic or winner-take-all CRNs. The original CRN-ICL paper shows that WTA systems can use threshold/default mechanisms, so the first-order tree-sum theory does not directly apply there.

Use **novel-class ICL accuracy** as the primary metric. Training and ordinary validation accuracy are secondary diagnostics.

Keep these objects separate:

- physical reaction topology `G`;
- input-encoding topology / mask `Omega`;
- trained functional topology;
- post-training mechanism diagnostics.

Use grouped or hierarchical inference because training seeds are nested inside topology/mask groups. Do not treat run-level seed rows as independent topology samples.

## Core Theory To Preserve

For a strongly connected directed first-order graph `G=(V,E)`, with edge rates

```math
k_e(z)=\exp(b_e+K_e^\top z),
```

the matrix-tree theorem gives

```math
\bar C_r(z)
=
\frac{\tau_r(z)}{\sum_s \tau_s(z)},
\qquad
\tau_r(z)=\sum_{T\in\mathcal T_r(G)}\prod_{e\in T}k_e(z).
```

Substituting the exponential encoding:

```math
\tau_r(z)
=
\sum_{T\in\mathcal T_r(G)}
\exp(\beta_T+\Theta_T^\top z),
\qquad
\beta_T=\sum_{e\in T}b_e,
\qquad
\Theta_T=\sum_{e\in T}K_e.
```

Therefore the computational basis is not the isolated edge-vector set `{K_e}`. It is the rooted tree-sum set:

```math
\{\Theta_T:T\in\mathcal T_r(G), r\in V\}.
```

Because concentrations are normalized, relative tree-score geometry and tree-difference geometry are often more relevant than raw tree vectors.

The current project has already established the scoped result:

> In tested first-order fixed-count regimes, topology-associated structural and functional variables explain residual novel-class ICL variation beyond raw trainable count. The strongest mechanism evidence is that trained models depend on branch/projection/tree alignment.

The new Phase 2 result sharpens the input-mask story:

> Input multiplicity should be measured in rooted-tree and tree-difference space, because the steady state computes with tree-sum projections.

## Required Source Reports

Read these before making changes:

1. `predictor_name_reconciliation.md`
2. `tree_level_multiplicity_reanalysis.md`
3. `gamma_toy_validation_report.md`
4. the prior topology-ICL first-order report / synthesis
5. the original CRN-ICL paper, especially Figure 3 and Appendix B.2-B.3
6. the Markov expressivity paper, especially input multiplicity, monotonicity, coefficient constraints, and sharpness sections

# Track A: Repair and Validate `gamma*_ICL`

## A0. Principle

The current `gamma*_ICL` probe failed the analytic toy gate. It should remain **gated off** for large topology selection.

The failure does not invalidate tree-polytope / branch-margin theory. It means the implementation, branch dataset, optimizer, or capacity definition does not yet reproduce known small-system results.

The original CRN-ICL paper gives the decisive sanity checks:

- `N_n=2, N_c=2, D=1`, both branches: should fail.
- `N_n=2, N_c=2, D=1`, one branch condition `z_q=max(z_1,z_2)`, corresponding to `M1>` and `M2>`: should pass without edge biases.
- `N_n=3, N_c=2, D=1`, complete directed three-species graph: should pass without edge biases.

The latest report shows:

- Toy A fails as expected.
- Toy B only passes when edge biases are allowed; no-bias fails.
- Toy C fails even with biases.

This is not acceptable, because the original analytic examples use `b_e=0` and show that the one-branch two-species case and both-branch three-species case are solvable.

## A1. Audit Branch Dataset Definitions

First check whether the toy datasets are defined correctly.

The one-branch condition from the paper is not "one class" and not a single active branch. It is:

```math
z_q=\max(z_1,z_2),
```

which contains two comparison branches:

```text
M1> : z_1=z_q>z_2
M2> : z_2=z_q>z_1
```

If the current report says Toy B has `active branches: [0]`, audit this carefully. That may indicate the one-branch task is being encoded incorrectly.

Deliverable:

```text
branch_dataset_audit.md
branch_dataset_audit.json
```

The audit must report:

- how each branch is generated;
- what labels are assigned;
- how many samples per branch;
- whether samples are delta-separated from ambiguous intersections;
- whether Toy B contains both `M1>` and `M2>`;
- whether Toy A contains `M1>`, `M1<`, `M2>`, `M2<`;
- whether Toy C uses the same branch set as Toy A.

## A2. Use Delta-Separated Analytic Branch Samples

Raw lower-tail margin can be destroyed by samples near branch intersections, especially near

```math
z_1=z_2=z_q.
```

Construct an analytic branch dataset with explicit margin separation. For `N_c=2,D=1`, use samples of the form:

```text
M1> : z = (c, c-u, c)
M1< : z = (c, c+u, c)
M2> : z = (c-u, c, c)
M2< : z = (c+u, c, c)
```

where `u >= delta > 0` and `c` can be sampled or set to zero. This avoids ambiguous near-intersection points.

Report results as a function of `delta` and input range.

## A3. Reproduce The Original Paper's Two-Species Analytic Construction

For the two-species network, with no edge biases:

```math
K_1 = a(1,-2,1),
\qquad
K_2 = b(-2,1,1),
```

with `a,b>0` for the `z_q=max(z_1,z_2)` one-branch condition.

Evaluate this hard-coded construction directly. Do not optimize yet.

Expected:

- Toy B one-branch, no bias: high accuracy and positive lower-tail margin on delta-separated data.
- Toy A both-branches, no bias: failure on at least one sign branch.

Deliverable:

```text
two_species_analytic_gamma_audit.md
two_species_analytic_gamma_audit.json
```

## A4. Reproduce The Original Paper's Three-Species Tree Sums

For the labeled three-species graph in the paper, the spanning-tree vectors must match:

```text
A1 = K1 + K2
A2 = K3 + K4
A3 = K1 + K3

B1 = K5 + K1
B2 = K4 + K6
B3 = K4 + K5

C1 = K6 + K3
C2 = K5 + K2
C3 = K6 + K2
```

Build a unit test or diagnostic that enumerates the rooted trees and verifies this exact list for the paper's edge labeling.

If the code produces a different list, the problem is likely tree orientation, edge labeling, or generator convention.

Deliverable:

```text
three_species_tree_sum_orientation_audit.md
three_species_tree_sum_orientation_audit.json
```

## A5. Evaluate The Analytic Three-Species Construction Directly

Use the construction from the original paper's Appendix B.3:

```math
K_2=0,
\qquad
K_3=0,
\qquad
K_1=v,
\qquad
K_4=-\hat M_{2>}+v,
\qquad
K_5=\hat M_{2>}-v,
\qquad
K_6=-v,
```

with

```math
v=\hat M_{1>}+\hat M_{2>}.
```

This creates the hexagonal tree-projection geometry that covers the four comparison branches.

Evaluate the hard-coded construction directly before optimizing.

Expected:

- Toy C three-species both-branches, no bias: high accuracy and positive lower-tail margin on delta-separated data.

Deliverable:

```text
three_species_analytic_gamma_audit.md
three_species_analytic_gamma_audit.json
```

## A6. Separate Ordering, Accuracy, and Margin

For every gamma toy report, output all of these separately:

1. classification accuracy;
2. branch ordering correctness;
3. mean margin;
4. p10 margin;
5. lower-tail CVaR / LCVaR margin;
6. branch-wise failure rates.

A model can have correct ordering but arbitrarily small raw margin near a branch boundary. Do not let a lower-tail margin failure hide correct branch ordering if the dataset includes near-ambiguous samples.

## A7. Add Analytic Warm Starts and Gradient Optimization

Random trials are not enough. Add:

- hard-coded analytic evaluation;
- analytic warm starts;
- projected gradient descent or Adam/L-BFGS optimization from the warm starts;
- no-bias and with-bias variants, clearly separated.

Bias variants must be named separately:

```text
gamma_no_bias
gamma_with_bias
```

Do not use a biased gamma result to claim capacity for the original no-bias first-order setting.

## A8. Gate Condition For Gamma Repair

`gamma*_ICL` is repaired only if all of the following pass:

1. Toy A no-bias both-branches fails as expected.
2. Toy B no-bias one-branch passes on delta-separated data using hard-coded analytic `K`.
3. Toy C no-bias three-species both-branches passes on delta-separated data using hard-coded analytic `K`.
4. The three-species tree-sum enumeration exactly matches the paper's `A_i,B_i,C_i` list.
5. The optimizer either preserves analytic warm-start solutions or can recover equivalent solutions.
6. The report separates accuracy, ordering, and lower-tail margin.

Final gamma repair deliverable:

```text
gamma_toy_repair_final_report.md
gamma_toy_repair_final_report.json
```

Until this gate passes, do not use `gamma*_ICL` for topology selection or large exact-control sweeps.

# Track B: Input-Multiplicity Causal Control

## B0. Principle

Phase 2 already provides a strong structural signal. This track can proceed even while Track A repairs `gamma*_ICL`.

The causal question is:

```text
At fixed physical graph G, input count, d_rel, and aggregate multiplicity, does tree-level or tree-difference comparison overlap causally affect novel-class ICL?
```

Do not use the failed `gamma*_ICL` as a selector. Use Phase 2 tree-level and tree-difference multiplicity metrics.

## B1. Metrics To Use

For input mask `Omega`, edge-level multiplicity is:

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

For each comparison pair `(i,q,d)`, use normalized root-conditioned overlap metrics, for example:

```math
\bar O^{\mathrm{tree}}_{r,i,q,d}
=
\frac{1}{|\mathcal T_r|}
\sum_{T\in\mathcal T_r}
\mathbf 1[A_{T,i,d}>0]\mathbf 1[A_{T,q,d}>0].
```

and same-root tree-difference overlap:

```math
\bar O^{\mathrm{diff}}_{r,i,q,d}
=
\frac{1}{|\mathcal P_r|}
\sum_{(T,T')\in\mathcal P_r}
\mathbf 1[A^{\mathrm{diff}}_{T,T',i,d}>0]
\mathbf 1[A^{\mathrm{diff}}_{T,T',q,d}>0].
```

Use normalized scores. Raw counts must not be used alone because raw overlap scales with the number of trees or tree pairs.

Recommended low-dimensional summaries:

```text
min_tree_overlap_comparison
mean_tree_overlap_comparison
gini_tree_overlap_comparison
min_tree_diff_overlap_comparison
mean_tree_diff_overlap_comparison
gini_tree_diff_overlap_comparison
coord_load_gini
edge_load_gini
comparison_coord_imbalance
```

Avoid overfitting by throwing every feature into one large regression. Phase 2 showed that the combined tree-plus-difference feature battery can overfit and produce poor LOO performance.

## B2. Construct Causal Mask Families

Choose one or more fixed physical graphs `G`. For each fixed `G`, generate masks matched on:

```text
input-coupled count
aggregate M_mean
possibly exact M_alpha distribution where feasible
d_rel(G, Omega)
```

Then deliberately vary:

```text
min_tree_overlap_comparison
min_tree_diff_overlap_comparison
coord_load_gini
edge_load_gini
comparison_coord_imbalance
```

Construct at least these mask categories:

1. **High tree-diff comparison overlap, balanced coordinate load.**
2. **High tree-diff comparison overlap, imbalanced coordinate load.**
3. **Low tree-diff comparison overlap, balanced aggregate multiplicity.**
4. **Low tree-diff comparison overlap, high coordinate-load imbalance.**

The goal is to isolate whether comparison-coordinate overlap in the tree/difference basis causes ICL changes, not merely whether mask families differ.

## B3. Training Design

For each selected mask group:

- same physical graph `G`;
- same `N_n, m, N_c, D`;
- same input-coupled count;
- matched `d_rel`;
- matched or stratified aggregate multiplicity;
- at least 5 seeds per group; more if feasible;
- use novel-class ICL as primary outcome.

Report separately:

```text
mean seed novel-class ICL
best seed novel-class ICL
seed standard deviation
branch failures
trained branch margin
post-training tree/posterior metrics where available
```

Use grouped inference. Do not treat seeds as independent topology groups.

## B4. Statistical Tests

Compare models:

```text
raw count / input count baseline
edge-level multiplicity
tree-level multiplicity
tree-difference multiplicity
tree-difference + coordinate-load low-dimensional summary
```

Use:

- grouped LOO `R^2`;
- clustered bootstrap delta `R^2`;
- regression residualized by physical graph if multiple physical graphs are used;
- direct paired comparisons when masks are constructed as matched pairs.

Primary causal contrast:

```text
high tree-diff comparison overlap vs low tree-diff comparison overlap
```

under matched `G`, count, `d_rel`, and aggregate multiplicity.

## B5. Deliverables

```text
tree_multiplicity_causal_mask_library.md
tree_multiplicity_causal_mask_library.json
input_multiplicity_causal_control_training_plan.md
input_multiplicity_causal_control_report.md
input_multiplicity_causal_control_report.json
```

The final report must state whether the Phase 2 reanalysis signal survives causal control.

# Track C: Exact-Degree / Exact-d_rel / Exact-Multiplicity Normal-Fan Expansion

This track should start only after Track B has produced a clean causal-control library, and ideally after Track A has at least produced a working analytic gamma diagnostic. It does **not** require using `gamma*_ICL` as a selector if gamma remains gated off.

## C1. Objective

Scale the existing exact-degree normal-fan pilot. The previous pilot showed that one can fix:

```text
N_n, m, N_c, D
exact in-degree sequence
exact out-degree sequence
d_rel
```

while varying normal-fan / tree-polytope diagnostics. It had only four trained topology groups, so it was design feedback, not proof.

Now also fix or stratify by tree-level and tree-difference multiplicity distribution.

## C2. Controls

Fix:

```text
N_n
m
N_c
D
input count
exact in-degree sequence
exact out-degree sequence
d_rel
tree-level multiplicity distribution where feasible
tree-difference multiplicity distribution or matched summary bins
```

Vary:

```text
active-tree count
branch-tree NMI
normal-fan support / active tree coverage
tree-polytope support geometry
branch sharpness R_{r,b}
tree-diff comparison overlap if not fixed
```

If `gamma*_ICL` is still not repaired, do not include it as a selector. You may compute it and label it as gated/diagnostic only.

## C3. Deliverables

```text
exact_degree_exact_drel_exact_multiplicity_normal_fan_library.md
exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.md
exact_degree_exact_drel_exact_multiplicity_normal_fan_training_report.json
```

# Track D: Thermodynamic Fmax Experiment, Delayed

Do not start this until Tracks A-C are stable or explicitly requested.

Existing arbitrary directed exponential-rate CRNs are not valid for thermodynamic force-budget claims. A valid thermodynamic experiment requires reversible support and a parameterization such as:

```math
W_{ij}=\exp(E_j-B_{ij}+F_{ij}/2+\text{input drive}),
\qquad
B_{ij}=B_{ji},
\qquad
F_{ij}=-F_{ji},
\qquad
|F_{ij}|\le F_{\max}.
```

Before sweeping `Fmax`, verify:

- reversible edge support;
- detailed-balance behavior at `Fmax=0`;
- correct generator convention;
- stable steady-state solve;
- valid comparison to the first-order CRN-ICL task.

This is important but not the immediate next step.

# Non-Negotiable Rules

1. Do not use current `gamma*_ICL` for topology selection until Track A passes.
2. Do not interpret Phase 3 failure as failure of first-order topology expressivity.
3. Do not launch broad new sweeps before analytic gamma repair and/or input-multiplicity causal controls.
4. Do not use the bare name `tree_geometry`; use reconciled predictor names.
5. Do not treat edge-level multiplicity as sufficient; use tree-level and tree-difference multiplicity.
6. Do not use large combined feature batteries without regularization and small-sample safeguards.
7. Do not treat seeds as independent topology groups.
8. Do not conflate physical topology `G` with input mask `Omega`.
9. Do not make thermodynamic claims from arbitrary directed graphs.
10. Do not apply first-order matrix-tree theory to autocatalytic or WTA CRNs.
11. Do not claim motif uniqueness.
12. Use novel-class ICL accuracy as the primary outcome.

# Minimum Final Synthesis Required

At the end of this task, produce:

```text
post_phase3_markov_icl_synthesis.md
post_phase3_markov_icl_synthesis.json
```

It must answer:

1. Was `gamma*_ICL` repaired on analytic toy cases?
2. If not, exactly which diagnostic failed: branch data, tree orientation, analytic construction, decoder, optimizer, or margin definition?
3. Does tree-level or tree-difference multiplicity causally predict ICL under matched controls?
4. Does the causal result support the Phase 2 reanalysis, or was Phase 2 mostly family/regime confounding?
5. Which metric should be used for the next exact-degree normal-fan expansion?
6. Which claims are now supported, which are weakened, and which remain open?

The synthesis must separate:

```text
expressivity
trainability
mechanism
causal evidence
thermodynamic physics
```

# Success Criteria

A strong positive outcome for this phase is:

1. `gamma*_ICL` passes analytic toy validation, including no-bias Toy B and no-bias Toy C on delta-separated data; and/or
2. tree-level or tree-difference comparison overlap causally predicts novel-class ICL at fixed `G`, input count, `d_rel`, and aggregate multiplicity.

A useful partial success is:

- `gamma*_ICL` remains unrepaired, but the exact source of failure is localized; and
- the tree-level multiplicity causal control is completed and interpretable.

A useful negative result is:

- tree-level multiplicity does not survive matched causal control, which would mean the Phase 2 reanalysis signal was probably confounded by mask family, graph family, or regime-level structure.

Do not overstate any result. The goal is to sharpen the theory, not to force a positive conclusion.

# Bottom Line For The Agent

The next scientific question is no longer whether topology matters in a broad sense. The immediate question is:

```text
Does comparison-coordinate multiplicity in rooted-tree and tree-difference space causally control first-order CRN-ICL, and can a repaired lower-tail branch-margin capacity reproduce the known analytic toy cases?
```

Answer that before launching another large topology sweep.
