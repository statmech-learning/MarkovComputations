# Markov-ICL Theory Validation and Exact-Control Experiments

## Objective

Continue the first-order Markov-ICL expressivity project.

The updated report establishes that first-order CRNs with exponential input-dependent rates compute through rooted tree-sum projections

\[
\Theta_T = \sum_{e \in T} K_e,
\]

and that in tested fixed-count regimes, topology-associated structural and functional variables explain residual novel-class ICL variation beyond raw trainable count.

The current open problem is not whether topology matters in some broad sense. The open problem is:

\[
\boxed{
\text{Which branch-aware Markov expressivity functional of }(G,\Omega)\text{ predicts ICL before training?}
}
\]

The next phase should repair predictor definitions, lift input multiplicity into the tree-sum basis, validate lower-tail \(\gamma^*_{\rm ICL}\) on analytic toy cases, run a clean input-multiplicity causal control, scale the exact-degree / exact-\(d_{\rm rel}\) / exact-multiplicity normal-fan experiment, and only then attempt a thermodynamic \(F_{\max}\) sweep.

## Scope

Stay within first-order CRNs / Markov jump processes unless explicitly deriving a separate nonlinear theory.

Do not apply the first-order matrix-tree topology theory to autocatalytic or WTA CRNs.

Use novel-class ICL accuracy as the primary metric.

Keep physical topology \(G\) separate from input-encoding topology \(\Omega\).

Use grouped inference because seeds are nested inside topology/mask groups.

## Phase 1: Reconcile Predictor Naming and Regression Definitions

Before any new training, resolve the apparent naming mismatch:

- fixed \(m20\) "tree geometry" appears as \(R^2_{\rm LOO}=0.409\) in the structural section;
- fixed \(m20\) "tree geometry variables" appear as \(R^2_{\rm LOO}=0.158\) in the Markov reanalysis.

Produce:

- `predictor_name_reconciliation.md`
- `predictor_name_reconciliation.json`

For every predictor family, list:

- exact feature columns;
- target variable;
- unit of analysis: run-level, group-mean, group-best, or seed-std;
- grouping / LOO scheme;
- number of rows and groups;
- regularization or feature standardization;
- source script;
- source artifact;
- reason for any numerical discrepancy.

If two predictor families are different, rename them. For example:

- `tree_geometry_structural_full`
- `tree_geometry_markov_reanalysis_subset`
- `masked_tree_geometry_structural`
- `masked_tree_geometry_markov_reanalysis`

No new scientific claim should be made until this is clarified.

## Phase 2: Upgrade Edge-Level Multiplicity to Tree-Level Multiplicity

Implement tree-level and tree-difference-level input multiplicity metrics.

The edge-level metric is:

\[
M_\alpha = \sum_e \Omega_{e\alpha}.
\]

But the first-order CRN computes through rooted tree sums, so define:

\[
A_{T,\alpha}=\sum_{e\in T}\Omega_{e\alpha},
\]

and, for relative tree contrasts,

\[
A_{T,T',\alpha}^{\rm diff}
=
\sum_e |s_T(e)-s_{T'}(e)|\Omega_{e\alpha}.
\]

For each comparison pair \((i,q,d)\), compute root-conditioned tree overlap:

\[
\bar O_{r,i,q,d}^{\rm tree}
=
\frac{1}{|\mathcal T_r|}
\sum_{T\in\mathcal T_r}
\mathbf 1[A_{T,i,d}>0]
\mathbf 1[A_{T,q,d}>0].
\]

Also compute tree-difference overlap:

\[
\bar O_{r,i,q,d}^{\rm diff}
=
\frac{1}{|\mathcal P_r|}
\sum_{(T,T')\in\mathcal P_r}
\mathbf 1[A_{T,T',i,d}^{\rm diff}>0]
\mathbf 1[A_{T,T',q,d}^{\rm diff}>0],
\]

where \(\mathcal P_r\) should include same-root tree differences, and optionally cross-root tree pairs relevant to decoder competition.

Report both unweighted and normalized versions. Raw counts must be accompanied by normalized metrics because raw overlap scales with the number of trees.

Also implement learned/post-training versions using either \(|K_{e\alpha}|\) or tree posterior weights:

\[
P(T\mid r,z).
\]

Deliverables:

- `tree_level_multiplicity_metrics.py`
- `tree_level_multiplicity_reanalysis.md`
- `tree_level_multiplicity_reanalysis.json`

Required comparison:

\[
\text{edge-level multiplicity}
\quad\text{vs.}\quad
\text{tree-level multiplicity}
\quad\text{vs.}\quad
\text{tree-difference multiplicity}.
\]

Evaluate against:

- mean novel-class ICL;
- best-seed ICL;
- seed standard deviation;
- branch failures;
- trained branch margin.

## Phase 3: Validate \(\gamma^*_{\rm ICL}\) on Analytic Toy Cases

Do not use \(\gamma^*_{\rm ICL}\) for large sweeps until it passes toy validation.

Use the original paper's analytic cases.

### Toy A: Two Species, Both Branches

\[
N_n=2,\qquad N_c=2,\qquad D=1.
\]

This should fail for both branches because two vectors cannot cover all four branch directions.

Expected result:

\[
\gamma^*_{\rm ICL}\le 0
\]

or at least one branch has persistent negative lower-tail margin.

### Toy B: Two Species, One Branch

Same system, but restrict to one branch.

Expected result:

\[
\gamma^*_{\rm ICL}>0
\]

under reasonable norm budget.

### Toy C: Three Species, Both Branches

\[
N_n=3,\qquad N_c=2,\qquad D=1.
\]

Use the complete directed three-species graph.

Expected result:

\[
\gamma^*_{\rm ICL}>0
\]

or clear improvement over the two-species both-branch case.

Run exact, tropical, and hard-root variants.

Run with and without learned biases \(b_e\), because the original paper's analytic small-network setup sets \(b_e=0\). Label these separately:

- `gamma_no_bias`
- `gamma_with_bias`

Deliverables:

- `gamma_toy_validation_report.md`
- `gamma_toy_validation_report.json`

Do not proceed to large \(\gamma^*\)-based topology selection unless this phase passes.

## Phase 4: Input-Multiplicity Causal Control

Run a targeted causal control, not a broad sweep.

Hold fixed:

\[
G,\qquad \text{input count},\qquad d_{\rm rel},\qquad M_{\rm mean},
\]

and preferably physical graph \(G\) exactly.

Vary:

\[
\min_{i,d}\bar O_{i,q,d}^{\rm tree},
\]

\[
\min_{i,d}\bar O_{i,q,d}^{\rm diff},
\]

coordinate-load Gini, edge-load Gini, and comparison-coordinate imbalance:

\[
|M_{i,d}-M_{q,d}|.
\]

Construct masks that are matched on aggregate count but differ in comparison overlap:

- high comparison overlap, balanced coordinate load;
- high comparison overlap, imbalanced coordinate load;
- low comparison overlap, balanced aggregate multiplicity;
- low comparison overlap, high coordinate-load imbalance.

Train enough seeds for grouped inference.

Primary question:

\[
\boxed{
\text{Does comparison-coordinate tree/difference overlap causally predict novel-class ICL at fixed count and }d_{\rm rel}?
}
\]

Deliverables:

- `input_multiplicity_causal_control_report.md`
- `input_multiplicity_causal_control_report.json`

## Phase 5: Scale Exact-Degree / Exact-\(d_{\rm rel}\) / Exact-Multiplicity Normal-Fan Experiment

Scale the pilot design.

Fix:

\[
N_n,\qquad m,\qquad N_c,\qquad D,
\]

exact in-degree sequence, exact out-degree sequence, input count, \(d_{\rm rel}\), and multiplicity distribution.

Then deliberately vary:

- active-tree count;
- branch-tree NMI;
- tree-polytope support geometry;
- branch sharpness

\[
R_{r,b}
=
\max_{T\in\mathcal T_r}\Theta_T^\top u_b
-
\min_{T\in\mathcal T_r}\Theta_T^\top u_b;
\]

- structural analogues of \(R_{r,b}\) before training;
- lower-tail \(\gamma^*_{\rm ICL}\);
- tree-level and tree-difference multiplicity.

Train enough topology/mask groups for grouped inference. The goal is not a loose sweep. The goal is to test:

\[
\boxed{
\text{After controlling count, degree sequence, }d_{\rm rel},\text{ and multiplicity, does tree-polytope branch geometry still matter?}
}
\]

Use grouped LOO, clustered bootstrap, and held-out-family or held-out-base-graph checks where possible.

Deliverables:

- `exact_degree_exact_drel_exact_multiplicity_normal_fan_report.md`
- `exact_degree_exact_drel_exact_multiplicity_normal_fan_report.json`

## Phase 6: Thermodynamic \(F_{\max}\) Experiment Only After the Above

Do not use arbitrary directed exponential-rate models for thermodynamic force-budget claims.

First build a thermodynamic Markov model with reversible support:

\[
W_{ij}
=
\exp(E_j-B_{ij}+F_{ij}/2+\text{input drive}),
\]

with

\[
B_{ij}=B_{ji},
\qquad
F_{ij}=-F_{ji},
\qquad
|F_{ij}|\le F_{\max}.
\]

First verify:

- reversible edge support;
- detailed-balance behavior at \(F_{\max}=0\);
- correct generator convention;
- stable steady-state solve;
- valid comparison to the existing first-order CRN-ICL task.

Then sweep \(F_{\max}\) and test whether:

- novel-class ICL accuracy increases with \(F_{\max}\);
- lower-tail branch margin increases with \(F_{\max}\);
- branch sharpness increases with \(F_{\max}\);
- learned solutions use non-equilibrium cycle affinities.

Deliverables:

- `thermodynamic_fmax_experiment_report.md`
- `thermodynamic_fmax_experiment_report.json`

This phase is important but should come after the tree-level multiplicity, toy \(\gamma^*\), and exact-control topology phases.

## Required Final Synthesis

Produce a final synthesis that separates claims into:

- expressivity;
- trainability;
- mechanism;
- causal interventions;
- thermodynamics.

The final synthesis must explicitly say what is supported and what is not.

At minimum, include:

- `final_markov_icl_next_phase_synthesis.md`
- `final_markov_icl_next_phase_synthesis.json`

## Non-Negotiable Rules

- Do not run broad sweeps before Phase 1-3 pass.
- Do not treat edge-level multiplicity as sufficient; lift it to tree and tree-difference space.
- Do not use \(\gamma^*_{\rm ICL}\) for experiment selection until it passes analytic toy validation.
- Do not treat seeds as independent topologies; use grouped inference.
- Do not conflate physical topology \(G\) with input mask \(\Omega\).
- Do not make thermodynamic claims from arbitrary directed graphs.
- Do not apply first-order tree-sum theory to autocatalytic or WTA CRNs.
- Do not claim a universal scalar law unless held-out exact-control experiments support it.
- Do not claim motif uniqueness.
- Use novel-class ICL accuracy as the primary outcome.

## Success Criteria

A strong positive result would show that tree-level comparison multiplicity, branch sharpness, normal-fan diagnostics, and lower-tail \(\gamma^*_{\rm ICL}\) improve prediction of novel-class ICL beyond \(d_{\rm rel}\) and masked tree geometry under exact-count / exact-degree / exact-multiplicity controls.

A trainability result would show that \(\gamma^*_{\rm ICL}\) predicts best-seed ICL, while conditioning/redundancy/tree-entropy metrics predict mean-seed reliability or seed variance.

A thermodynamic result would show that lower-tail ICL branch margin and novel-class ICL improve as allowed non-equilibrium driving \(F_{\max}\) increases in a valid reversible-support Markov parameterization.

A negative result is still useful if it shows that existing tree geometry is already as predictive as these richer Markov-expressivity metrics.

## Bottom Line

This is the next long-running agent task, but it must be gated.

Most important instruction:

\[
\boxed{
\text{Do Steps 1-3 before launching expensive new training.}
}
\]

Then:

\[
\boxed{
\text{Use Step 4 as the first causal control, Step 5 as the main topology falsification test, and Step 6 as a later physical-thermodynamic extension.}
}
\]

This keeps the project from turning into another broad sweep and forces it to sharpen the actual theory.
