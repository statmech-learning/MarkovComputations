# Long-Running Agent Goal: Multi-Base Normal-Fan / Tree-Count Theory for First-Order Markov-ICL

## Executive objective

Continue the first-order Markov-ICL / topology-ICL project from the current May 13 state.

The original project question was:

```text
Can reaction/input topology be useful for predicting and understanding ICL in first-order CRNs?
```

The current answer is now more precise:

```text
Topology is useful for understanding mechanism now, and partially useful for prediction, but no universal scalar topology law has been established.
```

The next task is **not** to prove that topology matters broadly. That has already been established in a scoped first-order sense. The next task is to determine whether the current weak pre-training signal -- **normal-fan / active-tree / rooted-tree-count geometry** -- can be turned into a better exact-control theory, or whether topology's strongest role remains **post-training and mechanistic**.

The central next question is:

```text
After controlling raw count, exact degree sequence, d_rel, and input multiplicity, does total rooted-tree abundance, task-aligned normal-fan geometry, cross-root contrast geometry, or some combination of these predict first-order CRN ICL before training?
```

The current live experiment is the **multi-base exact-degree normal-fan / tree-count library**. It was built to separate rooted-tree abundance from task-aligned normal-fan geometry. This should be consumed and analyzed before launching any new broad sweep.

---

## Current state to preserve

### 1. Exact first-order theory

Stay in first-order CRNs / Markov jump processes with exponential input-dependent rates unless a separate nonlinear theory is explicitly derived.

For edge `e`,

\[
k_e(z)=\exp(b_e+K_e^\top z).
\]

For root/species `r`, the matrix-tree theorem gives

\[
\bar C_r(z)=\frac{\tau_r(z)}{\sum_s \tau_s(z)},
\qquad
\tau_r(z)=\sum_{T\in\mathcal T_r(G)}\prod_{e\in T}k_e(z).
\]

Substituting the exponential rate form,

\[
\tau_r(z)=\sum_{T\in\mathcal T_r(G)}\exp(\beta_T+\Theta_T^\top z),
\qquad
\Theta_T=\sum_{e\in T}K_e.
\]

Therefore the computational projection basis is

\[
\{\Theta_T:T\in\mathcal T_r(G), r\in V\},
\]

not the isolated edge-vector set

\[
\{K_e:e\in E\}.
\]

This is the exact theoretical basis of the project.

### 2. What is already supported

The following statements are currently supported:

1. In tested first-order fixed-count regimes, raw trainable count is incomplete.
2. Topology-associated tree geometry and masked-tree geometry explain residual novel-class ICL variation beyond raw count in several tested regimes.
3. `d_rel` is a useful rank proxy, but it is not a capacity law.
4. Edge-level input multiplicity is insufficient; useful input multiplicity must be lifted into rooted-tree and tree-difference space.
5. The repaired no-bias `gamma*_ICL` probe passes analytic toy cases, but it is not yet predictive in larger trained libraries.
6. Same-root tree-difference comparison overlap was predictive in existing fixed-m20 data but failed as a standalone causal knob in the first prospective exact-control mask experiment.
7. The best current pre-training signal under exact controls is weak normal-fan / active-tree / rooted-tree-count geometry.
8. The strongest evidence remains post-training mechanism evidence: trained successful models depend on branch/projection/tree organization, and statistic-preserving scrambles collapse selected high-performing models.

### 3. What is not supported

Do **not** claim any of the following:

- `d_rel` is the topology capacity law.
- Same-root tree-difference overlap causally controls ICL as a standalone knob.
- Repaired `gamma*_ICL` predicts trained ICL.
- A unique physical motif for ICL has been discovered.
- These first-order conclusions transfer automatically to autocatalytic or WTA CRNs.
- Any thermodynamic `F_max`, entropy-production, or force-budget law has been established.

---

## Required source reports to read first

Before doing new work, read and preserve the conclusions of the latest reports:

```text
latest_updates.pdf
topology_icl_first_order_report.pdf
post_phase3_markov_icl_synthesis.md
gamma_toy_repair_final_report.md
input_multiplicity_causal_control_report.md
tree_multiplicity_causal_mask_library.md
predictor_name_reconciliation.md
tree_level_multiplicity_reanalysis.md
MARKOV_ICL_NEXT_PHASE_GOAL (1).md
original CRN-ICL paper 2601.06712v1.pdf
Markov expressivity paper s41467-025-61873-0.pdf
```

If some files are unavailable in the working tree, locate the equivalent committed artifacts in `ICL/results/next_phase_stats/` or the corresponding report directories.

---

## Operational setup and hygiene

Repository and cluster assumptions from the prior handoff:

```text
Repository: statmech-learning/MarkovComputations
Main Engaging worktree: /home/aadarwal/repos/topology
Branch: topology
Remote: statmech
GitHub target: github.com:statmech-learning/MarkovComputations.git
```

Engaging access pattern:

```text
Use existing tmux pane only: icl:13.2
Do not SSH separately.
Do not open another pane/window.
```

Useful commands:

```bash
tmux load-buffer /tmp/script
tmux paste-buffer -t icl:13.2
tmux send-keys -t icl:13.2 C-m
tmux capture-pane -t icl:13.2 -p -S -200
```

Python environment:

```text
Use python3, not python.
Preferred venv:
/home/aadarwal/orcd/scratch/venvs/markov_icl/bin/python
```

Cluster limitations:

```text
rg may not be available; use grep, find, sed, or Python.
Do not assume pdflatex or latexmk exists on Engaging.
```

Git hygiene:

```bash
git status --short -uno
```

Do not reset, clean, or remove untracked files unless explicitly asked. Other agents may be using nearby outputs.

---

# Track 0: Orientation and live-job audit

## Why

The latest report says a multi-base exact-degree normal-fan / tree-count experiment has been constructed and submitted. The first submission failed due to environment issues, the script was patched to load Miniforge, and corrected Slurm job `13902539` had begun completing tasks. Before doing new analysis or training, determine the current status of that experiment.

## Tasks

1. Confirm current branch, commit, and worktree status.
2. Locate the multi-base library generator and generated artifacts:

```text
ICL/multibase_normal_fan_tree_count_library.py
ICL/results/next_phase_stats/multibase_normal_fan_tree_count_training_plan.md
ICL/results/next_phase_stats/multibase_normal_fan_tree_count_separation_library.md
ICL/results/next_phase_stats/multibase_normal_fan_tree_count_n5_m12_N3_D2/
```

3. Check Slurm status for the latest multi-base job if available.
4. Count how many of the expected 185 training result files exist.
5. Identify failed / missing task IDs.
6. If some tasks failed, classify failure reason before resubmitting anything.
7. Do not launch any new broad sweep.

## Deliverables

```text
multibase_live_job_audit.md
multibase_live_job_audit.json
```

The audit must include:

```text
current branch / commit
expected task count
completed result count
failed task IDs
missing task IDs
whether all 37 selected topology groups have complete seed coverage
whether any old failed system-python results are present and should be ignored
```

---

# Track 1: Consume the multi-base exact-control results

## Why

The latest report says this is the next clean test. It is designed to separate **total rooted-tree abundance** from **task-aligned normal-fan / active-tree geometry**.

The library controls:

```text
N_n = 5
m = 12
N_c = 3
D = 2
p = 8
d_rel = 88
full input coupling
M_alpha = m for every input coordinate
exact in/out degree sequence within each base graph
```

It uses seven base graph families, degree-preserving rewires, and paired selection arms:

```text
Arm A: fixed rooted-tree count, variable normal-fan score
Arm B: variable rooted-tree count, matched normal-fan score
```

The reported design selected:

```text
560 candidate topologies
37 selected topology groups
20 matched pairs
185 training tasks
```

## Core scientific question

```text
Is ICL helped by total rooted-tree abundance, task-aligned normal-fan geometry, both, or neither?
```

## Required analysis

Compute, at the topology-group level:

```text
mean novel-class ICL
best-seed novel-class ICL
seed standard deviation
train / validation / novel gaps
failure rate if applicable
```

Evaluate these predictor sets:

```text
controls/base only
tree count + base
normal fan + base
tree count + normal fan + base
cross-root metrics + tree count + normal fan + base
repaired gamma diagnostics + base
all interpretable low-dimensional summaries
```

Use grouped inference:

```text
grouped LOO R^2
clustered bootstrap by base graph
held-out-base checks
matched-pair contrasts for Arm A and Arm B
regime/base residualized correlations
```

Do not treat seed runs as independent topology samples.

## Arm-specific tests

### Arm A: fixed tree count, variable normal fan

Ask:

```text
At nearly fixed total rooted-tree count, does normal-fan / active-tree geometry change ICL?
```

Report paired contrasts:

```text
Delta normal-fan score
Delta mean ICL
Delta best-seed ICL
Delta seed std
paired bootstrap CI
```

### Arm B: variable tree count, matched normal fan

Ask:

```text
At matched normal-fan score, does total rooted-tree abundance change ICL?
```

Report paired contrasts:

```text
Delta tree count
Delta mean ICL
Delta best-seed ICL
Delta seed std
paired bootstrap CI
```

## Deliverables

```text
multibase_exact_control_results_report.md
multibase_exact_control_results_report.json
multibase_exact_control_results_table.csv
multibase_exact_control_pairwise_contrasts.csv
```

## Interpretation rules

- If Arm A is positive and Arm B is weak, normal-fan branch geometry is the stronger predictor.
- If Arm B is positive and Arm A is weak, total rooted-tree abundance is the stronger predictor.
- If both are positive, both abundance and branch geometry matter.
- If neither is positive, the current pre-training topology predictors remain weak; mechanism may be primarily training-dependent.

---

# Track 2: Build cross-root and decoder-aware contrast metrics

## Why

The first prospective exact-control test showed that **same-root tree-difference comparison overlap** is not a standalone causal knob. That does not mean tree differences are irrelevant. It may mean the metric was too narrow.

The steady state is normalized across roots:

\[
\bar C_r=\frac{\tau_r}{\sum_s \tau_s}.
\]

The decoder compares species/root concentrations. Therefore, many relevant decision contrasts involve tree-score differences across roots:

\[
T\in\mathcal T_r(G),\qquad T'\in\mathcal T_s(G),\qquad r\ne s.
\]

## Tasks

Implement pre-training metrics that include cross-root and decoder-aware contrasts.

### 1. Cross-root tree-difference coordinate participation

For input coordinate `alpha`, define whether coordinate `alpha` participates in a cross-root tree contrast:

\[
A^{cross}_{T,T',\alpha}=\sum_e |s_T(e)-s_{T'}(e)|\Omega_{e\alpha},
\quad T\in\mathcal T_r,\ T'\in\mathcal T_s,\ r\ne s.
\]

For each comparison pair `(i, q, d)`, compute normalized overlap:

\[
O^{cross}_{i,q,d}=\mathbb E_{r\ne s}\mathbb E_{T\in\mathcal T_r,T'\in\mathcal T_s}
\mathbf 1[A^{cross}_{T,T',i,d}>0]
\mathbf 1[A^{cross}_{T,T',q,d}>0].
\]

Report min/mean/Gini over `(i,d)`.

### 2. Root-pair contrast diversity

For every root pair `(r,s)`, compute:

```text
number of distinct cross-root tree-difference supports
effective rank of cross-root incidence differences
root-pair tree-count product |T_r| |T_s|
root-pair support entropy
```

Aggregate as:

```text
minimum over root pairs
mean over root pairs
entropy / Gini over root pairs
```

### 3. Decoder-agnostic root-pair coverage

Since decoder `B` is learned, do not assume a fixed root-to-label assignment. Instead compute whether each context label could be assigned to at least one root pair with adequate comparison support.

Possible diagnostics:

```text
max root-pair support per context position
minimum across context positions
assignment-score via Hungarian matching between labels and root/root-pair supports
```

### 4. Optional post-training decoder-aware version

If learned decoders and `K` tensors are available, compute weighted metrics using:

```text
absolute decoder weights |B_{ell,r}|
branch-conditioned tree posterior P(T | r,z)
learned |K_{e,alpha}| participation
```

But keep these clearly labeled as post-training mechanism diagnostics, not pre-training predictors.

## Deliverables

```text
cross_root_decoder_contrast_metrics.py
cross_root_decoder_contrast_reanalysis.md
cross_root_decoder_contrast_reanalysis.json
```

## Required comparisons

Compare cross-root metrics against:

```text
same-root tree-difference overlap
normal-fan pair metrics
tree count
active-tree count
repaired gamma diagnostics
existing tree/masked-tree geometry
```

Outcomes:

```text
mean novel-class ICL
best-seed novel-class ICL
seed standard deviation
branch failures, where available
trained branch margin, where available
post-training scramble/ablation sensitivity, where available
```

---

# Track 3: Diagnose why retrospective tree-difference worked but prospective control failed

## Why

The fixed-m20 retrospective data showed strong tree-difference multiplicity prediction, but the first prospective exact-control mask experiment did not support same-root tree-difference overlap as a standalone causal knob.

This mismatch must be explained before making further claims.

## Tasks

Compare the old fixed-m20 masks and the prospective exact-control masks in the same feature space.

Compute for both libraries:

```text
same-root tree-difference overlap
cross-root tree-difference overlap
rooted-tree count
normal-fan active-tree count
branch-tree NMI
edge-load Gini
coordinate-load Gini
input-edge count
physical graph/backbone identity
d_rel
root tree-count balance
condition number / effective rank where available
```

Ask:

1. Was the fixed-m20 tree-difference signal confounded by mask family?
2. Was it confounded by physical backbone?
3. Did the prospective library saturate overlap, so high/low were both above threshold?
4. Did prospective high/low groups differ in normal-fan or tree-count variables despite matching tree-difference overlap?
5. Did same-root overlap fail because cross-root contrast geometry is the actual relevant object?
6. Did seed variance or optimization reliability dominate the prospective contrast?

## Deliverables

```text
retrospective_vs_prospective_tree_difference_diagnostic.md
retrospective_vs_prospective_tree_difference_diagnostic.json
retrospective_vs_prospective_feature_table.csv
```

The final section must say whether tree-difference overlap should be:

```text
retired as a selector
kept as a secondary diagnostic
modified into cross-root / decoder-aware form
or used only within specific mask-family regimes
```

---

# Track 4: Mechanism follow-up on multi-base exact-control models

## Why

Pre-training predictors remain weak. The strongest evidence throughout the project has been post-training mechanism evidence: high-performing trained models rely on branch/projection/tree organization, and statistic-preserving scrambles destroy ICL.

Every new exact-control experiment should therefore include mechanism follow-up on selected models.

## Model selection

From the multi-base experiment, select:

```text
high-ICL models from high tree-count / high normal-fan groups
high-ICL models from low tree-count / high normal-fan groups
high-ICL models from high tree-count / low normal-fan groups
low-ICL matched controls from same base graph / degree sequence where possible
mid-ICL ambiguous cases
```

## Diagnostics

For selected trained models compute:

```text
branch-active-tree MI
branch-to-root MI
tree posterior entropy
branch-specific root concentration separation
projection alignment with comparison directions
trained branch margin
post-training tree-drive range R_{r,b}
input-edge ablation sensitivity
physical-edge ablation sensitivity
```

Run causal interventions:

```text
context-block shuffle
stat-preserving projection scramble
stat-preserving branch-alignment scramble
decoder-root permutation
selected input-edge ablations
selected physical-edge ablations
```

## Deliverables

```text
multibase_mechanism_followup_report.md
multibase_mechanism_followup_report.json
multibase_mechanism_diagnostics.csv
multibase_causal_scramble_results.csv
```

## Interpretation

If pre-training metrics remain weak but mechanism scrambles strongly collapse high-ICL models, the correct claim is:

```text
Topology supplies the computational tree-sum basis, but trained functional organization is the dominant explanatory level.
```

If a pre-training metric predicts which models later show strong branch/projection organization, that metric becomes a serious capacity candidate.

---

# Track 5: Refine gamma only as a diagnostic

## Why

Repaired no-bias `gamma*_ICL` passes analytic toy gates, but it is not predictive in fixed-m20 existing data. It should remain a diagnostic, not a selector, until it predicts held-out exact-control outcomes.

## Tasks

Use repaired gamma to ask narrower questions:

1. Does gamma correlate with best-seed ICL in the multi-base exact-control experiment?
2. Does gamma correlate with branch failures or trained branch margin?
3. Does gamma improve when cross-root/decoder-aware contrast metrics are added?
4. Does gamma fail specifically in graphs with high tree count but poor coefficient controllability?
5. Does gamma explain any analytic or mechanism case studies even if it fails broad prediction?

Do not use gamma alone to select new large sweeps.

## Deliverables

```text
gamma_multibase_diagnostic_report.md
gamma_multibase_diagnostic_report.json
```

---

# Track 6: Optional follow-up designs after multi-base analysis

Only run these after Tracks 1-5 are analyzed.

## Option A: More degree sequences

If the multi-base result suggests a tree-count or normal-fan signal, replicate with additional exact in/out degree sequences.

Goal:

```text
Determine whether the signal survives beyond one degree sequence.
```

## Option B: Paired graph construction

Construct explicit graph pairs that match total rooted-tree count but differ in normal-fan geometry, and pairs that match normal-fan geometry but differ in tree count.

Goal:

```text
Strengthen causal separation beyond library selection.
```

## Option C: Cross-root mask control

If cross-root contrast metrics outperform same-root metrics, run a prospective mask/topology control based on cross-root overlap rather than same-root tree-difference overlap.

Goal:

```text
Test whether decoder-aware contrast geometry is the missing input-mask variable.
```

## Option D: Trainability-focused runs

If best-seed and mean-seed diverge, run more seeds for selected graph groups.

Goal:

```text
Separate expressivity envelope from training reliability.
```

---

# Track 7: Thermodynamics remains delayed

Do not run or claim a thermodynamic `F_max` experiment unless the project explicitly shifts to thermodynamic Markov parameterization.

A valid thermodynamic experiment requires reversible support and a parameterization such as:

\[
W_{ij}=\exp(E_j-B_{ij}+F_{ij}/2+\text{input drive}),
\qquad
B_{ij}=B_{ji},
\qquad
F_{ij}=-F_{ji},
\qquad
|F_{ij}|\le F_{\max}.
\]

Before any sweep, verify:

```text
reversible edge support
detailed-balance behavior at F_max = 0
correct generator convention
stable steady-state solve
valid comparison to first-order CRN-ICL task
```

Until then, say:

```text
Thermodynamics remains untested.
No F_max / entropy-production claim is supported.
```

---

# Required final synthesis

At the end of this task, produce:

```text
post_multibase_exact_control_synthesis.md
post_multibase_exact_control_synthesis.json
```

The synthesis must answer:

1. Does total rooted-tree abundance predict ICL under multi-base exact controls?
2. Does task-aligned normal-fan / active-tree geometry predict ICL after controlling tree count?
3. Do cross-root / decoder-aware contrast metrics improve prediction over same-root tree-difference overlap?
4. Does repaired `gamma*_ICL` predict best-seed expressivity, mean-seed trainability, branch failures, or trained branch margins?
5. Why did fixed-m20 tree-difference overlap predict retrospectively but fail prospectively?
6. Are pre-training predictors strong enough to claim a topology capacity law, or only weak diagnostics?
7. Does post-training branch/projection/tree organization remain the strongest evidence channel?
8. What claims are now supported, weakened, or still open?
9. Was any thermodynamic claim tested? If not, explicitly state that thermodynamics remains untested.

The synthesis must separate claims into:

```text
exact theory
pre-training prediction
expressivity
trainability
post-training mechanism
causal interventions
thermodynamics
```

---

# Acceptance criteria

## Strong positive topology-capacity result

A strong positive result requires that, under multi-base exact controls, one or more pre-training metrics improves prediction of novel-class ICL beyond base controls and survives held-out-base or matched-pair testing. Candidate metrics:

```text
total rooted-tree count
task-aligned normal-fan score
active-tree count
cross-root contrast geometry
cross-root + normal-fan combined metrics
```

## Strong abundance result

If tree count predicts ICL while normal-fan geometry does not, the claim is:

```text
Rooted-tree abundance, not task-aligned normal-fan geometry, is the better current pre-training topology signal.
```

## Strong normal-fan result

If normal-fan geometry predicts ICL at fixed tree count, the claim is:

```text
Task-aligned rooted-tree-polytope / normal-fan geometry predicts ICL beyond abundance and rank.
```

## Mechanism-dominant result

If pre-training metrics remain weak but scrambles strongly collapse trained high-ICL models, the claim is:

```text
Topology determines the tree-sum computational basis, but trained functional organization is the dominant explanatory level.
```

## Useful negative result

If no pre-training metric survives exact controls, the claim is:

```text
No current scalar pre-training metric predicts first-order CRN ICL under exact controls; topology remains mechanistically important but not yet a predictive capacity law.
```

## Invalid result

The result is invalid if:

```text
seed runs are treated as independent topology samples;
thermodynamic claims are made from arbitrary directed graphs;
first-order tree-sum claims are applied to WTA/autocatalytic systems;
broad sweeps are interpreted causally without exact controls;
repaired gamma is claimed as a capacity law without held-out predictive success.
```

---

# Bottom line for the agent

The current project state is:

```text
First-order tree-sum basis: exact.
Raw count and d_rel: incomplete.
Same-root tree-difference overlap: useful retrospectively, failed as a standalone prospective causal knob.
Repaired gamma: analytic-toy valid, not yet predictive.
Normal-fan / active-tree / tree-count geometry: best weak pre-training signal.
Post-training branch/projection dependence: strongest evidence.
Thermodynamics: untested.
```

Your immediate goal is:

```text
Analyze the multi-base exact-control library to determine whether rooted-tree abundance, task-aligned normal-fan geometry, cross-root contrast structure, or none of these can predict first-order CRN ICL under exact controls.
```

Do that before launching any broader topology sweep or thermodynamic extension.
