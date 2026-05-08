# Topological Control of In-Context Learning in First-Order CRNs

This document explains the topology-ICL direction from first principles, what was implemented, what experiments were run, what the audit checked, and what the results mean. The goal is not to overstate the conclusions. The goal is to make clear what has actually been achieved, why each step follows from the theory, and how the evidence should be interpreted.

The scope is important:

> Everything here is about first-order CRNs with exponential input-dependent rates.

The matrix-tree theorem gives an exact topology-to-steady-state map for first-order networks. That is why this setting is mathematically clean. The results here should not be automatically transferred to autocatalytic or winner-take-all CRNs, whose steady-state geometry is governed by different nonlinear mechanisms.

---

## 1. Original Scientific Question

The original CRN-ICL paper showed that chemical reaction networks can perform in-context learning without transformer-style attention. The model takes a context/query vector, maps it into reaction rates, relaxes to a steady state, and decodes that steady state into a prediction.

The paper's mechanism was subspace projection: the learned reaction-rate projections separate regions of input space corresponding to which context item matches the query.

The open question was:

> Does reaction topology affect ICL beyond the raw number of trainable degrees of freedom?

The null hypothesis was:

> If the number of trainable input-rate parameters and decoder size are fixed, topology adds no predictive power.

The alternative hypothesis was:

> In first-order CRNs, topology constrains the rooted spanning-tree projection geometry available to the steady state, and that geometry affects ICL capacity, trainability, or both.

The goal was therefore not to cluster graph adjacencies or make a vague statement that topology matters. The goal was to connect topology to the exact mathematical object controlling first-order CRN steady states.

---

## 2. Why The Matrix-Tree Theorem Is The Right Starting Point

Consider a strongly connected directed first-order reaction graph

```math
G=(V,E), \qquad |V|=N_n, \quad |E|=m.
```

The input is the concatenated context/query vector

```math
z=(z_1,\ldots,z_{N_c},z_q),
```

and the model predicts which context item matches the query. The primary metric is novel-class ICL accuracy, because this tests whether the model learned contextual matching rather than memorizing training classes.

For a first-order CRN with exponential input-dependent rates,

```math
k_e(z)=\exp(b_e+K_e^\top z),
```

the matrix-tree theorem gives the steady-state concentration of species/root `r` as

```math
\bar C_r(z)=\frac{\tau_r(z)}{\sum_s\tau_s(z)},
```

where

```math
\tau_r(z)=\sum_{T\in\mathcal T_r(G)}\prod_{e\in T} k_e(z).
```

Here `T_r(G)` is the set of rooted directed spanning trees ending at root `r`.

Substituting the exponential rates gives

```math
\prod_{e\in T} k_e(z)
=
\exp\left(
\sum_{e\in T}b_e+
\left(\sum_{e\in T}K_e\right)^\top z
\right).
```

Define

```math
\beta_T=\sum_{e\in T}b_e,
\qquad
\Theta_T=\sum_{e\in T}K_e.
```

Then

```math
\tau_r(z)=
\sum_{T\in\mathcal T_r(G)}
\exp(\beta_T+\Theta_T^\top z).
```

This is the central theoretical object.

The effective projections available to the CRN are not only the edge vectors `K_e`. They are the rooted tree-sum vectors

```math
\Theta_T=\sum_{e\in T}K_e.
```

So topology matters, if it matters, because topology determines which spanning trees exist and therefore which sums of edge projections can appear in the steady-state computation.

That gives the project its conceptual chain:

```text
reaction graph
-> rooted spanning trees
-> tree-sum projection vectors
-> steady-state concentration geometry
-> linear decodability
-> novel-class ICL
```

---

## 3. First Structural Proxy: Relative Tree Geometry

For each rooted tree `T`, define a tree-incidence vector

```math
s_T\in\{0,1\}^m,
```

where `s_T[e]=1` if edge `e` appears in tree `T`.

The tree projection is

```math
\Theta_T=K^\top s_T.
```

But the steady state is normalized by the sum over roots. Therefore, common shifts to all tree scores mostly cancel. What matters is relative tree-score geometry. We represent that by tree-incidence differences:

```math
D_G=
\begin{bmatrix}
s_{T_1}-s_{T_0}\\
s_{T_2}-s_{T_0}\\
\vdots
\end{bmatrix}.
```

The basic topology-aware degree proxy is

```math
d_{\rm rel}(G)=p\,\mathrm{rank}(D_G),
\qquad
p=(N_c+1)D.
```

For input-encoding masks `Omega`, where `Omega_{e alpha}` says whether input coordinate `alpha` may modulate edge `e`, the masked version is

```math
d_{\rm rel}(G,\Omega)
=
\sum_{\alpha=1}^{p}
\mathrm{rank}\left(D_G\operatorname{diag}(\Omega_{\cdot\alpha})\right).
```

This was the first topology-aware replacement for raw parameter count.

But the project also treated `d_rel` as incomplete from the beginning. ICL is not merely the question, "How many relative tree directions exist?" It is the sharper question, "Can those directions separate the query/context comparison branches?"

---

## 4. Sharper Capacity Target: Branch-Margin Geometry

The stronger theory target is a graph/mask branch-margin capacity:

```math
\gamma^*(G,\Omega)
=
\max_{K,B}
\min_{b\in\mathcal B}
\mathrm{margin}_b(K,B;G,\Omega),
```

where `B` is the set of comparison branches. This capacity must include norm constraints on `K` and `B`; otherwise the margin can be made artificially large by scaling.

In the tropical or large-projection limit,

```math
\log \tau_r(z)
\approx
\max_{T\in\mathcal T_r(G)}
(\beta_T+\Theta_T^\top z).
```

So each root behaves like a max over rooted-tree projections. In that view, the relevant geometric object is the collection of rooted tree polytopes and their normal fans. A successful topology should allow learned tree projections whose normal cones cover the ICL comparison branches in a decodable way.

The code implemented practical approximations to this ideal object:

- linear branch-margin capacity,
- rank-weighted branch capacity,
- tropical rooted-tree random-feature capacity,
- rooted tree-polytope support summaries,
- normal-fan branch/tree coverage diagnostics,
- randomized and optimized `gamma_star` margin proxies under projection/decoder norm constraints.

These are not the final theory. They are probes that move beyond `d_rel` toward the desired capacity object.

---

## 5. Main Implementation Artifacts

The project added or used the following core machinery:

```text
ICL/topology_metrics.py
ICL/branch_margin_capacity.py
ICL/collect_branch_margin_capacity.py
ICL/clustered_topology_inference.py
ICL/analyze_topology_model.py
ICL/causal_topology_interventions.py
ICL/make_mechanism_isolation_plan.py
ICL/make_matched_motif_controls.py
ICL/compare_matched_motif_controls.py
ICL/verify_topology_completion.py
```

The main reports and evidence artifacts are:

```text
ICL/results/next_phase_stats/topology_icl_research_synthesis.md
ICL/results/next_phase_stats/next_phase_evidence_report.md
ICL/results/next_phase_stats/topology_goal_completion_audit.md
ICL/results/next_phase_stats/mechanism_isolation_evidence.md
ICL/results/next_phase_stats/stat_preserving_causal_stratified/summary.md
ICL/results/next_phase_stats/degree_rewire_normal_fan_n5_m12_N3_D2/normal_fan_training_results.md
```

The latest committed state after the work described here is:

```text
branch: topology
latest pushed commit: f8097d33
```

---

## 6. Original Fixed-Count Regime

The first controlled regime fixed:

```text
N_n = 6
m = 20
input-coupled count = 200
physical backbones = random, cycle, hub
selected masks/topologies = 16 per backbone
training seeds = 5
total runs = 240
topology/mask groups = 48
```

This regime tested whether topology-associated variables explain variation when raw physical edge count and input-coupled count are controlled.

At topology-group level, the predictive results were:

| Model | Groups | Group LOO R2 |
| --- | ---: | ---: |
| raw count | 48 | -0.043 |
| raw plus `d_rel` | 48 | 0.145 |
| masked tree geometry | 48 | 0.189 |
| tree geometry | 48 | 0.409 |

The interpretation is straightforward:

- raw count alone did not explain novel-class ICL variation;
- `d_rel` improved over raw count;
- richer tree geometry improved more;
- therefore topology-associated tree geometry contains information that raw parameter count misses in this fixed-count regime.

This does not prove a universal topology law. It shows that the raw degree-count story is incomplete in this tested setting.

---

## 7. Input-Encoding Topology

The project separated physical topology from input-encoding topology.

Physical topology asks which reactions exist. Input-encoding topology asks which input coordinates are allowed to modulate which reactions.

The input-mask study fixed:

```text
physical edge count = 20
input-coupled parameter count = 200
```

At mask-mean level:

| Candidate | Baseline | Baseline LOO R2 | Candidate LOO R2 | Delta |
| --- | --- | ---: | ---: | ---: |
| physical backbone | raw counts | -0.043 | 0.172 | 0.215 |
| masked geometry | raw counts | -0.043 | 0.189 | 0.232 |
| mask family label | raw counts | -0.043 | -0.166 | -0.123 |

The important point is that the detailed masked tree geometry mattered more than a coarse mask-family label. So the evidence was not merely that "some masks are better." It was that the mask's interaction with relative tree geometry had predictive signal.

---

## 8. Post-Training Mechanism Diagnostics

Pre-training topology says what a graph could use. Post-training diagnostics ask what the trained model actually used.

Projection-alignment diagnostics were much stronger than pre-training structural metrics:

| Scope | Baseline LOO R2 | Projection alignment LOO R2 |
| --- | ---: | ---: |
| run level | -0.008 | 0.740 |
| topology mean | -0.043 | 0.809 |
| topology best seed | -0.043 | 0.599 |

This is a key result.

It suggests the following picture:

```text
topology constrains the available tree-sum geometry,
but optimization determines whether a trained model organizes that geometry around the ICL branches.
```

So the mechanism is not fully explained by pre-training graph metrics. The trained representation matters substantially.

---

## 9. Hard Regimes Across More `N_n`, `m`, `N_c`, `D`

The next phase added harder regimes to avoid relying only on the original `N_n=6,m=20` setup.

The checked hard regimes include:

```text
hard_n4_m6_N3_D2
hard_n5_m8_N3_D2
hard_n5_m12_N3_D2
```

Each has 12 topology groups and 60 trained runs. These are small at the group level, so their regressions are noisier than the original 48-group regime.

The strongest hard-regime structural result was `hard_n5_m12_N3_D2`:

| Model | Group LOO R2 | Boot Delta R2 | Family Boot Delta R2 | Heldout R2 |
| --- | ---: | ---: | ---: | ---: |
| raw count | -0.190 | NA | NA | -0.233 |
| raw plus `d_rel` | -0.190 | 0.000 | 0.000 | -0.233 |
| tree geometry | 0.324 | 0.356 | 0.381 | 0.102 |
| masked tree geometry | 0.447 | 0.297 | 0.316 | 0.399 |
| rooted tree-polytope capacity | 0.081 | 0.307 | 0.337 | -0.235 |

This says that, in this harder setting, tree geometry and masked tree geometry again beat raw count. The held-out-family result was positive for masked tree geometry in this regime.

The other hard regimes were more mixed. That is useful rather than disappointing: it says topology signal is real in some controlled regimes, but it is not yet a universal, simple predictor across all graph families and scales.

---

## 10. `gamma_star` Capacity Probes

The `gamma_star` work was introduced because `d_rel` is too coarse. The goal was to approximate a norm-constrained branch-margin capacity and compare it against rank/tree proxies.

Two versions were implemented:

1. a randomized bounded `gamma_star` proxy;
2. an optimized Torch-based `gamma_star` proxy.

For the optimized pilot on `hard_n5_m12_N3_D2`:

| Predictor family | LOO R2 |
| --- | ---: |
| raw count | -0.190 |
| tree geometry | 0.324 |
| masked tree geometry | 0.447 |
| rooted tree-polytope | 0.081 |
| normal fan | -0.072 |
| optimized gamma capacity | -0.564 |

The optimized `gamma_star` proxy improved over the random gamma proxy as infrastructure, but it did not yet outperform tree geometry or masked tree geometry as a held-out predictor.

That result is important because it prevents overclaiming. We now have machinery for norm-constrained capacity probing, but the present surrogate is not the final capacity theory. The next improvement should target the exact branch-min margin more directly, because the current smooth surrogate can optimize mean branch behavior without improving the held-out low-percentile branch margin.

---

## 11. Mechanism-Isolating Contrasts

The external critique correctly said that broad sweeps are less useful unless they isolate specific mechanisms. We therefore created contrast evidence for the requested fixed-statistic comparisons.

The report is:

```text
ICL/results/next_phase_stats/mechanism_isolation_evidence.md
```

### 11.1 Same `d_rel`, Different Edge Participation

In `n5_m12_N3_D2`, the largest available same-`d_rel` edge-participation contrast was:

| Side | Topology | Edge participation Gini | Mean ICL | Best ICL |
| --- | --- | ---: | ---: | ---: |
| low | `g0040_cycle_chords_seed41` | 0.080 | 83.240 | 91.200 |
| high | `g0082_cycle_chords_seed83` | 0.317 | 83.000 | 90.400 |

Mean ICL delta, high minus low: `-0.240` percentage points.

This shows that edge participation heterogeneity alone did not drive ICL in this tiny contrast. That is a useful negative/neutral result.

### 11.2 Same Total Tree Count, Different Root Balance

In `n4_m6_N3_D2`, holding total rooted-tree count fixed:

| Side | Topology | Root tree-count Gini | Mean ICL | Best ICL |
| --- | --- | ---: | ---: | ---: |
| low | `g0004_cycle_chords_seed5` | 0.000 | 67.600 | 78.600 |
| high | `g0010_cycle_chords_seed12` | 0.313 | 71.360 | 80.000 |

Mean ICL delta, high minus low: `+3.760` points.

This means root imbalance was not simply bad in this small contrast. Again, this argues against a simplistic scalar law.

### 11.3 Same Physical Graph, Different Input Masks

For fixed physical graphs with fixed input-coupled count and fixed `d_rel`, input-load structure still changed outcomes.

Example, random physical graph:

| Side | Mask | Edge-load Gini | Mean ICL | Best ICL |
| --- | --- | ---: | ---: | ---: |
| low | balanced mask | 0.000 | 79.920 | 94.600 |
| high | edge-block mask | 0.500 | 86.880 | 93.400 |

Mean ICL delta: `+6.960` points.

This shows that, even with the same physical graph and same `d_rel`, how input coordinates are distributed across edges can matter.

### 11.4 Same Mask Count, Different Coordinate Load

Again using fixed physical graph, fixed input count, and fixed `d_rel`, coordinate-load heterogeneity gave nontrivial changes.

Example, random physical graph:

| Side | Mask | Coordinate-load Gini | Mean ICL | Best ICL |
| --- | --- | ---: | ---: | ---: |
| low | edge-block mask | 0.000 | 86.880 | 93.400 |
| high | entry-random mask | 0.135 | 70.400 | 85.200 |

Mean ICL delta: `-16.480` points.

This is one of the clearer examples that raw count and even `d_rel` are not sufficient to describe the input-encoding geometry.

---

## 12. Exact Degree Sequence, Different Normal Fan

The mechanism-isolation audit initially found this contrast missing. Existing data did not contain two topologies with the same exact degree sequence, same `d_rel`, and different normal-fan geometry.

So we built a targeted follow-up rather than another broad sweep.

The new artifact is:

```text
ICL/results/next_phase_stats/degree_rewire_normal_fan_n5_m12_N3_D2/
```

The construction was:

- choose a base `N_n=5,m=12,N_c=3,D=2` topology;
- generate 32 variants by directed double-edge swaps;
- preserve each node's exact in-degree and out-degree;
- preserve `d_rel=88` across all variants;
- compute normal-fan and branch/tree capacity probes;
- select four normal-fan extremes;
- train each for five seeds, giving 20 total runs.

The library summary:

| Property | Value |
| --- | --- |
| variants | 32 |
| `N_n` | 5 |
| `m` | 12 |
| `N_c` | 3 |
| `D` | 2 |
| exact in-degree sequence | `[3,2,2,3,2]` |
| exact out-degree sequence | `[3,2,2,3,2]` |
| `d_rel` values | `[88]` |
| root tree-count Gini range | `0.0` to `0.0` |
| edge participation Gini range | `0.100` to `0.162` |
| rooted tree count range | `55` to `85` |

The trained outcomes were:

| Group | Mean ICL | Best ICL | Seed Std | Branch-Tree NMI | Active Tree Count | Edge Participation Gini | Rooted Trees |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed2` | 84.320 | 94.400 | 8.764 | 0.136435 | 39.250 | 0.121212 | 55 |
| `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed3` | 89.360 | 97.600 | 10.209 | 0.160333 | 50.750 | 0.137255 | 85 |
| `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed31` | 97.480 | 98.400 | 1.063 | 0.155424 | 52.750 | 0.137255 | 85 |
| `degree_rewire_n5_m12_base_g0354_degree_balanced_seed9_seed34` | 86.760 | 96.000 | 8.686 | 0.135706 | 45.250 | 0.161538 | 65 |

The fixed-statistic contrasts were:

| Contrast Metric | Metric Delta | Mean ICL Delta | Best ICL Delta |
| --- | ---: | ---: | ---: |
| branch-tree normal-fan NMI | 0.024627 | +2.600 | +1.600 |
| active-tree count | 13.500 | +13.160 | +4.000 |

This is not a large statistical study because `n=4` topology groups. It is a constructive pilot. It shows that after holding exact degree sequence and `d_rel` fixed, normal-fan/tree-polytope diagnostics can vary, and the trained ICL outcomes can vary with them.

The strongest interpretation is not "normal-fan active tree count is now the capacity law." The grounded interpretation is:

> The exact-degree/d_rel controls are possible, and this pilot identifies a promising mechanism-isolating design for the next larger sweep.

---

## 13. Statistic-Preserving Causal Scrambles

Correlation does not prove mechanism. So the causal target was to scramble branch/tree or projection alignment while preserving coarse statistics.

The relevant code is:

```text
ICL/causal_topology_interventions.py
```

The intervention definitions explicitly preserve:

- physical graph,
- root tree counts,
- `d_rel`,
- total input-mask density,
- input-mask support where relevant,
- per-edge projection row norms for the projection scramble.

The stratified pilot evaluated six trained models from `n5_m12_N3_D2`: two low, two mid, and two high novel-class ICL runs.

| Bucket | Label | Reported Novel | Sampled Baseline | Branch-Align Delta Mean | Projection Delta Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| low | `g0267_hub_spoke_seed76_trainseed4` | 65.8 | 68.33 | -50.83 | -39.58 |
| low | `g0289_two_module_seed2_trainseed5` | 67.4 | 73.33 | -68.33 | -30.00 |
| mid | `g0040_cycle_chords_seed41_trainseed3` | 85.4 | 88.33 | -76.67 | -55.42 |
| mid | `g0187_random_sc_seed92_trainseed1` | 85.4 | 90.00 | -76.67 | -53.33 |
| high | `g0187_random_sc_seed92_trainseed2` | 98.0 | 98.33 | -95.83 | -67.92 |
| high | `g0354_degree_balanced_seed9_trainseed4` | 98.6 | 98.33 | -96.67 | -64.17 |

This is one of the strongest mechanistic results because the interventions preserve many coarse statistics but damage ICL heavily.

The grounded interpretation is:

> The trained models rely on the alignment between input branches, learned projections, and tree/root organization. Preserving coarse statistics while scrambling that alignment causes large accuracy drops.

Again, this is a pilot, not a final causal theorem. But it is much harder to dismiss than ordinary correlation.

---

## 14. Essential Motif Retraining

The project also asked whether dense trained graphs use smaller functional subgraphs.

Two retraining protocols were compared:

1. physical subgraph retraining, where important physical edges are retained and the model is retrained;
2. sparse input-mask retraining, where the physical graph stays but input modulation is restricted.

The original physical-subgraph retraining retained more ICL than sparse input masks:

| Source | Layout | Source Mean ICL | Retrain Mean ICL | Retrain Best ICL | Retention Mean/Max |
| --- | --- | ---: | ---: | ---: | ---: |
| random | physical subgraph | 89.57 | 65.69 | 90.00 | 0.733 / 0.822 |
| random | input mask | 88.69 | 55.37 | 72.20 | 0.625 / 0.684 |
| cycle | physical subgraph | 88.93 | 67.40 | 88.60 | 0.758 / 0.834 |
| cycle | input mask | 88.43 | 54.59 | 73.20 | 0.618 / 0.707 |
| hub | physical subgraph | 84.22 | 72.09 | 92.40 | 0.856 / 0.949 |
| hub | input mask | 82.46 | 52.40 | 65.20 | 0.637 / 0.702 |

This suggested that physical reaction topology can serve as an active computational substrate.

However, matched controls softened the claim:

| Backbone | Control Kind | Control Mean ICL | Source Motif Mean ICL | Control-Source Delta | Control Win Rate |
| --- | --- | ---: | ---: | ---: | ---: |
| random | degree rewire | 69.83 | 65.69 | +4.15 | 0.875 |
| random | random SC | 68.25 | 65.69 | +2.57 | 0.688 |
| cycle | degree rewire | 66.37 | 67.40 | -1.03 | 0.375 |
| cycle | random SC | 67.94 | 67.40 | +0.54 | 0.562 |
| hub | degree rewire | 71.20 | 72.09 | -0.89 | 0.500 |
| hub | random SC | 75.42 | 72.09 | +3.33 | 0.625 |

The corrected interpretation is:

> Small physical subgraphs can support ICL better than sparse input masks under this extraction protocol, but the extracted motifs are not uniquely superior to matched controls.

That is still useful: it says physical topology matters as a substrate, but we should not claim discovery of a universal motif family.

---

## 15. What The Audit Checked

The final audit mapped the active objective to concrete artifacts. The checklist passed with no missing items.

The audit verified:

| Requirement | Evidence |
| --- | --- |
| norm-constrained `gamma_star` capacity code exists | `ICL/branch_margin_capacity.py` |
| `gamma_star` compared against `d_rel` | `hard_n5_m12_gamma_star_opt_regression.json` |
| fixed-count hard regimes exist | `expanded_hard_sweeps/*/topology_seed_aggregates.csv` |
| same `d_rel`, different bottleneck/participation contrast | `mechanism_isolation_evidence.json` |
| same tree count, different root balance contrast | `mechanism_isolation_evidence.json` |
| same physical graph, permuted input masks contrast | `mechanism_isolation_evidence.json` |
| same mask count, different coordinate-load contrast | `mechanism_isolation_evidence.json` |
| same degree sequence, different normal-fan library | `degree_rewire_normal_fan_n5_m12_N3_D2/library_summary.json` |
| same degree sequence, different normal-fan capacity | `degree_rewire_normal_fan_n5_m12_N3_D2/branch_margin_capacity.csv` |
| same degree sequence, different normal-fan trained outcomes | `degree_rewire_normal_fan_n5_m12_N3_D2/normal_fan_training_summary.json` |
| statistic-preserving causal scrambles | `stat_preserving_causal_stratified/summary.json` |
| final explanatory reports | synthesis, evidence report, normal-fan training report |

The audit result was:

```text
MISSING none
```

The tracked worktree was clean after the final commit.

---

## 16. What Has Actually Been Achieved

The project has achieved five concrete things.

### 16.1 It connected topology to the exact first-order steady-state formula

The topology story is now grounded in the matrix-tree theorem. The relevant projections are rooted tree sums, not isolated edge vectors.

### 16.2 It showed that raw trainable count is incomplete in tested fixed-count regimes

In the original fixed-count setting, raw count had negative group-level LOO `R2`, while tree geometry and masked tree geometry had positive predictive value.

### 16.3 It separated physical topology from input-encoding topology

The experiments distinguish deleting physical edges from removing input modulation on edges. This matters because the physical graph controls the spanning trees, while the input mask controls which input coordinates can influence those tree scores.

### 16.4 It moved from rank proxies toward capacity probes

The `gamma_star`, tropical, and normal-fan probes are imperfect, but they establish infrastructure for asking the sharper branch-margin question.

### 16.5 It added mechanism-isolating and causal evidence

The project now has targeted contrasts and statistic-preserving scrambles. These are more informative than broad graph sweeps because they isolate specific topology-associated variables.

---

## 17. What The Results Imply

The safest interpretation is:

> In the tested first-order fixed-count regimes, topology-associated structural and functional variables explain residual novel-class ICL variation that raw trainable count does not explain.

A more mechanistic interpretation, still scoped to first-order CRNs, is:

> First-order CRN topology constrains the relative rooted-spanning-tree projection geometry available to the steady state. Successful trained models appear to exploit this geometry through branch-specific projection/root/tree organization. Scrambling that organization while preserving coarse statistics substantially damages ICL.

The results do not imply:

- `d_rel` is the final capacity law;
- normal-fan active-tree count is the final capacity law;
- topology alone determines ICL;
- extracted motifs are uniquely optimal;
- the theory applies automatically to autocatalytic or WTA CRNs.

The best current mental model is hierarchical:

```text
raw count
< relative tree rank
< masked tree spectrum and branch support
< tree-polytope / normal-fan branch coverage
< trained projection-root-tree organization
< causal dependence on that organization
```

The project has evidence at every level of that hierarchy, but the final branch-margin capacity theory remains open.

---

## 18. What Should Come Next

The most valuable next step is not simply more data. It is a larger version of the targeted exact-degree/normal-fan design.

The next sweep should:

- generate many exact-degree rewires per base graph;
- keep `N_n`, `m`, `N_c`, `D`, input count, degree sequence, and `d_rel` fixed;
- deliberately vary tree-polytope and normal-fan diagnostics;
- train enough topology groups for meaningful grouped inference;
- compare normal-fan/branch-margin predictors against `d_rel` and raw count;
- keep using statistic-preserving causal scrambles on trained high- and mid-ICL models.

The theoretical next step is to improve the `gamma_star` capacity object so it better approximates

```math
\max_{K,B}\min_{b\in\mathcal B}\mathrm{margin}_b(K,B;G,\Omega)
```

under explicit norm constraints, rather than relying on smooth surrogate objectives that may not optimize worst-branch margins.

---

## 19. Bottom Line

The project has not discovered a universal topology law for CRN ICL. What it has done is more precise:

1. It grounded first-order CRN topology in the matrix-tree theorem.
2. It showed that raw degree count is insufficient in controlled regimes.
3. It built structural, capacity, mechanism, and causal probes around rooted spanning-tree geometry.
4. It ran fixed-count and mechanism-isolating experiments, including an exact-degree-sequence normal-fan pilot.
5. It found that trained ICL depends strongly on branch/projection/tree alignment.

So the current state is best summarized as:

> We have a defensible first-order topology-mediated ICL research program. The evidence supports the relevance of rooted spanning-tree geometry and trained branch/tree organization, while also showing that no single coarse scalar such as `d_rel` is sufficient. The next theoretical target is a sharper norm-constrained branch-margin/tree-polytope capacity.
