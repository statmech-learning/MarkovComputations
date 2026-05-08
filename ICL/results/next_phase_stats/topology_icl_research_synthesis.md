# Topological Control of In-Context Learning in First-Order CRNs

This document is a standalone synthesis of the topology-ICL project as of commit `d7f8cdb` on branch `topology`. It explains the scientific question, the theory that motivated the experiments, what was implemented, what was tried, what the concrete results were, and what conclusions are defensible.

The document is intentionally scoped. The results here are about first-order chemical reaction networks with exponential input-dependent rates. They do not establish a topology theory for autocatalytic or winner-take-all CRNs.

## 1. Executive Summary

The project tested whether CRN topology affects in-context learning beyond raw trainable degree count.

The conservative conclusion is:

> In the tested first-order fixed-count regimes, topology-associated structural and functional variables explain residual novel-class ICL variation that raw count does not explain.

The stronger conclusions that are not justified yet are:

- `d_rel` is the final capacity law for first-order CRN ICL.
- Extracted essential motifs are uniquely optimal.
- The first-order matrix-tree topology story applies automatically to autocatalytic or WTA CRNs.

The main empirical pattern is:

- Raw count alone is not enough in fixed-count regimes.
- Pre-training tree geometry gives weak-to-moderate predictive signal.
- Post-training projection/tree organization gives strong explanatory signal.
- Causal scrambling of branch/projection/tree alignment destroys ICL by large margins.
- Essential physical subgraphs can retrain substantially better than sparse input masks, but matched motif controls show that the extracted motifs are not uniquely superior under the current protocol.

The mechanism supported by the evidence is:

```text
reaction graph
-> rooted spanning trees
-> tree-sum projection geometry
-> steady-state concentration geometry
-> branch/projection organization
-> novel-class ICL
```

## 2. Scientific Question

The original CRN-ICL paper argues that CRNs can perform in-context learning by learning input projections that separate comparison subspaces. The open question was whether graph topology matters beyond the raw number of trainable parameters.

The null hypothesis was:

> Once raw trainable degree count and decoder size are controlled, reaction topology adds no predictive power for novel-class ICL.

The alternative hypothesis was:

> First-order CRN topology constrains relative spanning-tree projection geometry, and that geometry affects ICL expressivity, trainability, or both.

This project is not a generic graph-clustering exercise. The graph is studied because, in first-order CRNs, it determines the rooted spanning-tree sums that appear exactly in the steady state.

## 3. Theoretical Foundation

Consider a strongly connected directed reaction graph

```math
G=(V,E), \qquad |V|=N_n,\quad |E|=m.
```

For first-order reactions, the concentration dynamics can be written as

```math
\dot C_n
=
\sum_{m\neq n}
\left(k_{m\to n} C_m-k_{n\to m} C_n\right).
```

The input vector is

```math
z=(z_1,\ldots,z_{N_c},z_q),
```

where context items and the query are concatenated. The model predicts which context item matches the query. The primary metric throughout the project is novel-class ICL accuracy, not training accuracy or ordinary validation accuracy.

For exponential rate encoding,

```math
k_e(z)=\exp(b_e+K_e^\top z).
```

By the matrix-tree theorem, the steady-state concentration for root/species `r` has the form

```math
\bar C_r(z)=\frac{\tau_r(z)}{\sum_s \tau_s(z)},
```

where

```math
\tau_r(z)=\sum_{T\in\mathcal T_r(G)} \prod_{e\in T} k_e(z).
```

With exponential rates, each tree contribution becomes

```math
\prod_{e\in T} k_e(z)
=
\exp\left(
\sum_{e\in T} b_e
+
\left(\sum_{e\in T}K_e\right)^\top z
\right).
```

Define

```math
\beta_T=\sum_{e\in T} b_e,
\qquad
\Theta_T=\sum_{e\in T}K_e.
```

Then

```math
\tau_r(z)
=
\sum_{T\in\mathcal T_r(G)}
\exp(\beta_T+\Theta_T^\top z).
```

This is the central object. The effective projection vectors are not just the learned edge vectors `K_e`; they are the rooted spanning-tree sums `Theta_T`.

Topology matters because it determines which tree sums are possible.

## 4. Structural Quantities

For each rooted spanning tree `T`, define its edge-incidence vector

```math
s_T\in\{0,1\}^m.
```

The tree-sum projection is

```math
\Theta_T = K^\top s_T.
```

Because steady-state concentrations are normalized, common shifts across all trees largely cancel. The more relevant geometry is relative tree-score geometry, represented by tree-incidence differences:

```math
D_G =
\begin{bmatrix}
s_{T_1}-s_{T_0}\\
s_{T_2}-s_{T_0}\\
\vdots
\end{bmatrix}.
```

The basic topology-aware capacity proxy is

```math
d_{\rm rel}(G)=p\,\mathrm{rank}(D_G),
\qquad
p=(N_c+1)D.
```

For an input-encoding mask `Omega`, where `Omega_{e alpha}` says whether input coordinate `alpha` is allowed to modulate edge `e`, the masked quantity is

```math
d_{\rm rel}(G,\Omega)
=
\sum_{\alpha=1}^{p}
\mathrm{rank}\left(D_G\mathrm{diag}(\Omega_{\cdot\alpha})\right).
```

The project also tracked:

- effective rank of tree-difference geometry,
- condition numbers,
- root tree-count balance,
- edge participation heterogeneity,
- input edge-load and coordinate-load heterogeneity,
- comparison-branch rank proxies,
- branch-margin and tree-polytope capacity proxies.

## 5. Branch-Separation View

ICL is not just high rank. It asks whether query/context comparison branches can be separated.

In the tropical or large-projection limit,

```math
\log \tau_r(z)
\approx
\max_{T\in\mathcal T_r(G)}
(\beta_T+\Theta_T^\top z).
```

In this view, each species behaves like a max over rooted-tree projections. A successful model should arrange these projections so that inputs on a given comparison branch activate the right root/tree ensemble and become linearly decodable.

The ideal but still-unsolved capacity object is something like

```math
\gamma^*(G,\Omega)
=
\max_{K,B}
\min_z
\left[
q_{\ell^*(z)}(z)
-
\max_{\ell\neq \ell^*(z)} q_\ell(z)
\right],
```

under norm constraints on `K` and `B`.

The project implemented practical proxies:

- linear branch-margin capacity,
- rank-weighted branch capacity,
- tropical rooted-tree random-feature capacity,
- rooted tree-polytope support summaries,
- sampled normal-fan branch/tree coverage diagnostics.

These are useful probes, not a final capacity theory.

## 6. Experimental Program

The work proceeded in layers.

### 6.1 Implementation and Audit

Implemented or verified:

- arbitrary strongly connected first-order CRN topologies,
- exponential input-dependent rates,
- physical graph topology separate from input-encoding topology,
- rooted spanning-tree structural metrics,
- input masks,
- grouped and clustered inference,
- branch-margin/tree-polytope probes,
- active-tree and projection-alignment diagnostics,
- causal intervention scripts,
- essential motif extraction and retraining,
- matched motif controls,
- strict completion verifier.

The completion audit is in:

```text
ICL/results/next_phase_stats/topology_goal_completion_audit.md
```

The strict verifier passed with:

```bash
python3 ICL/verify_topology_completion.py \
  --experiment next=ICL/results/next_phase_stats \
  --report_md ICL/results/next_phase_stats/next_phase_evidence_report.md \
  --report_json ICL/results/next_phase_stats/next_phase_evidence_report.json \
  --report_kind next_phase \
  --require_expanded_followups
```

### 6.2 Original Fixed-Count Regime

The original controlled regime fixed:

```text
N_n = 6
m = 20
input-coupled count = 200
physical backbones = random, cycle, hub
selected masks/topologies = 16 per backbone
training seeds = 5
total runs = 240
groups = 48
```

This was the cleanest test of topology beyond raw count because raw physical edge count and input-coupled count were controlled.

### 6.3 Expanded and Hard Regimes

Additional pilot and hard regimes were added:

```text
n5_m7_N2_D1:      60 training outputs
n5_m12_N2_D1:     60 training outputs
hard_n4_m6_N3_D2: 60 results, 60 models, 60 mechanisms, 60 causal files
hard_n5_m8_N3_D2: 60 results, 60 models, 60 mechanisms, 60 causal files
hard_n5_m12_N3_D2: 60 results, 60 models, 60 mechanisms, 60 causal files
```

The hard regimes used `N_c=3,D=2` and sparse/intermediate/dense graph settings to stress the structural theory beyond the original `N_n=6,m=20` setup.

### 6.4 Statistical Upgrade

The external critique correctly pointed out that 240 run-level rows are not 240 independent topologies. Training seeds are nested inside topology/mask groups.

The project therefore added:

- group-level regressions,
- leave-one-group-out `R^2`,
- clustered bootstrap delta `R^2`,
- family-cluster bootstrap for hard regimes,
- held-out graph-family prediction where feasible.

This is why the final evidence emphasizes group-level and clustered metrics.

## 7. Results: Original Fixed-Count Regime

The original fixed-count experiments gave the clearest structural result.

### 7.1 Raw Count Was Not Predictive

At topology-group level:

| Model | Groups | Group LOO `R^2` |
| --- | ---: | ---: |
| raw count | 48 | -0.043 |
| raw plus `d_rel` | 48 | 0.145 |
| masked tree geometry | 48 | 0.189 |
| tree geometry | 48 | 0.409 |

This supports the idea that topology-derived tree geometry explains residual ICL variation after raw count is controlled.

The original interpretation report called this a `strong_positive` result:

```text
Topology-derived predictors and trained functional diagnostics both improve over count baselines.
```

### 7.2 Input-Encoding Topology Matters

The input-mask report fixed:

```text
physical edge count = 20
input-coupled parameter count = 200
```

At mask-mean level:

| Candidate | Baseline | Baseline LOO `R^2` | Candidate LOO `R^2` | Delta |
| --- | --- | ---: | ---: | ---: |
| physical backbone | raw counts | -0.043 | 0.172 | 0.215 |
| masked geometry | raw counts | -0.043 | 0.189 | 0.232 |
| mask family label | raw counts | -0.043 | -0.166 | -0.123 |

This says the detailed mask/tree geometry mattered more than a coarse mask-family label.

### 7.3 Post-Training Projection Alignment Was Much Stronger

For the same original fixed-count regime:

| Scope | Baseline | Projection alignment LOO `R^2` |
| --- | ---: | ---: |
| run level | -0.008 | 0.740 |
| topology mean | -0.043 | 0.809 |
| topology best seed | -0.043 | 0.599 |

This is important. Pre-training topology metrics predict some of the variation, but the trained model's functional organization explains much more. That is exactly what one would expect if topology constrains the available geometry while optimization decides whether the model actually uses it.

## 8. Results: Mechanism Correlations

Mechanism metrics correlated strongly with novel-class ICL in the original fixed-count runs.

| Metric | random | cycle | hub |
| --- | ---: | ---: | ---: |
| relative tree dimension | 0.335 | 0.218 | 0.297 |
| masked relative tree effective rank | 0.198 | 0.193 | 0.230 |
| trained branch margin | 0.877 | 0.832 | 0.744 |
| branch-active-tree mutual information | 0.821 | 0.776 | 0.752 |
| posterior matched comparison gap | 0.640 | 0.566 | 0.488 |
| input-coupling ablation max loss | 0.747 | 0.761 | 0.844 |
| physical ablation max loss | 0.348 | 0.426 | 0.899 |

Interpretation:

- Structural metrics are predictive but modest.
- Trained branch margins and branch-active-tree organization are much stronger.
- Ablation losses show that trained models rely on particular input-modulated edges.
- Physical ablation matters most in the hub backbone, consistent with physical bottlenecks being more consequential there.

## 9. Results: Hard-Regime Grouped Inference

The hard regimes were smaller at the group level, with 12 topology groups each, so the numbers are noisier. They still provide useful stress tests.

### 9.1 Hard `n4_m6_N3_D2`

| Model | Group LOO `R^2` | Boot Delta `R^2` | Family Boot Delta `R^2` | Heldout `R^2` |
| --- | ---: | ---: | ---: | ---: |
| raw count | -0.190 | NA | NA | -1.034 |
| raw plus `d_rel` | 0.203 | 0.066 | 0.077 | 0.295 |
| masked tree geometry | -0.151 | 0.096 | 0.090 | 0.224 |
| tropical tree capacity plus `d_rel` | -0.645 | 0.172 | 0.145 | 0.153 |
| normal fan capacity plus `d_rel` | -0.610 | 0.116 | 0.108 | 0.314 |

This regime shows modest positive family-heldout performance for the simpler `d_rel` and normal-fan models, but LOO values for the richer models are unstable because there are only 12 groups.

### 9.2 Hard `n5_m8_N3_D2`

| Model | Group LOO `R^2` | Boot Delta `R^2` | Family Boot Delta `R^2` | Heldout `R^2` |
| --- | ---: | ---: | ---: | ---: |
| raw count | -0.190 | NA | NA | -0.497 |
| raw plus `d_rel` | -0.260 | 0.020 | 0.017 | -0.610 |
| tree geometry | 0.041 | 0.206 | 0.172 | -0.683 |
| masked tree geometry | -0.110 | 0.187 | 0.154 | -0.117 |
| tropical tree capacity plus `d_rel` | -0.960 | 0.216 | 0.173 | -1.979 |
| normal fan capacity plus `d_rel` | -2.161 | 0.202 | 0.171 | -1.571 |

This regime shows bootstrap improvement over raw count but poor held-out-family prediction. The safest interpretation is that topology-associated variables explain within-sample residual structure, but transfer across graph families is not established here.

### 9.3 Hard `n5_m12_N3_D2`

| Model | Group LOO `R^2` | Boot Delta `R^2` | Family Boot Delta `R^2` | Heldout `R^2` |
| --- | ---: | ---: | ---: | ---: |
| raw count | -0.190 | NA | NA | -0.233 |
| raw plus `d_rel` | -0.190 | 0.000 | 0.000 | -0.233 |
| tree geometry | 0.324 | 0.356 | 0.381 | 0.102 |
| masked tree geometry | 0.447 | 0.297 | 0.316 | 0.399 |
| tropical tree capacity plus `d_rel` | -0.251 | 0.393 | 0.417 | -1.499 |
| rooted tree-polytope capacity | 0.081 | 0.307 | 0.337 | -0.235 |
| normal fan capacity | -0.286 | 0.331 | 0.371 | -0.534 |

This is the strongest hard-regime structural result. Tree geometry and masked tree geometry perform clearly better than raw count, and masked tree geometry has positive held-out-family performance.

## 10. Results: Causal Alignment Interventions

The causal interventions ask whether trained models rely on branch/projection organization. The interventions include:

- shuffling context blocks,
- permuting decoder roots,
- permuting learned edge projections,
- permuting edge rate functions,
- randomizing `K` directions.

The table reports mean drops in novel-class ICL accuracy.

| Regime | Context shuffle | Decoder root perm | Edge projection perm | Edge rate perm | Randomize `K` direction |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | -61.95 | -55.75 | -50.65 | -50.61 | -50.82 |
| cycle | -61.34 | -56.77 | -51.31 | -51.53 | -51.14 |
| hub | -52.74 | -49.13 | -41.91 | -43.94 | -44.44 |
| hard `n4_m6_N3_D2` | -48.61 | -36.80 | -27.89 | -34.53 | -36.24 |
| hard `n5_m8_N3_D2` | -58.04 | -33.35 | -35.18 | -40.03 | -40.37 |
| hard `n5_m12_N3_D2` | -73.10 | -44.14 | -50.81 | -52.42 | -52.50 |

These are large drops. They support the claim that high-ICL models use structured context/projection/root organization. They do not merely exploit generic parameter count.

## 11. Results: Branch-Margin and Tree-Polytope Probes

The branch-margin capacity probes were introduced because `d_rel` asks only how many relative tree directions are available. ICL asks whether those directions can separate query-context branches.

In the original fixed-count backbones, the simple linear branch-capacity probe often saturated:

```text
linear accuracy mean = 0.975 for most mask families
coord_block = 0.561
```

In the harder `N_c=3,D=2` regimes, the linear/rank-weighted probe again saturated near:

```text
linear accuracy mean = 0.877
rank-weighted linear mean = 0.877
```

The tropical and normal-fan probes were less saturated and therefore more diagnostic. Examples:

| Regime/family | Tropical accuracy mean | Tropical root effective rank | Rooted support frac | Normal fan tree NMI | Normal fan active trees |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard `n4_m6`, cycle chords | 0.383 | 2.677 | 1.000 | 0.025 | 8.269 |
| hard `n5_m8`, random strongly connected | 0.399 | 2.899 | 1.000 | 0.044 | 16.000 |
| hard `n5_m12`, degree balanced | 0.425 | 3.681 | 1.000 | 0.097 | 68.833 |

Interpretation:

- The simple linear branch probe is too easy and saturates.
- The tropical/rooted-polytope/normal-fan probes are closer to the actual first-order tree mechanism.
- These probes improved the conceptual framing, but they are still approximations rather than the final capacity theory.

## 12. Results: Essential Motif Retraining

Essential subgraphs were extracted from trained models using input-ablation importance. Two retraining experiments were compared:

1. Physical subgraph retraining: keep an extracted physical edge subgraph and retrain from scratch.
2. Input-mask retraining: keep the physical graph but restrict input modulation according to an extracted sparse mask.

The original result was:

| Source | Layout | Source mean ICL | Retrain mean ICL | Retrain best ICL | Retention mean/max |
| --- | --- | ---: | ---: | ---: | ---: |
| random | physical subgraph | 89.57 | 65.69 | 90.00 | 0.733 / 0.822 |
| random | input mask | 88.69 | 55.37 | 72.20 | 0.625 / 0.684 |
| cycle | physical subgraph | 88.93 | 67.40 | 88.60 | 0.758 / 0.834 |
| cycle | input mask | 88.43 | 54.59 | 73.20 | 0.618 / 0.707 |
| hub | physical subgraph | 84.22 | 72.09 | 92.40 | 0.856 / 0.949 |
| hub | input mask | 82.46 | 52.40 | 65.20 | 0.637 / 0.702 |

This suggested that physical topology carries computational structure beyond input sparsity alone.

However, matched motif controls were then added. They compared extracted motifs against random strongly connected and degree-rewired controls matched on coarse statistics.

| Backbone | Control kind | Control mean ICL | Source motif mean ICL | Control-source delta | Control win rate |
| --- | --- | ---: | ---: | ---: | ---: |
| random | degree rewire | 69.83 | 65.69 | 4.15 | 0.875 |
| random | random SC | 68.25 | 65.69 | 2.57 | 0.688 |
| cycle | degree rewire | 66.37 | 67.40 | -1.03 | 0.375 |
| cycle | random SC | 67.94 | 67.40 | 0.54 | 0.562 |
| hub | degree rewire | 71.20 | 72.09 | -0.89 | 0.500 |
| hub | random SC | 75.42 | 72.09 | 3.33 | 0.625 |

So the corrected interpretation is:

- Small physical subgraphs can support ICL much better than sparse input masks under this protocol.
- The extracted motif edge sets are not uniquely superior to matched controls.
- The motif result is evidence for the importance of physical reaction topology as a computational substrate, not yet evidence for a unique discovered motif law.

## 13. What Was Tried and Why

### 13.1 Raw Count Baselines

Why: The original paper emphasized enough learnable degrees of freedom. Any topology claim must beat raw count.

Outcome: In fixed-count regimes, raw count had near-zero or negative predictive power. This made the topology question meaningful.

### 13.2 `d_rel` and Tree Geometry

Why: Matrix-tree theory says relative tree scores control steady-state ratios.

Outcome: `d_rel` and broader tree geometry improved over raw count. The effect was real but not complete.

### 13.3 Input-Encoding Masks

Why: The physical graph and the input-encoding graph are different. Removing input modulation from an edge is not the same as deleting the physical edge.

Outcome: Masked tree geometry predicted mean ICL better than raw count in the original regime and performed well in hard `n5_m12_N3_D2`.

### 13.4 Branch-Margin and Tree-Polytope Probes

Why: Rank is not the same as branch separability.

Outcome: The simplest branch probes saturated. Tropical and normal-fan probes were more informative but still approximate.

### 13.5 Grouped and Clustered Inference

Why: Training seeds are nested inside topology/mask groups; treating all runs as independent overstates confidence.

Outcome: The grouped results still supported the main conclusion in the original regime and parts of the hard regimes. Held-out-family generalization was mixed.

### 13.6 Post-Training Mechanistic Diagnostics

Why: Structural topology says what the graph could use; trained diagnostics say what it did use.

Outcome: Branch margins, branch-active-tree mutual information, projection alignment, and ablation losses strongly tracked ICL.

### 13.7 Causal Interventions

Why: Correlation is not enough. Scrambling branch/projection/root alignment should damage ICL if the proposed mechanism is real.

Outcome: The causal drops were large across original and hard regimes.

### 13.8 Essential Motif Retraining

Why: A dense trained model may use a smaller functional subgraph. Retraining extracted subgraphs tests whether those motifs are expressive and trainable on their own.

Outcome: Physical subgraphs retrained better than sparse input masks, but matched controls prevented overclaiming unique motif discovery.

## 14. Implications

### 14.1 Topology Matters, But Not as a Simple Scalar

The evidence does not support a one-number law like:

```text
ICL succeeds iff d_rel > threshold.
```

Instead, topology appears to matter through a hierarchy:

```text
raw edge/input count
< relative tree rank
< tree spectrum and masking geometry
< branch-margin/tree-polytope coverage
< trained branch/tree/projection organization
< causal functional dependence
```

### 14.2 Structural Metrics Predict Capacity Weakly to Moderately

Pre-training tree metrics help, especially in the original fixed-count regime and hard `n5_m12_N3_D2`.

But structural metrics alone do not fully explain trained ICL. This is expected because the same topology can train into different mechanisms across seeds.

### 14.3 Functional Diagnostics Explain Trained Models Strongly

Projection alignment and branch-active-tree organization explain much more variation than structural metrics.

This supports the mechanism:

```text
successful trained models organize tree-sum projections around comparison branches
```

rather than merely using arbitrary high-dimensional features.

### 14.4 Causal Evidence Supports Branch/Projection Alignment

Scrambling context blocks, decoder roots, edge projections, edge rate functions, or `K` directions causes large accuracy collapses.

This is the strongest mechanistic evidence. It says that the trained networks rely on the organization of projections across the reaction graph.

### 14.5 Physical Topology Is an Active Constraint

Physical subgraph retraining often retained more ICL than input-mask retraining. This supports the idea that physical reaction pathways matter, not just the presence of input-modulated parameters.

But matched controls show that the current motif extraction does not identify uniquely optimal motifs. The safer claim is that small matched physical subgraphs can be sufficient, not that the extracted motifs are special in a universal sense.

## 15. Limitations

1. The theory and experiments are first-order only.

   The matrix-tree theorem gives an exact topology-to-steady-state map only for first-order CRNs. Autocatalytic and WTA networks require different theory.

2. The hard-regime group counts are small.

   Each hard regime has 12 topology groups. This makes rich regressions unstable and held-out-family prediction difficult.

3. Branch-margin capacity is not solved.

   The implemented probes are useful diagnostics, but the true tree-polytope branch-coverage theory remains open.

4. Held-out-family prediction is mixed.

   Some regimes show positive heldout performance; others do not. This limits claims about transfer across graph families.

5. Motif uniqueness is not established.

   Matched controls often perform as well as or better than extracted motifs.

6. Dense training may still help.

   Essential subgraphs can retrain, but retention below 1.0 in many cases means dense graphs may provide optimization help or redundant pathways.

## 16. Defensible Conclusions

The project establishes a strong scoped result:

> In first-order CRNs with exponential input-dependent rates, topology-associated structural and functional variables explain residual novel-class ICL variation beyond raw count in the tested fixed-count regimes.

The project supports the mechanistic interpretation:

> First-order CRN topology constrains relative rooted-spanning-tree projection geometry. Successful trained models exploit this geometry through branch-specific projection/root/tree organization, and disrupting that organization causally damages ICL.

The project does not establish:

> A universal scalar topology capacity law.

or:

> A nonlinear CRN topology theory.

or:

> A unique essential motif family.

## 17. Recommended Next Research Step

The next phase should focus on the weakest remaining theoretical link:

> Replace coarse `d_rel`/rank/spectrum proxies with a sharper tree-polytope branch-coverage theory.

Concretely:

1. Expand fixed-count sweeps across more `N_n`, `m`, `N_c`, and `D`.
2. Increase the number of graph families per hard regime so held-out-family tests are meaningful.
3. Define and optimize a constrained branch-margin capacity:

```math
\gamma^*(G,\Omega)
=
\max_{K,B}
\min_z
\left[
q_{\ell^*(z)}(z)
-
\max_{\ell\neq\ell^*(z)}q_\ell(z)
\right].
```

4. In the tropical limit, study normal-fan coverage of rooted tree polytopes:

```math
\log \tau_r(z)
\approx
\max_{T\in\mathcal T_r(G)}
(\beta_T+\Theta_T^\top z).
```

5. Use causal interventions that preserve coarse statistics while scrambling branch/tree alignment.
6. Compare extracted motifs against controls matched not only on edge count and `d_rel`, but also on branch-margin capacity and normal-fan coverage.

## 18. Where the Evidence Lives

Primary final reports:

```text
ICL/results/next_phase_stats/next_phase_evidence_report.md
ICL/results/next_phase_stats/topology_goal_completion_audit.md
```

Original fixed-count reports:

```text
ICL/results/topology_research_report.md
ICL/results/topology_research_report_interpretation.md
ICL/results/input_mask_topology_report.md
ICL/results/input_mask_topology_report_interpretation.md
```

Hard regime outputs:

```text
ICL/results/expanded_hard_sweeps/n4_m6_N3_D2
ICL/results/expanded_hard_sweeps/n5_m8_N3_D2
ICL/results/expanded_hard_sweeps/n5_m12_N3_D2
```

Core code paths:

```text
ICL/topology_metrics.py
ICL/branch_margin_capacity.py
ICL/clustered_topology_inference.py
ICL/analyze_topology_model.py
ICL/causal_topology_interventions.py
ICL/make_matched_motif_controls.py
ICL/compare_matched_motif_controls.py
ICL/verify_topology_completion.py
ICL/run_expanded_hard_followups.py
```

Final verification state:

```text
local branch: topology @ d7f8cdb
remote branch: origin/topology @ d7f8cdb
strict verifier: passed
```
