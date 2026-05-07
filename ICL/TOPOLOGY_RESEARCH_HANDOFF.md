# Topology-ICL Research Handoff For Theory Critique

Date: 2026-05-07  
Branch: `topology`  
Repository: `statmech-learning/MarkovComputations`  
Primary metric throughout: `test_novel_classes`, the novel-class ICL accuracy.

Live HTTPS summary site:

```text
https://statmech-learning.github.io/MarkovComputations/topology-icl/
```

Verified report artifacts:

```text
ICL/results/topology_research_report.md
ICL/results/topology_research_report.json
ICL/results/topology_research_report_interpretation.md
ICL/results/topology_research_report_interpretation.json
ICL/results/input_mask_topology_report.md
ICL/results/input_mask_topology_report.json
ICL/results/input_mask_topology_report_interpretation.md
ICL/results/input_mask_topology_report_interpretation.json
```

Cluster worktree where the verified reports were generated:

```text
/home/aadarwal/repos/topology/ICL
```

The raw SLURM run directories and trained model outputs remain on Engaging.
The git repo contains the code, tests, verified Markdown reports, verified JSON
reports, and the static HTTPS site.

## 1. Question We Tested

The original theory asked whether first-order CRN topology controls in-context
learning beyond raw trainable degree count.

The null hypothesis was:

```text
At fixed N_n, N_c, D, physical edge count, input-coupled parameter count,
decoder size, and training protocol, topology-derived metrics add no predictive
power for novel-class ICL.
```

The alternative hypothesis was:

```text
First-order CRN topology changes the relative spanning-tree projection geometry
available to the steady state, and this geometry predicts and mechanistically
explains ICL beyond raw count.
```

The key first-order theory is the matrix-tree representation. For a strongly
connected directed reaction graph,

```text
Cbar_r(z) = tau_r(z) / sum_s tau_s(z)
tau_r(z) = sum_{T in T_r(G)} prod_{e in T} k_e(z)
```

With exponential rate encoding,

```text
k_e(z) = exp(b_e + K_e^T z)
prod_{e in T} k_e(z) = exp(beta_T + Theta_T^T z)
Theta_T = sum_{e in T} K_e
```

Thus the effective projections are not individual edge vectors `K_e`, but
tree-sum vectors `Theta_T`. Because the steady state is normalized, the more
relevant object is relative tree geometry, for example differences
`s_T - s_T0` in tree-incidence space and the rank/spectrum of the corresponding
tree-difference matrix.

The implemented project follows this chain:

```text
reaction graph
-> rooted spanning trees
-> relative tree-sum projection geometry
-> steady-state concentration geometry
-> linear decodability
-> novel-class ICL
```

## 2. What Was Built

### First-order topology-controlled CRN

Implemented explicit physical reaction topology and input-encoding topology:

```text
ICL/models/topology_markov_icl.py
ICL/run_topology_icl.py
ICL/input_mask_utils.py
```

Conventions:

- Physical reactions are directed edges `(source, target)`.
- The Markov generator uses `W[target, source] = k_{source->target}`.
- Columns of `W` sum to zero.
- Steady states solve `W p = 0`.
- Physical topology is the edge list.
- Input-encoding topology is a binary mask `input_mask[e, alpha]` saying which
  input coordinate may modulate edge `e`.
- Physical edge deletion and input-coupling deletion are kept separate.

### Matrix-tree structural metrics

Implemented in:

```text
ICL/topology_metrics.py
```

The metrics include:

- strong-connectivity checks,
- rooted arborescence enumeration,
- tree-incidence matrices,
- relative tree rank,
- `d_rel`,
- effective rank and condition number of tree-difference geometry,
- masked relative geometry under input-encoding masks,
- root tree-count balance,
- edge participation heterogeneity,
- mean shortest path,
- branch-aware comparison-capacity metrics,
- context/query common-subspace ranks,
- input-overlap support metrics.

The branch-aware metrics were added because global rank can be too coarse:
ICL needs capacity on query-context comparison branches, not just high total
dimension.

### Controlled graph and mask libraries

Implemented in:

```text
ICL/make_topology_library.py
ICL/make_input_mask_library.py
ICL/submit_topology_library_sweep.py
ICL/submit_topology_phase1.py
```

The final verified experiment used three fixed physical backbones:

```text
random_sc_n6_m20_seed3
cycle_chords_n6_m20_seed3
hub_spoke_n6_m20_seed63
```

Each backbone had:

```text
n_nodes = 6
n_edges = 20
input_coupled_parameter_count = 200
selected masks/topologies = 16
training seeds per selected item = 5
runs per backbone = 80
pooled source runs = 240
topology/mask seed-aggregate groups = 48
```

Mask families included:

```text
balanced
coord_block
edge_block
entry_random
high_participation_edges
low_participation_edges
```

The fixed-count design matters: raw edge count and raw input-coupled parameter
count cannot explain within-regime variation because they are fixed by
construction.

### Training, collection, aggregation, and regression

Implemented in:

```text
ICL/collect_topology_results.py
ICL/aggregate_topology_seeds.py
ICL/regress_topology_results.py
ICL/finalize_topology_sweep.py
ICL/make_input_mask_report.py
ICL/make_topology_research_report.py
ICL/finalize_topology_research_report.py
ICL/interpret_topology_report.py
```

The analysis distinguishes:

- run-level accuracy,
- topology/mask mean across seeds, interpreted mainly as trainability or
  optimization reliability,
- topology/mask best seed, interpreted more as expressivity,
- seed standard deviation, interpreted as optimization reliability.

### Post-training mechanism analysis

Implemented in:

```text
ICL/topology_analysis.py
ICL/analyze_topology_model.py
ICL/submit_topology_mechanisms.py
ICL/collect_mechanism_results.py
ICL/summarize_topology_mechanisms.py
```

Mechanism diagnostics include:

- active root and active tree assignment,
- branch-to-root mutual information,
- branch-to-tree mutual information,
- tree assignment purity and normalized MI,
- branch margins,
- target log-probability margins,
- tree-projection alignment with comparison directions,
- posterior matched comparison gaps,
- input-coupling ablations,
- physical edge ablations,
- edge sensitivity and functional edge importance.

### Essential physical subgraphs and input masks

Implemented in:

```text
ICL/extract_essential_subgraphs.py
ICL/extract_essential_input_masks.py
ICL/compare_essential_retrains.py
ICL/finalize_essential_physical_retrains.py
ICL/finalize_essential_inputmask_retrains.py
ICL/recover_essential_physical_retrains.py
ICL/recover_essential_inputmask_retrains.py
```

Two different essential-structure analyses were run:

1. Physical essential subgraphs:
   prune physical reaction edges based on trained-model importance while
   preserving strong connectivity where required, then retrain the resulting
   physical motif from scratch.

2. Essential input masks:
   keep the physical graph fixed, prune input-coupling rows/entries, then
   retrain the sparse input-encoding mask from scratch.

This distinction is central. Physical topology and input-encoding topology are
not treated as the same object.

### Audit, recovery, and verification

Implemented in:

```text
ICL/audit_topology_artifacts.py
ICL/verify_topology_completion.py
ICL/TOPOLOGY_README.md
ICL/TOPOLOGY_STATUS.md
ICL/TOPOLOGY_COMPLETION_AUDIT.md
```

The final reports were generated only after strict audit and verifier passes.
The verifiers check source results, mechanism outputs, extracted essential
structures, retrain outputs, report JSON consistency, and whether both physical
essential subgraphs and input-mask retrains are represented in the consolidated
research report.

## 3. Experiment Coverage

The consolidated verified report has:

| experiment | runs | groups | m | mean ICL | best ICL | mean seed std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 80 | 16 | 20 | 76.76 | 94.60 | 7.08 |
| cycle | 80 | 16 | 20 | 76.38 | 94.20 | 8.37 |
| hub | 80 | 16 | 20 | 69.87 | 91.40 | 8.66 |

The topology library summaries were:

| experiment | selected/candidates | families | d_rel values | mean effective rank | mean edge gini |
| --- | ---: | --- | --- | ---: | ---: |
| random | 16/322 | balanced, coord_block, edge_block, entry_random | 190, 200 | 180.084 | 0.105 |
| cycle | 16/322 | balanced, coord_block, edge_block, entry_random, high_participation_edges, low_participation_edges | 190, 200 | 179.991 | 0.095 |
| hub | 16/322 | balanced, coord_block, edge_block, entry_random, high_participation_edges, low_participation_edges | 190, 200 | 172.855 | 0.217 |

## 4. Main Result

The final conservative interpretation is:

```text
strong_positive
```

Meaning:

```text
Topology-derived predictors and trained functional diagnostics both improve
over count baselines.
```

The support rule used by the interpretation script was:

```text
candidate LOO R2 - baseline LOO R2 >= 0.05
and n >= 6
```

## 5. Count-Control Results

### Pooled run-level results

| model | n | LOO R2 | interpretation |
| --- | ---: | ---: | --- |
| edge_count | 240 | -0.008 | raw count explains essentially nothing |
| edge_plus_drel | 240 | 0.057 | relative tree dimension improves over count |
| input_plus_masked_geometry | 240 | 0.116 | masked tree geometry improves further |
| edge_plus_projection | 240 | 0.740 | post-training projection diagnostics are strongly explanatory |

### Pooled topology-mean results

| model | n | LOO R2 | interpretation |
| --- | ---: | ---: | --- |
| edge_count | 48 | -0.043 | count baseline fails |
| edge_plus_drel | 48 | 0.145 | pre-training relative tree dimension helps |
| input_plus_masked_geometry | 48 | 0.189 | masked pre-training geometry helps more |
| edge_plus_projection | 48 | 0.809 | trained projection diagnostics explain most topology-mean variation |

### Pooled topology-best results

| model | n | LOO R2 | interpretation |
| --- | ---: | ---: | --- |
| edge_count | 48 | -0.043 | count baseline fails |
| edge_plus_drel | 48 | 0.180 | relative tree dimension helps best-seed expressivity |
| input_plus_masked_geometry | 48 | -0.027 | masked geometry did not generalize in this best-seed regression |
| edge_plus_projection | 48 | 0.599 | trained projection diagnostics remain strong |

Important nuance: masked geometry is not uniformly strong. It helps run-level
and topology-mean prediction, but not the topology-best LOO regression in the
consolidated report. This should be interpreted as evidence that masked
geometry is a useful coarse pre-training proxy, not a complete capacity theory.

## 6. Fixed-Physical Input-Mask Results

The focused input-mask report holds both physical edge count and input-coupled
parameter count fixed. This is the cleanest count-control slice.

### Run-level fixed-input-count regressions

| model | n | LOO R2 |
| --- | ---: | ---: |
| raw_counts | 240 | -0.008 |
| physical_backbone | 240 | 0.068 |
| mask_family | 240 | 0.030 |
| physical_plus_family | 240 | 0.118 |
| d_rel | 240 | 0.057 |
| masked_geometry | 240 | 0.116 |
| physical_plus_masked_geometry | 240 | 0.138 |

### Mask-mean fixed-input-count regressions

| model | n | LOO R2 |
| --- | ---: | ---: |
| raw_counts | 48 | -0.043 |
| physical_backbone | 48 | 0.172 |
| mask_family | 48 | -0.166 |
| physical_plus_family | 48 | 0.309 |
| d_rel | 48 | 0.145 |
| masked_geometry | 48 | 0.189 |
| physical_plus_masked_geometry | 48 | 0.327 |

This is a strong argument against the simplest raw-degree-count explanation.
When count is held fixed, physical backbone and masked tree geometry still
explain residual novel-class ICL variation.

## 7. Mechanistic Results

Run-level Pearson correlations with novel-class ICL:

| metric | random | cycle | hub |
| --- | ---: | ---: | ---: |
| relative tree dimension `d_rel` | 0.335 | 0.218 | 0.297 |
| weakest context/query common tree-contrast rank | 0.120 | 0.179 | 0.255 |
| weakest comparison-branch paired rank upper bound | 0.301 | 0.224 | 0.280 |
| comparison-branch rank imbalance | -0.313 | -0.234 | -0.284 |
| masked relative tree effective rank | 0.198 | 0.193 | 0.230 |
| input mask coordinate-load heterogeneity | -0.248 | -0.229 | -0.267 |
| trained branch margin | 0.877 | 0.832 | 0.744 |
| branch-active-tree MI | 0.821 | 0.776 | 0.752 |
| posterior matched comparison gap | 0.640 | 0.566 | 0.488 |
| input-coupling ablation max loss | 0.747 | 0.761 | 0.844 |
| physical ablation max loss | 0.348 | 0.426 | 0.899 |

Interpretation:

- Pre-training topology metrics are positively related to ICL, but modestly.
- Post-training branch margin and branch-active-tree MI are strongly related
  to ICL.
- Successful trained models appear to organize computation by branch-specific
  tree/root structure.
- Ablation losses correlate strongly with ICL, especially input-coupling
  ablations and hub physical ablations.

Important caution: post-training mechanism metrics are explanatory diagnostics,
not pure pre-training predictors. They show what trained models used; they do
not by themselves prove that those structures were inevitable from topology
alone.

## 8. Essential Motif Retraining

Input-ablation 50%-coverage essential physical subgraphs or input-encoding
masks were extracted and retrained from scratch.

### Aggregate retention

| source experiment | type | joined motifs | source mean ICL | retrain mean ICL | retrain best ICL | retention mean/max | motif size |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| random | physical subgraph | 16 | 89.57 | 65.69 | 90.00 | 0.733/0.822 | edges 12.69/9/17 |
| random | input mask | 16 | 88.69 | 55.37 | 72.20 | 0.625/0.684 | edges 20.00/20/20 |
| cycle | physical subgraph | 16 | 88.93 | 67.40 | 88.60 | 0.758/0.834 | edges 13.25/10/16 |
| cycle | input mask | 16 | 88.43 | 54.59 | 73.20 | 0.618/0.707 | edges 20.00/20/20 |
| hub | physical subgraph | 16 | 84.22 | 72.09 | 92.40 | 0.856/0.949 | edges 15.25/12/19 |
| hub | input mask | 16 | 82.46 | 52.40 | 65.20 | 0.637/0.702 | edges 20.00/20/20 |

Physical essential subgraphs retrain substantially better than sparse input
masks alone. This suggests that the physical reaction pathways themselves carry
important computational structure, not merely the count of input-coupled
parameters.

### Best retrained physical motifs

| source | motif | size | d_rel | source ICL | retrain mean | retrain max | retention mean/max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | `essential_input_ablation_loss_n6_m16_0012` | 16 | 300 | 90.20 | 79.84 | 90.00 | 0.885/0.998 |
| cycle | `essential_input_ablation_loss_n6_m15_0043` | 15 | 280 | 93.40 | 80.40 | 88.60 | 0.861/0.949 |
| hub | `essential_input_ablation_loss_n6_m16_0067` | 16 | 300 | 89.60 | 86.00 | 92.40 | 0.960/1.031 |
| hub | `essential_input_ablation_loss_n6_m18_0016` | 18 | 340 | 81.80 | 83.68 | 92.20 | 1.023/1.127 |

The hub case includes motifs whose retrain performance exceeds the source
model average, which is useful evidence that dense original graphs may aid
discovery/training but are not always required for final computation.

### Essential input-mask retraining

Input-mask retrains retained less:

| source | retrain mean | retrain best | retention mean/max |
| --- | ---: | ---: | ---: |
| random | 55.37 | 72.20 | 0.625/0.684 |
| cycle | 54.59 | 73.20 | 0.618/0.707 |
| hub | 52.40 | 65.20 | 0.637/0.702 |

This does not mean input-encoding topology is irrelevant. It means the
particular sparse input masks extracted by input ablation were not sufficient
to recover the full trained computation from scratch as reliably as physical
subgraphs.

## 9. Scientific Interpretation

The cleanest supported statement is:

```text
In the tested first-order CRN regime, topology affects ICL beyond raw degree
count. The effect is visible in relative spanning-tree geometry before
training, and it becomes much clearer in trained-model active-tree,
projection-alignment, margin, ablation, and essential-motif analyses.
```

More specifically:

1. Raw count is not enough.
   In fixed-count regimes, raw edge/input-coupled parameter count has negative
   or near-zero leave-one-out R2.

2. Relative tree geometry helps.
   `d_rel` improves run-level and topology-level prediction over count. The
   improvement is modest but consistent enough to meet the report's
   pre-registered support threshold.

3. Masked/input-aware geometry helps but is incomplete.
   Masked geometry improves run-level and topology-mean prediction, especially
   when combined with physical backbone, but it is not uniformly reliable for
   topology-best prediction and can overfit small group counts when too many
   predictors are used.

4. Functional tree organization is strong.
   Branch-active-tree MI and trained branch margin correlate strongly with
   novel-class ICL. This supports the proposed mechanism that successful
   models assign comparison branches to active tree/root ensembles.

5. Physical motifs matter.
   Essential physical subgraphs retrain from scratch with much higher retention
   than sparse input masks. This suggests the physical graph is not merely a
   passive carrier for input-coupling degrees of freedom.

6. Dense graphs may aid training.
   Retention below 1.0 for most motifs means the dense source graph often
   provides optimization help or redundant pathways even when a smaller motif
   can express much of the computation.

## 10. What This Does Not Prove

The results should not be overclaimed.

1. This is first-order only.
   The matrix-tree theorem controls the first-order system. These results do
   not automatically transfer to autocatalytic or WTA CRNs.

2. The graph scale is limited.
   The final verified fixed-count study is centered on `n_nodes=6`,
   `n_edges=20`, and `input_coupled_parameter_count=200`. The theory should be
   tested across additional `N_n`, `m`, `N_c`, and `D`.

3. Pre-training metrics are not a complete theory.
   `d_rel` and masked geometry add predictive power, but they do not explain
   most variance by themselves. Post-training projection diagnostics explain
   much more.

4. Some regressions have small group counts.
   Pooled run-level `n=240` and topology-level `n=48` are the more reliable
   summaries. Per-family LOO regressions over 16 masks can be unstable and
   should mainly be read as diagnostics.

5. Functional diagnostics are partly circular.
   Margins, active-tree MI, projection alignment, and ablation loss are measured
   after training. They are excellent mechanism evidence, but they are not
   independent pre-training predictors.

6. Essential subgraph extraction is conditional on a trained model.
   A motif that works after extraction is evidence of a usable computational
   structure, not necessarily evidence that the motif is easy to find from
   scratch in all settings.

7. The raw model outputs are not committed.
   The repo contains verified summaries and report JSON. Full raw run
   directories remain on Engaging.

## 11. Suggested Critique Targets

The theory author should focus critique on these points.

### A. Is `d_rel` the right first-order capacity proxy?

The current `d_rel` and masked tree-geometry metrics are useful but coarse. A
stronger theory may need a branch-margin capacity measure:

```text
max over K,B of min branch margin under norm constraints
```

or an approximation to that optimization.

### B. Are the branch-aware ranks measuring the right comparison subspaces?

We added common-subspace and input-overlap metrics because global rank can miss
query/context pairing. The critic should inspect whether these are the right
linear-algebraic proxies for comparison branch coverage or whether a different
subspace arrangement is needed.

### C. How should root normalization be handled?

The implementation uses relative tree-difference geometry, but the final
logits depend on normalized steady-state concentrations and decoder `B`.
There may be additional equivalence classes or degeneracies beyond the current
rank/spectrum metrics.

### D. Does active-tree MI really identify the mechanism?

High branch-active-tree MI is strong evidence of branch-specific organization,
but it does not rule out distributed tree ensembles. The critic should compare
active-tree assignments with posterior entropy and tree ensemble weights.

### E. Why do physical subgraphs retrain better than input masks?

This is one of the most interesting results. Possible explanations:

- physical motifs preserve coherent tree-sum geometry,
- input masks preserve physical paths but damage comparison-aligned projection
  directions,
- physical motifs regularize optimization better,
- input-mask extraction selected edges by source-model importance but not by
  from-scratch trainability.

### F. How much is topology versus family/backbone identity?

The fixed-input report shows `physical_backbone` and
`physical_plus_masked_geometry` help. The next critique should ask whether
backbone identity is standing in for a deeper metric not yet included.

## 12. Concrete Next Experiments

Recommended next experiments:

1. Scale graph size and edge count.
   Repeat the fixed-count design for more `n_nodes` and `m` values to see
   whether the `d_rel` effect strengthens, saturates, or disappears.

2. Add norm-controlled branch-margin probes.
   Approximate `gamma*(G)` on sampled branches using tree-sum features and
   compare it to `d_rel`.

3. Stratify by tree-count balance and bottleneck participation.
   Current physical families differ in edge participation. Build matched
   libraries that isolate participation heterogeneity while holding rank fixed.

4. Train extracted physical motifs from scratch at larger seed counts.
   Some motifs retained nearly all source performance. More seeds can separate
   expressivity from trainability.

5. Compare active-tree and soft-posterior mechanisms.
   For successful models with high ICL but lower active-tree purity, test
   whether the computation is distributed over tree ensembles.

6. Re-run the same graph with permuted input masks.
   This would more sharply separate physical topology from input-encoding
   topology.

7. Extend only after first-order theory is tightened.
   Do not apply the matrix-tree rank story directly to autocatalytic or WTA
   systems without deriving the corresponding nonlinear geometry.

## 13. How To Reproduce The Verified Summary

Local dependency-light checks:

```bash
python3 -m unittest discover -s ICL/tests
python3 -m py_compile $(find ICL -name '*.py' -not -path '*/__pycache__/*')
git diff --check
```

These currently pass locally. The most recent full local run reported:

```text
Ran 85 tests in 2.616s
OK
```

Final report generation on the cluster was done through the guarded finalizer:

```bash
cd /home/aadarwal/repos/topology/ICL

python3 finalize_topology_research_report.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --output_md results/topology_research_report.md \
  --output_json results/topology_research_report.json
```

Focused input-mask reports were verified as:

```bash
python3 verify_topology_completion.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --report_kind input_mask \
  --report_md results/input_mask_topology_report.md \
  --report_json results/input_mask_topology_report.json
```

The consolidated research verifier audits both physical and input-mask
essential retrain layouts and requires both to appear in the final report JSON.

## 14. File Map For Review

Core theory and metrics:

```text
ICL/topology_metrics.py
ICL/topology_analysis.py
ICL/branch_margin_capacity.py
ICL/collect_branch_margin_capacity.py
```

`branch_margin_capacity.py` is the first implementation of the proposed
branch-margin proxy. It samples exact-copy branches, gates squared
query/context comparison features by common relative tree-contrast support, and
reports oracle plus norm-controlled linear margins. It is intentionally a
pre-training proxy, not a solution to the full nonconvex `max_{K,B}` capacity
problem.

`collect_branch_margin_capacity.py` applies this probe to every selected row
in a topology or input-mask library CSV, producing a flat predictor table that
can be joined to training results.

Model and training:

```text
ICL/models/topology_markov_icl.py
ICL/run_topology_icl.py
```

Library generation:

```text
ICL/make_topology_library.py
ICL/make_topology_sweep_plan.py
ICL/make_input_mask_library.py
ICL/input_mask_utils.py
```

The physical graph library now supports the original `cycle_chords`,
`random_sc`, `hub_spoke`, and `two_module` families plus next-phase
`degree_balanced`, `bottleneck_bridge`, and `redundant_paths` families for
broader fixed-count sweeps.

`make_topology_sweep_plan.py` generates the next expanded fixed-count regime
matrix across `N_n`, edge regime, `N_c`, and `D`, with library-generation
commands and dry-run training-array commands that should be inspected before
cluster submission.

Cluster orchestration:

```text
ICL/submit_topology_phase1.py
ICL/submit_topology_library_sweep.py
ICL/submit_topology_mechanisms.py
```

Post-training analysis:

```text
ICL/analyze_topology_model.py
ICL/causal_topology_interventions.py
ICL/submit_causal_interventions.py
ICL/collect_causal_interventions.py
ICL/collect_mechanism_results.py
ICL/summarize_topology_mechanisms.py
```

`causal_topology_interventions.py` is the first causal mechanism probe. It
loads a trained run, evaluates a fixed novel-class batch, then measures
accuracy and mechanism deltas after context-block coordinate shuffles, edge
projection/rate-function permutations, decoder root permutations, and random
`K`-direction controls that preserve effective row norms. This is the intended
test for whether branch/tree alignment is functional rather than merely
correlated with ICL.

`submit_causal_interventions.py` runs that probe across completed trained runs
through a dry-run-safe SLURM array path.

`collect_causal_interventions.py` flattens completed intervention reports into
CSV and summary JSON so accuracy drops can be compared across topology/mask
groups and intervention types.

Essential motifs:

```text
ICL/extract_essential_subgraphs.py
ICL/extract_essential_input_masks.py
ICL/make_matched_motif_controls.py
ICL/compare_essential_retrains.py
ICL/recover_essential_physical_retrains.py
ICL/recover_essential_inputmask_retrains.py
```

`make_matched_motif_controls.py` is the first matched-control baseline for the
essential physical motif claim. It samples random and directed degree-rewired
strongly connected controls with the same `N_n` and `m`, scores them against
the extracted motif on coarse tree-geometry features, and writes a retrainable
`selected.csv` for the standard library sweep path.

Reports and verification:

```text
ICL/make_input_mask_report.py
ICL/make_topology_research_report.py
ICL/finalize_topology_research_report.py
ICL/interpret_topology_report.py
ICL/clustered_topology_inference.py
ICL/TOPOLOGY_THEORY_AUDIT.md
ICL/audit_topology_artifacts.py
ICL/verify_topology_completion.py
```

`TOPOLOGY_THEORY_AUDIT.md` is the first mandatory next-phase audit. It records
that tree orientation, strong-connectivity handling, structural mask selection,
and novel-class metric plumbing are sound, while trainable base-rate biases,
nested seed dependence, and unmatched motif controls remain explicit caveats.

`clustered_topology_inference.py` is the first implementation of the
hierarchical-statistics upgrade requested by the critique. It does not add a
heavy mixed-effects dependency, but it avoids treating seed rows as independent
topologies by reporting group-level regressions, cluster-bootstrap deltas,
leave-one-backbone-out prediction, and residual decomposition by topology/mask
group.

Tests:

```text
ICL/tests/
```

Hosted site:

```text
docs/topology-icl/
.github/workflows/topology-icl-pages.yml
```

## 15. Bottom Line

The project now supports a positive but scoped conclusion:

```text
In the tested first-order fixed-count regimes, topology-associated structural
and functional variables explain residual variation in novel-class ICL that raw
trainable count does not explain. The clearest pre-training signal is relative
spanning-tree geometry, and the clearest trained-model mechanism signal is
branch-specific active tree/projection organization. Essential physical motifs
retain more ICL than sparse input-encoding masks under the current extraction
and retraining protocol, but this motif result still needs matched controls
before it can be treated as a causal topology claim.
```

The strongest open theoretical task is to replace coarse rank/spectrum proxies
with a branch-margin or tree-polytope capacity theory that predicts which
topologies are expressive and which are merely trainable.
