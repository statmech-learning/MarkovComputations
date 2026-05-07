# Topology-Aware First-Order CRN ICL

This directory contains the first implementation slice for studying whether
first-order CRN topology controls in-context learning beyond raw parameter
count.

The convention is:

- Physical reactions are directed edges `(source, target)`.
- The Markov generator uses `W[target, source] = k_{source->target}`.
- Columns of `W` sum to zero and steady states solve `W p = 0`.
- Physical topology is separate from input-encoding topology. The edge list
  says which basal reactions exist; `input_mask[e, alpha]` says which input
  coordinates may modulate edge `e`.

## Files

- `topology_metrics.py`: graph families, strong-connectivity checks, rooted
  arborescence enumeration, tree-incidence matrices, relative tree rank,
  effective rank, branch-wise comparison capacity, tree-count balance, and
  edge participation metrics. Supported controlled families include
  `cycle_chords`, `random_sc`, `hub_spoke`, `two_module`,
  `degree_balanced`, `bottleneck_bridge`, and `redundant_paths`.
- `models/topology_markov_icl.py`: first-order ICL model on an explicit directed
  reaction graph.
- `run_topology_icl.py`: train one topology-controlled run and save
  `results.pkl`, `topology.json`, `topology_metrics.json`, and `config.json`.
- `submit_topology_phase1.py`: SLURM array generator for matched graph-family
  sweeps.
- `make_topology_library.py`: generate strongly connected fixed-`m` topology
  candidates, compute pre-training matrix-tree metrics, and select a
  structurally diverse subset for training.
- `make_topology_sweep_plan.py`: write a multi-regime fixed-count sweep plan
  across `N_n`, `m`, `N_c`, and `D`, including library-generation and dry-run
  submission commands for the expanded graph families.
- `input_mask_utils.py`: validate edge-order-aligned binary input masks and
  summarize input-encoding support.
- `make_input_mask_library.py`: hold the physical graph fixed, generate
  matched-count input-encoding masks, compute masked relative tree geometry,
  and write a `selected.csv` consumable by the library sweep submitter.
- `submit_topology_library_sweep.py`: SLURM array generator for training the
  selected topology library through `run_topology_icl.py --edge_json`, with
  optional `input_mask_json` rows for explicit input-encoding topology.
- `topology_analysis.py`: post-training active-tree, tree-projection
  alignment, branch-wise margin, and edge-sensitivity utilities.
- `branch_margin_capacity.py`: pre-training sampled branch-margin capacity
  probe. It gates exact-copy comparison features by common context/query
  relative tree-contrast support, then reports oracle and norm-controlled
  linear margins. This is a conservative proxy for the proposed
  tree-polytope/branch-margin theory, not a solution to the full nonconvex
  `max_{K,B}` CRN capacity problem.
- `collect_branch_margin_capacity.py`: compute the branch-margin capacity
  probe for selected rows in a topology or input-mask library CSV and write a
  flat predictor table.
- `join_branch_margin_capacity.py`: left-join branch-margin capacity predictor
  tables onto seed-level topology results, with capacity columns prefixed for
  direct use by `regress_topology_results.py` and
  `clustered_topology_inference.py`.
- `TOPOLOGY_THEORY_AUDIT.md`: first mandatory implementation audit for the
  next theory phase. It checks tree orientation, trainable bias treatment,
  strong-connectivity handling, pre-training selection leakage, novel-class
  metric use, motif-matching limitations, and the current statistical/capacity
  caveats.
- `analyze_topology_model.py`: load a trained run and write
  `mechanism_metrics.json`.
- `causal_topology_interventions.py`: load a trained run and evaluate causal
  alignment-scrambling interventions on a fixed novel-class batch, including
  context-coordinate shuffles, edge projection/rate-function permutations,
  decoder row permutations, and random direction controls that preserve
  effective row norms.
- `collect_causal_interventions.py`: collect `causal_interventions.json` files
  into a flat CSV and summary JSON for mechanism-intervention reports.
- `submit_causal_interventions.py`: SLURM array generator for running causal
  intervention reports over completed trained topology runs.
- `submit_topology_mechanisms.py`: SLURM array generator for post-training
  active-tree, edge-sensitivity, and ablation analysis over completed runs.
- `finalize_topology_sweep.py`: convenience wrapper that collects completed
  training runs, runs regressions, optionally submits/collects mechanism
  analyses, and refreshes topology-level seed aggregates.
- `run_expanded_hard_followups.py`: guarded one-command wrapper for the current
  expanded hard pilots. It submits or collects mechanism/causal follow-ups,
  refreshes the next-phase report, and can run the strict expanded-follow-up
  verifier. It refuses to finalize source-light roots with no raw `results.pkl`
  files unless explicitly overridden.
- `collect_topology_results.py`: collect completed run directories into a flat
  CSV for regressions against raw degree count and topology-derived metrics.
- `collect_mechanism_results.py`: collect mechanism-analysis JSON files into a
  flat CSV table, including branch-to-root/tree assignment purity summaries
  and normalized mutual information from stored per-sample active roots and
  active trees. When per-sample margins are available, it also reports
  worst-branch mean margin and worst-branch accuracy.
- `summarize_topology_mechanisms.py`: join topology and mechanism result CSVs,
  then report overall and within-edge-count correlations.
- `aggregate_topology_seeds.py`: aggregate repeated training seeds for each
  topology into mean, best-seed, and seed-variance reports.
- `extract_essential_subgraphs.py`: convert trained-model edge importance or
  ablation scores into strongly connected candidate motifs that can be retrained
  through `submit_topology_library_sweep.py`.
- `extract_essential_input_masks.py`: convert trained-model edge importance or
  ablation scores into sparse input-encoding masks while keeping the physical
  reaction graph fixed.
- `compare_essential_retrains.py`: join extracted physical motif or input-mask
  source metadata with from-scratch retrain aggregates and report performance
  retention.
- `make_matched_motif_controls.py`: generate matched random or degree-rewired
  physical controls for extracted essential subgraphs, scored against each
  source motif on coarse tree-geometry features, and write a `selected.csv`
  that can be retrained through the standard library sweep submitter.
- `finalize_essential_inputmask_retrains.py`: guarded finalizer for extracted
  essential input-mask retrains; it refuses to collect incomplete retrain sets
  unless `--allow_partial` is supplied.
- `finalize_essential_physical_retrains.py`: guarded finalizer for extracted
  physical essential-subgraph retrains; it refreshes the physical retrain
  aggregate and comparison artifacts but leaves consolidated report generation
  as an explicit later step.
- `recover_essential_physical_retrains.py`: conservative wrapper for
  interrupted physical essential-subgraph retrains; it audits the physical
  layout, writes status manifests, submits only missing retrain runs when
  requested, and calls the guarded physical finalizer only after completion.
- `recover_essential_inputmask_retrains.py`: conservative wrapper for
  interrupted essential input-mask retrains; it audits source artifacts, writes
  status manifests, optionally submits only missing retrain runs, and calls the
  guarded finalizer, completion verifier, and interpretation script only after
  strict retrain audits.
- `make_input_mask_report.py`: focused report for fixed-physical-graph
  input-mask sweeps, including masked tree geometry, mechanism predictors,
  extracted essential masks, and retrain retention.
- `make_topology_research_report.py`: consolidate fixed-edge sweeps,
  mechanism summaries, seed aggregates, physical essential-subgraph retrains,
  and essential input-mask retrains into one Markdown/JSON progress report.
- `make_next_phase_evidence_report.py`: compact follow-up report builder for
  clustered inference, branch-margin capacity probes, causal interventions,
  and expanded pilot sweep status.
- `refresh_next_phase_report.py`: targeted report refresher for long-running
  cluster follow-ups. It updates only labeled sections supplied on the command
  line, preserves older report sections when source artifacts are not present
  in the active checkout, and re-renders the Markdown with the canonical
  next-phase report builder.
- `finalize_topology_research_report.py`: report-scoped finalizer that rebuilds
  the consolidated research report, runs the strict research verifier, and
  writes the conservative H0/H1 interpretation.
- `audit_topology_artifacts.py`: read-only audit of source runs, mechanism
  outputs, extracted essential masks, retrain outputs, manifests, and comparison
  files. Use this before recovering interrupted cluster arrays or running
  guarded finalizers. It supports input masks by default and physical essential
  subgraphs with `--essential_kind physical`.
- `verify_topology_completion.py`: final non-mutating completion gate that runs
  the strict artifact audit and validates either focused input-mask reports
  (`--report_kind input_mask`) or consolidated topology research reports
  (`--report_kind research`).
- `interpret_topology_report.py`: read a verified report JSON and write a
  conservative H0/H1 evidence summary comparing count baselines, topology
  predictors, mechanism predictors, and essential-retrain retention.
- `regress_topology_results.py`: dependency-light OLS diagnostics for testing
  whether tree-geometry and branch-margin capacity predictors improve on raw
  parameter count.
- `clustered_topology_inference.py`: dependency-light statistical upgrade for
  nested seed data. It reports group-level regressions, clustered bootstrap
  deltas over topology/mask groups, leave-one-family/backbone-out prediction,
  and random-intercept-style residual decomposition.
- `tests/`: dependency-light unit coverage for matrix-tree metrics, input-mask
  validation, topology/input-mask library generation, collection, regression,
  seed aggregation, mechanism summaries, essential subgraph/mask extraction,
  retrain comparison, reporting, artifact audit, and SLURM dry-run wrappers.

## Local Structural Checks

These do not require Torch:

```bash
python3 -m unittest discover -s ICL/tests
python3 ICL/submit_topology_phase1.py --phase smoke --dry-run
```

The unittest suite is intentionally synthetic and pure Python. It checks the
analysis/control plane used after cluster training: selected topology and input
mask CSVs are retrainable, comparison-branch capacity metrics survive
collection and reporting, mechanism summaries expose branch margins and
active-tree diagnostics, interrupted retrain arrays can be audited/requeued
without overwriting completed runs, and final reports recognize both
`essential_input50` physical subgraphs and `essential_inputmask50` input masks.

## Single Smoke Training Run

Run this in an environment with Torch:

```bash
cd ICL
python3 run_topology_icl.py \
  --output results/topology_smoke/random_sc_n4_m8_seed1 \
  --topology_family random_sc \
  --n_nodes 4 \
  --n_edges 8 \
  --seed 1 \
  --topology_seed 1 \
  --epochs 2 \
  --train_samples 80 \
  --val_samples 40 \
  --eval_frequency 1 \
  --n_eval_samples 20 \
  --test_samples 40 \
  --no_progress
```

Then run the mechanism pass:

```bash
python3 analyze_topology_model.py \
  --run_dir results/topology_smoke/random_sc_n4_m8_seed1 \
  --n_samples 40
```

The mechanism file includes branch-active-root/tree mutual information,
active-tree entropy, learned tree-sum projection alignment with
context-query comparison directions, matrix-tree edge sensitivity, an
essential-edge summary, and target log-probability margins on novel-class ICL
samples.

For heavier post-training diagnostics, add ablation flags:

```bash
python3 analyze_topology_model.py \
  --run_dir results/topology_smoke/random_sc_n4_m8_seed1 \
  --n_samples 200 \
  --ablate_input \
  --ablate_physical
```

## Branch-Margin Capacity Probe

The current global rank proxy `d_rel` is intentionally coarse: it measures how
many relative tree directions are available, not whether those directions can
separate query/context comparison branches. `branch_margin_capacity.py` is the
first lightweight pre-training probe for that gap.

Example:

```bash
python3 branch_margin_capacity.py \
  --edge_json results/input_mask_library_n6_m20_c200/topologies/random_sc_n6_m20_seed3.json \
  --input_mask_json results/input_mask_library_n6_m20_c200/masks/mask0001.json \
  --n_context 4 \
  --z_dim 4 \
  --train_samples 2000 \
  --test_samples 2000 \
  --output_json results/branch_margin_probe.json \
  --output_md results/branch_margin_probe.md
```

The probe samples exact-copy branches, computes per-branch/per-coordinate
common context/query support in the relative tree-contrast map, gates
`-(z_i-z_q)^2` comparison features by that support, and reports:

- oracle branch-comparison accuracy and margins,
- norm-controlled linear ridge accuracy and margins,
- common-rank support by branch and coordinate,
- sampled tropical rooted-tree random-feature separability,
- the corresponding `d_rel` and branch-rank metrics.

Use it as a branch-specific topology predictor to compare against `d_rel`; do
not interpret it as proof that a trained CRN will realize the optimum.

For a whole selected library:

```bash
python3 collect_branch_margin_capacity.py \
  --library_csv results/expanded_pilot_libraries/n5_m7_N2_D1/selected.csv \
  --output_csv results/expanded_pilot_libraries/n5_m7_N2_D1/branch_margin_capacity.csv \
  --output_json results/expanded_pilot_libraries/n5_m7_N2_D1/branch_margin_capacity_summary.json \
  --N 2 \
  --D 1 \
  --tree_feature_trials 12
```

The tropical random-feature fields are closer to the tree-polytope hypothesis
than the squared-distance support fields: they sample edge projections, compute
root-wise max or log-sum-exp tree scores, normalize root log weights, and fit a
linear decoder on those root features. Treat them as a stochastic lower-bound
probe for branch separability under the actual rooted-tree incidence structure.
They are not an optimized `max_{K,B}` capacity and should be compared across
several seeds/trial counts before drawing strong conclusions.

## Cluster-Aware Statistical Diagnostics

Seed-level rows are nested inside topology/mask groups, so run-level OLS should
not be treated as if every seed were an independent topology. Use
`clustered_topology_inference.py` on collected run CSVs:

```bash
python3 clustered_topology_inference.py \
  --run_csv results/input_mask_fixed_m20_random_sc_seed3_c200/topology_results.csv \
  --cluster_col topology_name \
  --family_col physical_topology_name \
  --n_bootstrap 1000 \
  --output_json results/clustered_topology_inference.json
```

The JSON includes:

- topology/mask group-level regressions for mean, best, and seed-variance
  targets,
- clustered bootstrap `R2` deltas versus the raw-count baseline,
- leave-one-physical-family-out prediction summaries,
- residual decomposition into between-group and within-group components.

## Matched Essential-Motif Controls

Extracted physical motifs are selected from trained models, so a strong motif
result requires controls matched on coarse tree geometry. Generate a retrainable
control library with:

```bash
python3 make_matched_motif_controls.py \
  --source_csv results/input_mask_fixed_m20_random_sc_seed3_c200/essential_input50/selected.csv \
  --output_root results/input_mask_fixed_m20_random_sc_seed3_c200/essential_input50_matched_controls \
  --N 4 \
  --D 4 \
  --control_kinds random_sc,degree_rewire \
  --controls_per_source 4 \
  --candidates_per_source 256
```

Then train `selected.csv` with `submit_topology_library_sweep.py` using the
same seeds and optimizer as the extracted motifs. Compare extracted motifs
against these controls before making causal claims about motif superiority.

## Causal Tree/Branch Alignment Interventions

Associations between active trees and ICL branches are not enough by
themselves. For trained high-ICL runs, directly scramble the learned alignment
while preserving simpler statistics:

```bash
python3 causal_topology_interventions.py \
  --run_dir results/input_mask_fixed_m20_random_sc_seed3_c200/<run_label> \
  --n_samples 500 \
  --n_repeats 5 \
  --interventions context_block_shuffle,edge_projection_permutation,edge_rate_function_permutation,decoder_root_permutation,randomize_K_direction
```

The script writes `causal_interventions.json` with baseline accuracy/mechanism
summaries and intervention deltas on the same sampled novel-class batch. A
large negative `target_accuracy_delta` under branch/tree-alignment scrambles is
stronger mechanism evidence than correlation alone.

Run the intervention pass over completed training runs with:

```bash
python3 submit_causal_interventions.py \
  --input_root results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --n_samples 500 \
  --n_repeats 5 \
  --array \
  --dry-run
```

Collect completed intervention reports with:

```bash
python3 collect_causal_interventions.py \
  --input_root results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --output_csv results/input_mask_fixed_m20_random_sc_seed3_c200/causal_interventions.csv \
  --output_json results/input_mask_fixed_m20_random_sc_seed3_c200/causal_interventions_summary.json
```

## Engaging SLURM Usage

Use a separate cluster worktree named `topology`, then from its `ICL` directory:

For broader fixed-count sweeps, first generate a regime plan:

```bash
python3 make_topology_sweep_plan.py \
  --output_csv results/expanded_topology_sweep_plan.csv \
  --commands_sh results/expanded_topology_sweep_plan.sh \
  --n_nodes 4:8 \
  --edge_regimes sparse,intermediate,dense \
  --n_context 2,3 \
  --z_dims 1,2
```

Inspect the CSV before running the generated commands. The submission commands
are written with `--dry-run` by default so the cluster agent can review array
sizes and output roots before removing that flag.

```bash
export SLURM_OUTPUT_BASE=/pool/<group>/<user>/topology_phase1
export SLURM_PARTITION=<partition>
export SLURM_ACCOUNT=<account>
export SLURM_TIME=08:00:00
export SLURM_MEM_PER_CPU=8G
export SLURM_EXTRA_SETUP='module load python/3.11; source ~/venvs/icl/bin/activate'
export TOPOLOGY_PYTHON=python

python3 - <<'PY'
import os
print("SLURM_EXTRA_SETUP:", os.environ.get("SLURM_EXTRA_SETUP", ""))
print("TOPOLOGY_PYTHON:", os.environ.get("TOPOLOGY_PYTHON", "python3"))
PY

python3 submit_topology_phase1.py --phase smoke --array --dry-run
python3 submit_topology_phase1.py --phase smoke --array
```

The post-training SLURM submitters insert a Torch import preflight into each
array task. If `TOPOLOGY_PYTHON` cannot import Torch after `SLURM_EXTRA_SETUP`,
the task fails before doing mechanism or causal analysis. This is intentional:
those jobs load trained `model.pt` files and should not run under a system
Python without Torch.

The full initial controlled sweep is:

```bash
python3 submit_topology_phase1.py --phase pilot --array --max-concurrent 20
python3 submit_topology_phase1.py --phase phase1 --array --max-concurrent 40
```

Use `pilot` before `phase1`: it uses the same output format and matched
topology controls, but shorter runs suitable for checking signal and runtime.

For the stricter fixed-edge-count test, first build a topology library and
select structurally diverse graphs before training:

```bash
python3 make_topology_library.py \
  --output_root results/topology_library_n6_m20 \
  --n_nodes 6 \
  --n_edges 20 \
  --candidate_seeds 1:80 \
  --select_topologies 16

python3 submit_topology_library_sweep.py \
  --library_csv results/topology_library_n6_m20/selected.csv \
  --output_root results/topology_fixed_m20 \
  --seeds 1,2 \
  --array \
  --max-concurrent 24
```

If an array is interrupted or only some tasks are missing, resubmit only the
unfinished outputs:

```bash
python3 submit_topology_library_sweep.py \
  --library_csv results/topology_library_n6_m20/selected.csv \
  --output_root results/topology_fixed_m20 \
  --seeds 1,2 \
  --missing_only \
  --array
```

Each run stores pre-training structural predictors next to training/test
results, so regression analysis can compare novel-class ICL accuracy against
raw degree count, input-coupled degree count, global `d_rel`, weakest-branch
comparison `d_rel`, masked effective rank, root imbalance, and bottleneck
metrics. `comparison_branch_d_rel_min` is a loose input-mask diagnostic: for
each context position and feature dimension it takes the smaller
coordinate-wise relative rank between the context coordinate and matching query
coordinate, then sums over dimensions. `comparison_branch_common_d_rel_min` is
stricter and usually more mechanistic: it measures the intersection rank of the
context/query relative tree-contrast subspaces, so disjoint high-rank context
and query supports are not counted as comparison capacity. Together these
expose masks with high total capacity but poor paired support for one
query-context comparison branch.

To separate input-encoding topology from physical reaction topology, choose one
fixed physical graph and generate explicit masks with the same number of
coupled input parameters:

```bash
python3 make_input_mask_library.py \
  --edge_json results/topology_library_n6_m20/topologies/g0000_random_sc_seed1.json \
  --output_root results/input_mask_library_n6_m20_c200 \
  --coupled_counts 200 \
  --candidate_seeds 1:60 \
  --select_masks 16

python3 submit_topology_library_sweep.py \
  --library_csv results/input_mask_library_n6_m20_c200/selected.csv \
  --output_root results/input_mask_fixed_graph_c200 \
  --seeds 1,2 \
  --array \
  --max-concurrent 24
```

Rows from `make_input_mask_library.py` keep `edge_json` fixed and vary only
`input_mask_json`. The runner validates that mask rows match the physical edge
order exactly, then stores the canonical mask in each run's `topology.json`.

After jobs finish, collect the result table:

```bash
python3 collect_topology_results.py \
  --input_root "$SLURM_OUTPUT_BASE" \
  --output_csv "$SLURM_OUTPUT_BASE/topology_results.csv"
```

Then run the first nested-model diagnostic:

```bash
python3 regress_topology_results.py \
  --input_csv "$SLURM_OUTPUT_BASE/topology_results.csv" \
  --output_json "$SLURM_OUTPUT_BASE/topology_regression.json"
```

For a full post-training mechanism pass over completed runs:

```bash
python3 submit_topology_mechanisms.py \
  --input_root "$SLURM_OUTPUT_BASE" \
  --n_samples 500 \
  --ablate_input \
  --ablate_physical \
  --array \
  --max-concurrent 20

python3 collect_mechanism_results.py \
  --input_root "$SLURM_OUTPUT_BASE" \
  --output_csv "$SLURM_OUTPUT_BASE/mechanism_results.csv"

python3 summarize_topology_mechanisms.py \
  --topology_csv "$SLURM_OUTPUT_BASE/topology_results.csv" \
  --mechanism_csv "$SLURM_OUTPUT_BASE/mechanism_results.csv" \
  --output_json "$SLURM_OUTPUT_BASE/mechanism_summary.json"

python3 aggregate_topology_seeds.py \
  --topology_csv "$SLURM_OUTPUT_BASE/topology_results.csv" \
  --mechanism_csv "$SLURM_OUTPUT_BASE/mechanism_results.csv" \
  --output_csv "$SLURM_OUTPUT_BASE/topology_seed_aggregates.csv" \
  --output_json "$SLURM_OUTPUT_BASE/topology_seed_aggregates.json"
```

The same collection/regression/aggregation workflow can be run as one command:

```bash
python3 finalize_topology_sweep.py \
  --input_root "$SLURM_OUTPUT_BASE"
```

To submit mechanism analysis for a completed sweep, then later collect it:

```bash
python3 finalize_topology_sweep.py \
  --input_root "$SLURM_OUTPUT_BASE" \
  --submit_mechanisms \
  --ablate_input \
  --ablate_physical \
  --job_python "$TOPOLOGY_PYTHON"

python3 finalize_topology_sweep.py \
  --input_root "$SLURM_OUTPUT_BASE" \
  --collect_mechanisms
```

To submit and collect both mechanism decompositions and causal
tree/branch-alignment interventions for a completed sweep:

```bash
python3 finalize_topology_sweep.py \
  --input_root "$SLURM_OUTPUT_BASE" \
  --submit_mechanisms \
  --submit_causal \
  --ablate_input \
  --ablate_physical \
  --device cpu \
  --job_python "$TOPOLOGY_PYTHON" \
  --max-concurrent 20

python3 finalize_topology_sweep.py \
  --input_root "$SLURM_OUTPUT_BASE" \
  --collect_mechanisms \
  --collect_causal
```

For the currently tracked expanded hard pilots, run the same follow-up over
each root after syncing the `topology` branch on Engaging:

```bash
python3 run_expanded_hard_followups.py --status

python3 run_expanded_hard_followups.py \
  --submit_followups \
  --device cpu \
  --job_python "$TOPOLOGY_PYTHON" \
  --max-concurrent 20

# After the SLURM arrays finish:
python3 run_expanded_hard_followups.py \
  --collect_followups \
  --refresh_report \
  --strict_verify \
  --device cpu \
  --job_python "$TOPOLOGY_PYTHON"
```

The wrapper refuses to submit or collect follow-ups from a source-light
checkout with no raw `results.pkl` files, because the lower-level finalizer
recollects topology results at startup. To refresh only selected report
sections by hand from the `ICL/` directory, use the underlying report
refresher. It updates only labels supplied on the command line; by default it
ignores all-zero expanded roots so a source-light checkout does not erase
previously recorded completed-run counts.

```bash
python3 refresh_next_phase_report.py \
  --report_json results/next_phase_stats/next_phase_evidence_report.json \
  --clustered_json hard_n4_m6_N3_D2=results/expanded_hard_stats/n4_m6_N3_D2_branch_capacity_clustered_inference.json \
  --clustered_json hard_n5_m8_N3_D2=results/expanded_hard_stats/n5_m8_N3_D2_branch_capacity_clustered_inference.json \
  --clustered_json hard_n5_m12_N3_D2=results/expanded_hard_stats/n5_m12_N3_D2_branch_capacity_clustered_inference.json \
  --branch_capacity_json hard_n4_m6_N3_D2=results/expanded_hard_libraries/n4_m6_N3_D2/branch_margin_capacity_summary.json \
  --branch_capacity_json hard_n5_m8_N3_D2=results/expanded_hard_libraries/n5_m8_N3_D2/branch_margin_capacity_summary.json \
  --branch_capacity_json hard_n5_m12_N3_D2=results/expanded_hard_libraries/n5_m12_N3_D2/branch_margin_capacity_summary.json \
  --expanded_root hard_n4_m6_N3_D2=results/expanded_hard_sweeps/n4_m6_N3_D2 \
  --expanded_root hard_n5_m8_N3_D2=results/expanded_hard_sweeps/n5_m8_N3_D2 \
  --expanded_root hard_n5_m12_N3_D2=results/expanded_hard_sweeps/n5_m12_N3_D2 \
  --output_json results/next_phase_stats/next_phase_evidence_report.json \
  --output_md results/next_phase_stats/next_phase_evidence_report.md
```

When causal summaries exist, add matching `--causal_json
label=results/.../causal_interventions_summary.json` arguments before
re-rendering the report.

To test whether a trained dense topology discovered a sparse retrainable
motif, extract essential subgraphs and feed the resulting `selected.csv` back
into the existing library-sweep trainer:

```bash
python3 extract_essential_subgraphs.py \
  --input_root "$SLURM_OUTPUT_BASE" \
  --topology_csv "$SLURM_OUTPUT_BASE/topology_results.csv" \
  --output_root "$SLURM_OUTPUT_BASE/essential_input50" \
  --importance_source input_ablation_loss \
  --coverage_fraction 0.5 \
  --select_topologies 16

python3 submit_topology_library_sweep.py \
  --library_csv "$SLURM_OUTPUT_BASE/essential_input50/selected.csv" \
  --output_root "$SLURM_OUTPUT_BASE/essential_input50_retrain" \
  --seeds 1,2 \
  --array \
  --max-concurrent 24

# Audit or recover an interrupted array without overwriting completed runs.
python3 submit_topology_library_sweep.py \
  --library_csv "$SLURM_OUTPUT_BASE/essential_input50/selected.csv" \
  --output_root "$SLURM_OUTPUT_BASE/essential_input50_retrain" \
  --seeds 1,2 \
  --status_only \
  --manifest_csv "$SLURM_OUTPUT_BASE/essential_input50_retrain/task_manifest.csv"

python3 submit_topology_library_sweep.py \
  --library_csv "$SLURM_OUTPUT_BASE/essential_input50/selected.csv" \
  --output_root "$SLURM_OUTPUT_BASE/essential_input50_retrain" \
  --seeds 1,2 \
  --missing_only \
  --clean \
  --array \
  --max-concurrent 24

python3 finalize_essential_physical_retrains.py \
  --experiment physical="$SLURM_OUTPUT_BASE" \
  --seeds 1,2
```

The physical finalizer runs `finalize_topology_sweep.py` on
`essential_input50_retrain` and then `compare_essential_retrains.py` with the
physical layout defaults. It does not regenerate the consolidated research
report; run `make_topology_research_report.py` explicitly after all physical and
input-mask retrain comparisons you want to include are finalized.

The same physical path can be recovered with a single wrapper. Use `--dry-run`
first to inspect the audit, status, missing-only submit, guarded finalizer, and
strict physical audit commands:

```bash
python3 recover_essential_physical_retrains.py \
  --experiment physical="$SLURM_OUTPUT_BASE" \
  --seeds 1,2 \
  --submit_missing \
  --finalize_if_complete \
  --max-concurrent 24 \
  --dry-run
```

As with the input-mask recovery wrapper, run once with `--submit_missing`, then
rerun with `--finalize_if_complete` after the missing jobs finish.

For essential input-mask retrains, where the physical graph stays fixed and
only the learned input-coupling rows are pruned, the final collection can be
done in one guarded step after all retrain jobs finish:

```bash
python3 audit_topology_artifacts.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --require_source_results \
  --require_mechanisms \
  --require_essential_inputmask

for root in \
  results/input_mask_fixed_m20_random_sc_seed3_c200 \
  results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  results/input_mask_fixed_m20_hub_spoke_seed63_c200
do
  python3 submit_topology_library_sweep.py \
    --library_csv "$root/essential_inputmask50/selected.csv" \
    --output_root "$root/essential_inputmask50_retrain" \
    --seeds 1,2,3,4,5 \
    --status_only \
    --manifest_csv "$root/essential_inputmask50_retrain/task_manifest.csv"

  python3 submit_topology_library_sweep.py \
    --library_csv "$root/essential_inputmask50/selected.csv" \
    --output_root "$root/essential_inputmask50_retrain" \
    --seeds 1,2,3,4,5 \
    --missing_only \
    --clean \
    --array \
    --max-concurrent 16
done

python3 audit_topology_artifacts.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --require_source_results \
  --require_mechanisms \
  --require_essential_inputmask \
  --require_essential_retrains \
  --strict

python3 finalize_essential_inputmask_retrains.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --output_md results/input_mask_topology_report.md \
  --output_json results/input_mask_topology_report.json
```

The same recovery/finalization path can be driven by one wrapper. Use `--dry-run`
first to inspect the audit, status, missing-only submit, strict audit, and
finalizer commands:

```bash
python3 recover_essential_inputmask_retrains.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --submit_missing \
  --finalize_if_complete \
  --output_md results/input_mask_topology_report.md \
  --output_json results/input_mask_topology_report.json \
  --max-concurrent 16 \
  --dry-run
```

For execution, use the same wrapper in two passes. First submit only missing
retrain runs:

```bash
python3 recover_essential_inputmask_retrains.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --submit_missing \
  --max-concurrent 16
```

After the missing jobs finish, finalize through the guarded exact-path
completion check, then run the strict post-finalization audit:

```bash
python3 recover_essential_inputmask_retrains.py \
  --experiment random=results/input_mask_fixed_m20_random_sc_seed3_c200 \
  --experiment cycle=results/input_mask_fixed_m20_cycle_chords_seed3_c200 \
  --experiment hub=results/input_mask_fixed_m20_hub_spoke_seed63_c200 \
  --seeds 1,2,3,4,5 \
  --finalize_if_complete \
  --output_md results/input_mask_topology_report.md \
  --output_json results/input_mask_topology_report.json
```

If there are no missing retrain tasks, the first pass only writes the status
manifests and skips array creation. If retrains are incomplete, the second pass
stops in `finalize_essential_inputmask_retrains.py` before overwriting the
report. The finalizer checks exact outputs named
`<topology_id>_trainseed<seed>/results.pkl`; stale extra outputs fail unless
`--allow_extra` is supplied for diagnostics.

To consolidate completed sweeps into one auditable progress artifact and write
the conservative H0/H1 interpretation:

```bash
python3 finalize_topology_research_report.py \
  --experiment m20=results/topology_fixed_m20_library \
  --experiment m12=results/topology_fixed_m12_library \
  --seeds 1,2,3,4,5 \
  --output_md results/topology_research_report.md \
  --output_json results/topology_research_report.json
```

This wrapper first runs `make_topology_research_report.py`, then
`verify_topology_completion.py --report_kind research`, then
`interpret_topology_report.py --report_kind research`. When multiple
experiments are supplied, the report also includes pooled cross-regime
regressions that compare edge-count predictors with tree-geometry, mechanism,
and projection-alignment predictors.

If an experiment root contains `essential_input50/retrain_comparison.json` or
`essential_inputmask50/retrain_comparison.json`, the same consolidated report
will include the retrain-retention tables for those physical subgraphs or
input masks. This lets the final report cover both expressive minimal physical
motifs and sparse input-encoding motifs without changing the reporting command.
