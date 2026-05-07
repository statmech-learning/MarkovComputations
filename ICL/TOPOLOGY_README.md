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
  edge participation metrics.
- `models/topology_markov_icl.py`: first-order ICL model on an explicit directed
  reaction graph.
- `run_topology_icl.py`: train one topology-controlled run and save
  `results.pkl`, `topology.json`, `topology_metrics.json`, and `config.json`.
- `submit_topology_phase1.py`: SLURM array generator for matched graph-family
  sweeps.
- `make_topology_library.py`: generate strongly connected fixed-`m` topology
  candidates, compute pre-training matrix-tree metrics, and select a
  structurally diverse subset for training.
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
- `analyze_topology_model.py`: load a trained run and write
  `mechanism_metrics.json`.
- `submit_topology_mechanisms.py`: SLURM array generator for post-training
  active-tree, edge-sensitivity, and ablation analysis over completed runs.
- `finalize_topology_sweep.py`: convenience wrapper that collects completed
  training runs, runs regressions, optionally submits/collects mechanism
  analyses, and refreshes topology-level seed aggregates.
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
- `finalize_essential_inputmask_retrains.py`: guarded finalizer for extracted
  essential input-mask retrains; it refuses to collect incomplete retrain sets
  unless `--allow_partial` is supplied.
- `recover_essential_inputmask_retrains.py`: conservative wrapper for
  interrupted essential input-mask retrains; it audits source artifacts, writes
  status manifests, optionally submits only missing retrain runs, and calls the
  guarded finalizer only after a strict retrain audit.
- `make_input_mask_report.py`: focused report for fixed-physical-graph
  input-mask sweeps, including masked tree geometry, mechanism predictors,
  extracted essential masks, and retrain retention.
- `make_topology_research_report.py`: consolidate fixed-edge sweeps,
  mechanism summaries, seed aggregates, physical essential-subgraph retrains,
  and essential input-mask retrains into one Markdown/JSON progress report.
- `audit_topology_artifacts.py`: read-only audit of source runs, mechanism
  outputs, extracted essential masks, retrain outputs, manifests, and comparison
  files. Use this before recovering interrupted cluster arrays or running
  guarded finalizers. It supports input masks by default and physical essential
  subgraphs with `--essential_kind physical`.
- `verify_topology_completion.py`: final non-mutating completion gate that runs
  the strict artifact audit and validates either focused input-mask reports
  (`--report_kind input_mask`) or consolidated topology research reports
  (`--report_kind research`).
- `regress_topology_results.py`: dependency-light OLS diagnostics for testing
  whether tree-geometry predictors improve on raw parameter count.
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

## Engaging SLURM Usage

Use a separate cluster worktree named `topology`, then from its `ICL` directory:

```bash
export SLURM_OUTPUT_BASE=/pool/<group>/<user>/topology_phase1
export SLURM_PARTITION=<partition>
export SLURM_ACCOUNT=<account>
export SLURM_TIME=08:00:00
export SLURM_MEM_PER_CPU=8G
export SLURM_EXTRA_SETUP='module load python/3.11; source ~/venvs/icl/bin/activate'

python3 submit_topology_phase1.py --phase smoke --array --dry-run
python3 submit_topology_phase1.py --phase smoke --array
```

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
  --ablate_physical

python3 finalize_topology_sweep.py \
  --input_root "$SLURM_OUTPUT_BASE" \
  --collect_mechanisms
```

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

python3 compare_essential_retrains.py \
  --base_root "$SLURM_OUTPUT_BASE" \
  --output_csv "$SLURM_OUTPUT_BASE/essential_input50/retrain_comparison.csv" \
  --output_json "$SLURM_OUTPUT_BASE/essential_input50/retrain_comparison.json"
```

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

To consolidate completed sweeps into one auditable progress artifact:

```bash
python3 make_topology_research_report.py \
  --experiment m20=results/topology_fixed_m20_library \
  --experiment m12=results/topology_fixed_m12_library \
  --output_md results/topology_research_report.md \
  --output_json results/topology_research_report.json
```

When multiple experiments are supplied, the report also includes pooled
cross-regime regressions that compare edge-count predictors with tree-geometry,
mechanism, and projection-alignment predictors.

If an experiment root contains `essential_input50/retrain_comparison.json` or
`essential_inputmask50/retrain_comparison.json`, the same consolidated report
will include the retrain-retention tables for those physical subgraphs or
input masks. This lets the final report cover both expressive minimal physical
motifs and sparse input-encoding motifs without changing the reporting command.
