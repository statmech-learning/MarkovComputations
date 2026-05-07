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
  effective rank, tree-count balance, and edge participation metrics.
- `models/topology_markov_icl.py`: first-order ICL model on an explicit directed
  reaction graph.
- `run_topology_icl.py`: train one topology-controlled run and save
  `results.pkl`, `topology.json`, `topology_metrics.json`, and `config.json`.
- `submit_topology_phase1.py`: SLURM array generator for matched graph-family
  sweeps.
- `make_topology_library.py`: generate strongly connected fixed-`m` topology
  candidates, compute pre-training matrix-tree metrics, and select a
  structurally diverse subset for training.
- `submit_topology_library_sweep.py`: SLURM array generator for training the
  selected topology library through `run_topology_icl.py --edge_json`.
- `topology_analysis.py`: post-training active-tree, tree-projection
  alignment, and edge-sensitivity utilities.
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
  flat CSV table.
- `summarize_topology_mechanisms.py`: join topology and mechanism result CSVs,
  then report overall and within-edge-count correlations.
- `aggregate_topology_seeds.py`: aggregate repeated training seeds for each
  topology into mean, best-seed, and seed-variance reports.
- `extract_essential_subgraphs.py`: convert trained-model edge importance or
  ablation scores into strongly connected candidate motifs that can be retrained
  through `submit_topology_library_sweep.py`.
- `compare_essential_retrains.py`: join extracted motif source metadata with
  from-scratch retrain aggregates and report performance retention.
- `make_topology_research_report.py`: consolidate fixed-edge sweeps,
  mechanism summaries, seed aggregates, and essential motif retrain comparisons
  into one Markdown/JSON progress report.
- `regress_topology_results.py`: dependency-light OLS diagnostics for testing
  whether tree-geometry predictors improve on raw parameter count.
- `tests/test_topology_metrics.py`: exact small-graph matrix-tree checks.

## Local Structural Checks

These do not require Torch:

```bash
python3 -m unittest discover -s ICL/tests
python3 ICL/submit_topology_phase1.py --phase smoke --dry-run
```

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
raw degree count, `d_rel`, effective rank, root imbalance, and bottleneck
metrics.

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

python3 compare_essential_retrains.py \
  --base_root "$SLURM_OUTPUT_BASE" \
  --output_csv "$SLURM_OUTPUT_BASE/essential_input50/retrain_comparison.csv" \
  --output_json "$SLURM_OUTPUT_BASE/essential_input50/retrain_comparison.json"
```

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
