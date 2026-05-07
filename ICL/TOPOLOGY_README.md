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
- `topology_analysis.py`: post-training active-tree and edge-sensitivity
  utilities.
- `analyze_topology_model.py`: load a trained run and write
  `mechanism_metrics.json`.
- `collect_topology_results.py`: collect completed run directories into a flat
  CSV for regressions against raw degree count and topology-derived metrics.
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
active-tree entropy, matrix-tree edge sensitivity, an essential-edge summary,
and target log-probability margins on novel-class ICL samples.

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
python3 submit_topology_phase1.py --phase phase1 --array --max-concurrent 40
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
