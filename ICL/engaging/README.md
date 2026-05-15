# Running WTA-ICL on MIT Engaging

Reproduces the Winner-Takes-All ICL paper results on the MIT Engaging cluster
(`mit_normal` partition). The training entry point is `../run_icl_wta.py`.

## What these jobs reproduce

The paper's WTA checkpoints were trained with this config (hardcoded as
defaults in `run_icl_wta.py`):

```
K=128, L=128, D=4, N=4, B=1, epsilon=0.001
R0=2.0, softplus activations, beta_softplus=10.0, learn_K/beta=True
epochs=1000, lr=0.0025, batch_size=20, train_samples=50000, val_samples=5000
method=soft, temperature=0.1
```

Each job varies three values: `n_nodes` (param1), `rho_all` (param2),
`seed` (param3). The two reference checkpoints are `8_1.0_30` and `12_1.0_20`.
Runtime is ~20-70 min per job, CPU-only.

## One-time setup on Engaging

1. **Get the repo onto Engaging:**
   ```bash
   git clone https://github.com/statmech-learning/MarkovComputations.git
   ```
2. **Python env** — you said you already have one with `torch` + `numpy`.
   Note the activation command (venv: `source <env>/bin/activate`,
   conda: `source activate <env>`).
3. **Fill in the placeholders** (marked `<<< EDIT` / `EDIT ME`) in
   `job_wta_template.sh` and `submit_jobs_engaging.py`:
   - `ACCOUNT` — your Engaging SLURM account
   - env activation command
   - path to `MarkovComputations/ICL` on Engaging
   - `OUTPUT_BASE` — where results should be written

## Step 1 — single test job

```bash
cd MarkovComputations/ICL/engaging
sbatch job_wta_template.sh          # reproduces the 8_1.0_30 checkpoint
squeue -u $USER                     # watch it
```

Output lands in `ICL/results/wta_8_1.0_30/` (`model.pt`, `results.pkl`,
`params.json`, `training_batch.out`).

## Step 2 — verify a finished run

```bash
python ../verify_checkpoints.py results/wta_8_1.0_30
```

Re-loads the weights and re-runs evaluation; expect novel-class ICL ~97-99%.

## Step 3 — the sweep

Edit `JOBS` in `submit_jobs_engaging.py` (default = the two paper checkpoints),
then:

```bash
python submit_jobs_engaging.py --dry-run   # write scripts, inspect them
python submit_jobs_engaging.py             # write + sbatch
```

Each `(n_nodes, rho_all, seed)` triple gets its own `OUTPUT_BASE/<n>_<r>_<s>/`
directory with its job script and logs.

## Notes

- Jobs are CPU-only; `mit_normal` with `--cpus-per-task=4 --mem=8G` is ample.
- Retrained weights will **not** be bit-identical to the originals (different
  RNG history), but the metrics reproduce within ~1-2% — that is the result
  being replicated.
- `run_icl_wta.py` also accepts named args (`--n_nodes`, `--seed`, `--lr`,
  `--epochs`, ...) — see `python ../run_icl_wta.py --help`.
