# Markov-nudging project (pre-ICL)

This directory holds the older "MarkovComputations" research line: energy-based
"contrastive nudging" training of Markov chains as classifiers for MNIST,
Iris, glycans, etc. It is **not** used to reproduce the ICL paper
(`arXiv-2601.06712v1`); for that, see the parent `ICL/` directory instead.

## Contents

- `MarkovComputations.py` — the core library. Defines `WeightMatrix`,
  `InputData`, contrastive update rules, MNIST/Iris loaders. Built on
  NumPy + JAX + NetworkX + TensorFlow (used only for the MNIST loader).
- `generalized_hopfield_model.py` — `GHN` class (Generalized Hopfield
  Network), used by the two `MNIST_*GH*` notebooks.
- `training/` — CLI training scripts. `Training.py` is the local single-run
  version; `TrainingMidway*.py` are UChicago "Midway" cluster variants with
  different network architectures (perceptron, stacked, MI). `RunMidwayJobs.py`
  is the SLURM submitter for the Midway scripts.
- `MNIST/` — MNIST classification experiment notebooks.
- `dev_notebooks/` — development notebooks for each Training variant
  (`Dev_Stacked`, `Dev_Perceptron`, `Dev_MI`, `Dev_Glycan`), plus
  `Training.ipynb`, `Gaussians_Training.ipynb`, and `AnalyzeData.ipynb`
  (which inspects the `Dirs/` SLURM outputs).
- `mathematica/` — `MemoryPatterns.nb` and `Debugging.nb`, symbolic
  derivations supporting the library.
- `scratch/` — ad-hoc development and debugging scripts (eigenvalue autodiff
  experiments, sparse-solver tests, NetworkX graph tests).

## Running

The Python scripts in `training/` and the notebooks in `MNIST/` and
`dev_notebooks/` all do `from MarkovComputations import ...`. After the
reorg, each script and notebook adds the parent directory to `sys.path`
itself, so you can run them from any cwd without setting `PYTHONPATH`:

```bash
python markov-nudging/training/Training.py --output ./run_out
```

Or open any of the notebooks in Jupyter — the first cell does the path
fixup automatically.

### Cluster paths

`Training.py` now takes `--output <dir>` (defaults to `./output_training`)
instead of writing to a hardcoded path. `RunMidwayJobs.py` reads cluster
configuration from environment variables and falls back to the original
UChicago Midway defaults if they are unset:

```bash
export SLURM_OUTPUT_BASE=/path/to/results
export SLURM_PARTITION=mit_normal       # e.g. on MIT Engaging
export SLURM_ACCOUNT=<your-pi-account>
export SLURM_TIME=02:00:00
export SLURM_MEM_PER_CPU=2G
python markov-nudging/training/RunMidwayJobs.py
```
