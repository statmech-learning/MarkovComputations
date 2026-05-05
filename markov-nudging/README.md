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

The training scripts and notebooks all do `from MarkovComputations import ...`.
Because `MarkovComputations.py` lives at the top of this folder (not at the
top of each subdir), you must run them with this directory on `PYTHONPATH`.
For example:

```bash
cd markov-nudging
PYTHONPATH=. python training/Training.py
```

Or from a notebook:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # adjust depth as needed
from MarkovComputations import WeightMatrix, ...
```

The `TrainingMidway*.py` scripts and `RunMidwayJobs.py` also have hardcoded
`/project/svaikunt/...` paths from the UChicago cluster, so they will need
`output_dir` edits before they can run anywhere else.
