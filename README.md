# MarkovComputations

This repository contains two separate research projects sharing a folder.

## Layout

```
.
├── ICL/                    # ICL paper code (PyTorch). Self-contained.
├── arXiv-2601.06712v1/     # LaTeX sources of the ICL paper.
├── markov-nudging/         # Older project: energy-based "nudging" training
│                           # of Markov chains for MNIST / Iris / glycans /
│                           # Hopfield. Built on NumPy + JAX + NetworkX.
└── deprecated/             # Broken or orphan files kept for reference.
```

## ICL paper — `arXiv-2601.06712v1`

"In-context learning emerges in chemical reaction networks without attention"
(Floyd, Lopez Rios, Dinner, Vaikuntanathan).

All code needed to reproduce the paper lives under `ICL/`. The three chemical
models studied in the paper map to three model classes:

- Linear / first-order (Markov chain, default)  → `ICL/models/markov_icl.py`
- Autocatalytic (second-order)                  → `ICL/models/nonlinear_markov_icl.py`
- Winner-take-all                               → `ICL/models/wta_icl.py`

Entry points:

- `ICL/run_icl.py` — train the linear Markov-chain model (paper defaults).
- `ICL/run_icl_nlm.py` — train the autocatalytic model.
- `ICL/run_icl_local_wta.ipynb` — train the WTA model (notebook).
- `ICL/submit_jobs.py` — SLURM batch sweep for `run_icl.py`.

Shared library: `ICL/data_generation.py`, `ICL/datasets.py`,
`ICL/training.py`, `ICL/evaluation.py`, `ICL/config.py`.

## Older project — `markov-nudging/`

Contrastive "nudging" training of Markov-chain classifiers (pre-ICL work).
Not required for the ICL paper. See `markov-nudging/README.md` for details.

## `deprecated/`

Five files that are either broken or are pre-refactor monolithic duplicates
of code now living under `ICL/`. Safe to delete; kept in case the contents
are useful as a reference. See chat history for details on each file.
