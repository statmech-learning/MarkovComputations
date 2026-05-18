"""
Shared foundation for WTA-ICL topology analysis.

Every analysis module (m1_*.py .. m6_*.py) and run_all.py imports ONLY from
this file, so that checkpoint discovery, data regeneration, and the
instrumented forward pass have a single, consistent implementation.

>>> THIS FILE IS THE FROZEN CONTRACT. Module agents must NOT modify it. <<<

Standard header for every module/script in this package:

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import core

IMPORTANT empirical fact (verified on all 4 reference checkpoints):
The model's "winner-take-all" is SOFT. The steady-state concentration Y
spreads across ~2.5-3 species (the top species holds only ~56-68% of total
Y). So Y is a point on a simplex, NOT a one-hot vector, and the routing
q = Y @ B is a graded mixture of B-rows. Treat the discrete "winner" as a
summary statistic, and always measure how concentrated Y actually is.
Two distinct "winner" notions are provided and they disagree ~24% of the
time -- pick the right one for your analysis:
    winner       = argmin_j (beta_j / f_j)   -- the WTA *selection rule*
    dom_species  = argmax_j Y_j              -- the species that *dominates* Y

Public API:
    discover_checkpoints() -> list[Checkpoint]
    load_checkpoint(path)  -> Checkpoint
    build_gmm(seed)        -> GaussianMixtureModel
    make_eval_sets(...)    -> {'in_dist': [...], 'novel': [...]}
    instrument(model, eval_set, temperature) -> Trace
    get_traces(checkpoint, eval_sets=None)   -> {'in_dist': Trace, 'novel': Trace}
    load_all(n_eval=N_EVAL) -> (list[Checkpoint], dict[label -> {split: Trace}])
    setup_style() / module_outdir(name) / save_fig(fig, outdir, name)
"""

import os
import sys
import glob
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# --- make the ICL package importable ---------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
ICL_DIR = _THIS_DIR.parent                        # .../MarkovComputations/ICL
REPO_DIR = ICL_DIR.parent
if str(ICL_DIR) not in sys.path:
    sys.path.insert(0, str(ICL_DIR))

from data_generation import GaussianMixtureModel, generate_icl_gmm_data   # noqa: E402
from models.wta_icl import load_model                                     # noqa: E402

# --- locations --------------------------------------------------------------
CHECKPOINT_ROOTS = [
    ICL_DIR / "paper_checkpoints" / "WTA_params_nodes_8+12",
    ICL_DIR / "results" / "wta_n_nodes_rhoall_seed",
]
OUTDIR = ICL_DIR / "results" / "topology_analysis"

# Fixed seed for the analysis eval set: EVERY checkpoint is probed on
# identical inputs, which is required for cross-model comparison (m6).
ANALYSIS_SEED = 777
N_EVAL = 2000

# All current checkpoints share this config (paper config). build_gmm /
# make_eval_sets use it; instrument reads temperature from each checkpoint.
SHARED_CONFIG = dict(K=128, L=128, D=4, N=4, B=1, epsilon=0.001,
                     exact_copy=True, shuffle_context=True, unique_labels=False)


# ===========================================================================
# Checkpoints
# ===========================================================================
@dataclass
class Checkpoint:
    label: str          # e.g. "paper-n8-s30", "engaging-n12-s31"
    path: Path
    origin: str         # "paper" | "engaging"
    n_nodes: int
    seed: int
    model: object       # WinnerTakesAllICL, eval mode, weights loaded
    params: dict
    results: dict       # stored {'in_dist', 'novel_classes'}
    history: dict

    def __repr__(self):
        return (f"Checkpoint({self.label}, n_nodes={self.n_nodes}, "
                f"novel={self.results.get('novel_classes', '?')}%)")


def load_checkpoint(path) -> Checkpoint:
    """Load one run directory (must contain model.pt + results.pkl)."""
    path = Path(path).resolve()
    with open(path / "results.pkl", "rb") as f:
        res = pickle.load(f)
    params = res["params"]
    model = load_model(params, str(path) + os.sep, print_creation=False)
    model.eval()
    origin = "paper" if "paper_checkpoints" in str(path) else "engaging"
    n_nodes, seed = int(params["n_nodes"]), int(params["seed"])
    return Checkpoint(
        label=f"{origin}-n{n_nodes}-s{seed}", path=path, origin=origin,
        n_nodes=n_nodes, seed=seed, model=model, params=params,
        results=res.get("results", {}), history=res.get("history", {}))


def discover_checkpoints() -> list:
    """Find every run dir (model.pt + results.pkl) under the checkpoint roots,
    sorted by (n_nodes, origin, seed)."""
    found, seen = [], set()
    for root in CHECKPOINT_ROOTS:
        if not root.exists():
            continue
        for mp in sorted(glob.glob(str(root / "**" / "model.pt"), recursive=True)):
            d = Path(mp).parent
            if not (d / "results.pkl").exists():
                continue
            ck = load_checkpoint(d)
            if ck.label in seen:                       # safety net
                ck.label = f"{ck.label}-{d.name}"
            seen.add(ck.label)
            found.append(ck)
    found.sort(key=lambda c: (c.n_nodes, c.origin, c.seed))
    return found


# ===========================================================================
# Evaluation data
# ===========================================================================
def build_gmm(seed=ANALYSIS_SEED) -> GaussianMixtureModel:
    """Build the analysis GMM (paper config) with a given seed."""
    c = SHARED_CONFIG
    return GaussianMixtureModel(K=c["K"], D=c["D"], L=c["L"],
                                epsilon=c["epsilon"], seed=seed,
                                offset=0.0, use_offset=False)


def make_eval_sets(n_samples=N_EVAL, seed=ANALYSIS_SEED) -> dict:
    """Canonical in-distribution and novel-class evaluation sets.

    Uses a FIXED seed so every checkpoint is probed on identical inputs.
    Returns {'in_dist': [...], 'novel': [...]}; each element is a
    (z_seq (N+1,D), labels (N,), target) tuple.
    """
    c = SHARED_CONFIG
    out = {}
    for split, novel in (("in_dist", False), ("novel", True)):
        gmm = build_gmm(seed)         # rebuild -> identical RNG state per split
        out[split] = generate_icl_gmm_data(
            gmm, n_samples, c["N"], novel_classes=novel,
            exact_copy=c["exact_copy"], B=c["B"], L=c["L"],
            shuffle_context=c["shuffle_context"], unique_labels=c["unique_labels"])
    return out


# ===========================================================================
# Instrumented forward pass  -- the SINGLE definition of winner/attention/etc.
# ===========================================================================
@dataclass
class Trace:
    """Instrumented forward-pass results for M examples (one checkpoint/split).

    See the module docstring: the WTA is SOFT -- Y is a simplex point, not a
    one-hot. `winner` and `dom_species` are two summary statistics of it.
    """
    label: str
    split: str               # "in_dist" | "novel"
    n_nodes: int
    z_flat: np.ndarray       # (M, (N+1)*D)  flattened input
    f: np.ndarray            # (M, n)        reaction rates f_j
    ratios: np.ndarray       # (M, n)        beta_j / f_j
    winner: np.ndarray       # (M,)          argmin_j ratio -- WTA SELECTION rule
    dom_species: np.ndarray  # (M,)          argmax_j Y_j   -- DOMINANT species in Y
    softmin_w: np.ndarray    # (M, n)        softmin weights at eval tau
    Y: np.ndarray            # (M, n)        steady-state concentrations
    Y_frac: np.ndarray       # (M, n)        Y normalised to the simplex
    q: np.ndarray            # (M, N)        context-position scores (Y @ B)
    attention: np.ndarray    # (M, N)        softmax attention over context
    pred_pos: np.ndarray     # (M,)          attended context position
    true_pos: np.ndarray     # (M,)          position the query actually copies
    pred_label: np.ndarray   # (M,)          predicted label
    target: np.ndarray       # (M,)          true label
    correct: np.ndarray      # (M,) bool     pred_label == target
    K: np.ndarray            # (n,)          carrying capacities K_j
    beta: np.ndarray         # (n,)          competition rates beta_j

    @property
    def n(self):
        return int(self.winner.shape[0])

    @property
    def accuracy(self):
        return 100.0 * float(self.correct.mean())

    @property
    def peak_share(self):
        """Per-example fraction of total Y held by the dominant species (M,)."""
        return self.Y_frac.max(axis=1)


@torch.no_grad()
def instrument(model, eval_set, temperature) -> Trace:
    """Run the instrumented forward pass over an eval set.

    eval_set: list of (z_seq, labels, target) tuples (as from make_eval_sets,
              or built by a module for probing).
    Every module MUST obtain winner/Y/attention/etc. through this function.
    """
    model.eval()
    z_seq = torch.stack([e[0] for e in eval_set]).float()       # (M, N+1, D)
    labels = torch.stack([e[1] for e in eval_set]).float()      # (M, N)
    target = torch.tensor([float(e[2]) for e in eval_set])      # (M,)
    M, Np1, D = z_seq.shape
    N = Np1 - 1
    z_flat = z_seq.reshape(M, -1)

    f, K, beta = model.compute_reaction_rates(z_flat, labels)   # (M, n) each
    ratios = beta / (f + 1e-10)
    winner = ratios.argmin(dim=1)

    tau = model.get_annealed_params()["tau"]                    # eval tau
    softmin_w = torch.softmax(-ratios / tau, dim=1)
    Y = model.winner_takes_all_softplus(f, K, beta, tau=tau,
                                        beta_softplus=model.beta_softplus)
    Y_frac = Y / (Y.sum(dim=1, keepdim=True) + 1e-12)
    dom_species = Y.argmax(dim=1)

    q = Y @ model.B
    attention = torch.softmax(q / temperature, dim=1)
    pred_pos = attention.argmax(dim=1)
    pred_label = labels[torch.arange(M), pred_pos]

    # which context position the query copies (exact_copy -> exact match)
    query = z_seq[:, N, :]
    dist = (z_seq[:, :N, :] - query[:, None, :]).pow(2).sum(-1)  # (M, N)
    true_pos = dist.argmin(dim=1)
    correct = pred_label.long() == target.long()

    return Trace(
        label="", split="", n_nodes=int(model.n_nodes),
        z_flat=z_flat.numpy(), f=f.numpy(), ratios=ratios.numpy(),
        winner=winner.numpy(), dom_species=dom_species.numpy(),
        softmin_w=softmin_w.numpy(), Y=Y.numpy(), Y_frac=Y_frac.numpy(),
        q=q.numpy(), attention=attention.numpy(),
        pred_pos=pred_pos.numpy(), true_pos=true_pos.numpy(),
        pred_label=pred_label.numpy(), target=target.numpy(),
        correct=correct.numpy(), K=K[0].numpy(), beta=beta[0].numpy())


def get_traces(checkpoint, eval_sets=None) -> dict:
    """Return {'in_dist': Trace, 'novel': Trace} for a checkpoint."""
    if eval_sets is None:
        eval_sets = make_eval_sets()
    temperature = float(checkpoint.params["temperature"])
    out = {}
    for split, data in eval_sets.items():
        tr = instrument(checkpoint.model, data, temperature)
        tr.label, tr.split = checkpoint.label, split
        out[split] = tr
    return out


def load_all(n_eval=N_EVAL):
    """Discover all checkpoints, build the shared eval sets, instrument each.

    Returns (checkpoints, traces) where
        traces[label] = {'in_dist': Trace, 'novel': Trace}.
    """
    checkpoints = discover_checkpoints()
    eval_sets = make_eval_sets(n_samples=n_eval)
    traces = {ck.label: get_traces(ck, eval_sets) for ck in checkpoints}
    return checkpoints, traces


# ===========================================================================
# Output helpers
# ===========================================================================
def setup_style():
    """Configure matplotlib (Agg backend) and return the pyplot module."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                         "font.size": 9, "axes.titlesize": 10})
    return plt


def module_outdir(module_name) -> Path:
    """Create and return OUTDIR/<module_name>/ for a module's figures."""
    d = OUTDIR / module_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig, outdir, name) -> Path:
    """Save a figure into outdir and close it. Returns the path."""
    import matplotlib.pyplot as plt
    path = Path(outdir) / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    cks, traces = load_all(n_eval=1000)
    print(f"Discovered {len(cks)} checkpoint(s):\n")
    for ck in cks:
        tr = traces[ck.label]
        print(f"  {ck.label:18s} n_nodes={ck.n_nodes:2d}  "
              f"stored novel={str(ck.results.get('novel_classes','?')):>6s}%  "
              f"re-eval novel={tr['novel'].accuracy:5.1f}%  "
              f"in_dist={tr['in_dist'].accuracy:5.1f}%")
    if cks:
        tr = traces[cks[0].label]["novel"]
        dom_vs_win = 100.0 * (tr.dom_species == tr.winner).mean()
        print(f"\nSanity ({cks[0].label}, novel split, M={tr.n}):")
        print(f"  Y peak share (dominant species): mean={tr.peak_share.mean():.3f} "
              f"-- SOFT mixture, not one-hot")
        print(f"  species with >5% of Y          : "
              f"mean={(tr.Y_frac > 0.05).sum(1).mean():.2f}")
        print(f"  winner(argmin ratio)==dom_species: {dom_vs_win:.1f}%")
        print(f"  routing accuracy (pred==true pos): "
              f"{100.0*(tr.pred_pos==tr.true_pos).mean():.1f}%")
