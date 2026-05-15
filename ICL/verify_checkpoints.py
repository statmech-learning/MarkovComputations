"""
Verify trained WTA-ICL checkpoints reproduce their reported metrics.

For each run directory (containing model.pt + results.pkl), this:
  1. reads the stored hyperparameters (`params`) and stored test results,
  2. rebuilds the Gaussian Mixture Model with the run's seed,
  3. loads the trained weights into a fresh WinnerTakesAllICL model,
  4. re-runs test_icl / evaluate_iwl and compares against the stored numbers.

Re-evaluation is statistical (fresh test samples), so a match within ~1-2%
confirms the checkpoint is intact and usable.

Usage:
    python verify_checkpoints.py /path/to/run_dir [more_run_dirs ...]
"""

import sys
import os
import pickle
import argparse

import torch

ICL_DIR = os.path.dirname(os.path.abspath(__file__))
if ICL_DIR not in sys.path:
    sys.path.insert(0, ICL_DIR)

from data_generation import GaussianMixtureModel
from evaluation import test_icl, evaluate_iwl
from models.wta_icl import load_model


def verify(run_dir, n_samples=1000):
    run_dir = os.path.abspath(run_dir)
    with open(os.path.join(run_dir, 'results.pkl'), 'rb') as f:
        res = pickle.load(f)

    p = res['params']
    stored = res['results']

    print("=" * 70)
    print(f"RUN: {run_dir}")
    print(f"  n_nodes={p['n_nodes']}  seed={p['seed']}  "
          f"K={p['K']} L={p['L']} D={p['D']} N={p['N']} B={p['B']}")
    print(f"  method={p['method']}  temperature={p['temperature']}  "
          f"epsilon={p['epsilon']}")
    print(f"  STORED results: in_dist={stored['in_dist']:.2f}%  "
          f"novel_classes={stored['novel_classes']:.2f}%")
    print("=" * 70)

    device = torch.device('cpu')

    # Rebuild the GMM with the same seed used for this run.
    gmm = GaussianMixtureModel(
        K=p['K'], D=p['D'], L=p['L'], epsilon=p['epsilon'],
        seed=p['seed'], offset=p.get('offset', 1.0),
        use_offset=p.get('use_offset', False),
    )

    # Load the trained weights (load_model expects a trailing separator).
    model = load_model(p, run_dir + os.sep, print_creation=True)

    # Re-run the same evaluation used at training time.
    re_eval = test_icl(
        model, gmm, p['N'], device, n_samples=n_samples,
        exact_copy=p['exact_copy'], B=p['B'], method=p['method'],
        L=p['L'], temperature=p['temperature'],
        shuffle_context=p['shuffle_context'],
    )
    iwl = evaluate_iwl(
        model, gmm, p['N'], device, n_eval_samples=n_samples,
        L=p['L'], method=p['method'], temperature=p['temperature'],
        shuffle_context=p['shuffle_context'],
    )

    print("\n" + "-" * 70)
    print("RESULT COMPARISON (stored vs. fresh re-evaluation)")
    print(f"  in_dist        : {stored['in_dist']:7.2f}%  ->  {re_eval['in_dist']:7.2f}%")
    print(f"  novel_classes  : {stored['novel_classes']:7.2f}%  ->  {re_eval['novel_classes']:7.2f}%")
    print(f"  iwl (re-eval)  :          ->  {iwl:7.2f}%  (expected near-chance ~{100.0/p['L']:.1f}%)")
    print("-" * 70)

    ok = (abs(stored['novel_classes'] - re_eval['novel_classes']) < 3.0 and
          abs(stored['in_dist'] - re_eval['in_dist']) < 3.0)
    print("  VERDICT:", "PASS - checkpoint reproduces reported metrics"
          if ok else "CHECK  - re-eval drifted >3% from stored")
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dirs', nargs='+', help='Run directories to verify')
    ap.add_argument('--n_samples', type=int, default=1000)
    args = ap.parse_args()

    results = []
    for d in args.run_dirs:
        results.append(verify(d, n_samples=args.n_samples))
        print()

    print("=" * 70)
    print(f"SUMMARY: {sum(results)}/{len(results)} checkpoints verified")
    print("=" * 70)
    sys.exit(0 if all(results) else 1)
