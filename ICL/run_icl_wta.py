"""
WTA-ICL training runner (reconstructed).

This is a clean CLI re-implementation of the `run_icl_wta.py` script that
produced the paper's Winner-Takes-All ICL checkpoints. It was reconstructed
from the saved `results.pkl` hyperparameters and the refactored ICL modules
(`data_generation`, `datasets`, `models.wta_icl`, `training`, `evaluation`).

The hyperparameter defaults below reproduce the paper configuration:
    K=128, L=128, D=4, N=4, B=1, epsilon=0.001
    R0=2.0, softplus activations, beta_softplus=10.0, learn_K/beta=True
    epochs=1000, lr=0.0025, batch_size=20, train_samples=50000, val_samples=5000
    method='soft', temperature=0.1

Legacy sweep interface (matches the original SLURM jobs):
    python run_icl_wta.py --param1 <n_nodes> --param2 <rho_all> --param3 <seed> --output <dir>

Named interface (equivalent):
    python run_icl_wta.py --n_nodes 8 --sparsity_rho_all 1.0 --seed 30 --output <dir>

Output (in --output dir): model.pt, results.pkl, params.json
"""

import os
import sys
import time
import json
import pickle
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader

ICL_DIR = os.path.dirname(os.path.abspath(__file__))
if ICL_DIR not in sys.path:
    sys.path.insert(0, ICL_DIR)

from data_generation import GaussianMixtureModel, generate_icl_gmm_data
from datasets import ICLGMMDataset, collate_fn
from models.wta_icl import WinnerTakesAllICL
from training import train_model
from evaluation import test_icl


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- Legacy sweep aliases (param1=n_nodes, param2=rho_all, param3=seed) ---
    p.add_argument('--param1', type=int, default=None, help='alias for --n_nodes')
    p.add_argument('--param2', type=float, default=None, help='alias for --sparsity_rho_all')
    p.add_argument('--param3', type=int, default=None, help='alias for --seed')

    # --- Data ---
    p.add_argument('--K', type=int, default=128, help='Number of GMM classes')
    p.add_argument('--L', type=int, default=128, help='Number of output label classes')
    p.add_argument('--D', type=int, default=4, help='Feature dimension')
    p.add_argument('--N', type=int, default=4, help='Number of context examples')
    p.add_argument('--B', type=int, default=1, help='Burstiness')
    p.add_argument('--epsilon', type=float, default=0.001, help='Within-class noise scale')
    p.add_argument('--offset', type=float, default=0.0, help='Class-mean offset (unused unless use_offset)')
    p.add_argument('--exact_copy', type=lambda s: s.lower() != 'false', default=True)
    p.add_argument('--shuffle_context', type=lambda s: s.lower() != 'false', default=True)
    p.add_argument('--unique_labels', type=lambda s: s.lower() == 'true', default=False)
    p.add_argument('--train_samples', type=int, default=50000)
    p.add_argument('--val_samples', type=int, default=5000)

    # --- Model ---
    p.add_argument('--n_nodes', type=int, default=8, help='Number of chemical species Y_j')
    p.add_argument('--R0', type=float, default=2.0)
    p.add_argument('--activation', type=str, default='softplus', choices=['softplus', 'relu'])
    p.add_argument('--f_activation', type=str, default='softplus', choices=['softplus', 'exp'])
    p.add_argument('--beta_softplus', type=float, default=10.0)
    p.add_argument('--learn_K', type=lambda s: s.lower() != 'false', default=True)
    p.add_argument('--learn_beta', type=lambda s: s.lower() != 'false', default=True)
    p.add_argument('--sparsity_rho_edge', type=float, default=1.0)
    p.add_argument('--sparsity_rho_all', type=float, default=1.0)

    # --- Annealing (disabled by default for the paper config) ---
    p.add_argument('--use_annealing', type=lambda s: s.lower() == 'true', default=False)
    p.add_argument('--tau_start', type=float, default=0.2)
    p.add_argument('--tau_end', type=float, default=0.01)
    p.add_argument('--beta_softplus_start', type=float, default=2.0)
    p.add_argument('--beta_softplus_end', type=float, default=10.0)
    p.add_argument('--anneal_epochs', type=int, default=100)

    # --- Training ---
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr', type=float, default=0.0025)
    p.add_argument('--batch_size', type=int, default=20)
    p.add_argument('--method', type=str, default='soft')
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--eval_frequency', type=int, default=10)
    p.add_argument('--n_eval_samples', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)

    # --- I/O ---
    p.add_argument('--output', type=str, required=True, help='Output directory')

    args = p.parse_args()

    # Legacy aliases take precedence when supplied.
    if args.param1 is not None:
        args.n_nodes = args.param1
    if args.param2 is not None:
        args.sparsity_rho_all = args.param2
    if args.param3 is not None:
        args.seed = args.param3
    return args


def build_params(args):
    """Assemble the `params` dict saved in results.pkl (paper key schema)."""
    return {
        'K': args.K, 'L': args.L, 'D': args.D, 'N': args.N, 'B': args.B,
        'epsilon': args.epsilon, 'seed': args.seed,
        'exact_copy': args.exact_copy, 'shuffle_context': args.shuffle_context,
        'offset': args.offset, 'min_max_choice': None,
        'unique_labels': args.unique_labels,
        'n_nodes': args.n_nodes, 'R0': args.R0,
        'activation': args.activation, 'f_activation': args.f_activation,
        'beta_softplus': args.beta_softplus,
        'learn_K': args.learn_K, 'learn_beta': args.learn_beta,
        'use_annealing': args.use_annealing,
        'tau_start': args.tau_start, 'tau_end': args.tau_end,
        'beta_softplus_start': args.beta_softplus_start,
        'beta_softplus_end': args.beta_softplus_end,
        'anneal_epochs': args.anneal_epochs,
        'sparsity_rho_edge': args.sparsity_rho_edge,
        'sparsity_rho_all': args.sparsity_rho_all,
        'epochs': args.epochs, 'lr': args.lr, 'batch_size': args.batch_size,
        'train_samples': args.train_samples, 'val_samples': args.val_samples,
        'method': args.method, 'temperature': args.temperature,
    }


def main():
    args = parse_args()
    p = build_params(args)

    print("=" * 70)
    print("WTA ICL - CLASSIFICATION (Softmax Output)")
    print("=" * 70)
    print(f"K={p['K']}, D={p['D']}, N={p['N']}, B={p['B']}, nodes={p['n_nodes']}")
    print(f"Method: {p['method']}, Temperature: {p['temperature']}")
    print(f"seed={p['seed']}, rho_all={p['sparsity_rho_all']}, "
          f"epochs={p['epochs']}, lr={p['lr']}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # === GMM ===
    print("\nCreating GMM with discrete labels...")
    gmm = GaussianMixtureModel(
        K=args.K, D=args.D, L=args.L, epsilon=args.epsilon,
        seed=args.seed, offset=args.offset, use_offset=False,
    )
    print(f"  GMM: {args.K} classes with labels randomly assigned "
          f"from {{1, ..., {args.L}}}")
    print(f"  First 10 class labels: "
          f"{gmm.class_to_label[:min(10, args.K)].numpy()}")

    # === Data ===
    print("\nGenerating data...")
    train_data = generate_icl_gmm_data(
        gmm, args.train_samples, args.N, novel_classes=False,
        exact_copy=args.exact_copy, B=args.B, L=args.L,
        shuffle_context=args.shuffle_context, unique_labels=args.unique_labels,
    )
    val_data = generate_icl_gmm_data(
        gmm, args.val_samples, args.N, novel_classes=False,
        exact_copy=args.exact_copy, B=args.B, L=args.L,
        shuffle_context=args.shuffle_context, unique_labels=args.unique_labels,
    )
    train_loader = DataLoader(ICLGMMDataset(train_data), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ICLGMMDataset(val_data), batch_size=args.batch_size,
                            collate_fn=collate_fn)
    print(f"  Train: {len(train_data)} samples | Val: {len(val_data)} samples")

    # === Model ===
    print("\nCreating model...")
    model = WinnerTakesAllICL(
        n_nodes=args.n_nodes, z_dim=args.D, L=args.L, N=args.N,
        use_label_mod=False, alpha=1.0, R0=args.R0,
        activation=args.activation, beta_softplus=args.beta_softplus,
        f_activation=args.f_activation, learn_K=args.learn_K,
        learn_beta=args.learn_beta, sparsity_rho_edge=args.sparsity_rho_edge,
        sparsity_rho_all=args.sparsity_rho_all, print_creation=True,
        use_annealing=args.use_annealing, tau_start=args.tau_start,
        tau_end=args.tau_end, beta_softplus_start=args.beta_softplus_start,
        beta_softplus_end=args.beta_softplus_end, anneal_epochs=args.anneal_epochs,
    )

    # === Train ===
    print("\nTraining...")
    print("=" * 70)
    start = time.time()
    history = train_model(
        model, train_loader, val_loader, device,
        n_epochs=args.epochs, lr=args.lr, method=args.method,
        temperature=args.temperature, gmm=gmm, N=args.N, B=args.B, L=args.L,
        exact_copy=args.exact_copy, eval_frequency=args.eval_frequency,
        n_eval_samples=args.n_eval_samples, unique_labels=args.unique_labels,
    )
    training_time = time.time() - start
    print(f"\nTraining time: {training_time:.2f} seconds")

    # === Test ===
    results = test_icl(
        model, gmm, args.N, device, n_samples=1000,
        exact_copy=args.exact_copy, B=args.B, method=args.method,
        L=args.L, temperature=args.temperature,
        shuffle_context=args.shuffle_context,
    )

    # === Save ===
    os.makedirs(args.output, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output, 'model.pt'))
    with open(os.path.join(args.output, 'results.pkl'), 'wb') as f:
        pickle.dump({
            'results': results,
            'history': history,
            'params': p,
            'execution_time': training_time,
        }, f)
    with open(os.path.join(args.output, 'params.json'), 'w') as f:
        json.dump(p, f, indent=2)

    print(f"\n✓ Saved model to {os.path.join(args.output, 'model.pt')}")
    print(f"✓ Saved results to {os.path.join(args.output, 'results.pkl')}")
    print(f"✓ Execution Time: {training_time:.2f} seconds")


if __name__ == '__main__':
    main()
