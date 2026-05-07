#################################################
################  Import things #################
#################################################

import torch
import numpy as np
import pickle
import argparse
from torch.utils.data import DataLoader
import os
import time

from data_generation_regression import generate_icl_regression_data
from datasets_regression import ICLRegressionDataset, collate_fn_regression
from models import MatrixTreeMarkovICLRegression
from training_regression import train_model_regression
from evaluation_regression import test_regression


parser = argparse.ArgumentParser(description="Run Markov ICL regression experiment.")
parser.add_argument("--param1", type=int, required=True, help="Number of Markov nodes")
parser.add_argument("--param2", type=int, required=False, default=0, help="Random seed")
parser.add_argument("--output", type=str, required=True, help="Output directory")
parser.add_argument(
    "--context_scorer",
    type=str,
    choices=["linear", "mlp"],
    default="linear",
    help="Regression head type",
)
parser.add_argument(
    "--mlp_depth",
    type=int,
    default=2,
    help="Depth of MLP regression head (only when --context_scorer mlp)",
)
parser.add_argument(
    "--mlp_width",
    type=int,
    default=64,
    help="Hidden width of MLP regression head (only when --context_scorer mlp)",
)
args = parser.parse_args()

output_dir = args.output

# ============================================================
# Data parameters
# ============================================================
D = 8
N = 3
noise_std = 0.1
task_scale = 1.0
y_pad = 0.0
seed = args.param2

# ============================================================
# Model parameters
# ============================================================
n_nodes = args.param1
z_dim = D + 1  # x plus y-slot
transform_func = "exp"
learn_base_rates = True
context_scorer_type = args.context_scorer
mlp_depth = args.mlp_depth
mlp_width = args.mlp_width

# ============================================================
# Sparsity parameters
# ============================================================
sparsity_rho_edge = 1.0
sparsity_rho_all = 1.0
sparsity_rho_edge_base_W = 1.0
base_mask_value = float("-inf")

# ============================================================
# Training / inference
# ============================================================
epochs = 1000
lr = 0.0025
batch_size = 50
train_samples = 250
val_samples = 5000
method = "direct_solve"

params = {
    "D": D,
    "N": N,
    "noise_std": noise_std,
    "task_scale": task_scale,
    "y_pad": y_pad,
    "seed": seed,
    "n_nodes": n_nodes,
    "z_dim": z_dim,
    "transform_func": transform_func,
    "learn_base_rates": learn_base_rates,
    "context_scorer_type": context_scorer_type,
    "mlp_depth": mlp_depth,
    "mlp_width": mlp_width,
    "sparsity_rho_edge": sparsity_rho_edge,
    "sparsity_rho_all": sparsity_rho_all,
    "sparsity_rho_edge_base_W": sparsity_rho_edge_base_W,
    "base_mask_value": base_mask_value,
    "epochs": epochs,
    "lr": lr,
    "batch_size": batch_size,
    "train_samples": train_samples,
    "val_samples": val_samples,
    "method": method,
}

print("=" * 70)
print("MARKOV ICL - REGRESSION")
print("=" * 70)
print(f"D={D}, N={N}, nodes={n_nodes}, z_dim={z_dim}")
print(f"Method: {method}")
print(f"Regression head: {context_scorer_type} (depth={mlp_depth}, width={mlp_width})")
print("=" * 70)

torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

print("Generating data...")
train_data = generate_icl_regression_data(
    n_samples=train_samples,
    N=N,
    D=D,
    noise_std=noise_std,
    task_scale=task_scale,
    y_pad=y_pad,
    seed=seed,
)
val_data = generate_icl_regression_data(
    n_samples=val_samples,
    N=N,
    D=D,
    noise_std=noise_std,
    task_scale=task_scale,
    y_pad=y_pad,
)

train_loader = DataLoader(
    ICLRegressionDataset(train_data),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn_regression,
)
val_loader = DataLoader(
    ICLRegressionDataset(val_data),
    batch_size=batch_size,
    collate_fn=collate_fn_regression,
)

print("\nCreating model...")
model = MatrixTreeMarkovICLRegression(
    n_nodes=n_nodes,
    z_dim=z_dim,
    N=N,
    learn_base_rates=learn_base_rates,
    transform_func=transform_func,
    sparsity_rho_edge=sparsity_rho_edge,
    sparsity_rho_all=sparsity_rho_all,
    sparsity_rho_edge_base_W=sparsity_rho_edge_base_W,
    base_mask_value=base_mask_value,
    context_scorer_type=context_scorer_type,
    mlp_depth=mlp_depth,
    mlp_width=mlp_width,
)

start_time = time.time()
print("\nTraining...")
print("=" * 70)
history = train_model_regression(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    n_epochs=epochs,
    lr=lr,
    method=method,
)
end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

results = test_regression(
    model=model,
    N=N,
    D=D,
    device=device,
    n_samples=1000,
    noise_std=noise_std,
    task_scale=task_scale,
    y_pad=y_pad,
    method=method,
)

os.makedirs(output_dir, exist_ok=True)
model_path = f"{output_dir}/model.pt"
torch.save(model.state_dict(), model_path)

results_data = {
    "results": results,
    "history": history,
    "params": params,
    "execution_time": end_time - start_time,
}
results_path = f"{output_dir}/results.pkl"
with open(results_path, "wb") as file:
    pickle.dump(results_data, file)

print(f"\nSaved model to {model_path}")
print(f"Saved results to {results_path}")
print(f"Execution Time: {end_time - start_time:.2f} seconds")
