"""Analyze active trees and edge sensitivities for a trained topology ICL run."""

import argparse
import json
import os

import numpy as np
import torch

from data_generation import GaussianMixtureModel, generate_icl_gmm_data
from models import TopologyMatrixTreeMarkovICL
from topology_analysis import analyze_batch, edge_ablation_scan
from topology_metrics import topology_matrices


def choose_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--ablate_input", action="store_true")
    parser.add_argument("--ablate_physical", action="store_true")
    parser.add_argument("--physical_epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.run_dir, "topology.json")) as f:
        topology = json.load(f)

    device = choose_device(args.device)
    input_mask = np.asarray(topology["input_mask"], dtype=float)
    model = TopologyMatrixTreeMarkovICL(
        n_nodes=topology["n_nodes"],
        z_dim=config["D"],
        L=config["L"],
        N=config["N"],
        edges=topology["edges"],
        input_mask=input_mask,
        use_label_mod=config.get("use_label_mod", False),
        learn_base_rates=config.get("learn_base_rates", True),
        transform_func=config.get("transform_func", "exp"),
        print_creation=False,
    )
    state = torch.load(os.path.join(args.run_dir, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    gmm = GaussianMixtureModel(
        K=config["K"],
        D=config["D"],
        L=config["L"],
        epsilon=config["epsilon"],
        seed=config["seed"],
        offset=0.0,
    )
    data = generate_icl_gmm_data(
        gmm,
        args.n_samples,
        config["N"],
        novel_classes=True,
        exact_copy=config.get("exact_copy", True),
        B=config.get("B", 1),
        L=config["L"],
        shuffle_context=True,
        unique_labels=config.get("unique_labels", False),
    )
    z_seq = torch.stack([item[0] for item in data]).to(device)
    labels = torch.stack([item[1] for item in data]).to(device)
    targets = torch.as_tensor([int(item[2].item()) for item in data], device=device)

    matrices = topology_matrices(topology["n_nodes"], topology["edges"])
    metrics = analyze_batch(
        model,
        z_seq,
        labels,
        target_labels=targets,
        matrices=matrices,
        method=config.get("method", "direct_solve"),
        temperature=config.get("temperature", 1.0),
    )
    metrics["n_samples"] = args.n_samples
    metrics["run_dir"] = args.run_dir
    metrics["topology_name"] = topology.get("name", "custom")

    if args.ablate_input:
        metrics["input_edge_ablation"] = edge_ablation_scan(
            model,
            z_seq,
            labels,
            targets,
            mode="input",
            method=config.get("method", "direct_solve"),
            temperature=config.get("temperature", 1.0),
        )
    if args.ablate_physical:
        metrics["physical_edge_ablation"] = edge_ablation_scan(
            model,
            z_seq,
            labels,
            targets,
            mode="physical",
            method=config.get("method", "direct_solve"),
            temperature=config.get("temperature", 1.0),
            physical_epsilon=args.physical_epsilon,
        )

    output = args.output or os.path.join(args.run_dir, "mechanism_metrics.json")
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote mechanism metrics to {output}")
    print(
        "branch MI: "
        f"root={metrics['branch_active_root_mi']:.4f}, "
        f"tree={metrics['branch_active_tree_mi']:.4f}; "
        f"edge importance max={metrics['edge_importance_max']:.4e}"
    )


if __name__ == "__main__":
    main()
