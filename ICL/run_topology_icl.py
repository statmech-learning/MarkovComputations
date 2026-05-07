"""Train a first-order CRN ICL model on a controlled directed topology.

This runner is the Phase 1/2 entrypoint for the topology project. It keeps
physical reaction topology explicit and separate from input-encoding sparsity:

* ``edges`` define the basal first-order reaction graph.
* ``input_mask`` defines which input coordinates may modulate each edge rate.
* novel-class ICL accuracy remains the primary metric.
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_generation import GaussianMixtureModel, generate_icl_gmm_data
from datasets import ICLGMMDataset, collate_fn
from evaluation import test_icl
from input_mask_utils import input_mask_summary, load_input_mask_json
from models import TopologyMatrixTreeMarkovICL
from topology_metrics import (
    compute_topology_metrics,
    graph_from_family,
    is_strongly_connected,
    normalize_edges,
)
from training import train_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--exact_copy", action="store_true", default=True)
    parser.add_argument("--no_exact_copy", dest="exact_copy", action="store_false")
    parser.add_argument("--unique_labels", action="store_true")

    parser.add_argument("--n_nodes", type=int, default=6)
    parser.add_argument(
        "--topology_family",
        type=str,
        default="complete",
        choices=[
            "complete",
            "directed_cycle",
            "bidirected_cycle",
            "cycle_chords",
            "random_sc",
            "hub_spoke",
            "two_module",
        ],
    )
    parser.add_argument("--n_edges", type=int, default=None)
    parser.add_argument("--topology_seed", type=int, default=0)
    parser.add_argument(
        "--edge_json",
        type=str,
        default=None,
        help="Optional JSON file with {'n_nodes': int, 'edges': [[source,target], ...]}.",
    )
    parser.add_argument("--allow_not_strongly_connected", action="store_true")

    parser.add_argument("--input_rho_edge", type=float, default=1.0)
    parser.add_argument("--input_rho_all", type=float, default=1.0)
    parser.add_argument("--input_mask_seed", type=int, default=0)
    parser.add_argument(
        "--input_mask_json",
        type=str,
        default=None,
        help="Optional JSON file containing an explicit binary input_mask aligned to edge order.",
    )

    parser.add_argument("--transform_func", type=str, default="exp", choices=["exp", "softplus", "relu"])
    parser.add_argument("--learn_base_rates", action="store_true", default=True)
    parser.add_argument("--freeze_base_rates", dest="learn_base_rates", action="store_false")
    parser.add_argument("--use_label_mod", action="store_true")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--train_samples", type=int, default=25000)
    parser.add_argument("--val_samples", type=int, default=5000)
    parser.add_argument("--eval_frequency", type=int, default=10)
    parser.add_argument("--n_eval_samples", type=int, default=500)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--method", type=str, default="direct_solve", choices=["direct_solve", "linear_solver", "matrix_tree"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--print_creation", action="store_true")

    return parser.parse_args()


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


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


def load_topology(args):
    if args.edge_json:
        with open(args.edge_json) as f:
            payload = json.load(f)
        n_nodes = int(payload.get("n_nodes", args.n_nodes))
        edges = normalize_edges(n_nodes, payload["edges"])
        return n_nodes, edges, payload.get("name", "edge_json")

    spec = graph_from_family(
        args.topology_family,
        n_nodes=args.n_nodes,
        n_edges=args.n_edges,
        seed=args.topology_seed,
    )
    return spec.n_nodes, spec.edges, spec.name


def make_input_mask(n_edges, p, rho_edge, rho_all, seed):
    if not 0.0 <= rho_edge <= 1.0:
        raise ValueError("--input_rho_edge must be in [0, 1]")
    if not 0.0 <= rho_all <= 1.0:
        raise ValueError("--input_rho_all must be in [0, 1]")

    rng = np.random.default_rng(seed)
    edge_mask = (rng.random((n_edges, 1)) < rho_edge).astype(float)
    coord_mask = (rng.random((n_edges, p)) < rho_all).astype(float)
    mask = edge_mask * coord_mask
    if rho_edge == 1.0 and rho_all == 1.0:
        mask = np.ones((n_edges, p), dtype=float)
    return mask


def load_input_mask(args, n_nodes, edges, p):
    if args.input_mask_json:
        mask, metadata = load_input_mask_json(args.input_mask_json, n_nodes, edges, p)
        return mask.astype(float), str(metadata.get("name", "input_mask_json")), metadata

    mask = make_input_mask(
        len(edges),
        p,
        args.input_rho_edge,
        args.input_rho_all,
        seed=args.input_mask_seed,
    )
    if args.input_rho_edge == 1.0 and args.input_rho_all == 1.0:
        name = "full"
    else:
        name = (
            f"random_rhoedge{args.input_rho_edge:g}"
            f"_rhoall{args.input_rho_all:g}_seed{args.input_mask_seed}"
        )
    return mask, name, {
        "name": name,
        "rho_edge": args.input_rho_edge,
        "rho_all": args.input_rho_all,
        "seed": args.input_mask_seed,
    }


def main():
    args = parse_args()
    params = vars(args).copy()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_nodes, edges, topology_name = load_topology(args)
    p = (args.N + 1) * args.D
    input_mask, input_mask_name, input_mask_metadata = load_input_mask(args, n_nodes, edges, p)
    if input_mask_name == "full":
        run_topology_name = topology_name
    elif input_mask_name.startswith(f"{topology_name}__mask"):
        run_topology_name = input_mask_name
    else:
        run_topology_name = f"{topology_name}__mask_{input_mask_name}"

    if not is_strongly_connected(n_nodes, edges) and not args.allow_not_strongly_connected:
        raise SystemExit(
            "Topology is not strongly connected. Use a strongly connected graph "
            "or pass --allow_not_strongly_connected for diagnostics only."
        )

    topology_metrics = compute_topology_metrics(
        n_nodes=n_nodes,
        edges=edges,
        p=p,
        input_mask=input_mask,
    )
    topology_metrics["topology_name"] = run_topology_name
    topology_metrics["physical_topology_name"] = topology_name
    topology_metrics["input_mask_name"] = input_mask_name
    topology_metrics["input_mask_family"] = input_mask_metadata.get("mask_family", "")
    topology_metrics["input_mask_seed"] = input_mask_metadata.get("seed", "")
    topology_metrics.update(input_mask_summary(input_mask))
    topology_metrics["raw_physical_parameter_count"] = int(len(edges) * p)
    topology_metrics["n_req"] = int(2 * args.N * (args.N + 1) * args.D)
    topology_metrics["d_rel_minus_n_req"] = int(topology_metrics["d_rel"] - topology_metrics["n_req"])

    device = choose_device(args.device)

    print("=" * 70, flush=True)
    print("TOPOLOGY FIRST-ORDER CRN ICL", flush=True)
    print("=" * 70, flush=True)
    print(json.dumps(json_ready(params), indent=2), flush=True)
    print(f"Topology: {run_topology_name}, nodes={n_nodes}, edges={len(edges)}", flush=True)
    print(
        "Input mask: "
        f"{input_mask_name}, coupled={topology_metrics['input_coupled_parameter_count']}/"
        f"{len(edges) * p}",
        flush=True,
    )
    print(
        "Structural: "
        f"rank_D={topology_metrics['rank_D']}, "
        f"d_rel={topology_metrics['d_rel']}, "
        f"n_req={topology_metrics['n_req']}, "
        f"effective_rank_D_masked={topology_metrics['effective_rank_D_masked']:.3f}",
        flush=True,
    )
    print(f"Device: {device}", flush=True)

    gmm = GaussianMixtureModel(
        K=args.K,
        D=args.D,
        L=args.L,
        epsilon=args.epsilon,
        seed=args.seed,
        offset=0.0,
    )

    train_data = generate_icl_gmm_data(
        gmm,
        args.train_samples,
        args.N,
        novel_classes=False,
        exact_copy=args.exact_copy,
        B=args.B,
        L=args.L,
        shuffle_context=True,
        unique_labels=args.unique_labels,
    )
    val_data = generate_icl_gmm_data(
        gmm,
        args.val_samples,
        args.N,
        novel_classes=False,
        exact_copy=args.exact_copy,
        B=args.B,
        L=args.L,
        shuffle_context=True,
        unique_labels=args.unique_labels,
    )
    train_loader = DataLoader(
        ICLGMMDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ICLGMMDataset(val_data),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    model = TopologyMatrixTreeMarkovICL(
        n_nodes=n_nodes,
        z_dim=args.D,
        L=args.L,
        N=args.N,
        edges=edges,
        input_mask=input_mask,
        use_label_mod=args.use_label_mod,
        learn_base_rates=args.learn_base_rates,
        transform_func=args.transform_func,
        print_creation=args.print_creation,
    )

    start_time = time.time()
    history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        n_epochs=args.epochs,
        lr=args.lr,
        method=args.method,
        temperature=args.temperature,
        gmm=gmm,
        N=args.N,
        B=args.B,
        L=args.L,
        exact_copy=args.exact_copy,
        eval_frequency=args.eval_frequency,
        n_eval_samples=args.n_eval_samples,
        unique_labels=args.unique_labels,
        show_progress=not args.no_progress,
    )
    execution_time = time.time() - start_time

    results = test_icl(
        model,
        gmm,
        args.N,
        device,
        n_samples=args.test_samples,
        exact_copy=args.exact_copy,
        B=args.B,
        method=args.method,
        L=args.L,
        temperature=args.temperature,
        shuffle_context=True,
        unique_labels=args.unique_labels,
    )

    os.makedirs(args.output, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output, "model.pt"))

    topology_payload = {
        "name": run_topology_name,
        "physical_topology_name": topology_name,
        "input_mask_name": input_mask_name,
        "input_mask_metadata": json_ready(input_mask_metadata),
        "n_nodes": n_nodes,
        "edges": [list(edge) for edge in edges],
        "input_mask": input_mask.astype(int).tolist(),
    }
    with open(os.path.join(args.output, "topology.json"), "w") as f:
        json.dump(topology_payload, f, indent=2)
    with open(os.path.join(args.output, "topology_metrics.json"), "w") as f:
        json.dump(json_ready(topology_metrics), f, indent=2)
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(json_ready(params), f, indent=2)
    with open(os.path.join(args.output, "results.pkl"), "wb") as f:
        pickle.dump(
            {
                "results": results,
                "history": history,
                "params": params,
                "topology": topology_payload,
                "topology_metrics": topology_metrics,
                "execution_time": execution_time,
            },
            f,
        )

    print(f"Saved results to {args.output}", flush=True)
    print(f"Execution time: {execution_time:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
