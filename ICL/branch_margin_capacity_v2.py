"""Lower-tail branch-margin capacity probes for first-order Markov ICL.

The first branch-margin capacity attempt in ``branch_margin_capacity.py`` is a
useful structural proxy, but it mostly scores average comparison behavior.  This
module targets the sharper objective requested by the Markov-ICL handoff:

    max_theta min_branch LCVaR_alpha[margin_theta(z) | z in branch]

under explicit norm constraints and input masks.  Three variants are included:

``exact``
    Root features are exact log-sum-exp matrix-tree numerators.

``tropical``
    Root features are max-over-tree tropical numerators.

``hard_root``
    A structural-compatibility probe that assigns labels directly to roots and
    scores root-feature margins without a learned decoder.

This is still a finite-sample nonconvex probe, not a theorem.  It is designed to
produce branch-wise lower-tail margins and failures for comparison with
``d_rel``, masked geometry, and normal-fan summaries.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from input_mask_utils import load_input_mask_json
from topology_metrics import graph_from_family, normalize_edges, topology_matrices


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_edge_json(path: str) -> Tuple[int, Tuple[Tuple[int, int], ...], str]:
    with open(path) as handle:
        payload = json.load(handle)
    n_nodes = int(payload["n_nodes"])
    edges = normalize_edges(n_nodes, payload["edges"])
    return n_nodes, edges, str(payload.get("name", Path(path).stem))


def sample_exact_copy_branches(
    n_samples: int,
    n_context: int,
    z_dim: int,
    seed: int = 0,
    query_noise: float = 0.0,
    context_scale: float = 1.0,
    branch_subset: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample flattened exact-copy context/query data and branch labels."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_context <= 1:
        raise ValueError("n_context must be at least 2")
    if z_dim <= 0:
        raise ValueError("z_dim must be positive")
    if branch_subset is None:
        active_branches = np.arange(n_context, dtype=int)
    else:
        active_branches = np.asarray(sorted({int(item) for item in branch_subset}), dtype=int)
        if active_branches.size == 0:
            raise ValueError("branch_subset must contain at least one branch")
        if np.any(active_branches < 0) or np.any(active_branches >= n_context):
            raise ValueError(f"branch_subset entries must be in 0..{n_context - 1}")
    rng = np.random.default_rng(seed)
    contexts = rng.normal(0.0, context_scale, size=(n_samples, n_context, z_dim))
    labels = rng.choice(active_branches, size=n_samples)
    queries = contexts[np.arange(n_samples), labels].copy()
    if query_noise:
        queries += rng.normal(0.0, query_noise, size=queries.shape)
    z = np.concatenate([contexts.reshape(n_samples, n_context * z_dim), queries], axis=1)
    return z, labels.astype(int)


def lse(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-sum-exp."""

    if values.shape[axis] == 0:
        raise ValueError("cannot log-sum-exp an empty axis")
    max_value = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_value)
    return np.squeeze(max_value, axis=axis) + np.log(np.sum(shifted, axis=axis))


def project_rows_to_l2_radius(matrix: np.ndarray, radius: float) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if radius == 0:
        return np.zeros_like(matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    scale = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
    return matrix * scale


def project_frobenius_radius(matrix: np.ndarray, radius: float) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if radius == 0:
        return np.zeros_like(matrix)
    norm = float(np.linalg.norm(matrix))
    if norm <= radius or norm <= 1e-12:
        return matrix
    return matrix * (radius / norm)


def tree_table_from_arborescences(
    arborescences: Mapping[int, Sequence[Sequence[int]]],
    n_edges: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(tree_roots, tree_incidence)`` arrays."""

    roots = []
    rows = []
    for root in sorted(arborescences):
        for tree in arborescences[root]:
            row = np.zeros(n_edges, dtype=float)
            row[list(tree)] = 1.0
            roots.append(int(root))
            rows.append(row)
    if not rows:
        return np.zeros(0, dtype=int), np.zeros((0, n_edges), dtype=float)
    return np.asarray(roots, dtype=int), np.vstack(rows)


def finite_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Matrix product with defensive finite cleanup for report generation."""

    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        product = left @ right
    if np.all(np.isfinite(product)):
        return product
    return np.nan_to_num(product, nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def root_feature_matrix(
    z: np.ndarray,
    K: np.ndarray,
    edge_bias: np.ndarray,
    tree_roots: np.ndarray,
    tree_incidence: np.ndarray,
    n_nodes: int,
    variant: str,
) -> np.ndarray:
    """Compute exact or tropical root log-numerator features."""

    tree_beta = finite_matmul(tree_incidence, edge_bias)
    tree_theta = finite_matmul(tree_incidence, K)
    tree_scores = finite_matmul(z, tree_theta.T) + tree_beta[None, :]
    features = np.full((z.shape[0], n_nodes), -np.inf, dtype=float)
    for root in range(n_nodes):
        cols = np.where(tree_roots == root)[0]
        if cols.size == 0:
            continue
        if variant == "exact":
            features[:, root] = lse(tree_scores[:, cols], axis=1)
        elif variant in ("tropical", "hard_root"):
            features[:, root] = np.max(tree_scores[:, cols], axis=1)
        else:
            raise ValueError(f"unknown variant {variant!r}")
    return features


def score_logits(
    features: np.ndarray,
    decoder: Optional[np.ndarray],
    labels: np.ndarray,
    variant: str,
    root_assignment: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if variant == "hard_root":
        if root_assignment is None:
            raise ValueError("hard_root requires root_assignment")
        logits = np.zeros((features.shape[0], len(root_assignment)), dtype=float)
        for label, root in enumerate(root_assignment):
            logits[:, label] = features[:, int(root)]
        return logits
    if decoder is None:
        raise ValueError("decoder is required for exact/tropical variants")
    return features @ decoder


def margin_vector(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    correct = logits[np.arange(logits.shape[0]), labels]
    masked = logits.copy()
    masked[np.arange(logits.shape[0]), labels] = -np.inf
    incorrect = np.max(masked, axis=1)
    return correct - incorrect


def lower_tail_mean(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("-inf")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    k = max(1, int(math.ceil(alpha * values.size)))
    return float(np.mean(np.sort(values)[:k]))


def summarize_branch_margins(
    margins: np.ndarray,
    labels: np.ndarray,
    n_context: int,
    alpha: float,
    active_branches: Optional[Sequence[int]] = None,
) -> dict:
    by_branch = []
    lcvars = []
    failures = []
    accuracies = []
    if active_branches is None:
        branch_iter = list(range(n_context))
    else:
        branch_iter = sorted({int(item) for item in active_branches})
    for branch in branch_iter:
        branch_values = margins[labels == branch]
        lcvar = lower_tail_mean(branch_values, alpha)
        failure_rate = float(np.mean(branch_values <= 0.0)) if branch_values.size else 1.0
        accuracy = float(np.mean(branch_values > 0.0)) if branch_values.size else 0.0
        lcvars.append(lcvar)
        failures.append(failure_rate)
        accuracies.append(accuracy)
        by_branch.append(
            {
                "branch": branch,
                "n": int(branch_values.size),
                "margin_mean": float(np.mean(branch_values)) if branch_values.size else None,
                "margin_min": float(np.min(branch_values)) if branch_values.size else None,
                "margin_lcvar": lcvar,
                "failure_rate": failure_rate,
                "accuracy": accuracy,
            }
        )
    return {
        "objective": float(np.min(lcvars)) if lcvars else float("-inf"),
        "branch_margin_lcvar_min": float(np.min(lcvars)) if lcvars else float("-inf"),
        "branch_margin_lcvar_mean": float(np.mean(lcvars)) if lcvars else float("-inf"),
        "branch_failure_rate_max": float(np.max(failures)) if failures else 1.0,
        "branch_accuracy_min": float(np.min(accuracies)) if accuracies else 0.0,
        "accuracy": float(np.mean(margins > 0.0)) if margins.size else 0.0,
        "margin_mean": float(np.mean(margins)) if margins.size else float("-inf"),
        "margin_p10": float(np.quantile(margins, 0.10)) if margins.size else float("-inf"),
        "active_branches": branch_iter,
        "by_branch": by_branch,
    }


def branch_direction_matrix(n_context: int, z_dim: int) -> np.ndarray:
    """Return context-query comparison directions in flattened input space."""

    p = (n_context + 1) * z_dim
    directions = []
    query_offset = n_context * z_dim
    for branch in range(n_context):
        context_offset = branch * z_dim
        direction = np.zeros(p, dtype=float)
        for dim in range(z_dim):
            direction[context_offset + dim] = 1.0
            direction[query_offset + dim] = -1.0
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            direction /= norm
        directions.append(direction)
    return np.vstack(directions)


def tree_drive_range_summary(
    K: np.ndarray,
    tree_roots: np.ndarray,
    tree_incidence: np.ndarray,
    n_nodes: int,
    n_context: int,
    z_dim: int,
) -> dict:
    directions = branch_direction_matrix(n_context, z_dim)
    tree_theta = finite_matmul(tree_incidence, K)
    values = []
    rows = []
    for root in range(n_nodes):
        root_thetas = tree_theta[tree_roots == root]
        if root_thetas.size == 0:
            continue
        for branch, direction in enumerate(directions):
            drives = finite_matmul(root_thetas, direction)
            value = float(np.max(drives) - np.min(drives))
            values.append(value)
            rows.append({"root": root, "branch": branch, "tree_drive_range": value})
    return {
        "tree_drive_range_min": float(np.min(values)) if values else 0.0,
        "tree_drive_range_mean": float(np.mean(values)) if values else 0.0,
        "tree_drive_range_max": float(np.max(values)) if values else 0.0,
        "tree_drive_ranges": rows,
    }


def random_parameters(
    rng: np.random.Generator,
    n_edges: int,
    p: int,
    n_nodes: int,
    n_context: int,
    input_mask: np.ndarray,
    projection_radius: float,
    decoder_radius: float,
    edge_bias_radius: float,
    variant: str,
    root_assignment: Optional[Sequence[int]] = None,
) -> dict:
    K = rng.normal(size=(n_edges, p))
    K = project_rows_to_l2_radius(K, projection_radius)
    K = K * input_mask
    edge_bias = rng.uniform(-edge_bias_radius, edge_bias_radius, size=n_edges)
    decoder = None
    if variant != "hard_root":
        decoder = rng.normal(size=(n_nodes, n_context))
        decoder = project_frobenius_radius(decoder, decoder_radius)
    return {
        "K": K,
        "edge_bias": edge_bias,
        "decoder": decoder,
        "root_assignment": list(root_assignment) if root_assignment is not None else None,
    }


def enumerate_root_assignments(n_nodes: int, n_context: int, max_assignments: int) -> list:
    if n_nodes < n_context:
        return []
    assignments = list(itertools.permutations(range(n_nodes), n_context))
    return [list(item) for item in assignments[:max_assignments]]


def evaluate_parameters(
    params: Mapping[str, object],
    z: np.ndarray,
    labels: np.ndarray,
    tree_roots: np.ndarray,
    tree_incidence: np.ndarray,
    n_nodes: int,
    n_context: int,
    z_dim: int,
    variant: str,
    alpha: float,
    active_branches: Optional[Sequence[int]] = None,
) -> dict:
    features = root_feature_matrix(
        z,
        np.asarray(params["K"], dtype=float),
        np.asarray(params["edge_bias"], dtype=float),
        tree_roots,
        tree_incidence,
        n_nodes=n_nodes,
        variant=variant,
    )
    logits = score_logits(
        features,
        None if params.get("decoder") is None else np.asarray(params["decoder"], dtype=float),
        labels,
        variant=variant,
        root_assignment=params.get("root_assignment"),
    )
    margins = margin_vector(logits, labels)
    summary = summarize_branch_margins(
        margins,
        labels,
        n_context=n_context,
        alpha=alpha,
        active_branches=active_branches,
    )
    summary.update(
        tree_drive_range_summary(
            np.asarray(params["K"], dtype=float),
            tree_roots,
            tree_incidence,
            n_nodes=n_nodes,
            n_context=n_context,
            z_dim=z_dim,
        )
    )
    return summary


def lower_tail_capacity_probe(
    n_nodes: int,
    edges: Iterable[Sequence[int]],
    n_context: int,
    z_dim: int,
    input_mask: Optional[np.ndarray] = None,
    variant: str = "exact",
    n_samples: int = 600,
    trials: int = 64,
    seed: int = 0,
    alpha: float = 0.10,
    projection_radius: float = 1.0,
    decoder_radius: float = 1.0,
    edge_bias_radius: float = 0.0,
    max_trees_per_root: Optional[int] = None,
    max_root_assignments: int = 60,
    branch_subset: Optional[Sequence[int]] = None,
) -> dict:
    """Run a finite-sample lower-tail capacity probe."""

    if variant not in {"exact", "tropical", "hard_root"}:
        raise ValueError("variant must be exact, tropical, or hard_root")
    p = (n_context + 1) * z_dim
    edge_tuple = normalize_edges(n_nodes, edges)
    if input_mask is None:
        mask = np.ones((len(edge_tuple), p), dtype=float)
    else:
        mask = np.asarray(input_mask, dtype=float)
        if mask.shape != (len(edge_tuple), p):
            raise ValueError(f"input_mask must have shape {(len(edge_tuple), p)}")

    mats = topology_matrices(n_nodes, edge_tuple, max_trees_per_root=max_trees_per_root)
    tree_roots, tree_incidence = tree_table_from_arborescences(
        mats["arborescences"],
        len(edge_tuple),
    )
    if tree_incidence.shape[0] == 0:
        raise ValueError("no rooted trees were enumerated")

    z, labels = sample_exact_copy_branches(
        n_samples=n_samples,
        n_context=n_context,
        z_dim=z_dim,
        seed=seed,
        branch_subset=branch_subset,
    )
    active_branches = sorted({int(item) for item in (branch_subset if branch_subset is not None else range(n_context))})

    rng = np.random.default_rng(seed + 10007)
    if variant == "hard_root":
        root_assignments = enumerate_root_assignments(
            n_nodes,
            n_context,
            max_assignments=max_root_assignments,
        )
        if not root_assignments:
            raise ValueError("hard_root requires n_nodes >= n_context")
    else:
        root_assignments = [None]

    best = None
    best_params = None
    trial_rows = []
    total_trials = 0
    for assignment in root_assignments:
        for _ in range(trials):
            total_trials += 1
            params = random_parameters(
                rng,
                len(edge_tuple),
                p,
                n_nodes,
                n_context,
                mask,
                projection_radius=projection_radius,
                decoder_radius=decoder_radius,
                edge_bias_radius=edge_bias_radius,
                variant=variant,
                root_assignment=assignment,
            )
            summary = evaluate_parameters(
                params,
                z,
                labels,
                tree_roots,
                tree_incidence,
                n_nodes=n_nodes,
                n_context=n_context,
                z_dim=z_dim,
                variant=variant,
                alpha=alpha,
                active_branches=active_branches,
            )
            row = {
                "objective": summary["objective"],
                "accuracy": summary["accuracy"],
                "branch_failure_rate_max": summary["branch_failure_rate_max"],
                "root_assignment": assignment,
            }
            trial_rows.append(row)
            if best is None or summary["objective"] > best["objective"]:
                best = summary
                best_params = params

    if best is None or best_params is None:
        raise RuntimeError("capacity probe did not evaluate any trials")

    return {
        "variant": variant,
        "n_nodes": n_nodes,
        "n_edges": len(edge_tuple),
        "n_context": n_context,
        "z_dim": z_dim,
        "p": p,
        "n_samples": n_samples,
        "trials_per_assignment": trials,
        "total_trials": total_trials,
        "alpha": alpha,
        "projection_radius": projection_radius,
        "decoder_radius": decoder_radius,
        "edge_bias_radius": edge_bias_radius,
        "max_trees_per_root": max_trees_per_root,
        "active_branches": active_branches,
        "n_trees_total": int(tree_incidence.shape[0]),
        "input_mask_density": float(np.mean(mask > 0)),
        "input_multiplicity_min": float(np.min(mask.sum(axis=0))),
        "input_multiplicity_mean": float(np.mean(mask.sum(axis=0))),
        "input_multiplicity_max": float(np.max(mask.sum(axis=0))),
        "best": best,
        "best_root_assignment": best_params.get("root_assignment"),
        "trial_objective_mean": float(np.mean([row["objective"] for row in trial_rows])),
        "trial_objective_max": float(np.max([row["objective"] for row in trial_rows])),
        "trial_rows": trial_rows,
    }


def markdown_report(results: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Branch-Margin Capacity V2",
        "",
        "Finite-sample lower-tail branch-margin probes. These are nonconvex probes, not capacity theorems.",
        "",
        "| variant | objective | acc | worst failure | p10 margin | drive range mean | trials |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        best = result["best"]
        lines.append(
            "| {variant} | {objective:.4f} | {accuracy:.3f} | {failure:.3f} | {p10:.4f} | {drive:.4f} | {trials} |".format(
                variant=result["variant"],
                objective=float(best["objective"]),
                accuracy=float(best["accuracy"]),
                failure=float(best["branch_failure_rate_max"]),
                p10=float(best["margin_p10"]),
                drive=float(best["tree_drive_range_mean"]),
                trials=int(result["total_trials"]),
            )
        )
    lines.extend(["", "## Branch Details", ""])
    for result in results:
        lines.append(f"### {result['variant']}")
        lines.append("")
        lines.append("| branch | n | LCVaR margin | mean margin | failure | accuracy |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in result["best"]["by_branch"]:
            lines.append(
                "| {branch} | {n} | {lcvar:.4f} | {mean:.4f} | {failure:.3f} | {acc:.3f} |".format(
                    branch=int(row["branch"]),
                    n=int(row["n"]),
                    lcvar=float(row["margin_lcvar"]),
                    mean=float(row["margin_mean"]),
                    failure=float(row["failure_rate"]),
                    acc=float(row["accuracy"]),
                )
            )
        lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge_json", type=str, default=None)
    parser.add_argument(
        "--topology_family",
        type=str,
        default="cycle_chords",
        choices=[
            "complete",
            "directed_cycle",
            "bidirected_cycle",
            "cycle_chords",
            "random_sc",
            "hub_spoke",
            "two_module",
            "degree_balanced",
            "bottleneck_bridge",
            "redundant_paths",
        ],
    )
    parser.add_argument("--n_nodes", type=int, default=5)
    parser.add_argument("--n_edges", type=int, default=12)
    parser.add_argument("--topology_seed", type=int, default=1)
    parser.add_argument("--input_mask_json", type=str, default=None)
    parser.add_argument("--n_context", type=int, default=3)
    parser.add_argument("--z_dim", type=int, default=2)
    parser.add_argument("--variants", type=str, default="exact,tropical,hard_root")
    parser.add_argument("--n_samples", type=int, default=600)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--projection_radius", type=float, default=1.0)
    parser.add_argument("--decoder_radius", type=float, default=1.0)
    parser.add_argument("--edge_bias_radius", type=float, default=0.0)
    parser.add_argument("--max_trees_per_root", type=int, default=None)
    parser.add_argument("--max_root_assignments", type=int, default=24)
    parser.add_argument(
        "--branch_subset",
        type=str,
        default=None,
        help="Comma-separated branch labels to sample and include in the lower-tail objective.",
    )
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.edge_json:
        n_nodes, edges, topology_name = load_edge_json(args.edge_json)
    else:
        spec = graph_from_family(
            args.topology_family,
            n_nodes=args.n_nodes,
            n_edges=args.n_edges,
            seed=args.topology_seed,
        )
        n_nodes, edges, topology_name = spec.n_nodes, spec.edges, spec.name

    p = (args.n_context + 1) * args.z_dim
    input_mask = None
    if args.input_mask_json:
        input_mask, _ = load_input_mask_json(args.input_mask_json, n_nodes, edges, p)

    branch_subset = None
    if args.branch_subset:
        branch_subset = [int(item.strip()) for item in args.branch_subset.split(",") if item.strip()]

    results = []
    for variant in [item.strip() for item in args.variants.split(",") if item.strip()]:
        results.append(
            lower_tail_capacity_probe(
                n_nodes=n_nodes,
                edges=edges,
                n_context=args.n_context,
                z_dim=args.z_dim,
                input_mask=input_mask,
                variant=variant,
                n_samples=args.n_samples,
                trials=args.trials,
                seed=args.seed,
                alpha=args.alpha,
                projection_radius=args.projection_radius,
                decoder_radius=args.decoder_radius,
                edge_bias_radius=args.edge_bias_radius,
                max_trees_per_root=args.max_trees_per_root,
                max_root_assignments=args.max_root_assignments,
                branch_subset=branch_subset,
            )
        )

    payload = {
        "topology_name": topology_name,
        "n_nodes": n_nodes,
        "edges": [list(edge) for edge in edges],
        "results": results,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as handle:
            json.dump(json_ready(payload), handle, indent=2)
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(markdown_report(results) + "\n")
    if not args.output_json and not args.output_md:
        print(json.dumps(json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
