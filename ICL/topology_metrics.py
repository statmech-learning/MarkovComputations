"""Topology utilities for first-order CRN ICL experiments.

The first-order Markov CRN uses directed reactions ``source -> target``.  For
the stationary distribution of a column-sum-zero generator, the matrix-tree
numerator for root ``r`` is the sum over directed spanning trees whose edges
lead every non-root node to ``r``.  This module keeps that convention explicit:
edge tuples are always ``(source, target)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Edge = Tuple[int, int]
Tree = Tuple[int, ...]


@dataclass(frozen=True)
class TopologySpec:
    """Serializable graph specification."""

    n_nodes: int
    edges: Tuple[Edge, ...]
    name: str = "custom"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_nodes": self.n_nodes,
            "edges": [list(edge) for edge in self.edges],
        }


def normalize_edges(n_nodes: int, edges: Iterable[Sequence[int]]) -> Tuple[Edge, ...]:
    """Validate, de-duplicate, and normalize directed edges."""

    normalized: List[Edge] = []
    seen = set()
    for raw in edges:
        if len(raw) != 2:
            raise ValueError(f"Edge must have length 2, got {raw!r}")
        source, target = int(raw[0]), int(raw[1])
        if not 0 <= source < n_nodes or not 0 <= target < n_nodes:
            raise ValueError(f"Edge {(source, target)} outside 0..{n_nodes - 1}")
        if source == target:
            continue
        edge = (source, target)
        if edge not in seen:
            seen.add(edge)
            normalized.append(edge)
    return tuple(normalized)


def adjacency_lists(n_nodes: int, edges: Sequence[Edge]) -> Tuple[List[List[int]], List[List[int]]]:
    outgoing = [[] for _ in range(n_nodes)]
    incoming = [[] for _ in range(n_nodes)]
    for idx, (source, target) in enumerate(edges):
        outgoing[source].append(idx)
        incoming[target].append(idx)
    return outgoing, incoming


def _reachable_from(start: int, n_nodes: int, adjacency: Sequence[Sequence[int]]) -> set:
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def is_strongly_connected(n_nodes: int, edges: Sequence[Edge]) -> bool:
    """Return whether every node can reach every other node."""

    if n_nodes == 0:
        return False
    graph = [[] for _ in range(n_nodes)]
    reverse = [[] for _ in range(n_nodes)]
    for source, target in edges:
        graph[source].append(target)
        reverse[target].append(source)
    return (
        len(_reachable_from(0, n_nodes, graph)) == n_nodes
        and len(_reachable_from(0, n_nodes, reverse)) == n_nodes
    )


def complete_digraph(n_nodes: int) -> TopologySpec:
    edges = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
    return TopologySpec(n_nodes=n_nodes, edges=tuple(edges), name="complete")


def directed_cycle(n_nodes: int) -> TopologySpec:
    edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    return TopologySpec(n_nodes=n_nodes, edges=tuple(edges), name="directed_cycle")


def bidirected_cycle(n_nodes: int) -> TopologySpec:
    edges = []
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        edges.append((i, j))
        edges.append((j, i))
    return TopologySpec(n_nodes=n_nodes, edges=normalize_edges(n_nodes, edges), name="bidirected_cycle")


def cycle_plus_chords(n_nodes: int, n_edges: int, seed: int = 0) -> TopologySpec:
    """Directed cycle with deterministic random chords up to ``n_edges``."""

    base = list(directed_cycle(n_nodes).edges)
    all_edges = list(complete_digraph(n_nodes).edges)
    candidates = [edge for edge in all_edges if edge not in base]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    edges = normalize_edges(n_nodes, base + candidates[: max(0, n_edges - len(base))])
    if len(edges) != n_edges:
        raise ValueError(f"Cannot build {n_edges} unique edges for n_nodes={n_nodes}")
    return TopologySpec(n_nodes=n_nodes, edges=edges, name="cycle_chords")


def random_strongly_connected_digraph(
    n_nodes: int,
    n_edges: int,
    seed: int = 0,
    max_tries: int = 10000,
) -> TopologySpec:
    """Sample a strongly connected directed graph with exactly ``n_edges``."""

    min_edges = n_nodes
    max_edges = n_nodes * (n_nodes - 1)
    if not min_edges <= n_edges <= max_edges:
        raise ValueError(f"n_edges must be in [{min_edges}, {max_edges}], got {n_edges}")

    rng = random.Random(seed)
    cycle_edges = list(directed_cycle(n_nodes).edges)
    candidates = [edge for edge in complete_digraph(n_nodes).edges if edge not in cycle_edges]

    for _ in range(max_tries):
        extra = rng.sample(candidates, n_edges - n_nodes)
        edges = normalize_edges(n_nodes, cycle_edges + extra)
        if is_strongly_connected(n_nodes, edges):
            return TopologySpec(n_nodes=n_nodes, edges=edges, name="random_sc")
    raise RuntimeError(f"Failed to sample strongly connected graph after {max_tries} attempts")


def hub_spoke_strong(n_nodes: int, n_edges: int, hub: int = 0, seed: int = 0) -> TopologySpec:
    """Hub-heavy strongly connected graph, padded with random non-hub edges."""

    base = []
    for node in range(n_nodes):
        if node != hub:
            base.append((hub, node))
            base.append((node, hub))
    if n_edges < len(base):
        raise ValueError(f"Hub graph needs at least {len(base)} edges")
    candidates = [edge for edge in complete_digraph(n_nodes).edges if edge not in base]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    edges = normalize_edges(n_nodes, base + candidates[: n_edges - len(base)])
    return TopologySpec(n_nodes=n_nodes, edges=edges, name="hub_spoke")


def two_module_bridge(n_nodes: int, n_edges: int, seed: int = 0) -> TopologySpec:
    """Two dense modules joined by a small bidirectional bridge, then padded."""

    if n_nodes < 4:
        raise ValueError("two_module_bridge requires at least 4 nodes")
    split = n_nodes // 2
    left = list(range(split))
    right = list(range(split, n_nodes))
    base = []
    for group in (left, right):
        for source in group:
            for target in group:
                if source != target:
                    base.append((source, target))
    base.extend([(left[-1], right[0]), (right[0], left[-1])])
    base = list(normalize_edges(n_nodes, base))
    if n_edges < len(base):
        raise ValueError(f"Two-module graph needs at least {len(base)} edges")
    candidates = [edge for edge in complete_digraph(n_nodes).edges if edge not in base]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    edges = normalize_edges(n_nodes, base + candidates[: n_edges - len(base)])
    return TopologySpec(n_nodes=n_nodes, edges=edges, name="two_module")


def graph_from_family(
    family: str,
    n_nodes: int,
    n_edges: Optional[int] = None,
    seed: int = 0,
) -> TopologySpec:
    """Build a named topology family."""

    if family == "complete":
        spec = complete_digraph(n_nodes)
        if n_edges is not None and n_edges != len(spec.edges):
            raise ValueError("complete graph edge count is fixed")
        return spec
    if family == "directed_cycle":
        spec = directed_cycle(n_nodes)
        if n_edges is not None and n_edges != len(spec.edges):
            raise ValueError("directed_cycle edge count is fixed")
        return spec
    if family == "bidirected_cycle":
        spec = bidirected_cycle(n_nodes)
        if n_edges is not None and n_edges != len(spec.edges):
            raise ValueError("bidirected_cycle edge count is fixed")
        return spec
    if n_edges is None:
        raise ValueError(f"family {family!r} requires n_edges")
    if family == "cycle_chords":
        return cycle_plus_chords(n_nodes, n_edges, seed=seed)
    if family == "random_sc":
        return random_strongly_connected_digraph(n_nodes, n_edges, seed=seed)
    if family == "hub_spoke":
        return hub_spoke_strong(n_nodes, n_edges, seed=seed)
    if family == "two_module":
        return two_module_bridge(n_nodes, n_edges, seed=seed)
    raise ValueError(f"Unknown topology family: {family}")


def _tree_reaches_root(selected: Mapping[int, int], edges: Sequence[Edge], node: int, root: int) -> bool:
    seen = set()
    current = node
    while current != root:
        if current in seen or current not in selected:
            return False
        seen.add(current)
        edge_idx = selected[current]
        current = edges[edge_idx][1]
    return True


def enumerate_rooted_arborescences(
    n_nodes: int,
    edges: Sequence[Edge],
    root: int,
    max_trees: Optional[int] = None,
) -> List[Tree]:
    """Enumerate directed spanning trees leading to ``root``.

    Each non-root node contributes exactly one outgoing edge. The selected
    edges are valid iff following outgoing edges from every non-root reaches
    the root. This is intended for small graphs used in topology diagnostics.
    """

    if not 0 <= root < n_nodes:
        raise ValueError(f"Invalid root {root}")
    outgoing, _ = adjacency_lists(n_nodes, edges)
    non_roots = [node for node in range(n_nodes) if node != root]
    choices = []
    for node in non_roots:
        if not outgoing[node]:
            return []
        choices.append(outgoing[node])

    trees: List[Tree] = []
    for selected_edges in product(*choices):
        selected = dict(zip(non_roots, selected_edges))
        if all(_tree_reaches_root(selected, edges, node, root) for node in non_roots):
            trees.append(tuple(sorted(selected_edges)))
            if max_trees is not None and len(trees) >= max_trees:
                break
    return trees


def enumerate_arborescences(
    n_nodes: int,
    edges: Sequence[Edge],
    max_trees_per_root: Optional[int] = None,
) -> Dict[int, List[Tree]]:
    return {
        root: enumerate_rooted_arborescences(
            n_nodes, edges, root, max_trees=max_trees_per_root
        )
        for root in range(n_nodes)
    }


def tree_counts_by_determinant(n_nodes: int, edges: Sequence[Edge]) -> List[int]:
    """Count rooted arborescences by the directed matrix-tree theorem."""

    generator = np.zeros((n_nodes, n_nodes), dtype=float)
    for source, target in edges:
        generator[target, source] += 1.0
    col_sums = generator.sum(axis=0)
    generator -= np.diag(col_sums)

    counts = []
    for root in range(n_nodes):
        keep = [idx for idx in range(n_nodes) if idx != root]
        minor = generator[np.ix_(keep, keep)]
        counts.append(int(round(abs(np.linalg.det(minor)))))
    return counts


def tree_numerators_by_determinant(
    n_nodes: int,
    edges: Sequence[Edge],
    rates: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Weighted matrix-tree numerators by cofactors."""

    edge_tuple = normalize_edges(n_nodes, edges)
    if rates is None:
        rate_arr = np.ones(len(edge_tuple), dtype=float)
    else:
        rate_arr = np.asarray(rates, dtype=float)
        if rate_arr.shape != (len(edge_tuple),):
            raise ValueError(f"rates must have shape ({len(edge_tuple)},)")

    generator = np.zeros((n_nodes, n_nodes), dtype=float)
    for rate, (source, target) in zip(rate_arr, edge_tuple):
        generator[target, source] += rate
    col_sums = generator.sum(axis=0)
    generator -= np.diag(col_sums)

    numerators = []
    for root in range(n_nodes):
        keep = [idx for idx in range(n_nodes) if idx != root]
        minor = generator[np.ix_(keep, keep)]
        numerators.append(abs(np.linalg.det(minor)))
    return np.asarray(numerators, dtype=float)


def tree_numerators_by_enumeration(
    n_nodes: int,
    edges: Sequence[Edge],
    rates: Sequence[float],
    arborescences: Optional[Mapping[int, Sequence[Tree]]] = None,
) -> np.ndarray:
    """Weighted matrix-tree numerators by explicit tree products."""

    edge_tuple = normalize_edges(n_nodes, edges)
    rate_arr = np.asarray(rates, dtype=float)
    if rate_arr.shape != (len(edge_tuple),):
        raise ValueError(f"rates must have shape ({len(edge_tuple)},)")
    if arborescences is None:
        arborescences = enumerate_arborescences(n_nodes, edge_tuple)

    numerators = np.zeros(n_nodes, dtype=float)
    for root in range(n_nodes):
        total = 0.0
        for tree in arborescences[root]:
            total += float(np.prod(rate_arr[list(tree)]))
        numerators[root] = total
    return numerators


def incidence_matrix(arborescences: Mapping[int, Sequence[Tree]], n_edges: int) -> np.ndarray:
    rows = []
    for root in sorted(arborescences):
        for tree in arborescences[root]:
            row = np.zeros(n_edges, dtype=float)
            row[list(tree)] = 1.0
            rows.append(row)
    if not rows:
        return np.zeros((0, n_edges), dtype=float)
    return np.vstack(rows)


def relative_tree_matrix(incidence: np.ndarray) -> np.ndarray:
    if incidence.shape[0] <= 1:
        return np.zeros((0, incidence.shape[1]), dtype=float)
    return incidence[1:, :] - incidence[0:1, :]


def centered_tree_matrix(incidence: np.ndarray) -> np.ndarray:
    """Return an origin-free matrix with the same row span as tree contrasts."""

    if incidence.shape[0] == 0:
        return np.zeros((0, incidence.shape[1]), dtype=float)
    return incidence - incidence.mean(axis=0, keepdims=True)


def svd_metrics(matrix: np.ndarray, tol: float = 1e-9) -> dict:
    if matrix.size == 0:
        return {
            "rank": 0,
            "singular_values": [],
            "effective_rank": 0.0,
            "condition_number": math.inf,
        }
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular_values > tol))
    positive = singular_values[singular_values > tol]
    if positive.size == 0:
        effective_rank = 0.0
        condition_number = math.inf
    else:
        weights = positive**2
        weights = weights / weights.sum()
        effective_rank = float(np.exp(-np.sum(weights * np.log(weights))))
        condition_number = float(positive.max() / positive.min())
    return {
        "rank": rank,
        "singular_values": singular_values.tolist(),
        "effective_rank": effective_rank,
        "condition_number": condition_number,
    }


def gini(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.allclose(arr.sum(), 0.0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    weights = np.arange(1, n + 1)
    return float((2.0 * np.sum(weights * arr) / (n * arr.sum())) - ((n + 1.0) / n))


def edge_participation(incidence: np.ndarray) -> np.ndarray:
    if incidence.shape[0] == 0:
        return np.zeros(incidence.shape[1], dtype=float)
    return incidence.mean(axis=0)


def masked_relative_dimension(D_matrix: np.ndarray, input_mask: Optional[np.ndarray], p: int) -> int:
    """Compute sum_alpha rank(D_G diag(Omega[:, alpha]))."""

    return int(masked_relative_svd_metrics(D_matrix, input_mask, p)["rank"])


def masked_relative_svd_metrics(
    D_matrix: np.ndarray,
    input_mask: Optional[np.ndarray],
    p: int,
) -> dict:
    """SVD metrics for the block-diagonal masked relative tree map.

    With an input mask Omega, coordinate alpha contributes the linear map
    ``D_G diag(Omega[:, alpha])``. The full input-encoding map is block
    diagonal across input coordinates, so its singular values are the union of
    the coordinate-specific singular values.
    """

    if input_mask is None:
        base = svd_metrics(D_matrix)
        singular_values = np.asarray(base["singular_values"] * int(p), dtype=float)
    else:
        mask = np.asarray(input_mask, dtype=float)
        if mask.ndim != 2:
            raise ValueError("input_mask must have shape (n_edges, p)")
        if mask.shape[1] != p or mask.shape[0] != D_matrix.shape[1]:
            raise ValueError(
                f"input_mask shape {mask.shape} incompatible with D {D_matrix.shape} and p={p}"
            )
        values = []
        for alpha in range(p):
            values.extend(svd_metrics(D_matrix * mask[:, alpha][None, :])["singular_values"])
        singular_values = np.asarray(values, dtype=float)

    positive = singular_values[singular_values > 1e-9]
    if positive.size == 0:
        rank = 0
        effective_rank = 0.0
        condition_number = math.inf
    else:
        rank = int(positive.size)
        weights = positive**2
        weights = weights / weights.sum()
        effective_rank = float(np.exp(-np.sum(weights * np.log(weights))))
        condition_number = float(positive.max() / positive.min())
    return {
        "rank": rank,
        "singular_values": singular_values.tolist(),
        "effective_rank": effective_rank,
        "condition_number": condition_number,
    }


def _coordinate_relative_map(
    D_matrix: np.ndarray,
    input_mask: Optional[np.ndarray],
    alpha: int,
) -> np.ndarray:
    if input_mask is None:
        return D_matrix
    return D_matrix * np.asarray(input_mask[:, alpha], dtype=float)[None, :]


def subspace_intersection_rank(
    left: np.ndarray,
    right: np.ndarray,
    tol: float = 1e-9,
) -> int:
    """Dimension of the intersection of two column spaces."""

    rank_left = svd_metrics(left, tol=tol)["rank"]
    rank_right = svd_metrics(right, tol=tol)["rank"]
    union_rank = svd_metrics(np.concatenate([left, right], axis=1), tol=tol)["rank"]
    return int(rank_left + rank_right - union_rank)


def comparison_branch_rank_metrics(
    D_matrix: np.ndarray,
    input_mask: Optional[np.ndarray],
    p: int,
    n_context: int,
    z_dim: int,
) -> dict:
    """Balanced rank support for each context-query comparison branch.

    Global ``d_rel`` can be high even if a mask underserves one context item.
    This proxy computes a coordinate-wise relative rank for every input
    coordinate, then for each branch sums ``min(rank(context_i_dim),
    rank(query_dim))`` across feature dimensions. The minimum enforces paired
    context/query support for the comparison direction rather than rewarding
    query-only or context-only capacity.

    The stricter ``comparison_branch_common_d_rel_*`` fields measure the
    intersection of the context and query coordinate column spaces.  Those
    common relative tree directions are the ones that can support coefficients
    with equal-and-opposite context/query dependence.
    """

    if n_context <= 0 or z_dim <= 0:
        raise ValueError("n_context and z_dim must be positive")
    expected_p = (n_context + 1) * z_dim
    if expected_p != p:
        raise ValueError(f"(n_context + 1) * z_dim = {expected_p}, expected p={p}")

    mask = None
    if input_mask is None:
        coord_ranks = np.full(p, svd_metrics(D_matrix)["rank"], dtype=float)
        coord_counts = np.full(p, D_matrix.shape[1], dtype=float)
    else:
        mask = np.asarray(input_mask, dtype=float)
        if mask.ndim != 2:
            raise ValueError("input_mask must have shape (n_edges, p)")
        if mask.shape[1] != p or mask.shape[0] != D_matrix.shape[1]:
            raise ValueError(
                f"input_mask shape {mask.shape} incompatible with D {D_matrix.shape} and p={p}"
            )
        coord_ranks = np.asarray(
            [
                svd_metrics(D_matrix * mask[:, alpha][None, :])["rank"]
                for alpha in range(p)
            ],
            dtype=float,
        )
        coord_counts = mask.sum(axis=0)

    query_offset = n_context * z_dim
    branch_ranks = []
    branch_common_ranks = []
    branch_counts = []
    branch_overlap_counts = []
    for branch in range(n_context):
        context_offset = branch * z_dim
        rank_total = 0.0
        common_rank_total = 0.0
        count_total = 0.0
        overlap_count_total = 0.0
        for dim in range(z_dim):
            context_idx = context_offset + dim
            query_idx = query_offset + dim
            context_map = _coordinate_relative_map(D_matrix, mask, context_idx)
            query_map = _coordinate_relative_map(D_matrix, mask, query_idx)
            rank_total += min(coord_ranks[context_idx], coord_ranks[query_idx])
            common_rank_total += subspace_intersection_rank(context_map, query_map)
            count_total += min(coord_counts[context_idx], coord_counts[query_idx])
            if input_mask is None:
                overlap_count_total += D_matrix.shape[1]
            else:
                overlap_count_total += float(
                    np.sum(
                        (mask[:, context_idx] > 0)
                        & (mask[:, query_idx] > 0)
                    )
                )
        branch_ranks.append(rank_total)
        branch_common_ranks.append(common_rank_total)
        branch_counts.append(count_total)
        branch_overlap_counts.append(overlap_count_total)

    branch_ranks = np.asarray(branch_ranks, dtype=float)
    branch_common_ranks = np.asarray(branch_common_ranks, dtype=float)
    branch_counts = np.asarray(branch_counts, dtype=float)
    branch_overlap_counts = np.asarray(branch_overlap_counts, dtype=float)
    return {
        "comparison_branch_d_rel_values": [int(value) for value in branch_ranks],
        "comparison_branch_d_rel_min": int(branch_ranks.min()) if branch_ranks.size else 0,
        "comparison_branch_d_rel_mean": float(branch_ranks.mean()) if branch_ranks.size else 0.0,
        "comparison_branch_d_rel_max": int(branch_ranks.max()) if branch_ranks.size else 0,
        "comparison_branch_d_rel_gini": gini(branch_ranks),
        "comparison_branch_common_d_rel_values": [int(value) for value in branch_common_ranks],
        "comparison_branch_common_d_rel_min": (
            int(branch_common_ranks.min()) if branch_common_ranks.size else 0
        ),
        "comparison_branch_common_d_rel_mean": (
            float(branch_common_ranks.mean()) if branch_common_ranks.size else 0.0
        ),
        "comparison_branch_common_d_rel_max": (
            int(branch_common_ranks.max()) if branch_common_ranks.size else 0
        ),
        "comparison_branch_common_d_rel_gini": gini(branch_common_ranks),
        "comparison_branch_input_count_values": [int(value) for value in branch_counts],
        "comparison_branch_input_count_min": int(branch_counts.min()) if branch_counts.size else 0,
        "comparison_branch_input_count_mean": float(branch_counts.mean()) if branch_counts.size else 0.0,
        "comparison_branch_input_count_max": int(branch_counts.max()) if branch_counts.size else 0,
        "comparison_branch_input_count_gini": gini(branch_counts),
        "comparison_branch_input_overlap_values": [int(value) for value in branch_overlap_counts],
        "comparison_branch_input_overlap_min": (
            int(branch_overlap_counts.min()) if branch_overlap_counts.size else 0
        ),
        "comparison_branch_input_overlap_mean": (
            float(branch_overlap_counts.mean()) if branch_overlap_counts.size else 0.0
        ),
        "comparison_branch_input_overlap_max": (
            int(branch_overlap_counts.max()) if branch_overlap_counts.size else 0
        ),
        "comparison_branch_input_overlap_gini": gini(branch_overlap_counts),
    }


def graph_degree_stats(n_nodes: int, edges: Sequence[Edge]) -> dict:
    indeg = np.zeros(n_nodes, dtype=int)
    outdeg = np.zeros(n_nodes, dtype=int)
    for source, target in edges:
        outdeg[source] += 1
        indeg[target] += 1
    return {
        "in_degree": indeg.tolist(),
        "out_degree": outdeg.tolist(),
        "in_degree_cv": float(indeg.std() / indeg.mean()) if indeg.mean() else 0.0,
        "out_degree_cv": float(outdeg.std() / outdeg.mean()) if outdeg.mean() else 0.0,
    }


def mean_shortest_path(n_nodes: int, edges: Sequence[Edge]) -> Optional[float]:
    adjacency = [[] for _ in range(n_nodes)]
    for source, target in edges:
        adjacency[source].append(target)
    distances = []
    for start in range(n_nodes):
        dist = {start: 0}
        frontier = [start]
        for node in frontier:
            for nxt in adjacency[node]:
                if nxt not in dist:
                    dist[nxt] = dist[node] + 1
                    frontier.append(nxt)
        if len(dist) != n_nodes:
            return None
        distances.extend(distance for node, distance in dist.items() if node != start)
    return float(np.mean(distances)) if distances else 0.0


def compute_topology_metrics(
    n_nodes: int,
    edges: Iterable[Sequence[int]],
    p: int,
    input_mask: Optional[np.ndarray] = None,
    n_context: Optional[int] = None,
    z_dim: Optional[int] = None,
    max_trees_per_root: Optional[int] = None,
) -> dict:
    """Compute pre-training topology predictors for a first-order CRN."""

    edge_tuple = normalize_edges(n_nodes, edges)
    strongly_connected = is_strongly_connected(n_nodes, edge_tuple)
    arborescences = enumerate_arborescences(
        n_nodes, edge_tuple, max_trees_per_root=max_trees_per_root
    )
    M = incidence_matrix(arborescences, len(edge_tuple))
    D_anchor = relative_tree_matrix(M)
    D_centered = centered_tree_matrix(M)
    M_stats = svd_metrics(M)
    D_anchor_stats = svd_metrics(D_anchor)
    D_stats = svd_metrics(D_centered)
    D_masked_stats = masked_relative_svd_metrics(D_centered, input_mask, p)

    enum_counts = [len(arborescences[root]) for root in range(n_nodes)]
    det_counts = tree_counts_by_determinant(n_nodes, edge_tuple)
    participation = edge_participation(M)
    count_arr = np.asarray(enum_counts, dtype=float)

    metrics = {
        "n_nodes": n_nodes,
        "n_edges": len(edge_tuple),
        "p": int(p),
        "strongly_connected": strongly_connected,
        "edges": [list(edge) for edge in edge_tuple],
        "tree_counts_enum": enum_counts,
        "tree_counts_det": det_counts,
        "n_trees_total_enum": int(sum(enum_counts)),
        "rank_M": M_stats["rank"],
        "rank_D": D_stats["rank"],
        "rank_D_anchor": D_anchor_stats["rank"],
        "d_rel": D_masked_stats["rank"],
        "effective_rank_D": D_stats["effective_rank"],
        "condition_number_D": D_stats["condition_number"],
        "effective_rank_D_masked": D_masked_stats["effective_rank"],
        "condition_number_D_masked": D_masked_stats["condition_number"],
        "singular_values_D_masked": D_masked_stats["singular_values"],
        "singular_values_D": D_stats["singular_values"],
        "singular_values_D_anchor": D_anchor_stats["singular_values"],
        "root_tree_count_cv": float(count_arr.std() / count_arr.mean()) if count_arr.mean() else 0.0,
        "root_tree_count_gini": gini(enum_counts),
        "edge_participation": participation.tolist(),
        "edge_participation_mean": float(participation.mean()) if participation.size else 0.0,
        "edge_participation_var": float(participation.var()) if participation.size else 0.0,
        "edge_participation_gini": gini(participation),
        "bottleneck_edge_fraction_095": float(np.mean(participation >= 0.95)) if participation.size else 0.0,
        "mean_shortest_path": mean_shortest_path(n_nodes, edge_tuple),
    }
    if n_context is not None or z_dim is not None:
        if n_context is None or z_dim is None:
            raise ValueError("Provide both n_context and z_dim for branch comparison metrics")
        metrics.update(
            comparison_branch_rank_metrics(
                D_centered,
                input_mask,
                p=p,
                n_context=n_context,
                z_dim=z_dim,
            )
        )
    metrics.update(graph_degree_stats(n_nodes, edge_tuple))
    return metrics


def topology_matrices(
    n_nodes: int,
    edges: Iterable[Sequence[int]],
    max_trees_per_root: Optional[int] = None,
) -> dict:
    """Return explicit arborescences and incidence matrices for analysis."""

    edge_tuple = normalize_edges(n_nodes, edges)
    arborescences = enumerate_arborescences(
        n_nodes, edge_tuple, max_trees_per_root=max_trees_per_root
    )
    M = incidence_matrix(arborescences, len(edge_tuple))
    D = relative_tree_matrix(M)
    return {
        "edges": edge_tuple,
        "arborescences": arborescences,
        "M": M,
        "D": D,
    }
