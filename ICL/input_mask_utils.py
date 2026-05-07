"""Utilities for first-order CRN input-encoding masks.

Input masks are edge-order dependent: row ``e`` controls which coordinates of
the flattened context/query vector may modulate physical edge ``edges[e]``.
Keep this validation outside torch-dependent runner code so topology libraries
and tests can use it on lightweight environments.
"""

from __future__ import annotations

import json
import math
import os
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

from topology_metrics import Edge, gini, normalize_edges


def _binary_value(value, path: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)
        raise ValueError(f"{path} must be binary, got {value!r}")
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        if math.isfinite(parsed) and parsed in (0.0, 1.0):
            return int(parsed)
        raise ValueError(f"{path} must be binary and finite, got {value!r}")
    raise ValueError(f"{path} must be 0/1 or bool, got {type(value).__name__}")


def validate_input_mask(mask_like, n_edges: int, p: int, name: str = "input_mask") -> np.ndarray:
    """Validate a rectangular binary mask and return an integer ndarray."""

    if n_edges < 0 or p < 0:
        raise ValueError("n_edges and p must be nonnegative")
    if not isinstance(mask_like, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a 2-D array-like object")
    if len(mask_like) != n_edges:
        raise ValueError(f"{name} must have {n_edges} rows, got {len(mask_like)}")

    rows = []
    for row_idx, row in enumerate(mask_like):
        if not isinstance(row, (list, tuple, np.ndarray)):
            raise ValueError(f"{name}[{row_idx}] must be a row array")
        if len(row) != p:
            raise ValueError(f"{name}[{row_idx}] must have {p} columns, got {len(row)}")
        rows.append([_binary_value(value, f"{name}[{row_idx}][{col_idx}]") for col_idx, value in enumerate(row)])
    return np.asarray(rows, dtype=int)


def _payload_mask(payload):
    if isinstance(payload, Mapping):
        if "input_mask" not in payload:
            raise ValueError("input mask JSON object must contain 'input_mask'")
        return payload["input_mask"]
    return payload


def _validate_payload_edges(payload: Mapping, n_nodes: int, edges: Sequence[Edge]) -> None:
    if "edges" not in payload:
        return
    raw_edges = payload["edges"]
    if not isinstance(raw_edges, (list, tuple)):
        raise ValueError("input mask JSON 'edges' must be a list")
    payload_n_nodes = int(payload.get("n_nodes", n_nodes))
    if payload_n_nodes != n_nodes:
        raise ValueError(f"input mask JSON n_nodes={payload_n_nodes} does not match {n_nodes}")
    normalized = normalize_edges(n_nodes, raw_edges)
    if len(normalized) != len(raw_edges):
        raise ValueError("input mask JSON edges contain duplicates, self-loops, or invalid entries")
    if tuple(normalized) != tuple(edges):
        raise ValueError("input mask JSON edge order does not match the physical topology")


def load_input_mask_json(
    path: str,
    n_nodes: int,
    edges: Iterable[Sequence[int]],
    p: int,
) -> Tuple[np.ndarray, dict]:
    """Load and validate an input mask JSON file.

    The file may be either a bare 2-D mask or an object containing
    ``input_mask`` plus optional ``name``, ``n_nodes``, ``edges``, and ``p``.
    When edges are present they must match the normalized physical edge order
    exactly, because mask rows are edge-order dependent.
    """

    edge_tuple = normalize_edges(n_nodes, edges)
    with open(path) as f:
        payload = json.load(f)

    metadata = {}
    if isinstance(payload, Mapping):
        metadata = {key: value for key, value in payload.items() if key != "input_mask"}
        if "p" in payload and int(payload["p"]) != p:
            raise ValueError(f"input mask JSON p={payload['p']} does not match {p}")
        if "n_edges" in payload and int(payload["n_edges"]) != len(edge_tuple):
            raise ValueError(
                f"input mask JSON n_edges={payload['n_edges']} does not match {len(edge_tuple)}"
            )
        _validate_payload_edges(payload, n_nodes, edge_tuple)

    mask = validate_input_mask(_payload_mask(payload), len(edge_tuple), p)
    metadata.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    metadata["source_path"] = os.path.abspath(path)
    return mask, metadata


def input_mask_summary(mask_like) -> dict:
    """Return lightweight structural summaries of an input-encoding mask."""

    mask = np.asarray(mask_like, dtype=int)
    if mask.ndim != 2:
        raise ValueError("input_mask must be 2-D")
    n_edges, p = mask.shape
    edge_load = mask.sum(axis=1) if n_edges else np.asarray([], dtype=int)
    coord_load = mask.sum(axis=0) if p else np.asarray([], dtype=int)
    coupled = int(mask.sum())
    return {
        "input_coupled_parameter_count": coupled,
        "input_coupled_edge_count": int(np.count_nonzero(edge_load)),
        "input_coupled_coord_count": int(np.count_nonzero(coord_load)),
        "input_parameter_density": float(coupled / mask.size) if mask.size else 0.0,
        "input_edge_density": float(np.count_nonzero(edge_load) / n_edges) if n_edges else 0.0,
        "input_coord_density": float(np.count_nonzero(coord_load) / p) if p else 0.0,
        "input_edge_load_gini": gini(edge_load),
        "input_coord_load_gini": gini(coord_load),
        "input_edge_load_min": int(edge_load.min()) if edge_load.size else 0,
        "input_edge_load_max": int(edge_load.max()) if edge_load.size else 0,
        "input_coord_load_min": int(coord_load.min()) if coord_load.size else 0,
        "input_coord_load_max": int(coord_load.max()) if coord_load.size else 0,
    }
