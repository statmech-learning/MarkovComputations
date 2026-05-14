"""Decoder-aware cross-root contrast summaries for first-order Markov ICL.

The same-root tree-difference metrics are useful diagnostics, but the normalized
steady state compares rooted-tree numerators across species.  This module wraps
the existing cross-root tree-contrast implementation with low-dimensional
decoder-agnostic summaries that do not assume a fixed root-to-label decoder
before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cross_root_tree_contrast_metrics import cross_root_metrics_for_topology, json_ready


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _gini(values: Sequence[float]) -> float | None:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return None
    min_value = float(vals.min())
    if min_value < 0.0:
        vals = vals - min_value
    total = float(vals.sum())
    if total <= 1.0e-12:
        return 0.0
    vals = np.sort(vals)
    n = vals.size
    weighted = float(np.sum((np.arange(n) + 1.0) * vals))
    return float((2.0 * weighted / (n * total)) - ((n + 1.0) / n))


def _entropy01(values: Sequence[float]) -> float | None:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0], dtype=float)
    if vals.size == 0:
        return None
    probs = vals / float(vals.sum())
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(vals.size) if vals.size > 1 else 0.0


def decoder_aware_summaries(cross_metrics: Mapping[str, Any], n_context: int) -> dict[str, Any]:
    """Return decoder-agnostic root-pair coverage summaries.

    A learned decoder can choose which roots or root pairs to compare, so these
    summaries use best/top-k root-pair coverage rather than fixing an output
    species assignment.  They are structural pre-training diagnostics; learned
    decoder-weighted metrics should be labeled post-training.
    """

    root_pairs = cross_metrics.get("cross_per_root_pair") or []
    means = []
    mins = []
    counts = []
    products = []
    for row in root_pairs:
        mean_value = _finite_float(row.get("cross_overlap_norm_mean"))
        min_value = _finite_float(row.get("cross_overlap_norm_min"))
        sampled = _finite_float(row.get("cross_pair_count_sampled"))
        possible = _finite_float(row.get("cross_pair_count_possible"))
        if mean_value is not None:
            means.append(mean_value)
        if min_value is not None:
            mins.append(min_value)
        if sampled is not None:
            counts.append(sampled)
        if possible is not None:
            products.append(possible)

    sorted_means = sorted(means, reverse=True)
    top_k = sorted_means[: max(1, min(int(n_context), len(sorted_means)))]
    usable = [value for value in means if value > 0.0]
    return {
        "decoder_root_pair_count": len(root_pairs),
        "decoder_root_pair_overlap_mean": float(np.mean(means)) if means else None,
        "decoder_root_pair_overlap_min": float(np.min(means)) if means else None,
        "decoder_root_pair_overlap_max": float(np.max(means)) if means else None,
        "decoder_root_pair_overlap_gini": _gini(means),
        "decoder_root_pair_overlap_entropy": _entropy01(means),
        "decoder_root_pair_min_overlap_mean": float(np.mean(mins)) if mins else None,
        "decoder_root_pair_min_overlap_min": float(np.min(mins)) if mins else None,
        "decoder_topk_assignment_score": float(np.mean(top_k)) if top_k else None,
        "decoder_best_assignment_score": float(sorted_means[0]) if sorted_means else None,
        "decoder_usable_root_pair_fraction": float(len(usable) / len(means)) if means else None,
        "decoder_root_pair_tree_product_mean": float(np.mean(products)) if products else None,
        "decoder_root_pair_tree_product_entropy": _entropy01(products),
        "decoder_root_pair_sample_count_mean": float(np.mean(counts)) if counts else None,
    }


def decoder_aware_metrics_for_topology(
    n_nodes: int,
    edges: Sequence[Sequence[int]],
    input_mask: np.ndarray,
    n_context: int,
    z_dim: int,
    max_pairs_per_root_pair: int | None = 50000,
) -> dict[str, Any]:
    cross = cross_root_metrics_for_topology(
        n_nodes=n_nodes,
        edges=edges,
        input_mask=np.asarray(input_mask, dtype=float),
        n_context=n_context,
        z_dim=z_dim,
        max_pairs_per_root_pair=max_pairs_per_root_pair,
    )
    cross.update(decoder_aware_summaries(cross, n_context=n_context))
    return cross


def load_topology(path: Path) -> tuple[int, list[list[int]], np.ndarray]:
    payload = json.loads(path.read_text())
    return int(payload["n_nodes"]), payload["edges"], np.asarray(payload["input_mask"], dtype=float)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key == "cross_per_root_pair":
                continue
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-json", action="append", default=[])
    parser.add_argument("--topology-dir", default=None)
    parser.add_argument("--n-context", type=int, required=True)
    parser.add_argument("--z-dim", type=int, required=True)
    parser.add_argument("--max-pairs-per-root-pair", type=int, default=50000)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    paths = [Path(item) for item in args.topology_json]
    if args.topology_dir:
        paths.extend(sorted(Path(args.topology_dir).glob("*.json")))
    if not paths:
        raise SystemExit("Provide --topology-json or --topology-dir")

    rows = []
    for path in paths:
        n_nodes, edges, mask = load_topology(path)
        metrics = decoder_aware_metrics_for_topology(
            n_nodes=n_nodes,
            edges=edges,
            input_mask=mask,
            n_context=args.n_context,
            z_dim=args.z_dim,
            max_pairs_per_root_pair=args.max_pairs_per_root_pair,
        )
        metrics["topology_json"] = str(path)
        metrics["topology_id"] = path.stem
        rows.append(metrics)

    payload = {"schema": "cross_root_decoder_contrast_metrics.v1", "rows": rows}
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    if args.out_csv:
        write_csv(Path(args.out_csv), rows)
    if not args.out_json and not args.out_csv:
        print(json.dumps(json_ready(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
