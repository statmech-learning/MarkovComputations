"""Compute branch-margin capacity probes for rows in a topology library CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import List, Optional

import numpy as np

from branch_margin_capacity import branch_margin_capacity, load_edge_json
from input_mask_utils import load_input_mask_json


FIELDS = [
    "topology_id",
    "topology_name",
    "family",
    "physical_topology_name",
    "mask_name",
    "input_mask_name",
    "edge_json",
    "input_mask_json",
    "n_nodes",
    "n_edges",
    "n_context",
    "z_dim",
    "p",
    "d_rel",
    "comparison_branch_common_d_rel_min",
    "comparison_branch_common_d_rel_mean",
    "support_fraction",
    "support_min",
    "support_mean",
    "support_max",
    "rank_mass_min",
    "rank_mass_mean",
    "rank_mass_max",
    "rank_mass_gini",
    "rank_dim_mass_min",
    "rank_dim_mass_mean",
    "rank_dim_mass_max",
    "rank_dim_mass_gini",
    "rank_nonzero_fraction",
    "rank_weight_entropy",
    "rank_weight_effective_entries",
    "rooted_polytope_n_roots",
    "rooted_polytope_common_rank_total",
    "rooted_polytope_common_rank_mean",
    "rooted_polytope_common_rank_max",
    "rooted_polytope_supported_branch_dim_fraction",
    "rooted_polytope_branch_root_support_min",
    "rooted_polytope_branch_root_support_mean",
    "rooted_polytope_branch_root_support_max",
    "rooted_polytope_branch_root_support_gini",
    "rooted_polytope_branch_best_rank_min",
    "rooted_polytope_branch_best_rank_mean",
    "rooted_polytope_branch_best_rank_max",
    "rooted_polytope_branch_best_rank_gini",
    "rooted_polytope_root_rank_mass_min",
    "rooted_polytope_root_rank_mass_mean",
    "rooted_polytope_root_rank_mass_max",
    "rooted_polytope_root_rank_mass_gini",
    "rooted_polytope_root_rank_mass_effective",
    "oracle_test_accuracy",
    "oracle_test_margin_mean",
    "oracle_test_margin_p10",
    "oracle_test_margin_finite_fraction",
    "rank_weighted_oracle_test_accuracy",
    "rank_weighted_oracle_test_margin_mean",
    "rank_weighted_oracle_test_margin_p10",
    "rank_weighted_oracle_test_margin_finite_fraction",
    "linear_test_accuracy",
    "linear_test_margin_mean",
    "linear_test_margin_p10",
    "linear_test_margin_finite_fraction",
    "rank_weighted_linear_test_accuracy",
    "rank_weighted_linear_test_margin_mean",
    "rank_weighted_linear_test_margin_p10",
    "rank_weighted_linear_test_margin_finite_fraction",
    "linear_weight_norm",
    "weighted_linear_weight_norm",
    "tropical_feature_trials",
    "tropical_feature_mode",
    "tropical_projection_radius",
    "tropical_edge_bias_scale",
    "tropical_linear_train_accuracy_mean",
    "tropical_linear_train_accuracy_max",
    "tropical_linear_train_accuracy_std",
    "tropical_linear_test_accuracy_mean",
    "tropical_linear_test_accuracy_max",
    "tropical_linear_test_accuracy_std",
    "tropical_linear_test_margin_p10_mean",
    "tropical_linear_test_margin_p10_max",
    "tropical_linear_test_margin_p10_std",
    "tropical_root_feature_effective_rank_mean",
    "tropical_root_feature_effective_rank_max",
    "tropical_root_feature_effective_rank_std",
    "tropical_root_feature_variance_mean",
    "tropical_root_feature_variance_max",
    "tropical_root_feature_variance_std",
    "normal_fan_trials",
    "normal_fan_branch_root_nmi_mean",
    "normal_fan_branch_root_nmi_max",
    "normal_fan_branch_root_nmi_std",
    "normal_fan_branch_tree_nmi_mean",
    "normal_fan_branch_tree_nmi_max",
    "normal_fan_branch_tree_nmi_std",
    "normal_fan_active_root_count_mean",
    "normal_fan_active_root_count_max",
    "normal_fan_active_root_count_std",
    "normal_fan_active_tree_count_mean",
    "normal_fan_active_tree_count_max",
    "normal_fan_active_tree_count_std",
    "normal_fan_branch_active_tree_count_min_mean",
    "normal_fan_branch_active_tree_count_min_max",
    "normal_fan_branch_active_tree_count_min_std",
    "gamma_star_proxy_trials",
    "gamma_star_projection_radius",
    "gamma_star_decoder_radius",
    "gamma_star_edge_bias_radius",
    "gamma_star_mode",
    "gamma_star_selected_trial",
    "gamma_star_selected_train_branch_margin_p10_min",
    "gamma_star_selected_train_branch_margin_mean_min",
    "gamma_star_selected_test_branch_margin_p10_min",
    "gamma_star_selected_test_branch_margin_mean_min",
    "gamma_star_selected_test_sample_margin_min",
    "gamma_star_selected_test_accuracy",
    "gamma_star_selected_projection_norm",
    "gamma_star_selected_decoder_norm",
    "gamma_star_train_branch_margin_p10_min_mean",
    "gamma_star_train_branch_margin_p10_min_max",
    "gamma_star_train_branch_margin_p10_min_std",
    "gamma_star_test_branch_margin_p10_min_mean",
    "gamma_star_test_branch_margin_p10_min_max",
    "gamma_star_test_branch_margin_p10_min_std",
    "gamma_star_test_branch_margin_mean_min_mean",
    "gamma_star_test_branch_margin_mean_min_max",
    "gamma_star_test_branch_margin_mean_min_std",
    "gamma_star_test_accuracy_mean",
    "gamma_star_test_accuracy_max",
    "gamma_star_test_accuracy_std",
    "gamma_star_projection_norm_mean",
    "gamma_star_projection_norm_max",
    "gamma_star_projection_norm_std",
    "gamma_star_decoder_norm_mean",
    "gamma_star_decoder_norm_max",
    "gamma_star_decoder_norm_std",
]


def finite_or_empty(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def selected_rows(path: str, selected_only: bool = True) -> List[dict]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if selected_only:
        rows = [row for row in rows if str(row.get("selected", "1")) in {"1", "True", "true"}]
    return rows


def resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path in (None, ""):
        return None
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        if "/topologies/" in path:
            candidate = os.path.join(base_dir, "topologies", os.path.basename(path))
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def row_name(row: dict) -> str:
    return row.get("topology_name") or row.get("topology_id") or row.get("mask_name") or ""


def capacity_row(row: dict, base_dir: str, args) -> dict:
    edge_path = resolve_path(row.get("edge_json"), base_dir)
    if not edge_path:
        raise ValueError("row is missing edge_json")
    n_nodes, edges, topology_name = load_edge_json(edge_path)
    p = (args.N + 1) * args.D
    input_mask = None
    input_mask_name = "full"
    mask_path = resolve_path(row.get("input_mask_json"), base_dir)
    if mask_path:
        input_mask, metadata = load_input_mask_json(mask_path, n_nodes, edges, p)
        input_mask_name = str(metadata.get("name", os.path.splitext(os.path.basename(mask_path))[0]))

    result = branch_margin_capacity(
        n_nodes=n_nodes,
        edges=edges,
        n_context=args.N,
        z_dim=args.D,
        input_mask=input_mask,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        seed=args.seed,
        query_noise=args.query_noise,
        ridge=args.ridge,
        l2_radius=args.l2_radius,
        max_trees_per_root=args.max_trees_per_root,
        tree_feature_trials=args.tree_feature_trials,
        tree_feature_mode=args.tree_feature_mode,
        tree_feature_projection_radius=args.tree_feature_projection_radius,
        tree_feature_bias_scale=args.tree_feature_bias_scale,
        gamma_star_trials=args.gamma_star_trials,
        gamma_star_projection_radius=args.gamma_star_projection_radius,
        gamma_star_decoder_radius=args.gamma_star_decoder_radius,
        gamma_star_edge_bias_radius=args.gamma_star_edge_bias_radius,
    )
    return {
        "topology_id": row.get("topology_id", ""),
        "topology_name": row_name(row) or topology_name,
        "family": row.get("family", row.get("mask_family", "")),
        "physical_topology_name": row.get("physical_topology_name", topology_name),
        "mask_name": row.get("mask_name", ""),
        "input_mask_name": input_mask_name,
        "edge_json": row.get("edge_json") or edge_path,
        "input_mask_json": row.get("input_mask_json") or mask_path or "",
        **{field: result.get(field) for field in FIELDS if field not in {
            "topology_id",
            "topology_name",
            "family",
            "physical_topology_name",
            "mask_name",
            "input_mask_name",
            "edge_json",
            "input_mask_json",
        }},
    }


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: finite_or_empty(row.get(field)) for field in FIELDS} for row in rows)


def summary(rows: List[dict]) -> dict:
    by_family = {}
    for row in rows:
        family = row.get("family", "")
        by_family.setdefault(family, []).append(row)
    out = {"n_rows": len(rows), "families": {}}
    for family, items in sorted(by_family.items()):
        values = []
        weighted_values = []
        for item in items:
            try:
                values.append(float(item["linear_test_accuracy"]))
            except (TypeError, ValueError):
                pass
            try:
                weighted_values.append(float(item["rank_weighted_linear_test_accuracy"]))
            except (TypeError, ValueError):
                pass
        out["families"][family] = {
            "n": len(items),
            "linear_test_accuracy_mean": float(np.mean(values)) if values else None,
            "linear_test_accuracy_max": float(np.max(values)) if values else None,
            "rank_weighted_linear_test_accuracy_mean": float(np.mean(weighted_values)) if weighted_values else None,
            "rank_weighted_linear_test_accuracy_max": float(np.max(weighted_values)) if weighted_values else None,
            "tropical_linear_test_accuracy_mean": family_float_mean(items, "tropical_linear_test_accuracy_mean"),
            "tropical_linear_test_accuracy_max": family_float_max(items, "tropical_linear_test_accuracy_max"),
            "tropical_root_feature_effective_rank_mean": family_float_mean(
                items,
                "tropical_root_feature_effective_rank_mean",
            ),
            "rooted_polytope_supported_branch_dim_fraction_mean": family_float_mean(
                items,
                "rooted_polytope_supported_branch_dim_fraction",
            ),
            "rooted_polytope_branch_best_rank_min_mean": family_float_mean(
                items,
                "rooted_polytope_branch_best_rank_min",
            ),
            "normal_fan_branch_tree_nmi_mean": family_float_mean(
                items,
                "normal_fan_branch_tree_nmi_mean",
            ),
            "normal_fan_active_tree_count_mean": family_float_mean(
                items,
                "normal_fan_active_tree_count_mean",
            ),
            "gamma_star_selected_test_branch_margin_p10_min_mean": family_float_mean(
                items,
                "gamma_star_selected_test_branch_margin_p10_min",
            ),
            "gamma_star_selected_test_accuracy_mean": family_float_mean(
                items,
                "gamma_star_selected_test_accuracy",
            ),
        }
    return out


def family_values(items: List[dict], key: str) -> List[float]:
    values = []
    for item in items:
        try:
            value = float(item[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def family_float_mean(items: List[dict], key: str):
    values = family_values(items, key)
    return float(np.mean(values)) if values else None


def family_float_max(items: List[dict], key: str):
    values = family_values(items, key)
    return float(np.max(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query_noise", type=float, default=0.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--l2_radius", type=float, default=1.0)
    parser.add_argument("--max_trees_per_root", type=int, default=None)
    parser.add_argument("--tree_feature_trials", type=int, default=8)
    parser.add_argument("--tree_feature_mode", choices=["max", "logsumexp"], default="max")
    parser.add_argument("--tree_feature_projection_radius", type=float, default=1.0)
    parser.add_argument("--tree_feature_bias_scale", type=float, default=0.0)
    parser.add_argument("--gamma_star_trials", type=int, default=32)
    parser.add_argument("--gamma_star_projection_radius", type=float, default=1.0)
    parser.add_argument("--gamma_star_decoder_radius", type=float, default=1.0)
    parser.add_argument("--gamma_star_edge_bias_radius", type=float, default=0.0)
    parser.add_argument("--include_unselected", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args.library_csv))
    rows = []
    for row in selected_rows(args.library_csv, selected_only=not args.include_unselected):
        rows.append(capacity_row(row, base_dir, args))
    write_csv(args.output_csv, rows)
    report = summary(rows)
    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(report, handle, indent=2)
    print(f"Wrote {len(rows)} branch-margin capacity rows to {args.output_csv}")
    if args.output_json:
        print(f"Wrote branch-margin capacity summary to {args.output_json}")


if __name__ == "__main__":
    main()
