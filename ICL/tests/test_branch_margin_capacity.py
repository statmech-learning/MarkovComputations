import os
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from branch_margin_capacity import (  # noqa: E402
    branch_margin_capacity,
    comparison_feature_matrix,
    coordinate_common_rank_matrix,
    markdown_report,
    normalized_rank_weights,
    oracle_branch_scores,
    rank_geometry_summary,
    rooted_common_rank_tensor,
    rooted_polytope_support_summary,
    sample_exact_copy_branches,
    summarize_margin_scores,
    tropical_root_feature_matrix,
    tropical_tree_feature_capacity,
    weighted_comparison_feature_matrix,
)
from topology_metrics import complete_digraph, compute_topology_metrics, topology_matrices, centered_tree_matrix  # noqa: E402


class BranchMarginCapacityTests(unittest.TestCase):
    def test_common_rank_support_matches_branch_metric(self):
        n_nodes = 3
        n_context = 2
        z_dim = 1
        edges = complete_digraph(n_nodes).edges
        mats = topology_matrices(n_nodes, edges)
        D = centered_tree_matrix(mats["M"])
        p = (n_context + 1) * z_dim

        full = coordinate_common_rank_matrix(D, None, n_context=n_context, z_dim=z_dim)
        metrics = compute_topology_metrics(
            n_nodes,
            edges,
            p=p,
            n_context=n_context,
            z_dim=z_dim,
        )
        self.assertEqual(full.tolist(), [[metrics["rank_D"]], [metrics["rank_D"]]])

        mask = np.zeros((len(edges), p), dtype=int)
        mask[:, 0] = 1
        mask[:, n_context * z_dim] = 1
        masked = coordinate_common_rank_matrix(D, mask, n_context=n_context, z_dim=z_dim)
        self.assertGreater(masked[0, 0], 0)
        self.assertEqual(masked[1, 0], 0)

    def test_exact_copy_oracle_detects_missing_branch_support(self):
        z, labels = sample_exact_copy_branches(
            n_samples=400,
            n_context=2,
            z_dim=1,
            seed=3,
        )
        full_support = np.ones((2, 1), dtype=bool)
        full_features = comparison_feature_matrix(z, full_support)
        full_scores = oracle_branch_scores(full_features, n_context=2, z_dim=1, support=full_support)
        self.assertGreater(np.mean(np.argmax(full_scores, axis=1) == labels), 0.99)

        missing_branch = np.asarray([[1], [0]], dtype=bool)
        masked_features = comparison_feature_matrix(z, missing_branch)
        masked_scores = oracle_branch_scores(
            masked_features,
            n_context=2,
            z_dim=1,
            support=missing_branch,
        )
        branch_one = labels == 1
        self.assertTrue(np.all(np.isneginf(masked_scores[:, 1])))
        self.assertEqual(np.mean(np.argmax(masked_scores[branch_one], axis=1) == labels[branch_one]), 0.0)

    def test_branch_margin_capacity_reports_topology_gated_margin(self):
        result = branch_margin_capacity(
            n_nodes=3,
            edges=complete_digraph(3).edges,
            n_context=2,
            z_dim=1,
            train_samples=300,
            test_samples=300,
            seed=5,
        )
        self.assertEqual(result["probe_kind"], "topology_gated_squared_comparison_margin")
        self.assertEqual(result["support_min"], 1)
        self.assertGreater(result["oracle_test_accuracy"], 0.99)
        self.assertIn("linear_test_margin_mean", result)
        self.assertIn("rank_weighted_oracle_test_margin_mean", result)
        self.assertIn("tropical_linear_test_accuracy_mean", result)
        self.assertIn("rooted_polytope_supported_branch_dim_fraction", result)
        self.assertIn("rooted_common_rank_by_root_branch_dim", result)
        self.assertIn("rank_mass_gini", result)
        self.assertIn("d_rel", result)
        self.assertIn("Branch-Margin Capacity Probe", markdown_report(result))

    def test_rooted_polytope_support_summarizes_per_root_common_rank(self):
        n_nodes = 3
        n_context = 2
        z_dim = 1
        edges = complete_digraph(n_nodes).edges
        mats = topology_matrices(n_nodes, edges)
        ranks = rooted_common_rank_tensor(
            mats["arborescences"],
            n_edges=len(edges),
            input_mask=None,
            n_context=n_context,
            z_dim=z_dim,
        )
        self.assertEqual(ranks.shape, (n_nodes, n_context, z_dim))
        self.assertTrue(np.all(ranks > 0))

        summary = rooted_polytope_support_summary(ranks)
        self.assertEqual(summary["rooted_polytope_n_roots"], n_nodes)
        self.assertEqual(summary["rooted_polytope_supported_branch_dim_fraction"], 1.0)
        self.assertEqual(summary["rooted_polytope_branch_root_support_min"], float(n_nodes))
        self.assertGreater(summary["rooted_polytope_root_rank_mass_effective"], 1.0)

    def test_tropical_tree_features_have_root_shape_and_capacity_summary(self):
        n_nodes = 3
        n_context = 2
        z_dim = 1
        edges = complete_digraph(n_nodes).edges
        mats = topology_matrices(n_nodes, edges)
        z, labels = sample_exact_copy_branches(
            n_samples=40,
            n_context=n_context,
            z_dim=z_dim,
            seed=11,
        )
        p = (n_context + 1) * z_dim
        projections = np.ones((len(edges), p), dtype=float)
        features = tropical_root_feature_matrix(
            z,
            mats["arborescences"],
            n_edges=len(edges),
            edge_projections=projections,
        )
        self.assertEqual(features.shape, (40, n_nodes))
        self.assertTrue(np.all(np.isfinite(features)))

        summary = tropical_tree_feature_capacity(
            mats["arborescences"],
            n_edges=len(edges),
            input_dim=p,
            train_z=z,
            train_labels=labels,
            test_z=z,
            test_labels=labels,
            n_trials=3,
            seed=12,
        )
        self.assertEqual(summary["tropical_feature_trials"], 3)
        self.assertGreaterEqual(summary["tropical_linear_test_accuracy_mean"], 0.0)
        self.assertLessEqual(summary["tropical_linear_test_accuracy_mean"], 1.0)
        self.assertIn("tropical_root_feature_effective_rank_mean", summary)

    def test_rank_weighted_features_preserve_rank_strength(self):
        z, _ = sample_exact_copy_branches(
            n_samples=5,
            n_context=2,
            z_dim=2,
            seed=7,
        )
        ranks = np.asarray([[2, 0], [1, 4]], dtype=float)
        weights = normalized_rank_weights(ranks)
        self.assertEqual(weights.tolist(), [[0.5, 0.0], [0.25, 1.0]])
        features = weighted_comparison_feature_matrix(z, weights)
        self.assertEqual(features.shape, (5, 4))
        self.assertTrue(np.all(features[:, 1] == 0.0))

        summary = rank_geometry_summary(ranks)
        self.assertEqual(summary["rank_mass_min"], 2.0)
        self.assertEqual(summary["rank_mass_max"], 5.0)
        self.assertGreater(summary["rank_weight_effective_entries"], 1.0)

    def test_margin_summary_handles_nonfinite_margins(self):
        scores = np.asarray([[0.0, -np.inf], [-np.inf, -np.inf]])
        labels = np.asarray([0, 1])
        summary = summarize_margin_scores(scores, labels, "toy")
        self.assertEqual(summary["toy_accuracy"], 0.5)
        self.assertEqual(summary["toy_margin_finite_fraction"], 0.0)
        self.assertIsNone(summary["toy_margin_mean"])


class CollectBranchMarginCapacityPathTests(unittest.TestCase):
    def test_stale_absolute_topology_path_falls_back_to_library_topologies(self):
        from collect_branch_margin_capacity import resolve_path  # noqa: E402

        with tempfile.TemporaryDirectory() as tmpdir:
            topologies = os.path.join(tmpdir, "topologies")
            os.makedirs(topologies)
            local = os.path.join(topologies, "graph.json")
            with open(local, "w") as handle:
                handle.write("{}")
            stale = "/home/user/repos/topology/ICL/results/library/topologies/graph.json"
            self.assertEqual(resolve_path(stale, tmpdir), os.path.abspath(local))


if __name__ == "__main__":
    unittest.main()
