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
    oracle_branch_scores,
    sample_exact_copy_branches,
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
        self.assertIn("d_rel", result)
        self.assertIn("Branch-Margin Capacity Probe", markdown_report(result))


if __name__ == "__main__":
    unittest.main()
