import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from topology_metrics import (  # noqa: E402
    complete_digraph,
    comparison_branch_rank_metrics,
    compute_topology_metrics,
    directed_cycle,
    enumerate_arborescences,
    incidence_matrix,
    masked_relative_svd_metrics,
    tree_numerators_by_determinant,
    tree_numerators_by_enumeration,
    topology_matrices,
    tree_counts_by_determinant,
)


class TopologyMetricsTests(unittest.TestCase):
    def assert_tree_counts(self, name, n_nodes, edges, expected):
        arborescences = enumerate_arborescences(n_nodes, edges)
        enum_counts = [len(arborescences[root]) for root in range(n_nodes)]
        det_counts = tree_counts_by_determinant(n_nodes, edges)
        self.assertEqual(enum_counts, expected, name)
        self.assertEqual(det_counts, expected, name)

        M = incidence_matrix(arborescences, len(edges))
        self.assertTrue(np.all(M.sum(axis=1) == n_nodes - 1), name)

    def test_small_tree_counts_match_cofactors(self):
        self.assert_tree_counts("2-node reversible", 2, [(0, 1), (1, 0)], [1, 1])
        self.assert_tree_counts("3-node complete", 3, complete_digraph(3).edges, [3, 3, 3])
        self.assert_tree_counts("4-node directed cycle", 4, directed_cycle(4).edges, [1, 1, 1, 1])

    def test_relative_dimension_respects_input_mask(self):
        n_nodes = 3
        edges = complete_digraph(n_nodes).edges
        p = 4
        full = compute_topology_metrics(n_nodes, edges, p=p)
        zero_mask = np.zeros((len(edges), p), dtype=float)
        masked = compute_topology_metrics(n_nodes, edges, p=p, input_mask=zero_mask)
        self.assertEqual(full["d_rel"], full["rank_D"] * p)
        self.assertEqual(masked["d_rel"], 0)
        self.assertEqual(masked["effective_rank_D_masked"], 0.0)

    def test_masked_svd_metrics_matches_repeated_full_map(self):
        mats = topology_matrices(3, complete_digraph(3).edges)
        p = 3
        stats = masked_relative_svd_metrics(mats["D"], input_mask=None, p=p)
        one_coord = masked_relative_svd_metrics(mats["D"], input_mask=None, p=1)
        self.assertEqual(stats["rank"], p * one_coord["rank"])

    def test_comparison_branch_rank_metrics_detects_neglected_branch(self):
        n_nodes = 3
        edges = complete_digraph(n_nodes).edges
        n_context = 2
        z_dim = 2
        p = (n_context + 1) * z_dim
        full = compute_topology_metrics(
            n_nodes,
            edges,
            p=p,
            n_context=n_context,
            z_dim=z_dim,
        )
        self.assertEqual(
            full["comparison_branch_d_rel_values"],
            [full["rank_D"] * z_dim, full["rank_D"] * z_dim],
        )
        self.assertEqual(
            full["comparison_branch_common_d_rel_values"],
            [full["rank_D"] * z_dim, full["rank_D"] * z_dim],
        )

        mask = np.zeros((len(edges), p), dtype=int)
        mask[:, 0:z_dim] = 1
        mask[:, n_context * z_dim : (n_context + 1) * z_dim] = 1
        masked = compute_topology_metrics(
            n_nodes,
            edges,
            p=p,
            input_mask=mask,
            n_context=n_context,
            z_dim=z_dim,
        )
        self.assertEqual(masked["comparison_branch_d_rel_values"], [full["rank_D"] * z_dim, 0])
        self.assertEqual(
            masked["comparison_branch_common_d_rel_values"],
            [full["rank_D"] * z_dim, 0],
        )
        self.assertEqual(masked["comparison_branch_d_rel_min"], 0)
        self.assertEqual(masked["comparison_branch_common_d_rel_min"], 0)
        self.assertGreater(masked["comparison_branch_d_rel_gini"], 0.0)

    def test_comparison_branch_common_rank_detects_disjoint_subspaces(self):
        D_matrix = np.eye(2)
        mask = np.asarray(
            [
                [1, 0],
                [0, 1],
            ],
            dtype=int,
        )
        metrics = comparison_branch_rank_metrics(
            D_matrix,
            mask,
            p=2,
            n_context=1,
            z_dim=1,
        )
        self.assertEqual(metrics["comparison_branch_d_rel_values"], [1])
        self.assertEqual(metrics["comparison_branch_common_d_rel_values"], [0])
        self.assertEqual(metrics["comparison_branch_input_count_values"], [1])
        self.assertEqual(metrics["comparison_branch_input_overlap_values"], [0])

    def test_comparison_branch_rank_metrics_validates_context_shape(self):
        mats = topology_matrices(3, complete_digraph(3).edges)
        with self.assertRaises(ValueError):
            comparison_branch_rank_metrics(mats["D"], None, p=5, n_context=2, z_dim=2)

    def test_topology_matrices_shapes(self):
        mats = topology_matrices(3, complete_digraph(3).edges)
        self.assertEqual(mats["M"].shape, (9, 6))
        self.assertEqual(mats["D"].shape, (8, 6))

    def test_weighted_enumeration_matches_cofactors(self):
        n_nodes = 3
        edges = complete_digraph(n_nodes).edges
        rates = np.asarray([0.7, 1.3, 2.0, 0.5, 1.1, 1.7])
        enum = tree_numerators_by_enumeration(n_nodes, edges, rates)
        det = tree_numerators_by_determinant(n_nodes, edges, rates)
        np.testing.assert_allclose(enum, det, rtol=1e-10, atol=1e-10)

    def test_capped_tree_enumeration_is_flagged(self):
        metrics = compute_topology_metrics(
            4,
            complete_digraph(4).edges,
            p=2,
            max_trees_per_root=1,
        )
        self.assertFalse(metrics["tree_counts_match_det"])
        self.assertTrue(metrics["tree_enumeration_truncated"])
        self.assertEqual(metrics["max_trees_per_root"], 1)
        self.assertEqual(metrics["tree_counts_enum"], [1, 1, 1, 1])
        self.assertEqual(metrics["tree_counts_det"], [16, 16, 16, 16])


if __name__ == "__main__":
    unittest.main()
