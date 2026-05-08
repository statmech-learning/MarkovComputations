import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from topology_metrics import complete_digraph  # noqa: E402
from tree_level_multiplicity_metrics import (  # noqa: E402
    edge_level_multiplicity_summary,
    posterior_weighted_tree_overlap,
    tree_level_multiplicity_summary,
    tree_table_from_arborescences,
)
from topology_metrics import topology_matrices  # noqa: E402


class TreeLevelMultiplicityMetricsTests(unittest.TestCase):
    def test_full_mask_has_unit_comparison_overlap(self):
        n_nodes = 3
        edges = complete_digraph(n_nodes).edges
        n_context = 2
        z_dim = 1
        mask = np.ones((len(edges), (n_context + 1) * z_dim), dtype=float)
        summary = tree_level_multiplicity_summary(
            n_nodes,
            edges,
            mask,
            n_context=n_context,
            z_dim=z_dim,
            max_pairs_per_root=100,
        )
        self.assertEqual(summary["edge_overlap_norm_min"], 1.0)
        self.assertEqual(summary["tree_overlap_norm_min"], 1.0)
        self.assertEqual(summary["diff_overlap_norm_min"], 1.0)
        self.assertEqual(summary["tree_count_exact_total"], summary["tree_count_enumerated_total"])

    def test_masked_branch_zeroes_tree_overlap_for_that_branch(self):
        n_nodes = 3
        edges = complete_digraph(n_nodes).edges
        n_context = 2
        z_dim = 1
        p = (n_context + 1) * z_dim
        mask = np.ones((len(edges), p), dtype=float)
        mask[:, 1] = 0.0
        summary = tree_level_multiplicity_summary(
            n_nodes,
            edges,
            mask,
            n_context=n_context,
            z_dim=z_dim,
            max_pairs_per_root=100,
        )
        self.assertEqual(summary["edge_overlap_norm_min"], 0.0)
        self.assertEqual(summary["tree_overlap_norm_min"], 0.0)
        self.assertEqual(summary["diff_overlap_norm_min"], 0.0)

    def test_post_training_posterior_weighted_overlap(self):
        n_nodes = 3
        edges = complete_digraph(n_nodes).edges
        n_context = 2
        z_dim = 1
        mask = np.ones((len(edges), (n_context + 1) * z_dim), dtype=float)
        mats = topology_matrices(n_nodes, edges)
        roots, incidence = tree_table_from_arborescences(mats["arborescences"], len(edges))
        tree_loads = incidence @ mask
        weights = np.ones(tree_loads.shape[0], dtype=float)
        summary = posterior_weighted_tree_overlap(tree_loads, roots, weights, n_context, z_dim)
        self.assertEqual(summary["posterior_tree_overlap_norm_min"], 1.0)
        self.assertEqual(summary["posterior_tree_overlap_norm_mean"], 1.0)

    def test_edge_level_tracks_comparison_imbalance(self):
        mask = np.asarray(
            [
                [1, 0, 1],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        summary = edge_level_multiplicity_summary(mask, n_context=2, z_dim=1)
        self.assertEqual(summary["edge_overlap_norm_min"], 0.0)
        self.assertGreater(summary["edge_comparison_imbalance_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
