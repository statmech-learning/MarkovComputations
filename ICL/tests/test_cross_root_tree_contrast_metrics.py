import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cross_root_tree_contrast_metrics import cross_root_metrics_for_topology  # noqa: E402
from topology_metrics import complete_digraph  # noqa: E402


class CrossRootTreeContrastMetricsTests(unittest.TestCase):
    def test_full_mask_saturates_cross_root_overlap(self):
        n_nodes = 3
        edges = complete_digraph(n_nodes).edges
        n_context = 2
        z_dim = 1
        mask = np.ones((len(edges), (n_context + 1) * z_dim), dtype=float)
        summary = cross_root_metrics_for_topology(
            n_nodes,
            edges,
            mask,
            n_context=n_context,
            z_dim=z_dim,
            max_pairs_per_root_pair=100,
        )
        self.assertEqual(summary["cross_overlap_norm_min"], 1.0)
        self.assertEqual(summary["cross_best_root_pair_overlap_norm_min"], 1.0)
        self.assertGreater(summary["cross_contrast_effective_rank_mean"], 0.0)

    def test_missing_query_coordinate_zeroes_min_overlap(self):
        n_nodes = 3
        edges = complete_digraph(n_nodes).edges
        n_context = 2
        z_dim = 1
        p = (n_context + 1) * z_dim
        mask = np.ones((len(edges), p), dtype=float)
        mask[:, 1] = 0.0
        summary = cross_root_metrics_for_topology(
            n_nodes,
            edges,
            mask,
            n_context=n_context,
            z_dim=z_dim,
            max_pairs_per_root_pair=100,
        )
        self.assertEqual(summary["cross_overlap_norm_min"], 0.0)
        self.assertGreater(summary["cross_separation_norm_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
