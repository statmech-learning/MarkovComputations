import math
import os
import sys
import unittest

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from branch_margin_capacity_v2 import (  # noqa: E402
    finite_matmul,
    lower_tail_capacity_probe,
    lower_tail_mean,
    sample_exact_copy_branches,
    summarize_branch_margins,
)
from topology_metrics import complete_digraph  # noqa: E402


class BranchMarginCapacityV2Tests(unittest.TestCase):
    def test_lower_tail_mean_uses_requested_tail(self):
        values = np.asarray([4.0, 1.0, 2.0, 8.0])
        self.assertEqual(lower_tail_mean(values, 0.25), 1.0)
        self.assertEqual(lower_tail_mean(values, 0.50), 1.5)
        with self.assertRaises(ValueError):
            lower_tail_mean(values, 0.0)

    def test_branch_summary_reports_worst_branch_objective(self):
        margins = np.asarray([1.0, -0.5, 0.25, -1.0])
        labels = np.asarray([0, 0, 1, 1])
        summary = summarize_branch_margins(margins, labels, n_context=2, alpha=0.5)
        self.assertAlmostEqual(summary["branch_margin_lcvar_min"], -1.0)
        self.assertEqual(summary["branch_failure_rate_max"], 0.5)
        self.assertEqual(len(summary["by_branch"]), 2)

    def test_finite_matmul_cleans_nonfinite_products(self):
        left = np.asarray([[1.0, math.inf]])
        right = np.asarray([[1.0], [1.0]])
        product = finite_matmul(left, right)
        self.assertTrue(np.all(np.isfinite(product)))

    def test_exact_and_hard_root_probes_emit_finite_reports(self):
        edges = complete_digraph(3).edges
        for variant in ("exact", "hard_root"):
            result = lower_tail_capacity_probe(
                n_nodes=3,
                edges=edges,
                n_context=2,
                z_dim=1,
                variant=variant,
                n_samples=40,
                trials=2,
                seed=7,
                alpha=0.25,
                max_root_assignments=2,
            )
            self.assertEqual(result["variant"], variant)
            self.assertEqual(result["n_trees_total"], 9)
            self.assertTrue(math.isfinite(result["best"]["objective"]))
            self.assertEqual(len(result["best"]["by_branch"]), 2)
            self.assertGreaterEqual(result["best"]["accuracy"], 0.0)
            self.assertLessEqual(result["best"]["accuracy"], 1.0)

    def test_branch_subset_samples_only_requested_labels(self):
        _, labels = sample_exact_copy_branches(
            n_samples=200,
            n_context=3,
            z_dim=1,
            seed=2,
            branch_subset=[1],
        )
        self.assertEqual(set(labels.tolist()), {1})

    def test_summarize_branch_margins_respects_active_branches(self):
        margins = np.asarray([1.0, 1.0, -5.0, -5.0])
        labels = np.asarray([0, 0, 1, 1])
        active = summarize_branch_margins(margins, labels, n_context=2, alpha=0.5, active_branches=[0])
        all_branches = summarize_branch_margins(margins, labels, n_context=2, alpha=0.5)
        self.assertGreater(active["objective"], 0.0)
        self.assertLess(all_branches["objective"], 0.0)
        self.assertEqual(active["active_branches"], [0])

    def test_lower_tail_probe_records_active_branches(self):
        result = lower_tail_capacity_probe(
            n_nodes=2,
            edges=complete_digraph(2).edges,
            n_context=2,
            z_dim=1,
            variant="exact",
            n_samples=40,
            trials=2,
            seed=4,
            branch_subset=[0],
        )
        self.assertEqual(result["active_branches"], [0])
        self.assertEqual(result["best"]["active_branches"], [0])


if __name__ == "__main__":
    unittest.main()
