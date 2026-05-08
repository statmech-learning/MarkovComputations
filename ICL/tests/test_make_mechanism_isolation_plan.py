import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from make_mechanism_isolation_plan import build_plan, markdown_report  # noqa: E402


class MechanismIsolationPlanTests(unittest.TestCase):
    def test_build_plan_selects_same_drel_bottleneck_contrast(self):
        rows = [
            {
                "regime": "r",
                "topology_name": "low",
                "family": "a",
                "n_nodes": "5",
                "n_edges": "8",
                "d_rel": "120",
                "bottleneck_edge_fraction_095": "0.1",
                "n_trees_total_enum": "10",
                "root_tree_count_gini": "0.2",
            },
            {
                "regime": "r",
                "topology_name": "high",
                "family": "b",
                "n_nodes": "5",
                "n_edges": "8",
                "d_rel": "120",
                "bottleneck_edge_fraction_095": "0.7",
                "n_trees_total_enum": "10",
                "root_tree_count_gini": "0.8",
            },
        ]
        plan = build_plan(rows)
        by_name = {item["name"]: item for item in plan}
        bottleneck = by_name["same_drel_different_bottleneck_participation"]
        self.assertEqual(bottleneck["status"], "ready")
        self.assertEqual(bottleneck["low_item"], "low")
        self.assertEqual(bottleneck["high_item"], "high")
        self.assertAlmostEqual(bottleneck["contrast_delta"], 0.6)
        root_balance = by_name["same_tree_count_different_root_balance"]
        self.assertEqual(root_balance["status"], "ready")
        self.assertIn("Mechanism-Isolating", markdown_report(plan))

    def test_build_plan_reports_unavailable_when_columns_missing(self):
        plan = build_plan([{"regime": "r", "topology_name": "only"}])
        self.assertTrue(all(item["status"] == "unavailable" for item in plan))


if __name__ == "__main__":
    unittest.main()
