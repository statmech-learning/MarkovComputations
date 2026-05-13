import argparse
import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from multibase_normal_fan_tree_count_library import choose_pairs  # noqa: E402


def row(topology_id, tree_count, normal_fan_score):
    return {
        "topology_id": topology_id,
        "base_id": "base00",
        "n_trees_total_enum": float(tree_count),
        "normal_fan_score": float(normal_fan_score),
        "normal_fan_active_tree_count_mean": 10.0 + normal_fan_score,
        "normal_fan_branch_tree_nmi_mean": 0.1,
    }


class MultibaseNormalFanTreeCountLibraryTests(unittest.TestCase):
    def test_choose_pairs_builds_both_separation_arms(self):
        rows = [
            row("same_tree_low_nf", 10, -2.0),
            row("same_tree_high_nf", 10, 2.0),
            row("low_tree_matched_nf", 4, 0.02),
            row("high_tree_matched_nf", 20, 0.04),
        ]
        args = argparse.Namespace(
            arm_a_tree_tolerance=1.0,
            arm_b_normal_fan_tolerance=0.25,
            min_normal_fan_delta=1.0,
            min_tree_count_delta=8.0,
            pairs_per_arm_per_base=2,
        )
        pairs = choose_pairs(rows, args)
        arms = {pair["arm"] for pair in pairs}
        self.assertIn("arm_A_fixed_tree_count_variable_normal_fan", arms)
        self.assertIn("arm_B_variable_tree_count_matched_normal_fan", arms)
        self.assertTrue(any(row["selected"] == 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
