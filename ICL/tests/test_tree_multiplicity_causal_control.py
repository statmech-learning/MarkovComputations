import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tree_multiplicity_causal_control as control  # noqa: E402


class TreeMultiplicityCausalControlTests(unittest.TestCase):
    def test_fixed_m20_rows_load_with_controls(self):
        rows = control.load_fixed_m20_rows()
        self.assertEqual(len(rows), 48)
        self.assertEqual({row["input_coupled_parameter_count"] for row in rows}, {200})
        self.assertEqual({row["edge_M_mean"] for row in rows}, {10.0})

    def test_category_library_contains_required_strata(self):
        rows = control.load_fixed_m20_rows()
        control.assign_mask_categories(rows)
        categories = {row["causal_mask_category"] for row in rows}
        self.assertIn("high_tree_diff_overlap_balanced_coordinate_load", categories)
        self.assertIn("high_tree_diff_overlap_imbalanced_coordinate_load", categories)
        self.assertIn("low_tree_diff_overlap_balanced_aggregate_multiplicity", categories)
        self.assertIn("low_tree_diff_overlap_high_coordinate_load_imbalance", categories)

    def test_grouped_loo_prefers_tree_difference_over_edge_for_mean_icl(self):
        rows = control.load_fixed_m20_rows()
        results = control.model_results(rows, ["mean_novel_icl"])
        edge = control.find_model(results, "edge_level_multiplicity_plus_controls")
        diff = control.find_model(results, "tree_difference_multiplicity_plus_controls")
        self.assertGreater(diff["loo_r2"], edge["loo_r2"])


if __name__ == "__main__":
    unittest.main()
