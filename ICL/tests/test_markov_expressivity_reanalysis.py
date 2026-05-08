import math
import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from markov_expressivity_reanalysis import (  # noqa: E402
    aggregate_multiplicity_from_row,
    loo_r2,
    outcome_value,
    reversible_support_fraction,
    summarize_groups,
)


class MarkovExpressivityReanalysisTests(unittest.TestCase):
    def test_aggregate_multiplicity_handles_uniform_full_mask(self):
        row = {
            "p": "4",
            "n_edges": "2",
            "input_coupled_parameter_count": "8",
            "input_coupled_coord_count": "4",
            "input_coord_load_gini": "0",
        }
        metrics = aggregate_multiplicity_from_row(row)
        self.assertEqual(metrics["M_mean_aggregate"], 2.0)
        self.assertEqual(metrics["M_nonzero_mean_aggregate"], 2.0)
        self.assertEqual(metrics["M_zero_fraction_aggregate"], 0.0)
        self.assertAlmostEqual(metrics["M_sum_log_2M1_exact_if_uniform"], 4 * math.log(5.0))

    def test_summarize_groups_reads_aggregated_outcomes_and_normal_fan_fields(self):
        rows = [
            {
                "topology_name": "toy",
                "target_mean": "84.5",
                "target_std": "3.0",
                "p": "2",
                "input_coupled_parameter_count": "4",
                "input_coupled_coord_count": "2",
                "capacity_normal_fan_active_tree_count_mean": "11",
                "library_n_trees_total_enum_log": "1.7",
            }
        ]
        self.assertEqual(outcome_value(rows[0]), 84.5)
        groups = summarize_groups(rows)
        self.assertEqual(groups[0]["mean_novel_icl"], 84.5)
        self.assertEqual(groups[0]["capacity_normal_fan_active_tree_count_mean"], 11.0)
        self.assertEqual(groups[0]["library_n_trees_total_enum_log"], 1.7)

    def test_loo_r2_fits_simple_grouped_relationship(self):
        groups = [
            {"group": f"g{i}", "x": float(i), "y": 1.0 + 2.0 * float(i)}
            for i in range(6)
        ]
        result = loo_r2(groups, ["x"], "y")
        self.assertEqual(result["n_groups"], 6)
        self.assertGreater(result["loo_r2"], 0.99)

    def test_reversible_support_fraction_counts_directed_reverses(self):
        self.assertEqual(reversible_support_fraction([(0, 1), (1, 0)]), 1.0)
        self.assertEqual(reversible_support_fraction([(0, 1), (1, 2)]), 0.0)
        self.assertAlmostEqual(reversible_support_fraction([(0, 1), (1, 0), (1, 2)]), 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
