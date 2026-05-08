import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analytic_gamma_repair import (  # noqa: E402
    TOY_BRANCHES,
    build_reports,
    enumerate_paper_three_species_trees,
    evaluate_three_species,
    evaluate_two_species,
    make_delta_branch_dataset,
    three_species_analytic_k,
    two_species_analytic_k,
)


class Args:
    n_per_branch = 32
    delta = 0.25
    input_range = 1.0
    delta_sweep = [0.25]
    seed = 3
    two_scale = 4.0
    three_scale = 20.0
    optimizer_maxiter = 5


class AnalyticGammaRepairTests(unittest.TestCase):
    def test_toy_b_uses_two_max_branches(self):
        self.assertEqual(set(TOY_BRANCHES["toy_B_two_species_one_branch_max"]), {"M1>", "M2>"})

    def test_paper_three_species_tree_orientation(self):
        audit = enumerate_paper_three_species_trees()
        self.assertTrue(audit["passed"])

    def test_two_species_analytic_passes_max_branch_and_fails_both(self):
        toy_a = make_delta_branch_dataset("toy_A", TOY_BRANCHES["toy_A_two_species_both_branches"], 32, 0.25, 1.0, 4)
        toy_b = make_delta_branch_dataset("toy_B", TOY_BRANCHES["toy_B_two_species_one_branch_max"], 32, 0.25, 1.0, 5)
        k = two_species_analytic_k(4.0)
        self.assertLess(evaluate_two_species(toy_a, k)["classification_accuracy"], 0.75)
        self.assertEqual(evaluate_two_species(toy_b, k)["classification_accuracy"], 1.0)

    def test_three_species_analytic_passes_both_branches(self):
        toy_c = make_delta_branch_dataset("toy_C", TOY_BRANCHES["toy_C_three_species_both_branches"], 32, 0.25, 1.0, 6)
        metrics = evaluate_three_species(toy_c, three_species_analytic_k(20.0))
        self.assertEqual(metrics["classification_accuracy"], 1.0)
        self.assertEqual(metrics["branch_ordering_correctness"], 1.0)
        self.assertGreater(metrics["lcvar_margin"], 0.0)

    def test_final_gate_passes_with_short_optimizer(self):
        reports = build_reports(Args())
        self.assertTrue(reports["gamma_toy_repair_final_report"]["gamma_repaired"])


if __name__ == "__main__":
    unittest.main()
