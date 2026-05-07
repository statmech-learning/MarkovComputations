import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "regress_topology_results.py",
)


class RegressTopologyResultsTests(unittest.TestCase):
    def run_regression(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    def test_branch_capacity_predictor_set_is_reported(self):
        fieldnames = [
            "label",
            "topology_name",
            "n_edges",
            "raw_physical_parameter_count",
            "input_coupled_parameter_count",
            "d_rel",
            "comparison_branch_d_rel_min",
            "comparison_branch_d_rel_gini",
            "effective_rank_D",
            "effective_rank_D_masked",
            "condition_number_D",
            "condition_number_D_masked",
            "root_tree_count_gini",
            "edge_participation_gini",
            "edge_participation_var",
            "bottleneck_edge_fraction_095",
            "mean_shortest_path",
            "input_edge_load_gini",
            "input_coord_load_gini",
            "test_novel_classes",
        ]
        rows = []
        for idx in range(6):
            branch_min = 8 * idx
            rows.append(
                {
                    "label": f"run{idx}",
                    "topology_name": f"family{idx % 2}",
                    "n_edges": 20,
                    "raw_physical_parameter_count": 400,
                    "input_coupled_parameter_count": 200,
                    "d_rel": 160 + branch_min,
                    "comparison_branch_d_rel_min": branch_min,
                    "comparison_branch_d_rel_gini": 0.5 - 0.05 * idx,
                    "effective_rank_D": 10 + idx,
                    "effective_rank_D_masked": 12 + idx,
                    "condition_number_D": 10 + idx,
                    "condition_number_D_masked": 20 + idx,
                    "root_tree_count_gini": 0.1,
                    "edge_participation_gini": 0.2,
                    "edge_participation_var": 0.01,
                    "bottleneck_edge_fraction_095": 0.0,
                    "mean_shortest_path": 2.0,
                    "input_edge_load_gini": 0.3,
                    "input_coord_load_gini": 0.4,
                    "test_novel_classes": 50 + branch_min,
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = os.path.join(tmpdir, "topology_results.csv")
            output_json = os.path.join(tmpdir, "regression.json")
            with open(input_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            result = self.run_regression(
                [
                    "--input_csv",
                    input_csv,
                    "--output_json",
                    output_json,
                ]
            )
            with open(output_json) as f:
                report = json.load(f)

        self.assertIn("input_count_plus_branch_drel", result.stdout)
        model = report["models"]["input_count_plus_branch_drel"]
        self.assertEqual(model["n"], 6)
        self.assertIn("comparison_branch_d_rel_min", model["predictors"])
        self.assertIn("comparison_branch_d_rel_gini", model["predictors"])
        self.assertGreater(model["r2"], 0.9)


if __name__ == "__main__":
    unittest.main()
