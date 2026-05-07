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
            "comparison_branch_common_d_rel_min",
            "comparison_branch_common_d_rel_gini",
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
                    "comparison_branch_common_d_rel_min": branch_min,
                    "comparison_branch_common_d_rel_gini": 0.5 - 0.05 * idx,
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
        self.assertIn("comparison_branch_common_d_rel_min", model["predictors"])
        self.assertIn("comparison_branch_common_d_rel_gini", model["predictors"])
        self.assertIn("comparison_branch_d_rel_min", model["predictors"])
        self.assertIn("comparison_branch_d_rel_gini", model["predictors"])
        self.assertGreater(model["r2"], 0.9)
        self.assertIsNotNone(model["leave_one_out_r2"])

    def test_leave_one_out_uses_effective_rank_for_fixed_count_design(self):
        fieldnames = [
            "label",
            "topology_name",
            "n_edges",
            "raw_physical_parameter_count",
            "input_coupled_parameter_count",
            "d_rel",
            "comparison_branch_common_d_rel_min",
            "comparison_branch_common_d_rel_gini",
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
        for idx, branch_min in enumerate([0, 10, 20, 30]):
            rows.append(
                {
                    "label": f"run{idx}",
                    "topology_name": "fixed_count",
                    "n_edges": 20,
                    "raw_physical_parameter_count": 400,
                    "input_coupled_parameter_count": 200,
                    "d_rel": 100,
                    "comparison_branch_common_d_rel_min": branch_min,
                    "comparison_branch_common_d_rel_gini": 0.0,
                    "comparison_branch_d_rel_min": branch_min,
                    "comparison_branch_d_rel_gini": 0.0,
                    "effective_rank_D": 10,
                    "effective_rank_D_masked": 10,
                    "condition_number_D": 10,
                    "condition_number_D_masked": 10,
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
            self.run_regression(
                [
                    "--input_csv",
                    input_csv,
                    "--output_json",
                    output_json,
                ]
            )
            with open(output_json) as f:
                report = json.load(f)

        model = report["models"]["input_count_plus_branch_drel"]
        self.assertEqual(model["n"], 4)
        self.assertIsNotNone(model["leave_one_out_r2"])

    def test_branch_capacity_model_accepts_legacy_csv_without_common_rank(self):
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
        for idx, branch_min in enumerate([0, 10, 20, 30]):
            rows.append(
                {
                    "label": f"legacy{idx}",
                    "topology_name": "legacy_fixed_count",
                    "n_edges": 20,
                    "raw_physical_parameter_count": 400,
                    "input_coupled_parameter_count": 200,
                    "d_rel": 100,
                    "comparison_branch_d_rel_min": branch_min,
                    "comparison_branch_d_rel_gini": 0.0,
                    "effective_rank_D": 10,
                    "effective_rank_D_masked": 10,
                    "condition_number_D": 10,
                    "condition_number_D_masked": 10,
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
            input_csv = os.path.join(tmpdir, "legacy_topology_results.csv")
            output_json = os.path.join(tmpdir, "regression.json")
            with open(input_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            self.run_regression(
                [
                    "--input_csv",
                    input_csv,
                    "--output_json",
                    output_json,
                ]
            )
            with open(output_json) as f:
                report = json.load(f)

        model = report["models"]["input_count_plus_branch_drel"]
        self.assertEqual(model["n"], 4)
        self.assertGreater(model["r2"], 0.9)

    def test_branch_margin_capacity_predictor_set_uses_enriched_columns(self):
        fieldnames = [
            "label",
            "topology_name",
            "n_edges",
            "raw_physical_parameter_count",
            "input_coupled_parameter_count",
            "d_rel",
            "comparison_branch_common_d_rel_min",
            "comparison_branch_common_d_rel_gini",
            "capacity_support_fraction",
            "capacity_support_min",
            "capacity_linear_test_accuracy",
            "capacity_linear_test_margin_p10",
            "test_novel_classes",
        ]
        rows = []
        for idx in range(8):
            capacity = 0.1 * idx
            rows.append(
                {
                    "label": f"capacity{idx}",
                    "topology_name": f"capacity_topo{idx}",
                    "n_edges": 20,
                    "raw_physical_parameter_count": 400,
                    "input_coupled_parameter_count": 200,
                    "d_rel": 100 + idx,
                    "comparison_branch_common_d_rel_min": 3 * idx,
                    "comparison_branch_common_d_rel_gini": 0.0,
                    "capacity_support_fraction": 0.5 + 0.05 * idx,
                    "capacity_support_min": idx % 3,
                    "capacity_linear_test_accuracy": 0.45 + capacity,
                    "capacity_linear_test_margin_p10": -0.4 + capacity,
                    "test_novel_classes": 40 + 30 * capacity,
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = os.path.join(tmpdir, "capacity_topology_results.csv")
            output_json = os.path.join(tmpdir, "regression.json")
            with open(input_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            self.run_regression(
                [
                    "--input_csv",
                    input_csv,
                    "--output_json",
                    output_json,
                ]
            )
            with open(output_json) as f:
                report = json.load(f)

        model = report["models"]["branch_margin_capacity"]
        self.assertEqual(model["n"], 8)
        self.assertIn("capacity_linear_test_accuracy", model["predictors"])
        self.assertGreater(model["r2"], 0.9)

    def test_rooted_tree_polytope_capacity_predictor_set_uses_rooted_columns(self):
        fieldnames = [
            "label",
            "topology_name",
            "n_edges",
            "raw_physical_parameter_count",
            "input_coupled_parameter_count",
            "d_rel",
            "comparison_branch_common_d_rel_min",
            "comparison_branch_common_d_rel_gini",
            "capacity_rooted_polytope_supported_branch_dim_fraction",
            "capacity_rooted_polytope_branch_root_support_min",
            "capacity_rooted_polytope_branch_root_support_gini",
            "capacity_rooted_polytope_branch_best_rank_min",
            "capacity_rooted_polytope_root_rank_mass_gini",
            "test_novel_classes",
        ]
        rows = []
        for idx in range(8):
            rooted = 0.1 * idx
            rows.append(
                {
                    "label": f"rooted{idx}",
                    "topology_name": f"rooted_topo{idx}",
                    "n_edges": 20,
                    "raw_physical_parameter_count": 400,
                    "input_coupled_parameter_count": 200,
                    "d_rel": 100 + idx,
                    "comparison_branch_common_d_rel_min": idx,
                    "comparison_branch_common_d_rel_gini": 0.0,
                    "capacity_rooted_polytope_supported_branch_dim_fraction": 0.3 + rooted,
                    "capacity_rooted_polytope_branch_root_support_min": idx % 4,
                    "capacity_rooted_polytope_branch_root_support_gini": 0.2,
                    "capacity_rooted_polytope_branch_best_rank_min": 2 * idx,
                    "capacity_rooted_polytope_root_rank_mass_gini": 0.1,
                    "test_novel_classes": 35 + 25 * rooted,
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_csv = os.path.join(tmpdir, "rooted_capacity_topology_results.csv")
            output_json = os.path.join(tmpdir, "regression.json")
            with open(input_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            self.run_regression(
                [
                    "--input_csv",
                    input_csv,
                    "--output_json",
                    output_json,
                ]
            )
            with open(output_json) as f:
                report = json.load(f)

        model = report["models"]["rooted_tree_polytope_capacity"]
        self.assertEqual(model["n"], 8)
        self.assertIn("capacity_rooted_polytope_branch_best_rank_min", model["predictors"])
        self.assertGreater(model["r2"], 0.9)


if __name__ == "__main__":
    unittest.main()
