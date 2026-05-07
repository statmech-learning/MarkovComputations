import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "aggregate_topology_seeds.py",
)


class AggregateTopologySeedsTests(unittest.TestCase):
    def write_csv(self, path, fieldnames, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_aggregate(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_aggregates_seed_mean_max_variance_and_mechanism_metrics(self):
        topology_fields = [
            "label",
            "seed",
            "topology_name",
            "n_edges",
            "input_coupled_parameter_count",
            "d_rel",
            "comparison_branch_d_rel_min",
            "comparison_branch_d_rel_gini",
            "effective_rank_D_masked",
            "condition_number_D_masked",
            "input_edge_load_gini",
            "input_coord_load_gini",
            "test_novel_classes",
        ]
        mechanism_fields = [
            "label",
            "target_logprob_margin_mean",
            "target_logprob_margin_branch_mean_min",
            "branch_active_tree_mi",
            "branch_active_tree_nmi",
            "branch_active_tree_purity_mean",
            "input_ablation_max_loss",
        ]
        topology_rows = [
            {
                "label": "topo_a_seed1",
                "seed": 1,
                "topology_name": "topo_a",
                "n_edges": 20,
                "input_coupled_parameter_count": 200,
                "d_rel": 180,
                "comparison_branch_d_rel_min": 30,
                "comparison_branch_d_rel_gini": 0.1,
                "effective_rank_D_masked": 15.0,
                "condition_number_D_masked": 12.0,
                "input_edge_load_gini": 0.2,
                "input_coord_load_gini": 0.3,
                "test_novel_classes": 80.0,
            },
            {
                "label": "topo_a_seed2",
                "seed": 2,
                "topology_name": "topo_a",
                "n_edges": 20,
                "input_coupled_parameter_count": 200,
                "d_rel": 180,
                "comparison_branch_d_rel_min": 30,
                "comparison_branch_d_rel_gini": 0.1,
                "effective_rank_D_masked": 15.0,
                "condition_number_D_masked": 12.0,
                "input_edge_load_gini": 0.2,
                "input_coord_load_gini": 0.3,
                "test_novel_classes": 100.0,
            },
            {
                "label": "topo_b_seed1",
                "seed": 1,
                "topology_name": "topo_b",
                "n_edges": 20,
                "input_coupled_parameter_count": 200,
                "d_rel": 120,
                "comparison_branch_d_rel_min": 0,
                "comparison_branch_d_rel_gini": 0.8,
                "effective_rank_D_masked": 8.0,
                "condition_number_D_masked": 100.0,
                "input_edge_load_gini": 0.6,
                "input_coord_load_gini": 0.7,
                "test_novel_classes": 40.0,
            },
        ]
        mechanism_rows = [
            {
                "label": "topo_a_seed1",
                "target_logprob_margin_mean": 1.0,
                "target_logprob_margin_branch_mean_min": 0.5,
                "branch_active_tree_mi": 0.8,
                "branch_active_tree_nmi": 0.7,
                "branch_active_tree_purity_mean": 0.9,
                "input_ablation_max_loss": 10.0,
            },
            {
                "label": "topo_a_seed2",
                "target_logprob_margin_mean": 3.0,
                "target_logprob_margin_branch_mean_min": 1.5,
                "branch_active_tree_mi": 1.2,
                "branch_active_tree_nmi": 0.9,
                "branch_active_tree_purity_mean": 1.0,
                "input_ablation_max_loss": 14.0,
            },
            {
                "label": "topo_b_seed1",
                "target_logprob_margin_mean": -0.5,
                "target_logprob_margin_branch_mean_min": -1.0,
                "branch_active_tree_mi": 0.1,
                "branch_active_tree_nmi": 0.05,
                "branch_active_tree_purity_mean": 0.3,
                "input_ablation_max_loss": 1.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            topology_csv = os.path.join(tmpdir, "topology_results.csv")
            mechanism_csv = os.path.join(tmpdir, "mechanism_results.csv")
            output_csv = os.path.join(tmpdir, "aggregates.csv")
            output_json = os.path.join(tmpdir, "aggregates.json")
            self.write_csv(topology_csv, topology_fields, topology_rows)
            self.write_csv(mechanism_csv, mechanism_fields, mechanism_rows)
            result = self.run_aggregate(
                [
                    "--topology_csv",
                    topology_csv,
                    "--mechanism_csv",
                    mechanism_csv,
                    "--output_csv",
                    output_csv,
                    "--output_json",
                    output_json,
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(output_csv, newline="") as f:
                rows = {row["group"]: row for row in csv.DictReader(f)}
            with open(output_json) as f:
                report = json.load(f)

        self.assertIn("Wrote 2 topology groups", result.stdout)
        self.assertEqual(report["n_groups"], 2)
        topo_a = rows["topo_a"]
        self.assertEqual(topo_a["n_runs"], "2")
        self.assertEqual(topo_a["labels"], "topo_a_seed1;topo_a_seed2")
        self.assertAlmostEqual(float(topo_a["target_mean"]), 90.0)
        self.assertAlmostEqual(float(topo_a["target_max"]), 100.0)
        self.assertAlmostEqual(float(topo_a["target_std"]), 10.0)
        self.assertEqual(topo_a["n_mechanism_runs"], "2")
        self.assertAlmostEqual(float(topo_a["target_logprob_margin_mean_mean"]), 2.0)
        self.assertAlmostEqual(
            float(topo_a["target_logprob_margin_branch_mean_min_mean"]),
            1.0,
        )
        self.assertAlmostEqual(float(topo_a["branch_active_tree_mi_mean"]), 1.0)
        self.assertIn("masked_tree_geometry", report["regressions"]["target_mean"])


if __name__ == "__main__":
    unittest.main()
