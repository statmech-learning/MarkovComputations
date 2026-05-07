import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "summarize_topology_mechanisms.py",
)


class SummarizeTopologyMechanismsTests(unittest.TestCase):
    def write_csv(self, path, fieldnames, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_summary(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_reports_topology_and_mechanism_correlations(self):
        topology_fields = [
            "label",
            "topology_name",
            "n_edges",
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
            "bottleneck_edge_fraction_095",
            "mean_shortest_path",
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
            "posterior_matched_comparison_gap_mean",
            "input_ablation_max_loss",
            "physical_ablation_max_loss",
        ]
        topology_rows = []
        mechanism_rows = []
        for idx, (label, edges, d_rel, branch_rank, target) in enumerate(
            [
                ("a1", 10, 10, 2, 50),
                ("a2", 10, 20, 4, 70),
                ("b1", 12, 30, 6, 60),
                ("b2", 12, 40, 8, 80),
            ]
        ):
            topology_rows.append(
                {
                    "label": label,
                    "topology_name": f"topo_{label[0]}",
                    "n_edges": edges,
                    "input_coupled_parameter_count": 100,
                    "d_rel": d_rel,
                    "comparison_branch_d_rel_min": branch_rank,
                    "comparison_branch_d_rel_gini": 1.0 / branch_rank,
                    "effective_rank_D": d_rel / 2.0,
                    "effective_rank_D_masked": d_rel / 3.0,
                    "condition_number_D": 100.0 - d_rel,
                    "condition_number_D_masked": 110.0 - d_rel,
                    "root_tree_count_gini": 0.1,
                    "edge_participation_gini": 0.2,
                    "bottleneck_edge_fraction_095": 0.0,
                    "mean_shortest_path": 2.0,
                    "input_edge_load_gini": 0.3,
                    "input_coord_load_gini": 0.4,
                    "test_novel_classes": target,
                }
            )
            mechanism_rows.append(
                {
                    "label": label,
                    "target_logprob_margin_mean": idx + 1,
                    "target_logprob_margin_branch_mean_min": branch_rank / 2.0,
                    "branch_active_tree_mi": branch_rank / 4.0,
                    "branch_active_tree_nmi": branch_rank / 8.0,
                    "branch_active_tree_purity_mean": 0.5 + idx * 0.1,
                    "posterior_matched_comparison_gap_mean": idx * 0.2,
                    "input_ablation_max_loss": idx + 2,
                    "physical_ablation_max_loss": idx,
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            topology_csv = os.path.join(tmpdir, "topology_results.csv")
            mechanism_csv = os.path.join(tmpdir, "mechanism_results.csv")
            output_json = os.path.join(tmpdir, "mechanism_summary.json")
            self.write_csv(topology_csv, topology_fields, topology_rows)
            self.write_csv(mechanism_csv, mechanism_fields, mechanism_rows)
            result = self.run_summary(
                [
                    "--topology_csv",
                    topology_csv,
                    "--mechanism_csv",
                    mechanism_csv,
                    "--output_json",
                    output_json,
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(output_json) as f:
                report = json.load(f)

        self.assertIn("Rows joined: 4", result.stdout)
        self.assertEqual(report["n_joined_rows"], 4)
        self.assertIn("comparison_branch_d_rel_min", report["overall_correlations"])
        self.assertIn("target_logprob_margin_branch_mean_min", report["overall_correlations"])
        self.assertIn("branch_active_tree_mi", report["overall_correlations"])
        self.assertGreater(report["overall_correlations"]["comparison_branch_d_rel_min"], 0.4)
        self.assertAlmostEqual(
            report["within_edge_count_residual_correlations"]["comparison_branch_d_rel_min"],
            1.0,
        )
        self.assertEqual(report["by_edge_count"]["10.0"]["n"], 2)
        self.assertIn("topo_a", report["by_topology_name"])
        self.assertEqual(len(report["family_within_edge_count"]), 2)


if __name__ == "__main__":
    unittest.main()
