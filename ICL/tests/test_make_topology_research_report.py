import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "make_topology_research_report.py",
)


class MakeTopologyResearchReportTests(unittest.TestCase):
    def write_csv(self, path, fieldnames, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_report(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def make_experiment(self, tmpdir):
        root = os.path.join(tmpdir, "topology_fixed_toy")
        os.makedirs(root)

        run_fields = [
            "label",
            "topology_name",
            "n_nodes",
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
            "input_edge_load_gini",
            "input_coord_load_gini",
            "mean_shortest_path",
            "test_novel_classes",
        ]
        aggregate_fields = [
            "group",
            "topology_name",
            "n_runs",
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
            "input_edge_load_gini",
            "input_coord_load_gini",
            "mean_shortest_path",
            "target_mean",
            "target_max",
            "target_std",
            "target_logprob_margin_mean_mean",
            "target_logprob_margin_branch_mean_min_mean",
            "branch_active_tree_mi_mean",
            "posterior_matched_comparison_gap_mean_mean",
            "tree_comparison_energy_fraction_mean_mean",
            "active_tree_matched_comparison_gap_mean_mean",
            "input_ablation_max_loss_mean",
        ]
        mechanism_fields = [
            "label",
            "target_logprob_margin_mean",
            "target_logprob_margin_branch_mean_min",
            "branch_active_root_mi",
            "branch_active_tree_mi",
            "tree_comparison_energy_fraction_mean",
            "posterior_matched_comparison_gap_mean",
            "active_tree_matched_comparison_gap_mean",
            "input_ablation_max_loss",
            "physical_ablation_max_loss",
        ]

        runs = []
        aggregates = []
        mechanisms = []
        for idx in range(6):
            branch_min = 8 + idx * 4
            accuracy = 50.0 + idx * 7.0
            label = f"toy_topology_{idx}_seed1"
            runs.append(
                {
                    "label": label,
                    "topology_name": f"toy_topology_{idx}",
                    "n_nodes": 5,
                    "n_edges": 12 + (idx % 3) * 2,
                    "input_coupled_parameter_count": 120 + idx * 8,
                    "d_rel": 80 + branch_min,
                    "comparison_branch_d_rel_min": branch_min,
                    "comparison_branch_d_rel_gini": 0.5 - idx * 0.05,
                    "effective_rank_D": 7.0 + idx,
                    "effective_rank_D_masked": 6.0 + idx,
                    "condition_number_D": 40.0 - idx,
                    "condition_number_D_masked": 50.0 - idx,
                    "root_tree_count_gini": 0.1 + idx * 0.01,
                    "edge_participation_gini": 0.2 + idx * 0.01,
                    "input_edge_load_gini": 0.3 - idx * 0.02,
                    "input_coord_load_gini": 0.4 - idx * 0.03,
                    "mean_shortest_path": 1.5 + idx * 0.1,
                    "test_novel_classes": accuracy,
                }
            )
            aggregates.append(
                {
                    "group": f"toy_topology_{idx}",
                    "topology_name": f"toy_topology_{idx}",
                    "n_runs": 3,
                    "n_edges": 12 + (idx % 3) * 2,
                    "input_coupled_parameter_count": 120 + idx * 8,
                    "d_rel": 80 + branch_min,
                    "comparison_branch_d_rel_min": branch_min,
                    "comparison_branch_d_rel_gini": 0.5 - idx * 0.05,
                    "effective_rank_D": 7.0 + idx,
                    "effective_rank_D_masked": 6.0 + idx,
                    "condition_number_D": 40.0 - idx,
                    "condition_number_D_masked": 50.0 - idx,
                    "root_tree_count_gini": 0.1 + idx * 0.01,
                    "edge_participation_gini": 0.2 + idx * 0.01,
                    "input_edge_load_gini": 0.3 - idx * 0.02,
                    "input_coord_load_gini": 0.4 - idx * 0.03,
                    "mean_shortest_path": 1.5 + idx * 0.1,
                    "target_mean": accuracy - 2.0,
                    "target_max": accuracy + 1.0,
                    "target_std": 2.0 + idx * 0.2,
                    "target_logprob_margin_mean_mean": 0.2 + idx * 0.3,
                    "target_logprob_margin_branch_mean_min_mean": -0.1 + idx * 0.25,
                    "branch_active_tree_mi_mean": 0.1 + idx * 0.2,
                    "posterior_matched_comparison_gap_mean_mean": -0.2 + idx * 0.1,
                    "tree_comparison_energy_fraction_mean_mean": 0.3 + idx * 0.05,
                    "active_tree_matched_comparison_gap_mean_mean": -0.1 + idx * 0.12,
                    "input_ablation_max_loss_mean": 1.0 + idx * 2.0,
                }
            )
            mechanisms.append(
                {
                    "label": label,
                    "target_logprob_margin_mean": 0.1 + idx * 0.4,
                    "target_logprob_margin_branch_mean_min": -0.2 + idx * 0.3,
                    "branch_active_root_mi": 0.1 + idx * 0.1,
                    "branch_active_tree_mi": 0.2 + idx * 0.2,
                    "tree_comparison_energy_fraction_mean": 0.3 + idx * 0.05,
                    "posterior_matched_comparison_gap_mean": -0.2 + idx * 0.12,
                    "active_tree_matched_comparison_gap_mean": -0.1 + idx * 0.15,
                    "input_ablation_max_loss": 1.0 + idx * 2.0,
                    "physical_ablation_max_loss": 0.5 + idx,
                }
            )

        self.write_csv(os.path.join(root, "topology_results.csv"), run_fields, runs)
        self.write_csv(os.path.join(root, "topology_seed_aggregates.csv"), aggregate_fields, aggregates)
        self.write_csv(os.path.join(root, "mechanism_results.csv"), mechanism_fields, mechanisms)

        with open(os.path.join(root, "topology_regression.json"), "w") as f:
            json.dump(
                {
                    "models": {
                        "input_count_plus_branch_drel": {
                            "n": 6,
                            "r2": 0.9,
                            "leave_one_out_r2": 0.7,
                            "rmse": 2.0,
                            "predictors": [
                                "input_coupled_parameter_count",
                                "d_rel",
                                "comparison_branch_d_rel_min",
                                "comparison_branch_d_rel_gini",
                            ],
                        }
                    }
                },
                f,
            )
        with open(os.path.join(root, "topology_seed_aggregates.json"), "w") as f:
            json.dump(
                {
                    "regressions": {
                        "target_mean": {
                            "input_count_plus_branch_drel": {
                                "n": 6,
                                "r2": 0.88,
                                "leave_one_out_r2": 0.6,
                                "rmse": 3.0,
                                "predictors": [
                                    "input_coupled_parameter_count",
                                    "d_rel",
                                    "comparison_branch_d_rel_min",
                                    "comparison_branch_d_rel_gini",
                                ],
                            }
                        },
                        "target_max": {},
                        "target_std": {},
                    }
                },
                f,
            )
        with open(os.path.join(root, "mechanism_summary.json"), "w") as f:
            json.dump(
                {
                    "overall_correlations": {
                        "comparison_branch_d_rel_min": 0.75,
                        "target_logprob_margin_branch_mean_min": 0.8,
                    },
                    "within_edge_count_residual_correlations": {
                        "comparison_branch_d_rel_min": 0.25,
                        "target_logprob_margin_branch_mean_min": 0.3,
                    },
                },
                f,
            )

        essential_dir = os.path.join(root, "essential_inputmask50")
        comparison_fields = [
            "topology_name",
            "d_rel",
            "source_test_novel_classes_mean",
            "retrain_input_coupled_parameter_count",
            "retrain_target_mean",
            "retrain_target_max",
            "retrain_retention_mean",
            "retrain_retention_max",
        ]
        self.write_csv(
            os.path.join(essential_dir, "retrain_comparison.csv"),
            comparison_fields,
            [
                {
                    "topology_name": "essential_toy",
                    "d_rel": 24,
                    "source_test_novel_classes_mean": 82.0,
                    "retrain_input_coupled_parameter_count": 48,
                    "retrain_target_mean": 70.0,
                    "retrain_target_max": 76.0,
                    "retrain_retention_mean": 0.85,
                    "retrain_retention_max": 0.92,
                }
            ],
        )
        with open(os.path.join(essential_dir, "retrain_comparison.json"), "w") as f:
            json.dump(
                {
                    "n_joined": 1,
                    "source_mean_mean": 82.0,
                    "retrain_mean_mean": 70.0,
                    "retrain_max_best": 76.0,
                    "retention_mean_mean": 0.85,
                    "retention_max_mean": 0.92,
                    "retrain_input_coupled_parameter_count_mean": 48.0,
                },
                f,
            )
        retrain_root = os.path.join(root, "essential_inputmask50_retrain")
        self.write_csv(
            os.path.join(retrain_root, "topology_seed_aggregates.csv"),
            aggregate_fields,
            [dict(aggregates[0], group="essential_toy", topology_name="essential_toy")],
        )
        with open(os.path.join(retrain_root, "topology_seed_aggregates.json"), "w") as f:
            json.dump({"regressions": {"target_mean": {}, "target_max": {}, "target_std": {}}}, f)

        return root

    def test_report_includes_branch_metrics_pooled_models_and_essential_retrains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir)
            output_md = os.path.join(tmpdir, "report.md")
            output_json = os.path.join(tmpdir, "report.json")
            result = self.run_report(
                [
                    "--experiment",
                    f"toy={root}",
                    "--output_md",
                    output_md,
                    "--output_json",
                    output_json,
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(output_md) as f:
                markdown = f.read()
            with open(output_json) as f:
                payload = json.load(f)

        self.assertIn("Wrote", result.stdout)
        self.assertIn("weakest comparison-branch paired rank", markdown)
        self.assertIn("worst branch mean margin", markdown)
        self.assertIn("input_plus_branch_drel", markdown)
        self.assertIn("Essential Motif Retraining", markdown)
        self.assertIn("input mask", markdown)

        pooled_model = payload["pooled"]["run_level"]["input_plus_branch_drel"]
        self.assertIn("comparison_branch_d_rel_min", pooled_model["predictors"])
        self.assertIn("comparison_branch_d_rel_gini", pooled_model["predictors"])

        correlations = payload["experiments"][0]["mechanism_correlations"]
        self.assertIn("comparison_branch_d_rel_min", correlations)
        self.assertIn("target_logprob_margin_branch_mean_min", correlations)
        self.assertEqual(
            payload["experiments"][0]["essential_input50"]["comparison"]["n_joined"],
            1,
        )
        self.assertEqual(
            payload["experiments"][0]["essential_input50"]["source_dir"],
            "essential_inputmask50",
        )


if __name__ == "__main__":
    unittest.main()
