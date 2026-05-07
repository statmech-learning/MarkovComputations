import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "make_input_mask_report.py",
)


class MakeInputMaskReportTests(unittest.TestCase):
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
        root = os.path.join(tmpdir, "input_mask_fixed_toy")
        os.makedirs(root)
        run_fields = [
            "label",
            "topology_name",
            "physical_topology_name",
            "input_mask_name",
            "input_mask_family",
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
        aggregate_fields = [
            "group",
            "topology_name",
            "physical_topology_name",
            "input_mask_name",
            "input_mask_family",
            "n_runs",
            "n_edges",
            "input_coupled_parameter_count",
            "d_rel",
            "comparison_branch_d_rel_min",
            "comparison_branch_d_rel_gini",
            "effective_rank_D_masked",
            "condition_number_D_masked",
            "input_edge_load_gini",
            "input_coord_load_gini",
            "target_mean",
            "target_max",
            "target_std",
            "target_logprob_margin_mean_mean",
            "target_logprob_margin_branch_mean_min_mean",
            "branch_active_tree_mi_mean",
            "branch_active_tree_nmi_mean",
            "branch_active_tree_purity_mean_mean",
            "posterior_matched_comparison_gap_mean_mean",
            "input_ablation_max_loss_mean",
            "physical_ablation_max_loss_mean",
        ]
        mechanism_fields = [
            "label",
            "target_logprob_margin_mean",
            "target_logprob_margin_branch_mean_min",
            "target_logprob_margin_branch_mean_gini",
            "target_accuracy_branch_mean_min",
            "branch_active_root_mi",
            "branch_active_tree_mi",
            "branch_active_tree_nmi",
            "branch_active_tree_purity_mean",
            "posterior_matched_comparison_gap_mean",
            "input_ablation_max_loss",
            "physical_ablation_max_loss",
        ]
        runs = [
            {
                "label": "mask_a_seed1",
                "topology_name": "mask_a",
                "physical_topology_name": "random",
                "input_mask_name": "mask_a",
                "input_mask_family": "balanced",
                "n_edges": 20,
                "input_coupled_parameter_count": 200,
                "d_rel": 200,
                "comparison_branch_d_rel_min": 40,
                "comparison_branch_d_rel_gini": 0.0,
                "effective_rank_D_masked": 24.0,
                "condition_number_D_masked": 10.0,
                "input_edge_load_gini": 0.1,
                "input_coord_load_gini": 0.2,
                "test_novel_classes": 90.0,
            },
            {
                "label": "mask_b_seed1",
                "topology_name": "mask_b",
                "physical_topology_name": "random",
                "input_mask_name": "mask_b",
                "input_mask_family": "coord_block",
                "n_edges": 20,
                "input_coupled_parameter_count": 200,
                "d_rel": 160,
                "comparison_branch_d_rel_min": 0,
                "comparison_branch_d_rel_gini": 0.75,
                "effective_rank_D_masked": 12.0,
                "condition_number_D_masked": 100.0,
                "input_edge_load_gini": 0.4,
                "input_coord_load_gini": 0.8,
                "test_novel_classes": 55.0,
            },
        ]
        aggregates = [
            {
                "group": "mask_a",
                "topology_name": "mask_a",
                "physical_topology_name": "random",
                "input_mask_name": "mask_a",
                "input_mask_family": "balanced",
                "n_runs": 2,
                "n_edges": 20,
                "input_coupled_parameter_count": 200,
                "d_rel": 200,
                "comparison_branch_d_rel_min": 40,
                "comparison_branch_d_rel_gini": 0.0,
                "effective_rank_D_masked": 24.0,
                "condition_number_D_masked": 10.0,
                "input_edge_load_gini": 0.1,
                "input_coord_load_gini": 0.2,
                "target_mean": 88.0,
                "target_max": 92.0,
                "target_std": 3.0,
                "target_logprob_margin_mean_mean": 2.5,
                "target_logprob_margin_branch_mean_min_mean": 2.0,
                "branch_active_tree_mi_mean": 1.2,
                "branch_active_tree_nmi_mean": 0.8,
                "branch_active_tree_purity_mean_mean": 0.9,
                "posterior_matched_comparison_gap_mean_mean": 0.4,
                "input_ablation_max_loss_mean": 20.0,
                "physical_ablation_max_loss_mean": 5.0,
            },
            {
                "group": "mask_b",
                "topology_name": "mask_b",
                "physical_topology_name": "random",
                "input_mask_name": "mask_b",
                "input_mask_family": "coord_block",
                "n_runs": 2,
                "n_edges": 20,
                "input_coupled_parameter_count": 200,
                "d_rel": 160,
                "comparison_branch_d_rel_min": 0,
                "comparison_branch_d_rel_gini": 0.75,
                "effective_rank_D_masked": 12.0,
                "condition_number_D_masked": 100.0,
                "input_edge_load_gini": 0.4,
                "input_coord_load_gini": 0.8,
                "target_mean": 55.0,
                "target_max": 60.0,
                "target_std": 4.0,
                "target_logprob_margin_mean_mean": 0.5,
                "target_logprob_margin_branch_mean_min_mean": -0.2,
                "branch_active_tree_mi_mean": 0.2,
                "branch_active_tree_nmi_mean": 0.1,
                "branch_active_tree_purity_mean_mean": 0.4,
                "posterior_matched_comparison_gap_mean_mean": -0.1,
                "input_ablation_max_loss_mean": 2.0,
                "physical_ablation_max_loss_mean": 1.0,
            },
        ]
        mechanisms = [
            {
                "label": "mask_a_seed1",
                "target_logprob_margin_mean": 2.4,
                "target_logprob_margin_branch_mean_min": 1.9,
                "target_logprob_margin_branch_mean_gini": 0.1,
                "target_accuracy_branch_mean_min": 80.0,
                "branch_active_root_mi": 0.7,
                "branch_active_tree_mi": 1.1,
                "branch_active_tree_nmi": 0.75,
                "branch_active_tree_purity_mean": 0.9,
                "posterior_matched_comparison_gap_mean": 0.4,
                "input_ablation_max_loss": 18.0,
                "physical_ablation_max_loss": 4.0,
            },
            {
                "label": "mask_b_seed1",
                "target_logprob_margin_mean": 0.4,
                "target_logprob_margin_branch_mean_min": -0.3,
                "target_logprob_margin_branch_mean_gini": 0.5,
                "target_accuracy_branch_mean_min": 25.0,
                "branch_active_root_mi": 0.1,
                "branch_active_tree_mi": 0.2,
                "branch_active_tree_nmi": 0.1,
                "branch_active_tree_purity_mean": 0.4,
                "posterior_matched_comparison_gap_mean": -0.2,
                "input_ablation_max_loss": 2.0,
                "physical_ablation_max_loss": 1.0,
            },
        ]
        self.write_csv(os.path.join(root, "topology_results.csv"), run_fields, runs)
        self.write_csv(os.path.join(root, "topology_seed_aggregates.csv"), aggregate_fields, aggregates)
        self.write_csv(os.path.join(root, "mechanism_results.csv"), mechanism_fields, mechanisms)

        essential_dir = os.path.join(root, "essential_inputmask50")
        selected_fields = [
            "selected",
            "topology_name",
            "input_coupled_parameter_count",
            "source_input_coupled_parameter_count_mean",
            "raw_essential_edges",
            "d_rel",
            "comparison_branch_d_rel_min",
            "effective_rank_D_masked",
            "source_test_novel_classes_max",
        ]
        self.write_csv(
            os.path.join(essential_dir, "selected.csv"),
            selected_fields,
            [
                {
                    "selected": 1,
                    "topology_name": "ess_a",
                    "input_coupled_parameter_count": 60,
                    "source_input_coupled_parameter_count_mean": 200,
                    "raw_essential_edges": 4,
                    "d_rel": 60,
                    "comparison_branch_d_rel_min": 12,
                    "effective_rank_D_masked": 10.0,
                    "source_test_novel_classes_max": 92.0,
                }
            ],
        )
        self.write_csv(os.path.join(essential_dir, "library.csv"), selected_fields, [])
        with open(os.path.join(essential_dir, "retrain_comparison.json"), "w") as f:
            json.dump(
                {
                    "n_joined": 1,
                    "source_mean_mean": 90.0,
                    "retrain_mean_mean": 70.0,
                    "retrain_max_best": 75.0,
                    "retention_mean_mean": 0.78,
                    "retention_max_mean": 0.82,
                    "retrain_input_coupled_parameter_count_mean": 60.0,
                },
                f,
            )
        return root

    def test_report_includes_branch_capacity_and_margin_columns(self):
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
        self.assertIn("branch d_rel min", markdown)
        self.assertIn("worst branch margin", markdown)
        self.assertIn("tree NMI", markdown)
        self.assertIn("Extracted Essential Input Masks", markdown)
        correlations = payload["pooled"]["run_correlations"]
        self.assertIn("comparison_branch_d_rel_min", correlations)
        self.assertIn("target_logprob_margin_branch_mean_min", correlations)


if __name__ == "__main__":
    unittest.main()
