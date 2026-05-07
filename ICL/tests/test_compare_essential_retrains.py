import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "compare_essential_retrains.py",
)


class CompareEssentialRetrainsTests(unittest.TestCase):
    def write_csv(self, path, fieldnames, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_compare(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_joins_selected_motifs_to_retrain_aggregates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            selected_csv = os.path.join(tmpdir, "selected.csv")
            retrain_csv = os.path.join(tmpdir, "retrain_aggregates.csv")
            output_csv = os.path.join(tmpdir, "comparison.csv")
            output_json = os.path.join(tmpdir, "comparison.json")

            self.write_csv(
                selected_csv,
                [
                    "topology_name",
                    "source_labels",
                    "source_test_novel_classes_max",
                    "source_test_novel_classes_mean",
                    "source_target_accuracy_max",
                    "source_target_accuracy_mean",
                    "source_input_coupled_parameter_count_mean",
                ],
                [
                    {
                        "topology_name": "mask_a",
                        "source_labels": "run_a;run_b",
                        "source_test_novel_classes_max": 90.0,
                        "source_test_novel_classes_mean": 80.0,
                        "source_target_accuracy_max": 92.0,
                        "source_target_accuracy_mean": 82.0,
                        "source_input_coupled_parameter_count_mean": 200,
                    },
                    {
                        "topology_name": "mask_b",
                        "source_labels": "run_c",
                        "source_test_novel_classes_max": 70.0,
                        "source_test_novel_classes_mean": 60.0,
                        "source_target_accuracy_max": 72.0,
                        "source_target_accuracy_mean": 62.0,
                        "source_input_coupled_parameter_count_mean": 200,
                    },
                ],
            )
            self.write_csv(
                retrain_csv,
                [
                    "group",
                    "topology_name",
                    "n_edges",
                    "d_rel",
                    "comparison_branch_d_rel_min",
                    "comparison_branch_d_rel_gini",
                    "effective_rank_D",
                    "effective_rank_D_masked",
                    "input_coupled_parameter_count",
                    "target_max",
                    "target_mean",
                    "target_std",
                ],
                [
                    {
                        "group": "mask_a_group",
                        "topology_name": "mask_a",
                        "n_edges": 20,
                        "d_rel": 64,
                        "comparison_branch_d_rel_min": 12,
                        "comparison_branch_d_rel_gini": 0.25,
                        "effective_rank_D": 15.0,
                        "effective_rank_D_masked": 10.0,
                        "input_coupled_parameter_count": 48,
                        "target_max": 72.0,
                        "target_mean": 64.0,
                        "target_std": 4.0,
                    },
                    {
                        "group": "unmatched_group",
                        "topology_name": "not_selected",
                        "n_edges": 20,
                        "d_rel": 20,
                        "comparison_branch_d_rel_min": 0,
                        "comparison_branch_d_rel_gini": 1.0,
                        "effective_rank_D": 5.0,
                        "effective_rank_D_masked": 4.0,
                        "input_coupled_parameter_count": 12,
                        "target_max": 55.0,
                        "target_mean": 50.0,
                        "target_std": 2.0,
                    },
                ],
            )

            result = self.run_compare(
                [
                    "--selected_csv",
                    selected_csv,
                    "--retrain_aggregate_csv",
                    retrain_csv,
                    "--output_csv",
                    output_csv,
                    "--output_json",
                    output_json,
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(output_csv, newline="") as f:
                rows = list(csv.DictReader(f))
            with open(output_json) as f:
                summary = json.load(f)

        self.assertIn("Joined motifs: 1", result.stdout)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["topology_name"], "mask_a")
        self.assertEqual(row["topology_id"], "mask_a_group")
        self.assertEqual(row["retrain_input_coupled_parameter_count"], "48.0")
        self.assertAlmostEqual(float(row["retrain_retention_max"]), 0.8)
        self.assertAlmostEqual(float(row["retrain_retention_mean"]), 0.8)
        self.assertEqual(summary["n_joined"], 1)
        self.assertEqual(summary["retrain_input_coupled_parameter_count_mean"], 48.0)
        self.assertEqual(summary["comparison_branch_d_rel_min_mean"], 12.0)

    def test_zero_join_fails_with_clear_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            selected_csv = os.path.join(tmpdir, "selected.csv")
            retrain_csv = os.path.join(tmpdir, "retrain_aggregates.csv")
            output_csv = os.path.join(tmpdir, "comparison.csv")
            output_json = os.path.join(tmpdir, "comparison.json")

            self.write_csv(
                selected_csv,
                ["topology_name", "source_test_novel_classes_max", "source_test_novel_classes_mean"],
                [
                    {
                        "topology_name": "selected_mask",
                        "source_test_novel_classes_max": 90.0,
                        "source_test_novel_classes_mean": 80.0,
                    }
                ],
            )
            self.write_csv(
                retrain_csv,
                ["group", "topology_name", "target_max", "target_mean"],
                [
                    {
                        "group": "other_group",
                        "topology_name": "other_mask",
                        "target_max": 70.0,
                        "target_mean": 60.0,
                    }
                ],
            )

            result = self.run_compare(
                [
                    "--selected_csv",
                    selected_csv,
                    "--retrain_aggregate_csv",
                    retrain_csv,
                    "--output_csv",
                    output_csv,
                    "--output_json",
                    output_json,
                ]
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No retrained motifs joined by topology_name", result.stderr + result.stdout)
        self.assertIn("selected_mask", result.stderr + result.stdout)
        self.assertIn("other_mask", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
