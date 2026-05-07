import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "collect_mechanism_results.py",
)


class CollectMechanismResultsTests(unittest.TestCase):
    def run_collector(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    def test_branch_assignment_summaries_are_collected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_a")
            os.makedirs(run_dir)
            metrics = {
                "run_dir": run_dir,
                "topology_name": "toy",
                "n_samples": 5,
                "target_accuracy": 80.0,
                "target_logprob_margin": [2.0, 4.0, -1.0, 1.0, 2.0],
                "target_correct": [1, 1, 0, 1, 0],
                "branch_ids": [0, 0, 1, 1, 1],
                "active_root": [2, 2, 1, 1, 3],
                "active_tree": [10, 10, 11, 12, 12],
            }
            with open(os.path.join(run_dir, "mechanism_metrics.json"), "w") as f:
                json.dump(metrics, f)
            output_csv = os.path.join(tmpdir, "mechanism_results.csv")

            result = self.run_collector(
                [
                    "--input_root",
                    tmpdir,
                    "--output_csv",
                    output_csv,
                ]
            )
            with open(output_csv, newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertIn("Wrote 1 rows", result.stdout)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["n_branches_observed"], "2")
        self.assertEqual(row["active_root_unique_count"], "3")
        self.assertEqual(row["active_tree_unique_count"], "3")
        self.assertAlmostEqual(float(row["branch_active_root_purity_mean"]), 5.0 / 6.0)
        self.assertAlmostEqual(float(row["branch_active_tree_purity_mean"]), 5.0 / 6.0)
        self.assertAlmostEqual(float(row["branch_active_tree_purity_min"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(row["branch_active_tree_unique_mean"]), 1.5)
        self.assertAlmostEqual(float(row["branch_active_root_nmi"]), 1.0)
        self.assertAlmostEqual(float(row["branch_active_tree_nmi"]), 1.0)
        self.assertAlmostEqual(float(row["target_logprob_margin_branch_mean_min"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(row["target_logprob_margin_branch_mean_mean"]), 11.0 / 6.0)
        self.assertAlmostEqual(float(row["target_accuracy_branch_mean_min"]), 100.0 / 3.0)
        self.assertAlmostEqual(float(row["target_accuracy_branch_mean_mean"]), 200.0 / 3.0)
        margin_rows = json.loads(row["target_logprob_margin_by_branch"])
        self.assertEqual(margin_rows[0]["branch"], 0)
        self.assertAlmostEqual(margin_rows[1]["mean"], 2.0 / 3.0)
        assignments = json.loads(row["branch_active_tree_assignment"])
        self.assertEqual(assignments[0]["dominant"], 10)
        self.assertEqual(assignments[1]["dominant"], 12)


if __name__ == "__main__":
    unittest.main()
