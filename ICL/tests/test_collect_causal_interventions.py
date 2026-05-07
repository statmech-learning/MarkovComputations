import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "collect_causal_interventions.py",
)


class CollectCausalInterventionsTests(unittest.TestCase):
    def test_collects_reports_and_summarizes_deltas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_a")
            os.makedirs(run_dir)
            payload = {
                "run_dir": run_dir,
                "topology_name": "toy",
                "n_samples": 10,
                "n_repeats": 2,
                "baseline": {
                    "target_accuracy": 80.0,
                    "target_logprob_margin_mean": 1.5,
                    "branch_active_tree_mi": 0.4,
                    "branch_active_root_mi": 0.2,
                },
                "interventions": [
                    {
                        "intervention": "context_block_shuffle",
                        "repeat": 0,
                        "seed": 3,
                        "target_accuracy": 40.0,
                        "target_accuracy_delta": -40.0,
                        "target_logprob_margin_mean": 0.1,
                        "target_logprob_margin_mean_delta": -1.4,
                        "branch_active_tree_mi": 0.05,
                        "branch_active_tree_mi_delta": -0.35,
                    },
                    {
                        "intervention": "context_block_shuffle",
                        "repeat": 1,
                        "seed": 4,
                        "target_accuracy": 60.0,
                        "target_accuracy_delta": -20.0,
                    },
                ],
            }
            with open(os.path.join(run_dir, "causal_interventions.json"), "w") as handle:
                json.dump(payload, handle)

            output_csv = os.path.join(tmpdir, "causal.csv")
            output_json = os.path.join(tmpdir, "causal.json")
            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--input_root",
                    tmpdir,
                    "--output_csv",
                    output_csv,
                    "--output_json",
                    output_json,
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(output_csv, newline="") as handle:
                rows = list(csv.DictReader(handle))
            with open(output_json) as handle:
                summary = json.load(handle)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["label"], "run_a")
            self.assertEqual(rows[0]["baseline_target_accuracy"], "80.0")
            self.assertEqual(rows[0]["target_accuracy_delta"], "-40.0")
            self.assertEqual(summary["n_rows"], 2)
            self.assertEqual(summary["n_runs"], 1)
            self.assertAlmostEqual(
                summary["interventions"]["context_block_shuffle"]["target_accuracy_delta_mean"],
                -30.0,
            )
            self.assertIn("Wrote 2 intervention rows", result.stdout)


if __name__ == "__main__":
    unittest.main()
