import csv
import os
import subprocess
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from join_branch_margin_capacity import join_rows  # noqa: E402


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "join_branch_margin_capacity.py",
)


class JoinBranchMarginCapacityTests(unittest.TestCase):
    def test_join_replicates_capacity_across_seed_rows(self):
        topology_rows = [
            {
                "label": "topoA_seed1",
                "topology_name": "topoA",
                "test_novel_classes": "60",
            },
            {
                "label": "topoA_seed2",
                "topology_name": "topoA",
                "test_novel_classes": "70",
            },
            {
                "label": "topoB_seed1",
                "topology_name": "topoB",
                "test_novel_classes": "55",
            },
        ]
        capacity_rows = [
            {
                "topology_name": "topoA",
                "linear_test_accuracy": "0.91",
                "linear_test_margin_p10": "0.2",
                "support_fraction": "1.0",
            }
        ]
        rows, fieldnames, report = join_rows(
            topology_rows,
            capacity_rows,
            topology_keys=["topology_name"],
            capacity_keys=["topology_name"],
        )

        self.assertIn("capacity_linear_test_accuracy", fieldnames)
        self.assertEqual(report["n_matched_rows"], 2)
        self.assertEqual(report["n_missing_rows"], 1)
        self.assertEqual(rows[0]["capacity_linear_test_accuracy"], "0.91")
        self.assertEqual(rows[1]["capacity_linear_test_margin_p10"], "0.2")
        self.assertEqual(rows[2]["capacity_linear_test_accuracy"], "")

    def test_cli_writes_enriched_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            topology_csv = os.path.join(tmpdir, "topology.csv")
            capacity_csv = os.path.join(tmpdir, "capacity.csv")
            output_csv = os.path.join(tmpdir, "enriched.csv")
            with open(topology_csv, "w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["label", "topology_name", "test_novel_classes"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "label": "seed1",
                        "topology_name": "topoA",
                        "test_novel_classes": "60",
                    }
                )
            with open(capacity_csv, "w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["topology_name", "linear_test_accuracy", "support_min"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "topology_name": "topoA",
                        "linear_test_accuracy": "0.88",
                        "support_min": "1",
                    }
                )
            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--topology_csv",
                    topology_csv,
                    "--capacity_csv",
                    capacity_csv,
                    "--output_csv",
                    output_csv,
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            with open(output_csv, newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertIn("1/1 training rows matched", result.stdout)
        self.assertEqual(rows[0]["capacity_linear_test_accuracy"], "0.88")
        self.assertEqual(rows[0]["capacity_support_min"], "1")


if __name__ == "__main__":
    unittest.main()
