import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "collect_branch_margin_capacity.py",
)


class CollectBranchMarginCapacityTests(unittest.TestCase):
    def test_collects_selected_library_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edge_json = os.path.join(tmpdir, "edges.json")
            with open(edge_json, "w") as handle:
                json.dump(
                    {
                        "name": "complete3",
                        "n_nodes": 3,
                        "edges": [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
                    },
                    handle,
                )
            library_csv = os.path.join(tmpdir, "selected.csv")
            with open(library_csv, "w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["selected", "topology_id", "topology_name", "family", "edge_json"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "selected": "1",
                        "topology_id": "g0",
                        "topology_name": "complete3",
                        "family": "complete",
                        "edge_json": edge_json,
                    }
                )
            output_csv = os.path.join(tmpdir, "capacity.csv")
            output_json = os.path.join(tmpdir, "capacity.json")

            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--library_csv",
                    library_csv,
                    "--output_csv",
                    output_csv,
                    "--output_json",
                    output_json,
                    "--N",
                    "2",
                    "--D",
                    "1",
                    "--train_samples",
                    "100",
                    "--test_samples",
                    "100",
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

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["topology_id"], "g0")
        self.assertEqual(rows[0]["family"], "complete")
        self.assertEqual(rows[0]["n_nodes"], "3")
        self.assertEqual(rows[0]["n_context"], "2")
        self.assertTrue(float(rows[0]["oracle_test_accuracy"]) >= 0.0)
        self.assertEqual(summary["n_rows"], 1)
        self.assertIn("complete", summary["families"])
        self.assertIn("Wrote 1 branch-margin capacity rows", result.stdout)


if __name__ == "__main__":
    unittest.main()
