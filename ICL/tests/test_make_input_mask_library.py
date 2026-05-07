import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "make_input_mask_library.py",
)


class MakeInputMaskLibraryTests(unittest.TestCase):
    def write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    def run_library(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_generates_selected_retrainable_input_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edge_json = os.path.join(tmpdir, "cycle3.json")
            edges = [[0, 1], [1, 2], [2, 0]]
            self.write_json(
                edge_json,
                {
                    "name": "cycle3",
                    "family": "cycle",
                    "n_nodes": 3,
                    "edges": edges,
                },
            )
            output_root = os.path.join(tmpdir, "input_mask_library")
            result = self.run_library(
                [
                    "--edge_json",
                    edge_json,
                    "--output_root",
                    output_root,
                    "--N",
                    "2",
                    "--D",
                    "1",
                    "--families",
                    "entry_random,edge_block,balanced",
                    "--coupled_counts",
                    "3",
                    "--candidate_seeds",
                    "1:2",
                    "--select_masks",
                    "2",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            with open(os.path.join(output_root, "library.csv"), newline="") as f:
                library_rows = list(csv.DictReader(f))
            with open(os.path.join(output_root, "selected.csv"), newline="") as f:
                selected_rows = list(csv.DictReader(f))

            self.assertGreaterEqual(len(library_rows), 2)
            self.assertEqual(len(selected_rows), 2)
            self.assertEqual(sum(row["selected"] == "1" for row in library_rows), 2)

            for row in selected_rows:
                self.assertEqual(row["selected"], "1")
                self.assertEqual(row["physical_topology_name"], "cycle3")
                self.assertEqual(row["input_coupled_parameter_count"], "3")
                self.assertTrue(row["comparison_branch_d_rel_min"])
                self.assertEqual(row["edge_json"], os.path.abspath(edge_json))
                self.assertTrue(os.path.exists(row["input_mask_json"]))

                with open(row["input_mask_json"]) as f:
                    mask_payload = json.load(f)
                self.assertEqual(mask_payload["edges"], edges)
                self.assertEqual(mask_payload["mask_summary"]["input_coupled_parameter_count"], 3)
                self.assertEqual(mask_payload["topology_metrics"]["physical_topology_name"], "cycle3")

            self.assertIn("Generated candidate masks", result.stdout)
            self.assertIn("Selected masks: 2", result.stdout)


if __name__ == "__main__":
    unittest.main()
