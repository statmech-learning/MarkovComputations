import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "make_topology_library.py",
)


class MakeTopologyLibraryTests(unittest.TestCase):
    def run_library(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_generates_selected_strongly_connected_topologies_with_branch_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = os.path.join(tmpdir, "topology_library")
            result = self.run_library(
                [
                    "--output_root",
                    output_root,
                    "--n_nodes",
                    "4",
                    "--n_edges",
                    "6",
                    "--N",
                    "2",
                    "--D",
                    "1",
                    "--families",
                    "cycle_chords,random_sc",
                    "--candidate_seeds",
                    "1:3",
                    "--select_topologies",
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
                self.assertEqual(row["n_nodes"], "4")
                self.assertEqual(row["n_edges"], "6")
                self.assertEqual(row["p"], "3")
                self.assertIn(row["family"], {"cycle_chords", "random_sc"})
                self.assertTrue(row["comparison_branch_d_rel_min"])
                self.assertTrue(os.path.exists(row["edge_json"]))
                with open(row["edge_json"]) as f:
                    payload = json.load(f)
                self.assertEqual(payload["n_nodes"], 4)
                self.assertEqual(len(payload["edges"]), 6)
                self.assertTrue(payload["metrics"]["strongly_connected"])
                self.assertIn("comparison_branch_d_rel_min", payload["metrics"])

            self.assertIn("Generated candidates", result.stdout)
            self.assertIn("Selected topologies: 2", result.stdout)


if __name__ == "__main__":
    unittest.main()
