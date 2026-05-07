import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "make_matched_motif_controls.py",
)


class MakeMatchedMotifControlsTests(unittest.TestCase):
    def write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(payload, handle)

    def write_csv(self, path, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fieldnames = [
            "selected",
            "topology_id",
            "topology_name",
            "n_nodes",
            "n_edges",
            "p",
            "edge_json",
            "d_rel",
            "effective_rank_D",
            "root_tree_count_gini",
            "edge_participation_gini",
        ]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_script(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_generates_selected_matched_random_controls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edge_json = os.path.join(tmpdir, "motifs", "motif.json")
            self.write_json(
                edge_json,
                {
                    "name": "motif_cycle4_chord",
                    "n_nodes": 4,
                    "edges": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
                },
            )
            source_csv = os.path.join(tmpdir, "essential_input50", "selected.csv")
            self.write_csv(
                source_csv,
                [
                    {
                        "selected": "1",
                        "topology_id": "motif_a",
                        "topology_name": "motif_a",
                        "n_nodes": "4",
                        "n_edges": "5",
                        "p": "3",
                        "edge_json": edge_json,
                        "d_rel": "",
                        "effective_rank_D": "",
                        "root_tree_count_gini": "",
                        "edge_participation_gini": "",
                    }
                ],
            )

            output_root = os.path.join(tmpdir, "matched_controls")
            result = self.run_script(
                [
                    "--source_csv",
                    source_csv,
                    "--output_root",
                    output_root,
                    "--N",
                    "2",
                    "--D",
                    "1",
                    "--control_kinds",
                    "random_sc",
                    "--controls_per_source",
                    "2",
                    "--candidates_per_source",
                    "12",
                    "--seed",
                    "7",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            with open(os.path.join(output_root, "library.csv"), newline="") as handle:
                library_rows = list(csv.DictReader(handle))
            with open(os.path.join(output_root, "selected.csv"), newline="") as handle:
                selected_rows = list(csv.DictReader(handle))
            with open(os.path.join(output_root, "summary.json")) as handle:
                summary = json.load(handle)

            self.assertGreaterEqual(len(library_rows), 2)
            self.assertEqual(len(selected_rows), 2)
            self.assertEqual(summary["n_selected_controls"], 2)
            for row in selected_rows:
                self.assertEqual(row["selected"], "1")
                self.assertEqual(row["family"], "matched_motif_control")
                self.assertEqual(row["control_kind"], "random_sc")
                self.assertEqual(row["source_topology_id"], "motif_a")
                self.assertEqual(row["n_nodes"], "4")
                self.assertEqual(row["n_edges"], "5")
                self.assertTrue(row["match_score"])
                self.assertTrue(os.path.exists(row["edge_json"]))
                with open(row["edge_json"]) as handle:
                    payload = json.load(handle)
                self.assertEqual(payload["n_nodes"], 4)
                self.assertEqual(len(payload["edges"]), 5)
                self.assertEqual(payload["family"], "matched_motif_control")
                self.assertTrue(payload["metrics"]["strongly_connected"])

            self.assertIn("Selected matched controls: 2", result.stdout)


if __name__ == "__main__":
    unittest.main()
