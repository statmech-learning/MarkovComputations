import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "compare_matched_motif_controls.py",
)


class CompareMatchedMotifControlsTests(unittest.TestCase):
    def write_csv(self, path, fieldnames, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_compares_control_retrains_to_source_motif_retrains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_csv = os.path.join(tmpdir, "source.csv")
            source_retrain_csv = os.path.join(tmpdir, "source_retrain.csv")
            control_csv = os.path.join(tmpdir, "control.csv")
            control_retrain_csv = os.path.join(tmpdir, "control_retrain.csv")
            output_csv = os.path.join(tmpdir, "joined.csv")
            output_json = os.path.join(tmpdir, "summary.json")
            output_md = os.path.join(tmpdir, "summary.md")
            self.write_csv(
                source_csv,
                [
                    "topology_id",
                    "topology_name",
                    "source_test_novel_classes_mean",
                    "source_test_novel_classes_max",
                    "n_edges",
                    "d_rel",
                ],
                [
                    {
                        "topology_id": "motif0",
                        "topology_name": "motif_zero",
                        "source_test_novel_classes_mean": "90",
                        "source_test_novel_classes_max": "95",
                        "n_edges": "12",
                        "d_rel": "200",
                    }
                ],
            )
            self.write_csv(
                source_retrain_csv,
                ["group", "topology_name", "target_mean", "target_max"],
                [
                    {
                        "group": "motif_zero",
                        "topology_name": "motif_zero",
                        "target_mean": "72",
                        "target_max": "80",
                    }
                ],
            )
            self.write_csv(
                control_csv,
                [
                    "topology_id",
                    "topology_name",
                    "control_kind",
                    "source_topology_id",
                    "source_topology_name",
                    "match_score",
                    "n_edges",
                    "d_rel",
                ],
                [
                    {
                        "topology_id": "ctrl0",
                        "topology_name": "control_zero",
                        "control_kind": "random_sc",
                        "source_topology_id": "motif0",
                        "source_topology_name": "motif_zero",
                        "match_score": "0.1",
                        "n_edges": "12",
                        "d_rel": "198",
                    }
                ],
            )
            self.write_csv(
                control_retrain_csv,
                ["group", "topology_name", "target_mean", "target_max", "n_edges", "d_rel"],
                [
                    {
                        "group": "control_zero",
                        "topology_name": "control_zero",
                        "target_mean": "62",
                        "target_max": "70",
                        "n_edges": "12",
                        "d_rel": "198",
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--source_csv",
                    source_csv,
                    "--source_retrain_csv",
                    source_retrain_csv,
                    "--control_csv",
                    control_csv,
                    "--control_retrain_csv",
                    control_retrain_csv,
                    "--output_csv",
                    output_csv,
                    "--output_json",
                    output_json,
                    "--output_md",
                    output_md,
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            with open(output_csv, newline="") as handle:
                rows = list(csv.DictReader(handle))
            with open(output_json) as handle:
                summary = json.load(handle)
            with open(output_md) as handle:
                markdown = handle.read()

        self.assertIn("Joined matched controls: 1", result.stdout)
        self.assertEqual(rows[0]["control_minus_source_retrain_mean"], "-10.0")
        self.assertEqual(summary["n_joined"], 1)
        self.assertEqual(summary["by_control_kind"]["random_sc"]["control_win_rate_mean"], 0.0)
        self.assertIn("Matched Motif Control Comparison", markdown)


if __name__ == "__main__":
    unittest.main()
