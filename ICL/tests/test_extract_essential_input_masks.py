import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "extract_essential_input_masks.py",
)


class ExtractEssentialInputMasksTests(unittest.TestCase):
    def write_csv(self, path, fieldnames, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    def run_extractor(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def make_run(self, root, label, novel_accuracy):
        run_dir = os.path.join(root, label)
        edges = [[0, 1], [1, 2], [2, 0]]
        full_mask = [[1, 1, 1] for _ in edges]
        self.write_json(
            os.path.join(run_dir, "topology.json"),
            {
                "name": "cycle3_masked",
                "physical_topology_name": "cycle3",
                "n_nodes": 3,
                "edges": edges,
                "p": 3,
                "input_mask": full_mask,
            },
        )
        self.write_json(
            os.path.join(run_dir, "mechanism_metrics.json"),
            {
                "target_accuracy": novel_accuracy - 1.0,
                "input_edge_ablation": {
                    "accuracy_loss": [0.2, 5.0, 1.0],
                },
            },
        )
        return run_dir

    def test_extracts_retrainable_masks_without_pruning_physical_edges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = os.path.join(tmpdir, "source")
            self.make_run(input_root, "run_a", 88.0)
            self.make_run(input_root, "run_b", 85.0)
            topology_csv = os.path.join(tmpdir, "topology_results.csv")
            self.write_csv(
                topology_csv,
                ["label", "test_novel_classes"],
                [
                    {"label": "run_a", "test_novel_classes": 88.0},
                    {"label": "run_b", "test_novel_classes": 85.0},
                ],
            )
            output_root = os.path.join(tmpdir, "essential_inputmask50")
            result = self.run_extractor(
                [
                    "--input_root",
                    input_root,
                    "--output_root",
                    output_root,
                    "--topology_csv",
                    topology_csv,
                    "--top_k",
                    "1",
                    "--select_masks",
                    "1",
                    "--N",
                    "2",
                    "--D",
                    "1",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            with open(os.path.join(output_root, "summary.json")) as f:
                summary = json.load(f)
            with open(os.path.join(output_root, "selected.csv"), newline="") as f:
                selected_rows = list(csv.DictReader(f))
            with open(os.path.join(output_root, "library.csv"), newline="") as f:
                library_rows = list(csv.DictReader(f))

            self.assertEqual(summary["n_unique_masks"], 1)
            self.assertEqual(summary["n_selected_masks"], 1)
            self.assertEqual(len(library_rows), 1)
            self.assertEqual(len(selected_rows), 1)
            row = selected_rows[0]
            self.assertEqual(row["selected"], "1")
            self.assertEqual(row["raw_essential_edges"], "1.0")
            self.assertEqual(row["input_coupled_parameter_count"], "3")
            self.assertEqual(row["source_run_count"], "2")
            self.assertEqual(row["source_test_novel_classes_max"], "88.0")
            self.assertTrue(row["comparison_branch_d_rel_min"])
            self.assertEqual(row["comparison_branch_common_d_rel_source"], "artifact")
            self.assertEqual(row["comparison_branch_input_overlap_source"], "artifact")

            with open(row["edge_json"]) as f:
                edge_payload = json.load(f)
            with open(row["input_mask_json"]) as f:
                mask_payload = json.load(f)

            self.assertEqual(edge_payload["edges"], [[0, 1], [1, 2], [2, 0]])
            self.assertEqual(mask_payload["edges"], [[0, 1], [1, 2], [2, 0]])
            self.assertEqual(mask_payload["input_mask"], [[0, 0, 0], [1, 1, 1], [0, 0, 0]])
            self.assertEqual(mask_payload["mask_summary"]["input_coupled_edge_count"], 1)


if __name__ == "__main__":
    unittest.main()
