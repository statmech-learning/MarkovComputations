import csv
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "submit_topology_library_sweep.py",
)


class SubmitTopologyLibrarySweepTests(unittest.TestCase):
    def make_library(self, tmpdir):
        edge_json = os.path.join(tmpdir, "edge.json")
        mask_json = os.path.join(tmpdir, "mask.json")
        for path in (edge_json, mask_json):
            with open(path, "w") as f:
                f.write("{}\n")
        library_csv = os.path.join(tmpdir, "selected.csv")
        with open(library_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "selected",
                    "topology_id",
                    "topology_name",
                    "edge_json",
                    "input_mask_json",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "selected": "1",
                    "topology_id": "g0001",
                    "topology_name": "test_topology",
                    "edge_json": os.path.basename(edge_json),
                    "input_mask_json": os.path.basename(mask_json),
                }
            )
            writer.writerow(
                {
                    "selected": "0",
                    "topology_id": "skip",
                    "topology_name": "unselected",
                    "edge_json": os.path.basename(edge_json),
                    "input_mask_json": os.path.basename(mask_json),
                }
            )
        return library_csv

    def run_submitter(self, args, cwd):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )

    def run_submitter_unchecked(self, args, cwd):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            cwd=cwd,
            check=False,
            text=True,
            capture_output=True,
        )

    def test_status_manifest_reports_completed_and_missing_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library_csv = self.make_library(tmpdir)
            output_root = os.path.join(tmpdir, "runs")
            completed_dir = os.path.join(output_root, "g0001_trainseed1")
            os.makedirs(completed_dir)
            with open(os.path.join(completed_dir, "results.pkl"), "wb") as f:
                f.write(b"done")

            manifest = os.path.join(tmpdir, "manifest.csv")
            result = self.run_submitter(
                [
                    "--library_csv",
                    library_csv,
                    "--output_root",
                    output_root,
                    "--seeds",
                    "1,2",
                    "--status_only",
                    "--manifest_csv",
                    manifest,
                ],
                cwd=tmpdir,
            )

            self.assertIn("Expected tasks: 2", result.stdout)
            self.assertIn("Completed outputs: 1", result.stdout)
            self.assertIn("Missing outputs: 1", result.stdout)
            with open(manifest) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            completed = {row["train_seed"]: row["completed"] for row in rows}
            self.assertEqual(completed, {"1": "True", "2": "False"})
            self.assertTrue(all("--input_mask_json" in row["command"] for row in rows))

    def test_missing_only_array_metadata_excludes_completed_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            library_csv = self.make_library(tmpdir)
            output_root = os.path.join(tmpdir, "runs")
            completed_dir = os.path.join(output_root, "g0001_trainseed1")
            os.makedirs(completed_dir)
            with open(os.path.join(completed_dir, "results.pkl"), "wb") as f:
                f.write(b"done")

            result = self.run_submitter(
                [
                    "--library_csv",
                    library_csv,
                    "--output_root",
                    output_root,
                    "--seeds",
                    "1,2",
                    "--missing_only",
                    "--array",
                    "--dry-run",
                    "--clean",
                    "--max-concurrent",
                    "3",
                ],
                cwd=tmpdir,
            )

            self.assertIn("Tasks: 1", result.stdout)
            self.assertIn("Skipped existing outputs: 1", result.stdout)
            commands_path = os.path.join(output_root, "_array_meta", "commands.txt")
            outputs_path = os.path.join(output_root, "_array_meta", "outputs.txt")
            with open(commands_path) as f:
                commands = [line.strip() for line in f if line.strip()]
            with open(outputs_path) as f:
                outputs = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(commands), 1)
            self.assertEqual(len(outputs), 1)
            self.assertIn("--seed 2", commands[0])
            self.assertTrue(outputs[0].endswith("g0001_trainseed2"))

    def test_selected_rows_require_topology_id_before_status_or_submit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            edge_json = os.path.join(tmpdir, "edge.json")
            with open(edge_json, "w") as f:
                f.write("{}\n")
            library_csv = os.path.join(tmpdir, "selected.csv")
            with open(library_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["selected", "topology_id", "topology_name", "edge_json"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "selected": "1",
                        "topology_id": "",
                        "topology_name": "bad",
                        "edge_json": os.path.basename(edge_json),
                    }
                )

            result = self.run_submitter_unchecked(
                [
                    "--library_csv",
                    library_csv,
                    "--output_root",
                    os.path.join(tmpdir, "runs"),
                    "--seeds",
                    "1",
                    "--status_only",
                ],
                cwd=tmpdir,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing topology_id", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
