import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "recover_essential_physical_retrains.py",
)


class RecoverEssentialPhysicalRetrainsTests(unittest.TestCase):
    def touch(self, path, data=b""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    def write_csv(self, path, fieldnames, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_recovery(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_dry_run_prints_physical_audit_status_missing_and_finalize_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root_a = os.path.join(tmpdir, "random")
            root_b = os.path.join(tmpdir, "hub")
            result = self.run_recovery(
                [
                    "--experiment",
                    f"random={root_a}",
                    "--experiment",
                    f"hub={root_b}",
                    "--seeds",
                    "1,2",
                    "--submit_missing",
                    "--finalize_if_complete",
                    "--max-concurrent",
                    "7",
                    "--dry-run",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        stdout = result.stdout
        self.assertIn("audit_topology_artifacts.py", stdout)
        self.assertIn("--essential_directory essential_input50", stdout)
        self.assertIn("--retrain_directory essential_input50_retrain", stdout)
        self.assertIn("--essential_kind physical", stdout)
        self.assertIn("--require_source_results", stdout)
        self.assertIn("--require_mechanisms", stdout)
        self.assertIn("--require_essential", stdout)
        self.assertIn("--require_essential_retrains", stdout)
        self.assertIn("--strict", stdout)
        self.assertIn("submit_topology_library_sweep.py", stdout)
        self.assertIn("--status_only", stdout)
        self.assertIn("--missing_only", stdout)
        self.assertIn("--max-concurrent 7", stdout)
        self.assertIn("--dry-run", stdout)
        self.assertIn("finalize_essential_physical_retrains.py", stdout)
        self.assertNotIn("verify_topology_completion.py", stdout)
        self.assertNotIn("interpret_topology_report.py", stdout)
        self.assertNotIn("make_topology_research_report.py", stdout)
        self.assertIn("random=", stdout)
        self.assertIn("hub=", stdout)
        finalizer_pos = stdout.index("finalize_essential_physical_retrains.py")
        strict_pos = stdout.rindex("--require_essential_retrains")
        self.assertLess(finalizer_pos, strict_pos)

    def test_submit_missing_skips_array_when_manifest_has_no_missing_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "experiment")
            source_run = os.path.join(root, "source_run")
            self.touch(os.path.join(source_run, "results.pkl"), b"done")
            self.write_json(os.path.join(source_run, "topology_metrics.json"), {})
            self.write_json(os.path.join(source_run, "mechanism_metrics.json"), {})

            essential_dir = os.path.join(root, "essential_input50")
            edge_json = os.path.join(essential_dir, "edge.json")
            self.write_json(edge_json, {"n_nodes": 3, "edges": [[0, 1], [1, 2], [2, 0]]})
            rows = [
                {
                    "selected": "1",
                    "topology_id": "ess0",
                    "topology_name": "ess0",
                    "edge_json": edge_json,
                }
            ]
            self.write_csv(
                os.path.join(essential_dir, "library.csv"),
                ["selected", "topology_id", "topology_name", "edge_json"],
                rows,
            )
            self.write_csv(
                os.path.join(essential_dir, "selected.csv"),
                ["selected", "topology_id", "topology_name", "edge_json"],
                rows,
            )
            self.write_json(os.path.join(essential_dir, "summary.json"), {"n_selected_subgraphs": 1})

            retrain_root = os.path.join(root, "essential_input50_retrain")
            for seed in [1, 2]:
                self.touch(os.path.join(retrain_root, f"ess0_trainseed{seed}", "results.pkl"), b"done")

            result = self.run_recovery(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1,2",
                    "--submit_missing",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            manifest = os.path.join(retrain_root, "task_manifest.csv")
            with open(manifest, newline="") as f:
                manifest_rows = list(csv.DictReader(f))

        self.assertIn("Missing physical retrain tasks for", result.stdout)
        self.assertIn("0/2", result.stdout)
        self.assertIn("No missing physical retrain tasks", result.stdout)
        self.assertEqual(len(manifest_rows), 2)
        self.assertTrue(all(row["completed"] == "True" for row in manifest_rows))
        self.assertFalse(os.path.exists(os.path.join(retrain_root, "_array_meta")))


if __name__ == "__main__":
    unittest.main()
