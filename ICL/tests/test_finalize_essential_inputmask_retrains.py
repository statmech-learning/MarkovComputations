import csv
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "finalize_essential_inputmask_retrains.py",
)


class FinalizeEssentialInputMaskRetrainsTests(unittest.TestCase):
    def make_experiment(self, tmpdir, selected_count=2, completed_count=0):
        root = os.path.join(tmpdir, "experiment")
        selected_dir = os.path.join(root, "essential_inputmask50")
        retrain_dir = os.path.join(root, "essential_inputmask50_retrain")
        os.makedirs(selected_dir)
        os.makedirs(retrain_dir)

        selected_csv = os.path.join(selected_dir, "selected.csv")
        with open(selected_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["selected", "topology_id", "topology_name"])
            writer.writeheader()
            for idx in range(selected_count):
                writer.writerow(
                    {
                        "selected": "1",
                        "topology_id": f"mask{idx}",
                        "topology_name": f"mask{idx}",
                    }
                )
            writer.writerow(
                {
                    "selected": "0",
                    "topology_id": "unselected",
                    "topology_name": "unselected",
                }
            )

        for idx in range(completed_count):
            topology_idx = idx // 2
            seed = (idx % 2) + 1
            if topology_idx >= selected_count:
                break
            result_path = os.path.join(
                retrain_dir,
                f"mask{topology_idx}_trainseed{seed}",
                "results.pkl",
            )
            os.makedirs(os.path.dirname(result_path))
            with open(result_path, "wb") as f:
                f.write(b"done")
            for filename in ["topology_metrics.json", "config.json"]:
                with open(os.path.join(os.path.dirname(result_path), filename), "w") as f:
                    f.write("{}\n")
        return root

    def run_finalizer(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_incomplete_retrains_fail_without_allow_partial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=2, completed_count=3)
            result = self.run_finalizer(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1,2",
                    "--dry-run",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected=4 completed=3", result.stdout)
        self.assertIn("Retrain outputs are incomplete", result.stderr + result.stdout)

    def test_allow_partial_runs_dry_run_collection_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=2, completed_count=3)
            result = self.run_finalizer(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1,2",
                    "--allow_partial",
                    "--dry-run",
                ]
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("expected=4 completed=3", result.stdout)
        self.assertIn("finalize_topology_sweep.py", result.stdout)
        self.assertIn("compare_essential_retrains.py", result.stdout)

    def test_complete_retrains_refresh_report_in_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=2, completed_count=4)
            result = self.run_finalizer(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1,2",
                    "--dry-run",
                    "--output_md",
                    os.path.join(tmpdir, "report.md"),
                    "--output_json",
                    os.path.join(tmpdir, "report.json"),
                ]
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("expected=4 completed=4", result.stdout)
        self.assertIn("make_input_mask_report.py", result.stdout)

    def test_wrong_result_directories_do_not_satisfy_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=1, completed_count=0)
            wrong_dir = os.path.join(root, "essential_inputmask50_retrain", "wrong_label")
            os.makedirs(wrong_dir)
            with open(os.path.join(wrong_dir, "results.pkl"), "wb") as f:
                f.write(b"done")
            result = self.run_finalizer(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1",
                    "--dry-run",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected=1 completed=0 total_results=1", result.stdout)
        self.assertIn("mask0_trainseed1", result.stdout)

    def test_extra_result_directories_fail_without_allow_extra(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=1, completed_count=1)
            extra_dir = os.path.join(root, "essential_inputmask50_retrain", "stale_run")
            os.makedirs(extra_dir)
            with open(os.path.join(extra_dir, "results.pkl"), "wb") as f:
                f.write(b"done")
            result = self.run_finalizer(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1",
                    "--dry-run",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected=1 completed=1 total_results=2", result.stdout)
        self.assertIn("Unexpected retrain outputs", result.stdout)
        self.assertIn("stale_run", result.stdout)

    def test_allow_extra_permits_diagnostic_finalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=1, completed_count=1)
            extra_dir = os.path.join(root, "essential_inputmask50_retrain", "stale_run")
            os.makedirs(extra_dir)
            with open(os.path.join(extra_dir, "results.pkl"), "wb") as f:
                f.write(b"done")
            result = self.run_finalizer(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1",
                    "--dry-run",
                    "--allow_extra",
                ]
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("finalize_topology_sweep.py", result.stdout)

    def test_selected_rows_require_topology_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "experiment")
            selected_dir = os.path.join(root, "essential_inputmask50")
            retrain_dir = os.path.join(root, "essential_inputmask50_retrain")
            os.makedirs(selected_dir)
            os.makedirs(retrain_dir)
            with open(os.path.join(selected_dir, "selected.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["selected", "topology_name"])
                writer.writeheader()
                writer.writerow({"selected": "1", "topology_name": "mask0"})
            result = self.run_finalizer(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1",
                    "--dry-run",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing topology_id", result.stderr + result.stdout)

    def test_complete_results_without_sidecars_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=1, completed_count=1)
            os.remove(
                os.path.join(
                    root,
                    "essential_inputmask50_retrain",
                    "mask0_trainseed1",
                    "config.json",
                )
            )
            result = self.run_finalizer(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1",
                    "--dry-run",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required run-file", result.stdout)
        self.assertIn("Retrain run sidecars are incomplete", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
