import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "finalize_topology_sweep.py",
)


class FinalizeTopologySweepTests(unittest.TestCase):
    def run_finalizer(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_dry_run_prints_collect_regress_and_aggregate_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_finalizer(
                [
                    "--input_root",
                    tmpdir,
                    "--dry-run",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        self.assertIn("collect_topology_results.py", result.stdout)
        self.assertIn("regress_topology_results.py", result.stdout)
        self.assertIn("aggregate_topology_seeds.py", result.stdout)
        self.assertIn("topology_results.csv", result.stdout)
        self.assertIn("topology_regression.json", result.stdout)
        self.assertIn("topology_seed_aggregates.csv", result.stdout)
        self.assertNotIn("submit_topology_mechanisms.py", result.stdout)

    def test_submit_mechanisms_dry_run_passes_job_python(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_000")
            os.makedirs(run_dir)
            for filename in ("results.pkl", "config.json", "topology.json"):
                with open(os.path.join(run_dir, filename), "w") as handle:
                    handle.write("{}")

            result = self.run_finalizer(
                [
                    "--input_root",
                    tmpdir,
                    "--submit_mechanisms",
                    "--job_python",
                    "/env/bin/python",
                    "--skip_torch_check",
                    "--dry-run",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            commands_path = os.path.join(tmpdir, "_mechanism_array_meta", "commands.txt")
            script_path = os.path.join(tmpdir, "_mechanism_array_meta", "run_mechanism_task.sh")
            with open(commands_path) as handle:
                commands = handle.read()
            with open(script_path) as handle:
                script = handle.read()

        self.assertIn("submit_topology_mechanisms.py", result.stdout)
        self.assertIn("--python /env/bin/python", result.stdout)
        self.assertIn("--skip_torch_check", result.stdout)
        self.assertIn("/env/bin/python -u analyze_topology_model.py", commands)
        self.assertNotIn("import torch", script)


if __name__ == "__main__":
    unittest.main()
