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
            for filename in ("results.pkl", "config.json", "topology.json", "model.pt"):
                with open(os.path.join(run_dir, filename), "w") as handle:
                    handle.write("{}")
            with open(os.path.join(run_dir, "causal_interventions.json"), "w") as handle:
                handle.write('{"baseline": {}, "interventions": []}')

            result = self.run_finalizer(
                [
                    "--input_root",
                    tmpdir,
                    "--submit_mechanisms",
                    "--submit_causal",
                    "--collect_causal",
                    "--job_python",
                    "/env/bin/python",
                    "--device",
                    "cpu",
                    "--causal_n_samples",
                    "21",
                    "--causal_n_repeats",
                    "2",
                    "--causal_interventions",
                    "context_block_shuffle",
                    "--skip_torch_check",
                    "--dry-run",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            mechanism_commands_path = os.path.join(tmpdir, "_mechanism_array_meta", "commands.txt")
            mechanism_script_path = os.path.join(tmpdir, "_mechanism_array_meta", "run_mechanism_task.sh")
            with open(mechanism_commands_path) as handle:
                mechanism_commands = handle.read()
            with open(mechanism_script_path) as handle:
                mechanism_script = handle.read()

            causal_commands_path = os.path.join(tmpdir, "_causal_array_meta", "commands.txt")
            causal_script_path = os.path.join(tmpdir, "_causal_array_meta", "run_causal_task.sh")
            with open(causal_commands_path) as handle:
                causal_commands = handle.read()
            with open(causal_script_path) as handle:
                causal_script = handle.read()

        self.assertIn("submit_topology_mechanisms.py", result.stdout)
        self.assertIn("submit_causal_interventions.py", result.stdout)
        self.assertIn("collect_causal_interventions.py", result.stdout)
        self.assertIn("--python /env/bin/python", result.stdout)
        self.assertIn("--skip_torch_check", result.stdout)
        self.assertIn("/env/bin/python -u analyze_topology_model.py", mechanism_commands)
        self.assertIn("--device cpu", mechanism_commands)
        self.assertNotIn("import torch", mechanism_script)
        self.assertIn("/env/bin/python -u causal_topology_interventions.py", causal_commands)
        self.assertIn("--n_samples 21", causal_commands)
        self.assertIn("--n_repeats 2", causal_commands)
        self.assertIn("--interventions context_block_shuffle", causal_commands)
        self.assertNotIn("import torch", causal_script)


if __name__ == "__main__":
    unittest.main()
