import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "submit_causal_interventions.py",
)


class SubmitCausalInterventionsTests(unittest.TestCase):
    def touch(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write("")

    def test_writes_dry_run_array_for_completed_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_a")
            for name in ["results.pkl", "config.json", "topology.json", "model.pt"]:
                self.touch(os.path.join(run_dir, name))

            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--input_root",
                    tmpdir,
                    "--n_samples",
                    "12",
                    "--n_repeats",
                    "2",
                    "--interventions",
                    "context_block_shuffle",
                    "--python",
                    "/env/bin/python",
                    "--array",
                    "--dry-run",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            meta_dir = os.path.join(tmpdir, "_causal_array_meta")
            with open(os.path.join(meta_dir, "commands.txt")) as handle:
                commands = handle.read()
            with open(os.path.join(meta_dir, "run_causal_task.sh")) as handle:
                script = handle.read()

        self.assertIn("Tasks: 1", result.stdout)
        self.assertIn("Dry run. Submit with: sbatch", result.stdout)
        self.assertIn("causal_topology_interventions.py", commands)
        self.assertIn("/env/bin/python -u causal_topology_interventions.py", commands)
        self.assertIn("--n_samples 12", commands)
        self.assertIn("--n_repeats 2", commands)
        self.assertIn("--interventions context_block_shuffle", commands)
        self.assertIn("#SBATCH --job-name=topo_causal", script)
        self.assertIn("/env/bin/python - <<'PY'", script)
        self.assertIn("import torch", script)


if __name__ == "__main__":
    unittest.main()
