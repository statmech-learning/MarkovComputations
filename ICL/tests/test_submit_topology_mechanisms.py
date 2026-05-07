import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "submit_topology_mechanisms.py",
)


class SubmitTopologyMechanismsTests(unittest.TestCase):
    def run_submitter(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def touch(self, path, data="{}\n"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(data)

    def test_dry_run_array_writes_mechanism_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "topology_seed1")
            self.touch(os.path.join(run_dir, "results.pkl"), "done\n")
            self.touch(os.path.join(run_dir, "config.json"))
            self.touch(os.path.join(run_dir, "topology.json"))

            result = self.run_submitter(
                [
                    "--input_root",
                    tmpdir,
                    "--n_samples",
                    "17",
                    "--device",
                    "cpu",
                    "--ablate_input",
                    "--ablate_physical",
                    "--physical_epsilon",
                    "0.001",
                    "--max-concurrent",
                    "4",
                    "--array",
                    "--dry-run",
                    "--clean",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            meta_dir = os.path.join(tmpdir, "_mechanism_array_meta")
            commands_path = os.path.join(meta_dir, "commands.txt")
            outputs_path = os.path.join(meta_dir, "outputs.txt")
            script_path = os.path.join(meta_dir, "run_mechanism_task.sh")
            with open(commands_path) as f:
                commands = [line.strip() for line in f if line.strip()]
            with open(outputs_path) as f:
                outputs = [line.strip() for line in f if line.strip()]
            with open(script_path) as f:
                script = f.read()

        self.assertIn("Tasks: 1", result.stdout)
        self.assertIn("Dry run. Submit with: sbatch", result.stdout)
        self.assertEqual(len(commands), 1)
        self.assertIn("analyze_topology_model.py", commands[0])
        self.assertIn("--n_samples 17", commands[0])
        self.assertIn("--device cpu", commands[0])
        self.assertIn("--ablate_input", commands[0])
        self.assertIn("--ablate_physical", commands[0])
        self.assertIn("--physical_epsilon 0.001", commands[0])
        self.assertEqual(outputs, [os.path.join(run_dir, "mechanism_metrics.json")])
        self.assertIn("#SBATCH --array=0-0%4", script)


if __name__ == "__main__":
    unittest.main()
