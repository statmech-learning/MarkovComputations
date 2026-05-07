import csv
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "submit_topology_phase1.py",
)


class SubmitTopologyPhase1Tests(unittest.TestCase):
    def run_submitter(self, args, env):
        merged_env = os.environ.copy()
        merged_env.update(env)
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
            env=merged_env,
        )

    def test_smoke_dry_run_writes_index_and_array_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_submitter(
                [
                    "--phase",
                    "smoke",
                    "--seeds",
                    "1,2,3",
                    "--array",
                    "--dry-run",
                    "--max-concurrent",
                    "5",
                    "--clean",
                ],
                env={"SLURM_OUTPUT_BASE": tmpdir},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            index_csv = os.path.join(tmpdir, "_array_meta", "index.csv")
            script_path = os.path.join(tmpdir, "_array_meta", "run_task.sh")
            with open(index_csv, newline="") as f:
                rows = list(csv.DictReader(f))
            with open(script_path) as f:
                script = f.read()

        self.assertIn("Tasks: 6", result.stdout)
        self.assertIn("Dry run. Submit with: sbatch", result.stdout)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["seed"] for row in rows}, {"1", "2"})
        families = {row["topology_family"] for row in rows}
        self.assertEqual(families, {"complete", "cycle_chords", "random_sc"})

        complete_rows = [row for row in rows if row["topology_family"] == "complete"]
        sparse_rows = [row for row in rows if row["topology_family"] != "complete"]
        self.assertTrue(all(row["n_edges"] == "" for row in complete_rows))
        self.assertTrue(all(row["n_edges"] == "8" for row in sparse_rows))
        self.assertTrue(all("--no_progress" in row["command"] for row in rows))
        self.assertTrue(all("run_topology_icl.py" in row["command"] for row in rows))
        self.assertTrue(all("--n_edges" not in row["command"] for row in complete_rows))
        self.assertTrue(all("--n_edges 8" in row["command"] for row in sparse_rows))
        self.assertIn("#SBATCH --array=0-5%5", script)


if __name__ == "__main__":
    unittest.main()
