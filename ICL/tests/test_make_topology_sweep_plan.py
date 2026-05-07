import csv
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "make_topology_sweep_plan.py",
)


class MakeTopologySweepPlanTests(unittest.TestCase):
    def test_writes_regime_plan_and_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = os.path.join(tmpdir, "plan.csv")
            commands_sh = os.path.join(tmpdir, "commands.sh")
            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--output_csv",
                    output_csv,
                    "--commands_sh",
                    commands_sh,
                    "--output_root",
                    os.path.join(tmpdir, "libraries"),
                    "--train_root",
                    os.path.join(tmpdir, "sweeps"),
                    "--n_nodes",
                    "4,5",
                    "--edge_regimes",
                    "sparse,dense",
                    "--n_context",
                    "2",
                    "--z_dims",
                    "1,2",
                    "--input_coupled_count",
                    "0.5",
                    "--select_topologies",
                    "3",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(output_csv, newline="") as handle:
                rows = list(csv.DictReader(handle))
            with open(commands_sh) as handle:
                commands = handle.read()

        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["n_nodes"], "4")
        self.assertEqual(rows[0]["N"], "2")
        self.assertEqual(rows[0]["D"], "1")
        self.assertEqual(rows[0]["p"], "3")
        self.assertEqual(rows[0]["n_req"], "12")
        self.assertIn("make_topology_library.py", rows[0]["make_library_command"])
        self.assertIn("submit_topology_library_sweep.py", rows[0]["submit_command"])
        self.assertIn("--dry-run", rows[0]["submit_command"])
        self.assertIn("degree_balanced", rows[0]["make_library_command"])
        self.assertIn("make_topology_library.py", commands)
        self.assertIn("Wrote 8 topology sweep regimes", result.stdout)


if __name__ == "__main__":
    unittest.main()
