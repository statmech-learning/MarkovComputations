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


if __name__ == "__main__":
    unittest.main()
