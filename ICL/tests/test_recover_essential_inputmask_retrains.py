import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "recover_essential_inputmask_retrains.py",
)


class RecoverEssentialInputMaskRetrainsTests(unittest.TestCase):
    def run_recovery(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_dry_run_prints_audit_status_missing_and_finalize_commands(self):
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
                    "--output_md",
                    os.path.join(tmpdir, "report.md"),
                    "--output_json",
                    os.path.join(tmpdir, "report.json"),
                    "--max-concurrent",
                    "7",
                    "--dry-run",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)

        stdout = result.stdout
        self.assertIn("audit_topology_artifacts.py", stdout)
        self.assertIn("--require_source_results", stdout)
        self.assertIn("--require_mechanisms", stdout)
        self.assertIn("--require_essential_inputmask", stdout)
        self.assertIn("--require_essential_retrains", stdout)
        self.assertIn("--strict", stdout)
        self.assertIn("submit_topology_library_sweep.py", stdout)
        self.assertIn("--status_only", stdout)
        self.assertIn("--missing_only", stdout)
        self.assertIn("--max-concurrent 7", stdout)
        self.assertIn("--dry-run", stdout)
        self.assertIn("finalize_essential_inputmask_retrains.py", stdout)
        self.assertIn("random=", stdout)
        self.assertIn("hub=", stdout)

    def test_finalize_requires_both_report_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_recovery(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--finalize_if_complete",
                    "--dry-run",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Provide both --output_md and --output_json", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
