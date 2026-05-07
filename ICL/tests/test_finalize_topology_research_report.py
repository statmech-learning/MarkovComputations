import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "finalize_topology_research_report.py",
)


class FinalizeTopologyResearchReportTests(unittest.TestCase):
    def run_finalizer(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_dry_run_prints_report_verifier_and_interpretation_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root_a = os.path.join(tmpdir, "random")
            root_b = os.path.join(tmpdir, "cycle")
            output_md = os.path.join(tmpdir, "topology_report.md")
            output_json = os.path.join(tmpdir, "topology_report.json")
            result = self.run_finalizer(
                [
                    "--experiment",
                    f"random={root_a}",
                    "--experiment",
                    f"cycle={root_b}",
                    "--seeds",
                    "1,2",
                    "--target",
                    "test_novel_classes",
                    "--output_md",
                    output_md,
                    "--output_json",
                    output_json,
                    "--interpret_min_n",
                    "8",
                    "--interpret_delta",
                    "0.1",
                    "--dry-run",
                ]
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout
        self.assertIn("make_topology_research_report.py", stdout)
        self.assertIn("verify_topology_completion.py", stdout)
        self.assertIn("--report_kind research", stdout)
        self.assertIn("--seeds 1,2", stdout)
        self.assertIn("interpret_topology_report.py", stdout)
        self.assertIn("--report_kind research", stdout)
        self.assertIn("--min_n 8", stdout)
        self.assertIn("--delta 0.1", stdout)
        self.assertIn("topology_report_interpretation.md", stdout)
        self.assertIn("topology_report_interpretation.json", stdout)
        self.assertIn("random=", stdout)
        self.assertIn("cycle=", stdout)
        report_pos = stdout.index("make_topology_research_report.py")
        verifier_pos = stdout.index("verify_topology_completion.py")
        interpreter_pos = stdout.index("interpret_topology_report.py")
        self.assertLess(report_pos, verifier_pos)
        self.assertLess(verifier_pos, interpreter_pos)

    def test_skip_interpretation_omits_interpreter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_finalizer(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--output_md",
                    os.path.join(tmpdir, "report.md"),
                    "--output_json",
                    os.path.join(tmpdir, "report.json"),
                    "--skip_interpretation",
                    "--dry-run",
                ]
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("make_topology_research_report.py", result.stdout)
        self.assertIn("verify_topology_completion.py", result.stdout)
        self.assertNotIn("interpret_topology_report.py", result.stdout)

    def test_allow_unknown_provenance_is_forwarded_to_verifier_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.run_finalizer(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--output_md",
                    os.path.join(tmpdir, "report.md"),
                    "--output_json",
                    os.path.join(tmpdir, "report.json"),
                    "--allow_unknown_provenance",
                    "--dry-run",
                ]
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        verifier_line = next(
            line
            for line in result.stdout.splitlines()
            if "verify_topology_completion.py" in line
        )
        interpreter_line = next(
            line
            for line in result.stdout.splitlines()
            if "interpret_topology_report.py" in line
        )
        self.assertIn("--allow_unknown_provenance", verifier_line)
        self.assertNotIn("--allow_unknown_provenance", interpreter_line)


if __name__ == "__main__":
    unittest.main()
