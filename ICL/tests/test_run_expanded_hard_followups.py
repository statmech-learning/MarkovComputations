import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "run_expanded_hard_followups.py",
)


class RunExpandedHardFollowupsTests(unittest.TestCase):
    def run_script(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_status_reports_raw_and_summary_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run0")
            os.makedirs(run_dir)
            for filename in ("results.pkl", "mechanism_metrics.json", "causal_interventions.json"):
                with open(os.path.join(run_dir, filename), "w") as handle:
                    handle.write("{}")
            with open(os.path.join(tmpdir, "topology_results.csv"), "w") as handle:
                handle.write("run_id,test_novel_classes\nrun0,0.9\n")

            result = self.run_script(["--regime", f"hard={tmpdir}", "--status"])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("label,root,results_pkl,mechanisms,causal,topology_rows", result.stdout)
        self.assertIn(f"hard,{tmpdir},1,1,1,1", result.stdout)

    def test_submit_refuses_source_light_checkout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "topology_results.csv"), "w") as handle:
                handle.write("run_id,test_novel_classes\n")

            result = self.run_script(["--regime", f"hard={tmpdir}", "--submit_followups"])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to run follow-up finalization without raw results.pkl", result.stderr)

    def test_dry_run_prints_submit_collect_refresh_and_strict_verify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_json = os.path.join(tmpdir, "report.json")
            report_md = os.path.join(tmpdir, "report.md")

            result = self.run_script(
                [
                    "--regime",
                    f"hard={tmpdir}",
                    "--submit_followups",
                    "--collect_followups",
                    "--refresh_report",
                    "--strict_verify",
                    "--job_python",
                    "/env/bin/python",
                    "--report_json",
                    report_json,
                    "--report_md",
                    report_md,
                    "--dry-run",
                ]
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("finalize_topology_sweep.py", result.stdout)
        self.assertIn("--submit_mechanisms", result.stdout)
        self.assertIn("--submit_causal", result.stdout)
        self.assertIn("--collect_mechanisms", result.stdout)
        self.assertIn("--collect_causal", result.stdout)
        self.assertIn("--job_python /env/bin/python", result.stdout)
        self.assertIn("refresh_next_phase_report.py", result.stdout)
        self.assertIn("verify_topology_completion.py", result.stdout)
        self.assertIn("--require_expanded_followups", result.stdout)


if __name__ == "__main__":
    unittest.main()
