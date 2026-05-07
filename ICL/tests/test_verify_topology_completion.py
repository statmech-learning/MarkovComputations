import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "verify_topology_completion.py",
)


class VerifyTopologyCompletionTests(unittest.TestCase):
    def write_report(self, tmpdir, *, selected=1, joined=1, retrained=1, unknown=False):
        report_md = os.path.join(tmpdir, "report.md")
        report_json = os.path.join(tmpdir, "report.json")
        with open(report_md, "w") as f:
            f.write(
                "\n".join(
                    [
                        "# Input-Mask Topology-ICL Report",
                        "## Pooled Fixed-Input-Count Regressions",
                        "Common branch-rank source counts",
                        "Input-overlap source counts",
                        "## Essential Input-Mask Retraining",
                    ]
                )
            )
        source_key = "unknown" if unknown else "artifact"
        payload = {
            "target": "test_novel_classes",
            "experiments": [
                {
                    "name": "exp",
                    "run_summary": {"n": 2},
                    "aggregate_summary": {"n": 1},
                    "essential_inputmask50": {
                        "selected_summary": {"n_selected": selected},
                        "comparison": {"n_joined": joined},
                        "retrain_aggregate": {"n": retrained},
                    },
                }
            ],
            "pooled": {
                "run_summary": {"n": 2},
                "aggregate_summary": {"n": 1},
                "run_common_branch_source_counts": {source_key: 2},
                "aggregate_common_branch_source_counts": {source_key: 1},
                "run_input_overlap_source_counts": {source_key: 2},
                "aggregate_input_overlap_source_counts": {source_key: 1},
            },
        }
        with open(report_json, "w") as f:
            json.dump(payload, f)
        return report_md, report_json

    def write_research_report(self, tmpdir, *, joined=1, retrained=1):
        report_md = os.path.join(tmpdir, "research.md")
        report_json = os.path.join(tmpdir, "research.json")
        with open(report_md, "w") as f:
            f.write(
                "\n".join(
                    [
                        "# Topology-ICL Progress Report",
                        "## Pooled Fixed-Edge Regime Analysis",
                        "Common branch-rank source counts",
                        "Input-overlap source counts",
                        "## Essential Motif Retraining",
                    ]
                )
            )
        payload = {
            "target": "test_novel_classes",
            "experiments": [
                {
                    "name": "exp",
                    "run_summary": {"n_runs": 3},
                    "aggregate_summary": {"n_topology_groups": 1},
                    "essential_input50": {
                        "layouts": [
                            {
                                "label": "input mask",
                                "comparison": {"n_joined": joined},
                                "retrain_aggregate": {"n_topology_groups": retrained},
                            }
                        ]
                    },
                }
            ],
            "pooled": {
                "run_rows": 3,
                "aggregate_groups": 1,
                "retrain_groups": 1,
                "run_common_branch_source_counts": {"artifact": 3},
                "aggregate_common_branch_source_counts": {"artifact": 1},
                "run_input_overlap_source_counts": {"artifact": 3},
                "aggregate_input_overlap_source_counts": {"artifact": 1},
            },
        }
        with open(report_json, "w") as f:
            json.dump(payload, f)
        return report_md, report_json

    def run_verifier(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_valid_report_passes_with_audit_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_md, report_json = self.write_report(tmpdir)
            result = self.run_verifier(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--report_md",
                    report_md,
                    "--report_json",
                    report_json,
                    "--skip_audit",
                ]
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("Topology completion verification passed", result.stdout)

    def test_report_join_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_md, report_json = self.write_report(tmpdir, selected=2, joined=1, retrained=2)
            result = self.run_verifier(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--report_md",
                    report_md,
                    "--report_json",
                    report_json,
                    "--skip_audit",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("retrain comparison joined 1/2", result.stdout)

    def test_unknown_provenance_fails_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_md, report_json = self.write_report(tmpdir, unknown=True)
            result = self.run_verifier(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--report_md",
                    report_md,
                    "--report_json",
                    report_json,
                    "--skip_audit",
                ]
            )
            allowed = self.run_verifier(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--report_md",
                    report_md,
                    "--report_json",
                    report_json,
                    "--skip_audit",
                    "--allow_unknown_provenance",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown provenance", result.stdout)
        self.assertEqual(allowed.returncode, 0, allowed.stderr + allowed.stdout)

    def test_research_report_kind_checks_consolidated_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_md, report_json = self.write_research_report(tmpdir)
            result = self.run_verifier(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--report_md",
                    report_md,
                    "--report_json",
                    report_json,
                    "--report_kind",
                    "research",
                    "--skip_audit",
                ]
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_research_report_kind_fails_on_retrain_group_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_md, report_json = self.write_research_report(tmpdir, joined=2, retrained=1)
            result = self.run_verifier(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--report_md",
                    report_md,
                    "--report_json",
                    report_json,
                    "--report_kind",
                    "research",
                    "--skip_audit",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("retrain aggregate groups 1/2", result.stdout)


if __name__ == "__main__":
    unittest.main()
