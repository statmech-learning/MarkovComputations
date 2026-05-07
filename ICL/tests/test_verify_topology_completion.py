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

    def write_research_report(self, tmpdir, *, joined=1, retrained=1, layout_sources=None):
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
        if layout_sources is None:
            layout_sources = ["essential_input50", "essential_inputmask50"]
        labels = {
            "essential_input50": "physical subgraph",
            "essential_inputmask50": "input mask",
        }
        layouts = [
            {
                "label": labels.get(source_dir, source_dir),
                "source_dir": source_dir,
                "comparison": {"n_joined": joined},
                "retrain_aggregate": {"n_topology_groups": retrained},
            }
            for source_dir in layout_sources
        ]
        payload = {
            "target": "test_novel_classes",
            "experiments": [
                {
                    "name": "exp",
                    "run_summary": {"n_runs": 3},
                    "aggregate_summary": {"n_topology_groups": 1},
                    "essential_input50": {
                        "layouts": layouts
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

    def write_next_phase_report(
        self,
        tmpdir,
        *,
        include_hard=True,
        include_causal=True,
        hard_family_col="derived_graph_family",
    ):
        report_md = os.path.join(tmpdir, "next_phase.md")
        report_json = os.path.join(tmpdir, "next_phase.json")
        with open(report_md, "w") as f:
            hard_headings = []
            if include_hard:
                hard_headings = [
                    "### hard_n4_m6_N3_D2",
                    "Rows: `60`. Groups: `12`. Families: `4` via `derived_graph_family`.",
                    "### hard_n5_m8_N3_D2",
                    "Rows: `60`. Groups: `12`. Families: `6` via `derived_graph_family`.",
                    "### hard_n5_m12_N3_D2",
                    "Rows: `60`. Groups: `12`. Families: `7` via `derived_graph_family`.",
                ]
            f.write(
                "\n".join(
                    [
                        "# Next-Phase Topology-ICL Evidence Report",
                        "## Clustered And Group-Aware Inference",
                        "derived_graph_family",
                        *hard_headings,
                        "## Causal Alignment Interventions",
                        "## Branch-Margin Capacity Probes",
                        "## Matched Essential-Motif Controls",
                        "## Expanded Pilot Status",
                        "## Interpretation Guardrails",
                    ]
                )
            )
        labels = ["pooled_original", "pooled_branch_capacity"]
        if include_hard:
            labels.extend(["hard_n4_m6_N3_D2", "hard_n5_m8_N3_D2", "hard_n5_m12_N3_D2"])
        clustered = [
            {
                "label": label,
                "n_run_rows": 60,
                "n_clusters": 12,
                "family_col": hard_family_col if label.startswith("hard_") else "physical_topology_name",
                "models": {
                    "raw_count": {"group_loo_r2": -0.1},
                    "tree_geometry": {"group_loo_r2": 0.1},
                },
            }
            for label in labels
        ]
        capacity = [
            {
                "label": label,
                "n_rows": 12,
                "families": [{"family": "random_sc", "n": 1, "linear_test_accuracy_mean": 0.8}],
            }
            for label in labels
            if label.startswith("hard_")
        ]
        causal = []
        if include_causal:
            causal = [
                {
                    "label": "random",
                    "n_runs": 80,
                    "interventions": [{"intervention": "scramble", "n": 10, "target_accuracy_delta_mean": -50.0}],
                }
            ]
        matched = [
            {
                "label": "random",
                "n_joined": 32,
                "by_control_kind": {"random_sc": {"n": 16}},
            }
        ]
        expanded = [
            {
                "label": label,
                "results_pkl_count": 60,
                "mechanism_count": 0,
                "causal_count": 0,
            }
            for label in labels
            if label.startswith("hard_")
        ]
        payload = {
            "generated_at": "2026-05-07T00:00:00+00:00",
            "scope": "first-order CRNs with exponential input-dependent rates",
            "clustered_inference": clustered,
            "branch_margin_capacity": capacity,
            "causal_interventions": causal,
            "matched_motif_controls": matched,
            "expanded_pilot_status": expanded,
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

    def test_next_phase_report_kind_checks_current_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_md, report_json = self.write_next_phase_report(tmpdir)
            result = self.run_verifier(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--report_md",
                    report_md,
                    "--report_json",
                    report_json,
                    "--report_kind",
                    "next_phase",
                ]
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_next_phase_report_kind_fails_when_hard_regimes_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_md, report_json = self.write_next_phase_report(tmpdir, include_hard=False)
            result = self.run_verifier(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--report_md",
                    report_md,
                    "--report_json",
                    report_json,
                    "--report_kind",
                    "next_phase",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing hard_n4_m6_N3_D2", result.stdout)

    def test_next_phase_report_kind_fails_without_causal_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_md, report_json = self.write_next_phase_report(tmpdir, include_causal=False)
            result = self.run_verifier(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--report_md",
                    report_md,
                    "--report_json",
                    report_json,
                    "--report_kind",
                    "next_phase",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no causal intervention entries", result.stdout)

    def test_next_phase_report_kind_requires_derived_family_for_hard_regimes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_md, report_json = self.write_next_phase_report(
                tmpdir,
                hard_family_col="physical_topology_name",
            )
            result = self.run_verifier(
                [
                    "--experiment",
                    f"exp={tmpdir}",
                    "--report_md",
                    report_md,
                    "--report_json",
                    report_json,
                    "--report_kind",
                    "next_phase",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("heldout must use derived_graph_family", result.stdout)

    def test_research_report_kind_requires_both_essential_layouts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_md, report_json = self.write_research_report(
                tmpdir,
                layout_sources=["essential_inputmask50"],
            )
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
        self.assertIn("missing physical essential_input50", result.stdout)


if __name__ == "__main__":
    unittest.main()
