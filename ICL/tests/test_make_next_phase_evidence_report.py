import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "make_next_phase_evidence_report.py",
)


class MakeNextPhaseEvidenceReportTests(unittest.TestCase):
    def write_json(self, path, payload):
        with open(path, "w") as handle:
            json.dump(payload, handle)

    def test_builds_markdown_from_next_phase_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clustered_json = os.path.join(tmpdir, "clustered.json")
            causal_json = os.path.join(tmpdir, "causal.json")
            capacity_json = os.path.join(tmpdir, "capacity.json")
            matched_json = os.path.join(tmpdir, "matched.json")
            expanded_root = os.path.join(tmpdir, "expanded")
            os.makedirs(os.path.join(expanded_root, "run0"))
            with open(os.path.join(expanded_root, "run0", "results.pkl"), "w") as handle:
                handle.write("placeholder")
            self.write_json(
                clustered_json,
                {
                    "n_run_rows": 12,
                    "n_clusters": 4,
                    "n_families": 2,
                    "family_col": "derived_graph_family",
                    "group_level": {
                        "target_mean": {
                            "raw_count": {"n": 4, "leave_one_out_r2": -0.1},
                            "branch_margin_capacity": {"n": 4, "leave_one_out_r2": 0.4},
                        }
                    },
                    "cluster_bootstrap_run_level": {
                        "branch_margin_capacity": {
                            "delta_mean": 0.2,
                            "delta_ci95": [0.1, 0.3],
                            "prob_delta_positive": 1.0,
                        }
                    },
                    "family_cluster_bootstrap_run_level": {
                        "branch_margin_capacity": {
                            "delta_mean": 0.15,
                            "delta_ci95": [0.05, 0.25],
                            "prob_delta_positive": 1.0,
                        }
                    },
                    "leave_family_out_group_target_mean": {
                        "branch_margin_capacity": {"pooled_r2": 0.25, "pooled_rmse": 3.0}
                    },
                },
            )
            self.write_json(
                causal_json,
                {
                    "n_rows": 6,
                    "n_runs": 2,
                    "interventions": {
                        "edge_projection_permutation": {
                            "n": 6,
                            "target_accuracy_delta_mean": -40.0,
                            "target_accuracy_delta_min": -60.0,
                            "target_accuracy_delta_max": -10.0,
                        }
                    },
                },
            )
            self.write_json(
                capacity_json,
                {
                    "n_rows": 3,
                    "families": {
                        "random": {
                            "n": 3,
                            "linear_test_accuracy_mean": 0.8,
                            "linear_test_accuracy_max": 0.9,
                            "rooted_polytope_supported_branch_dim_fraction_mean": 1.0,
                            "normal_fan_branch_tree_nmi_mean": 0.2,
                        }
                    },
                },
            )
            self.write_json(
                matched_json,
                {
                    "n_joined": 4,
                    "overall": {"n_sources": 2},
                    "by_control_kind": {
                        "random_sc": {
                            "n": 2,
                            "n_sources": 2,
                            "control_target_mean_mean": 60.0,
                            "source_retrain_target_mean_mean": 70.0,
                            "control_minus_source_retrain_mean_mean": -10.0,
                            "control_win_rate_mean": 0.0,
                            "match_score_mean": 0.2,
                        }
                    },
                },
            )
            output_md = os.path.join(tmpdir, "report.md")
            output_json = os.path.join(tmpdir, "report.json")

            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--clustered_json",
                    f"pooled={clustered_json}",
                    "--causal_json",
                    f"random={causal_json}",
                    "--branch_capacity_json",
                    f"random={capacity_json}",
                    "--matched_motif_json",
                    f"random={matched_json}",
                    "--expanded_root",
                    f"pilot={expanded_root}",
                    "--output_md",
                    output_md,
                    "--output_json",
                    output_json,
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            with open(output_md) as handle:
                markdown = handle.read()
            with open(output_json) as handle:
                payload = json.load(handle)

        self.assertIn("Next-Phase Topology-ICL Evidence Report", markdown)
        self.assertIn("Families: `2` via `derived_graph_family`", markdown)
        self.assertIn("family boot delta R2", markdown)
        self.assertIn("branch_margin_capacity", markdown)
        self.assertIn("rooted support frac", markdown)
        self.assertIn("normal fan tree NMI", markdown)
        self.assertIn("edge_projection_permutation", markdown)
        self.assertIn("Matched Essential-Motif Controls", markdown)
        self.assertIn("Extracted motifs beat these matched controls", markdown)
        self.assertIn("random_sc", markdown)
        self.assertIn("results.pkl", markdown)
        self.assertEqual(payload["clustered_inference"][0]["n_clusters"], 4)
        self.assertEqual(payload["clustered_inference"][0]["family_col"], "derived_graph_family")
        self.assertEqual(payload["matched_motif_controls"][0]["n_joined"], 4)
        self.assertIn("Wrote next-phase evidence report", result.stdout)


if __name__ == "__main__":
    unittest.main()
