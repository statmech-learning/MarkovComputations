import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "refresh_next_phase_report.py",
)


class RefreshNextPhaseReportTests(unittest.TestCase):
    def write_json(self, path, payload):
        with open(path, "w") as handle:
            json.dump(payload, handle)

    def test_refresh_replaces_labeled_sections_and_rerenders_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_json = os.path.join(tmpdir, "report.json")
            clustered_json = os.path.join(tmpdir, "clustered.json")
            capacity_json = os.path.join(tmpdir, "capacity.json")
            expanded_root = os.path.join(tmpdir, "expanded")
            run_dir = os.path.join(expanded_root, "run0")
            os.makedirs(run_dir)
            for name in ("results.pkl", "mechanism_metrics.json", "causal_interventions.json"):
                with open(os.path.join(run_dir, name), "w") as handle:
                    handle.write("{}")

            self.write_json(
                report_json,
                {
                    "generated_at": "old",
                    "scope": "First-order CRNs with exponential input-dependent rates",
                    "clustered_inference": [
                        {
                            "label": "hard",
                            "path": "old",
                            "n_run_rows": 1,
                            "n_clusters": 1,
                            "n_families": 1,
                            "models": {"raw_count": {}},
                        }
                    ],
                    "causal_interventions": [],
                    "branch_margin_capacity": [],
                    "matched_motif_controls": [],
                    "expanded_pilot_status": [{"label": "hard", "results_pkl_count": 0}],
                },
            )
            self.write_json(
                clustered_json,
                {
                    "n_run_rows": 10,
                    "n_clusters": 2,
                    "n_families": 1,
                    "family_col": "derived_graph_family",
                    "group_level": {
                        "target_mean": {
                            "raw_count": {"n": 2, "leave_one_out_r2": -0.1},
                            "tree_geometry": {"n": 2, "leave_one_out_r2": 0.2},
                        }
                    },
                    "cluster_bootstrap_run_level": {
                        "tree_geometry": {"delta_mean": 0.3, "delta_ci95": [0.1, 0.5], "prob_delta_positive": 1.0}
                    },
                    "family_cluster_bootstrap_run_level": {
                        "tree_geometry": {"delta_mean": 0.25, "prob_delta_positive": 1.0}
                    },
                    "leave_family_out_group_target_mean": {
                        "tree_geometry": {"pooled_r2": 0.1}
                    },
                },
            )
            self.write_json(
                capacity_json,
                {
                    "n_rows": 2,
                    "families": {
                        "random_sc": {
                            "n": 2,
                            "linear_test_accuracy_mean": 0.8,
                            "tropical_linear_test_accuracy_mean": 0.4,
                        }
                    },
                },
            )
            output_json = os.path.join(tmpdir, "out.json")
            output_md = os.path.join(tmpdir, "out.md")

            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--report_json",
                    report_json,
                    "--clustered_json",
                    f"hard={clustered_json}",
                    "--branch_capacity_json",
                    f"hard={capacity_json}",
                    "--expanded_root",
                    f"hard={expanded_root}",
                    "--output_json",
                    output_json,
                    "--output_md",
                    output_md,
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            with open(output_json) as handle:
                payload = json.load(handle)
            with open(output_md) as handle:
                markdown = handle.read()

        self.assertNotEqual(payload["generated_at"], "old")
        self.assertEqual(payload["clustered_inference"][0]["n_run_rows"], 10)
        self.assertEqual(payload["branch_margin_capacity"][0]["n_rows"], 2)
        self.assertEqual(payload["expanded_pilot_status"][0]["mechanism_count"], 1)
        self.assertIn("family boot delta R2", markdown)
        self.assertIn("derived_graph_family", markdown)
        self.assertIn("Refreshed next-phase report JSON", result.stdout)

    def test_refresh_preserves_existing_expanded_status_when_source_has_no_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_json = os.path.join(tmpdir, "report.json")
            empty_expanded_root = os.path.join(tmpdir, "expanded")
            os.makedirs(empty_expanded_root)
            self.write_json(
                report_json,
                {
                    "generated_at": "old",
                    "scope": "First-order CRNs with exponential input-dependent rates",
                    "clustered_inference": [],
                    "causal_interventions": [],
                    "branch_margin_capacity": [],
                    "matched_motif_controls": [],
                    "expanded_pilot_status": [
                        {
                            "label": "hard",
                            "root": "results/expanded_hard_sweeps/hard",
                            "results_pkl_count": 60,
                            "mechanism_count": 0,
                            "causal_count": 0,
                        }
                    ],
                },
            )
            output_json = os.path.join(tmpdir, "out.json")
            output_md = os.path.join(tmpdir, "out.md")

            subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--report_json",
                    report_json,
                    "--expanded_root",
                    f"hard={empty_expanded_root}",
                    "--output_json",
                    output_json,
                    "--output_md",
                    output_md,
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            with open(output_json) as handle:
                payload = json.load(handle)

        self.assertEqual(payload["expanded_pilot_status"][0]["results_pkl_count"], 60)
        self.assertEqual(payload["expanded_pilot_status"][0]["root"], "results/expanded_hard_sweeps/hard")


if __name__ == "__main__":
    unittest.main()
