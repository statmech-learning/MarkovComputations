import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "interpret_topology_report.py",
)


def fit(n, loo, r2=None):
    return {
        "n": n,
        "leave_one_out_r2": loo,
        "r2": loo if r2 is None else r2,
        "rmse": 1.0,
    }


class InterpretTopologyReportTests(unittest.TestCase):
    def run_interpreter(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_input_mask_report_flags_structural_and_mechanism_support(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_json = os.path.join(tmpdir, "report.json")
            output_json = os.path.join(tmpdir, "interpretation.json")
            output_md = os.path.join(tmpdir, "interpretation.md")
            payload = {
                "target": "test_novel_classes",
                "experiments": [
                    {
                        "name": "toy",
                        "essential_inputmask50": {
                            "comparison": {
                                "n_joined": 1,
                                "retention_mean_mean": 0.8,
                                "retention_max_mean": 0.9,
                                "retrain_mean_mean": 70.0,
                                "retrain_max_best": 80.0,
                            }
                        },
                    }
                ],
                "pooled": {
                    "run_summary": {
                        "n": 8,
                        "n_edges_values": [20],
                        "input_coupled_parameter_count_values": [200],
                    },
                    "aggregate_summary": {"n": 4},
                    "run_regressions": {
                        "raw_counts": fit(8, 0.1),
                        "masked_geometry": fit(8, 0.3),
                        "mechanism": fit(8, 0.45),
                    },
                    "aggregate_target_mean": {
                        "raw_counts": fit(8, 0.05),
                        "physical_backbone": fit(8, 0.25),
                        "masked_geometry": fit(8, 0.35),
                        "mechanism": fit(8, 0.5),
                    },
                    "aggregate_target_max": {},
                    "aggregate_target_std": {},
                },
            }
            with open(report_json, "w") as f:
                json.dump(payload, f)

            result = self.run_interpreter(
                [
                    "--report_json",
                    report_json,
                    "--report_kind",
                    "input_mask",
                    "--output_json",
                    output_json,
                    "--output_md",
                    output_md,
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(output_json) as f:
                interpretation = json.load(f)
            with open(output_md) as f:
                markdown = f.read()

        self.assertEqual(interpretation["support_summary"]["verdict"], "strong_positive")
        self.assertTrue(interpretation["count_control"]["fixed_n_edges"])
        self.assertTrue(interpretation["count_control"]["fixed_input_coupled_parameter_count"])
        self.assertIn("Topology-ICL Interpretation", markdown)
        self.assertIn("masked_geometry", markdown)

    def test_research_report_without_supported_deltas_is_weak(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_json = os.path.join(tmpdir, "research.json")
            output_json = os.path.join(tmpdir, "interpretation.json")
            payload = {
                "target": "test_novel_classes",
                "experiments": [
                    {
                        "name": "toy",
                        "essential_input50": {
                            "layouts": [
                                {
                                    "label": "physical subgraph",
                                    "comparison": {
                                        "n_joined": 1,
                                        "retention_mean_mean": 0.4,
                                    },
                                }
                            ]
                        },
                    }
                ],
                "pooled": {
                    "run_rows": 10,
                    "aggregate_groups": 5,
                    "retrain_groups": 1,
                    "run_level": {
                        "edge_count": fit(10, 0.4),
                        "edge_plus_drel": fit(10, 0.39),
                        "edge_plus_mechanism": fit(10, 0.38),
                    },
                    "aggregate_target_mean": {
                        "edge_count": fit(10, 0.2),
                        "input_plus_masked_geometry": fit(10, 0.21),
                    },
                    "aggregate_target_max": {},
                    "retrain_target_mean": {},
                    "retrain_target_max": {},
                },
            }
            with open(report_json, "w") as f:
                json.dump(payload, f)

            result = self.run_interpreter(
                [
                    "--report_json",
                    report_json,
                    "--report_kind",
                    "research",
                    "--output_json",
                    output_json,
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(output_json) as f:
                interpretation = json.load(f)

        self.assertEqual(interpretation["support_summary"]["verdict"], "weak_or_negative")
        self.assertEqual(interpretation["report_kind"], "research")


if __name__ == "__main__":
    unittest.main()
