import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "audit_topology_artifacts.py",
)


class AuditTopologyArtifactsTests(unittest.TestCase):
    def write_csv(self, path, fieldnames, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def touch(self, path, data=b""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def make_experiment(
        self,
        tmpdir,
        selected_count=2,
        retrain_count=0,
        missing_reference=False,
        finalized=False,
    ):
        root = os.path.join(tmpdir, "exp")
        run_dir = os.path.join(root, "source_run")
        self.touch(os.path.join(run_dir, "results.pkl"), b"done")
        self.touch(os.path.join(run_dir, "topology_metrics.json"), b"{}")
        self.touch(os.path.join(run_dir, "mechanism_metrics.json"), b"{}")
        self.touch(os.path.join(root, "topology_seed_aggregates.csv"), b"group\n")

        essential_dir = os.path.join(root, "essential_inputmask50")
        ref_dir = os.path.join(essential_dir, "refs")
        os.makedirs(ref_dir, exist_ok=True)
        rows = []
        for idx in range(selected_count):
            edge_json = os.path.join(ref_dir, f"edge{idx}.json")
            mask_json = os.path.join(ref_dir, f"mask{idx}.json")
            if not missing_reference or idx > 0:
                with open(edge_json, "w") as f:
                    json.dump(
                        {
                            "n_nodes": 3,
                            "edges": [[0, 1], [1, 2], [2, 0]],
                        },
                        f,
                    )
                with open(mask_json, "w") as f:
                    json.dump(
                        {
                            "input_mask": [
                                [1, 1],
                                [1, 1],
                                [1, 1],
                            ]
                        },
                        f,
                    )
            rows.append(
                {
                    "selected": "1",
                    "topology_id": f"mask{idx}",
                    "topology_name": f"mask{idx}",
                    "edge_json": edge_json,
                    "input_mask_json": mask_json,
                }
            )
        rows.append(
                {
                    "selected": "0",
                    "topology_id": "not_selected",
                    "topology_name": "not_selected",
                    "edge_json": "",
                    "input_mask_json": "",
            }
        )
        self.write_csv(
            os.path.join(essential_dir, "library.csv"),
            ["selected", "topology_id", "topology_name", "edge_json", "input_mask_json"],
            rows,
        )
        self.write_csv(
            os.path.join(essential_dir, "selected.csv"),
            ["selected", "topology_id", "topology_name", "edge_json", "input_mask_json"],
            rows[:selected_count],
        )
        with open(os.path.join(essential_dir, "summary.json"), "w") as f:
            json.dump({"n_selected_masks": selected_count}, f)

        retrain_dir = os.path.join(root, "essential_inputmask50_retrain")
        for idx in range(retrain_count):
            topology_idx = idx // 2
            seed = (idx % 2) + 1
            if topology_idx >= selected_count:
                break
            self.touch(
                os.path.join(
                    retrain_dir,
                    f"mask{topology_idx}_trainseed{seed}",
                    "results.pkl",
                ),
                b"done",
            )
            for filename in ["topology_metrics.json", "config.json"]:
                self.touch(
                    os.path.join(
                        retrain_dir,
                        f"mask{topology_idx}_trainseed{seed}",
                        filename,
                    ),
                    b"{}",
                )
        self.write_csv(
            os.path.join(retrain_dir, "task_manifest.csv"),
            ["completed", "results_path"],
            [
                {
                    "completed": "True",
                    "results_path": f"mask{idx // 2}_trainseed{(idx % 2) + 1}/results.pkl",
                }
                for idx in range(retrain_count)
            ],
        )
        self.touch(os.path.join(retrain_dir, "_array_meta", "commands.txt"), b"cmd\n")

        if finalized:
            for name in [
                "topology_results.csv",
                "topology_regression.json",
                "topology_seed_aggregates.csv",
                "topology_seed_aggregates.json",
            ]:
                self.touch(os.path.join(retrain_dir, name), b"{}")
            self.write_csv(
                os.path.join(essential_dir, "retrain_comparison.csv"),
                ["topology_name"],
                [{"topology_name": f"mask{idx}"} for idx in range(selected_count)],
            )
            with open(os.path.join(essential_dir, "retrain_comparison.json"), "w") as f:
                json.dump({"n_joined": selected_count}, f)

        return root

    def run_audit(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=False,
            text=True,
            capture_output=True,
        )

    def test_reports_partial_retrain_counts_without_strict_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=2, retrain_count=1)
            output_json = os.path.join(tmpdir, "audit.json")
            result = self.run_audit(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1,2",
                    "--output_json",
                    output_json,
                ]
            )
            with open(output_json) as f:
                payload = json.load(f)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("completed=1 expected=4 missing=3", result.stdout)
        retrain = payload["experiments"][0]["essential_inputmask_retrain"]
        self.assertEqual(retrain["expected_results"], 4)
        self.assertEqual(retrain["completed_results"], 1)

    def test_strict_retrain_requirement_fails_on_incomplete_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=2, retrain_count=3)
            result = self.run_audit(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1,2",
                    "--require_essential_retrains",
                    "--strict",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("essential retrains incomplete: 3/4", result.stdout)

    def test_strict_retrain_requirement_passes_for_finalized_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(
                tmpdir,
                selected_count=2,
                retrain_count=4,
                finalized=True,
            )
            result = self.run_audit(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1,2",
                    "--require_source_results",
                    "--require_mechanisms",
                    "--require_essential_inputmask",
                    "--require_essential_retrains",
                    "--strict",
                ]
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("status: ok", result.stdout)

    def test_strict_retrain_requirement_fails_on_wrong_output_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=1, retrain_count=0)
            self.touch(
                os.path.join(root, "essential_inputmask50_retrain", "wrong_run", "results.pkl"),
                b"done",
            )
            result = self.run_audit(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1",
                    "--require_essential_retrains",
                    "--strict",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("essential retrains incomplete: 0/1", result.stdout)
        self.assertIn("unexpected results", result.stdout)

    def test_strict_retrain_requirement_fails_on_incomplete_comparison(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(
                tmpdir,
                selected_count=2,
                retrain_count=4,
                finalized=True,
            )
            essential_dir = os.path.join(root, "essential_inputmask50")
            self.write_csv(
                os.path.join(essential_dir, "retrain_comparison.csv"),
                ["topology_name"],
                [{"topology_name": "mask0"}],
            )
            with open(os.path.join(essential_dir, "retrain_comparison.json"), "w") as f:
                json.dump({"n_joined": 1}, f)
            result = self.run_audit(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1,2",
                    "--require_essential_retrains",
                    "--strict",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("comparison row count mismatch: 1/2", result.stdout)
        self.assertIn("comparison n_joined mismatch: 1/2", result.stdout)

    def test_strict_retrain_requirement_fails_on_missing_run_sidecars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(
                tmpdir,
                selected_count=1,
                retrain_count=1,
                finalized=True,
            )
            os.remove(
                os.path.join(
                    root,
                    "essential_inputmask50_retrain",
                    "mask0_trainseed1",
                    "config.json",
                )
            )
            result = self.run_audit(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1",
                    "--require_essential_retrains",
                    "--strict",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing required run sidecars", result.stdout)
        self.assertIn("missing_run_files=1", result.stdout)

    def test_strict_essential_requirement_fails_on_missing_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(
                tmpdir,
                selected_count=2,
                missing_reference=True,
            )
            result = self.run_audit(
                [
                    "--experiment",
                    f"exp={root}",
                    "--require_essential_inputmask",
                    "--strict",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing edge_json", result.stdout)
        self.assertIn("missing input_mask_json", result.stdout)

    def test_strict_essential_requirement_fails_on_invalid_references(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=1)
            ref_dir = os.path.join(root, "essential_inputmask50", "refs")
            with open(os.path.join(ref_dir, "edge0.json"), "w") as f:
                json.dump({}, f)
            with open(os.path.join(ref_dir, "mask0.json"), "w") as f:
                json.dump({"input_mask": [[1, 0], [0.5, 1]]}, f)
            result = self.run_audit(
                [
                    "--experiment",
                    f"exp={root}",
                    "--require_essential_inputmask",
                    "--strict",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid edge_json", result.stdout)
        self.assertIn("invalid input_mask_json", result.stdout)

    def test_strict_essential_requirement_fails_on_edge_mask_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=1)
            ref_dir = os.path.join(root, "essential_inputmask50", "refs")
            with open(os.path.join(ref_dir, "mask0.json"), "w") as f:
                json.dump({"input_mask": [[1, 0], [0, 1]]}, f)
            result = self.run_audit(
                [
                    "--experiment",
                    f"exp={root}",
                    "--require_essential_inputmask",
                    "--strict",
                ]
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("edge/input-mask mismatches", result.stdout)
        self.assertIn("edge_mask_pair_invalid=1", result.stdout)

    def test_strict_physical_essential_requirement_does_not_require_input_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = self.make_experiment(tmpdir, selected_count=0)
            essential_dir = os.path.join(root, "essential_input50")
            ref_dir = os.path.join(essential_dir, "refs")
            edge_json = os.path.join(ref_dir, "edge0.json")
            os.makedirs(ref_dir)
            with open(edge_json, "w") as f:
                json.dump({"n_nodes": 3, "edges": [[0, 1], [1, 2], [2, 0]]}, f)
            row = {
                "selected": "1",
                "topology_id": "ess0",
                "topology_name": "ess0",
                "edge_json": edge_json,
            }
            self.write_csv(
                os.path.join(essential_dir, "library.csv"),
                ["selected", "topology_id", "topology_name", "edge_json"],
                [row],
            )
            self.write_csv(
                os.path.join(essential_dir, "selected.csv"),
                ["selected", "topology_id", "topology_name", "edge_json"],
                [row],
            )
            with open(os.path.join(essential_dir, "summary.json"), "w") as f:
                json.dump({"n_selected_subgraphs": 1}, f)

            retrain_dir = os.path.join(root, "essential_input50_retrain")
            run_dir = os.path.join(retrain_dir, "ess0_trainseed1")
            for filename in ["results.pkl", "topology_metrics.json", "config.json"]:
                self.touch(os.path.join(run_dir, filename), b"{}")
            for name in [
                "topology_results.csv",
                "topology_regression.json",
                "topology_seed_aggregates.csv",
                "topology_seed_aggregates.json",
            ]:
                self.touch(os.path.join(retrain_dir, name), b"{}")
            self.write_csv(
                os.path.join(essential_dir, "retrain_comparison.csv"),
                ["topology_name"],
                [{"topology_name": "ess0"}],
            )
            with open(os.path.join(essential_dir, "retrain_comparison.json"), "w") as f:
                json.dump({"n_joined": 1}, f)
            other_layout_run = os.path.join(
                root,
                "essential_inputmask50_retrain",
                "mask0_trainseed1",
            )
            for filename in ["results.pkl", "topology_metrics.json", "config.json"]:
                self.touch(os.path.join(other_layout_run, filename), b"{}")

            result = self.run_audit(
                [
                    "--experiment",
                    f"exp={root}",
                    "--seeds",
                    "1",
                    "--essential_directory",
                    "essential_input50",
                    "--retrain_directory",
                    "essential_input50_retrain",
                    "--essential_kind",
                    "physical",
                    "--require_source_results",
                    "--require_mechanisms",
                    "--require_essential",
                    "--require_essential_retrains",
                    "--strict",
                ]
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("source: results=1 metrics=1 mechanisms=1", result.stdout)
        self.assertIn("kind=physical", result.stdout)
        self.assertIn("status: ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
