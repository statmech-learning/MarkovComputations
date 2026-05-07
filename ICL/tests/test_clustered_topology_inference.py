import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from clustered_topology_inference import (  # noqa: E402
    aggregate_seed_groups,
    graph_family_from_name,
    run_clustered_inference,
    with_derived_graph_family,
)


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "clustered_topology_inference.py",
)


class ClusteredTopologyInferenceTests(unittest.TestCase):
    def make_rows(self):
        rows = []
        for family_idx, family in enumerate(["random", "cycle", "hub"]):
            for topo_idx in range(4):
                d_rel = 100 + 20 * topo_idx + 5 * family_idx
                for seed in range(3):
                    rows.append(
                        {
                            "label": f"{family}_topo{topo_idx}_seed{seed}",
                            "topology_name": f"{family}_topo{topo_idx}",
                            "physical_topology_name": family,
                            "seed": seed,
                            "n_edges": 20,
                            "raw_physical_parameter_count": 400,
                            "input_coupled_parameter_count": 200,
                            "d_rel": d_rel,
                            "comparison_branch_common_d_rel_min": topo_idx * 4,
                            "comparison_branch_common_d_rel_gini": 0.5 - 0.05 * topo_idx,
                            "comparison_branch_d_rel_min": topo_idx * 4,
                            "comparison_branch_d_rel_gini": 0.5 - 0.05 * topo_idx,
                            "effective_rank_D": 10 + topo_idx,
                            "effective_rank_D_masked": 12 + topo_idx,
                            "condition_number_D": 10 + topo_idx,
                            "condition_number_D_masked": 15 + topo_idx,
                            "root_tree_count_gini": 0.1 + 0.01 * family_idx,
                            "edge_participation_gini": 0.2 + 0.01 * family_idx,
                            "edge_participation_var": 0.01,
                            "bottleneck_edge_fraction_095": 0.0,
                            "mean_shortest_path": 2.0,
                            "input_edge_load_gini": 0.3,
                            "input_coord_load_gini": 0.4,
                            "test_novel_classes": 40 + 0.2 * d_rel + seed,
                        }
                    )
        return rows

    def test_aggregates_nested_seed_rows(self):
        rows = self.make_rows()
        aggregate = aggregate_seed_groups(rows)
        self.assertEqual(len(aggregate), 12)
        first = aggregate[0]
        self.assertEqual(first["n_runs"], 3)
        self.assertIn("target_mean", first)
        self.assertIn("target_max", first)
        self.assertIn("target_std", first)

    def test_clustered_report_includes_group_bootstrap_and_family_holdout(self):
        report = run_clustered_inference(
            self.make_rows(),
            n_bootstrap=25,
            seed=7,
        )
        self.assertEqual(report["n_run_rows"], 36)
        self.assertEqual(report["n_clusters"], 12)
        self.assertEqual(report["n_families"], 3)
        self.assertIn("target_mean", report["group_level"])
        self.assertIn("raw_plus_drel", report["group_level"]["target_mean"])
        delta = report["cluster_bootstrap_run_level"]["raw_plus_drel"]
        self.assertEqual(delta["n_bootstrap_effective"], 25)
        self.assertGreater(delta["delta_mean"], 0.0)
        holdout = report["leave_family_out_group_target_mean"]["raw_plus_drel"]
        self.assertEqual(len(holdout["families"]), 3)
        self.assertIn("residual_std_between_cluster_means", report["residual_decomposition_run_level"]["raw_plus_drel"])

    def test_derives_graph_family_from_topology_instance_names(self):
        self.assertEqual(graph_family_from_name("cycle_chords_n5_m8_seed34"), "cycle_chords")
        self.assertEqual(graph_family_from_name("bottleneck_bridge_n5_m12_seed7"), "bottleneck_bridge")
        self.assertEqual(graph_family_from_name("random"), "random")

        rows = []
        for family in ["cycle_chords", "random_sc"]:
            for topo_idx in range(2):
                for seed in range(2):
                    rows.append(
                        {
                            "label": f"{family}_{topo_idx}_{seed}",
                            "topology_name": f"{family}_n5_m8_seed{topo_idx}",
                            "physical_topology_name": f"{family}_n5_m8_seed{topo_idx}",
                            "seed": seed,
                            "raw_physical_parameter_count": 100,
                            "input_coupled_parameter_count": 50,
                            "d_rel": topo_idx + 1,
                            "comparison_branch_common_d_rel_min": topo_idx,
                            "comparison_branch_common_d_rel_gini": 0.1,
                            "comparison_branch_d_rel_min": topo_idx,
                            "comparison_branch_d_rel_gini": 0.1,
                            "effective_rank_D": 2,
                            "effective_rank_D_masked": 2,
                            "condition_number_D": 1,
                            "condition_number_D_masked": 1,
                            "root_tree_count_gini": 0,
                            "edge_participation_gini": 0,
                            "edge_participation_var": 0,
                            "bottleneck_edge_fraction_095": 0,
                            "mean_shortest_path": 1,
                            "input_edge_load_gini": 0,
                            "input_coord_load_gini": 0,
                            "test_novel_classes": 50 + 10 * topo_idx,
                        }
                    )
        derived = with_derived_graph_family(rows)
        self.assertEqual(sorted({row["derived_graph_family"] for row in derived}), ["cycle_chords", "random_sc"])
        report = run_clustered_inference(
            rows,
            n_bootstrap=5,
            derive_graph_family=True,
        )
        self.assertEqual(report["family_col"], "derived_graph_family")
        self.assertEqual(report["n_families"], 2)

    def test_cli_writes_json(self):
        rows = self.make_rows()
        fieldnames = sorted({key for row in rows for key in row})
        with tempfile.TemporaryDirectory() as tmpdir:
            run_csv = os.path.join(tmpdir, "runs.csv")
            output_json = os.path.join(tmpdir, "clustered.json")
            with open(run_csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPT,
                    "--run_csv",
                    run_csv,
                    "--n_bootstrap",
                    "10",
                    "--derive_graph_family",
                    "--output_json",
                    output_json,
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("Cluster bootstrap", result.stdout)
            with open(output_json) as handle:
                payload = json.load(handle)
        self.assertEqual(payload["n_clusters"], 12)


if __name__ == "__main__":
    unittest.main()
