import csv
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "collect_topology_results.py",
)


class CollectTopologyResultsTests(unittest.TestCase):
    def run_collector(self, args):
        return subprocess.run(
            [sys.executable, SCRIPT, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    def test_backfills_branch_comparison_metrics_from_saved_topology(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "run_a")
            os.makedirs(run_dir)
            edges = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
            # N=2, D=1 gives coordinates [context0, context1, query].
            # Only context0 and query are coupled, so branch 1 has no paired
            # comparison rank support.
            input_mask = [[1, 0, 1] for _ in edges]
            with open(os.path.join(run_dir, "topology.json"), "w") as f:
                json.dump(
                    {
                        "name": "toy",
                        "n_nodes": 3,
                        "edges": edges,
                        "input_mask": input_mask,
                    },
                    f,
                )
            with open(os.path.join(run_dir, "topology_metrics.json"), "w") as f:
                json.dump({"topology_name": "toy", "n_nodes": 3, "n_edges": 6, "p": 3}, f)
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump({"N": 2, "D": 1, "seed": 7}, f)
            with open(os.path.join(run_dir, "results.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "history": {},
                        "results": {"novel_classes": 50.0},
                        "execution_time": 1.0,
                    },
                    f,
                )
            output_csv = os.path.join(tmpdir, "topology_results.csv")

            result = self.run_collector(
                [
                    "--input_root",
                    tmpdir,
                    "--output_csv",
                    output_csv,
                ]
            )
            with open(output_csv, newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertIn("Wrote 1 rows", result.stdout)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["comparison_branch_d_rel_min"], "0")
        self.assertGreater(float(row["comparison_branch_d_rel_max"]), 0.0)
        self.assertGreater(float(row["comparison_branch_d_rel_gini"]), 0.0)


if __name__ == "__main__":
    unittest.main()
