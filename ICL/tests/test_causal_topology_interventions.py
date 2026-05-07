import os
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from causal_topology_interventions import (
    context_block_permutation,
    metric_deltas,
    parse_interventions,
    permutation,
    random_effective_K_with_same_support,
)


SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "causal_topology_interventions.py",
)


class CausalTopologyInterventionsTests(unittest.TestCase):
    def test_context_block_shuffle_keeps_query_block_fixed(self):
        perm = context_block_permutation(n_context=3, z_dim=2, seed=5, include_query=False)
        self.assertEqual(sorted(perm.tolist()), list(range(8)))
        self.assertEqual(perm[-2:].tolist(), [6, 7])
        self.assertNotEqual(perm[:6].tolist(), list(range(6)))

    def test_all_block_shuffle_is_valid_permutation(self):
        perm = context_block_permutation(n_context=2, z_dim=3, seed=9, include_query=True)
        self.assertEqual(sorted(perm.tolist()), list(range(9)))

    def test_permutation_avoids_identity_when_possible(self):
        perm = permutation(seed=0, n=2)
        self.assertEqual(sorted(perm.tolist()), [0, 1])
        self.assertFalse(np.all(perm == np.arange(2)))

    def test_randomize_effective_K_preserves_support_and_row_norms(self):
        effective = np.asarray(
            [
                [3.0, 4.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        mask = np.asarray(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        randomized = random_effective_K_with_same_support(effective, mask, seed=2)
        np.testing.assert_allclose(randomized[mask == 0], 0.0)
        np.testing.assert_allclose(
            np.linalg.norm(randomized * mask, axis=1),
            np.linalg.norm(effective * mask, axis=1),
        )

    def test_metric_deltas(self):
        deltas = metric_deltas(
            {"target_accuracy": 60.0, "branch_active_tree_mi": 0.1},
            {"target_accuracy": 80.0, "branch_active_tree_mi": 0.3},
        )
        self.assertAlmostEqual(deltas["target_accuracy_delta"], -20.0)
        self.assertAlmostEqual(deltas["branch_active_tree_mi_delta"], -0.2)

    def test_parse_interventions_rejects_unknown(self):
        with self.assertRaises(ValueError):
            parse_interventions("context_block_shuffle,nope")

    def test_help_does_not_require_torch(self):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Causal interventions", result.stdout)


if __name__ == "__main__":
    unittest.main()
