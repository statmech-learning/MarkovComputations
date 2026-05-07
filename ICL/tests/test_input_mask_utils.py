import json
import os
import sys
import tempfile
import unittest

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from input_mask_utils import input_mask_summary, load_input_mask_json, validate_input_mask  # noqa: E402


class InputMaskUtilsTests(unittest.TestCase):
    def test_validate_accepts_binary_rectangular_mask(self):
        mask = validate_input_mask([[1, 0, True], [0.0, 1.0, False]], n_edges=2, p=3)
        np.testing.assert_array_equal(mask, np.asarray([[1, 0, 1], [0, 1, 0]]))

    def test_validate_rejects_bad_shape_and_values(self):
        with self.assertRaisesRegex(ValueError, "rows"):
            validate_input_mask([[1, 0]], n_edges=2, p=2)
        with self.assertRaisesRegex(ValueError, "columns"):
            validate_input_mask([[1], [0]], n_edges=2, p=2)
        for bad in (0.5, -1, 2, None, "1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_input_mask([[bad]], n_edges=1, p=1)

    def test_load_object_json_validates_edge_order(self):
        n_nodes = 3
        edges = [(0, 1), (1, 2), (2, 0)]
        payload = {
            "name": "cycle_mask",
            "n_nodes": n_nodes,
            "edges": [list(edge) for edge in edges],
            "p": 2,
            "input_mask": [[1, 0], [0, 1], [1, 1]],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mask.json")
            with open(path, "w") as f:
                json.dump(payload, f)
            mask, metadata = load_input_mask_json(path, n_nodes, edges, p=2)
        self.assertEqual(metadata["name"], "cycle_mask")
        np.testing.assert_array_equal(mask, np.asarray(payload["input_mask"]))

    def test_load_rejects_reordered_edges(self):
        n_nodes = 3
        edges = [(0, 1), (1, 2), (2, 0)]
        payload = {
            "n_nodes": n_nodes,
            "edges": [[1, 2], [0, 1], [2, 0]],
            "p": 2,
            "input_mask": [[1, 0], [0, 1], [1, 1]],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mask.json")
            with open(path, "w") as f:
                json.dump(payload, f)
            with self.assertRaisesRegex(ValueError, "edge order"):
                load_input_mask_json(path, n_nodes, edges, p=2)

    def test_load_accepts_bare_matrix_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bare.json")
            with open(path, "w") as f:
                json.dump([[1, 0], [0, 1]], f)
            mask, metadata = load_input_mask_json(path, 2, [(0, 1), (1, 0)], p=2)
        self.assertEqual(metadata["name"], "bare")
        np.testing.assert_array_equal(mask, np.asarray([[1, 0], [0, 1]]))

    def test_summary_counts_edge_and_coordinate_support(self):
        summary = input_mask_summary(np.asarray([[1, 0, 0], [1, 1, 0], [0, 0, 0]]))
        self.assertEqual(summary["input_coupled_parameter_count"], 3)
        self.assertEqual(summary["input_coupled_edge_count"], 2)
        self.assertEqual(summary["input_coupled_coord_count"], 2)


if __name__ == "__main__":
    unittest.main()
