"""Closed-geometry parity and actual Numba teacher graph compilation."""
import concurrent.futures
from pathlib import Path
import unittest

import numpy as np
import rind_phase2

# Supports testing the staged source before installing it in the sibling tree.
staged_source = Path(__file__).resolve().parents[2] / "src/rind_phase2"
if str(staged_source) not in rind_phase2.__path__:
    rind_phase2.__path__.insert(0, str(staged_source))

from rind_dataset.constants import PARAM_WIDTH
from rind_dataset.geometry import segment_intersects_obstacle as reference
from rind_phase2.region_geometry import segment_intersects_obstacle as compiled
from rind_phase2.regions import check_geometry_backend, response_regions, graph


class RegionGeometryTests(unittest.TestCase):
    def _obstacles(self):
        params = np.zeros((4, PARAM_WIDTH), dtype=np.float64)
        params[0, :5] = [8, 8, 3, 5, 0.37]
        params[1, :5] = [20, 8, 5, 3, -0.61]
        params[2, :6] = [3, 18, 12, 16, 8, 26]
        vertices = [(18, 18), (26, 18), (26, 26), (22, 22), (18, 26)]
        params[3, 0] = len(vertices)
        params[3, 1:1 + 2 * len(vertices)] = np.asarray(vertices).ravel()
        return params

    def test_all_primitives_match_canonical_segments(self):
        rng = np.random.default_rng(20261028)
        segments = rng.uniform(-4, 36, size=(2000, 4))
        for kind, params in enumerate(self._obstacles()):
            for px, py, qx, qy in segments:
                self.assertEqual(compiled(px, py, qx, qy, kind, params),
                                 reference(px, py, qx, qy, kind, params))

    def test_touching_inside_and_zero_length_segments(self):
        params = self._obstacles()
        params[0, :5] = [8, 8, 3, 5, 0]
        params[1, :5] = [20, 8, 5, 3, 0]
        segments = [(0., 3., 5., 3.), (5., 3., 5., 3.), (8., 8., 8., 8.),
                    (10., 11., 30., 11.), (15., 8., 15., 8.),
                    (3., 18., 12., 16.), (18., 18., 26., 18.),
                    (22., 22., 22., 22.), (-1., -1., -1., -1.)]
        for kind, row in enumerate(params):
            for segment in segments:
                self.assertEqual(compiled(*segment, kind, row), reference(*segment, kind, row))

    def test_preflight_compiles_graph_and_excludes_thin_obstacle(self):
        result = check_geometry_backend()
        self.assertEqual(result["geometry_checks"], 128)
        if hasattr(graph, "nopython_signatures"):
            self.assertTrue(result["nopython_compiled"])

    def test_parallel_first_observations_have_identical_graphs(self):
        check_geometry_backend()
        types = np.array([0], dtype=np.uint8)
        params = np.zeros((1, PARAM_WIDTH)); params[0, :5] = [1, .5, .1, .4, 0]
        def run(_):
            result = response_regions(np.ones((1, 2)), np.zeros((1, 2), dtype=bool),
                                      (0, 0), types, params)
            return result.values.tolist(), result.neighbors, result.excluded_edges
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(run, range(12)))
        self.assertTrue(all(value == ([1., 1.], [[], []], 1) for value in outputs))


if __name__ == "__main__":
    unittest.main()
