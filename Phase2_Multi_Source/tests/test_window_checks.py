"""Independent pixel-based checks of window validity and sampling semantics."""
import unittest
import numpy as np
from rind_phase2.window_checks import response_integrals, validate_windows


class RandomWindowTests(unittest.TestCase):
    def setUp(self):
        self.response = (np.indices((16, 16))[1] % 3).astype(np.uint8)
        self.obstacle = np.zeros((16, 16), dtype=bool)
        self.obstacle[2:6, 2:6] = True
        self.response[self.obstacle] = 0
        self.sources = np.array([[4.25, 2.0], [12.2, 13.7]])

    def test_window_checks_match_direct_pixels_and_continuous_geometry(self):
        for side in (4, 8):
            views, expected = [], []
            width = 16 - side + 1
            for y in range(width):
                for x in range(width):
                    contained = any(x <= px < x + side and y <= py < y + side for px, py in self.sources)
                    r = self.response[y:y + side, x:x + side]
                    free = ~self.obstacle[y:y + side, x:x + side]
                    values = r[free]
                    views.append((x, y, side))
                    expected.append(not contained and values.size > 0 and np.unique(values).size > 1)
            integrals = response_integrals(self.response, self.obstacle)
            for view, valid in zip(views, expected):
                actual = validate_windows(*integrals, self.sources, np.asarray([view], dtype=np.uint16))
                self.assertEqual(actual, valid)

    def test_integer_integrals_match_numpy_at_full_world_size(self):
        rng = np.random.default_rng(38)
        response = rng.integers(0, 5, size=(1024, 1024), dtype=np.uint8)
        obstacle = rng.random((1024, 1024)) < .25
        free = ~obstacle
        values = np.where(free, response, 0).astype(np.int32)
        for actual, image in zip(response_integrals(response, obstacle), (free, values, values * values)):
            expected = np.zeros((1025, 1025), dtype=np.int32)
            expected[1:, 1:] = image.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
            np.testing.assert_array_equal(actual, expected)

    def test_obstacles_do_not_make_constant_free_response_informative(self):
        for level in (0, 1, 2, 4):
            response = np.full((16, 16), level, dtype=np.uint8)
            response[self.obstacle] = 0
            views = np.asarray([[x, y, 4] for y in range(13) for x in range(13)], dtype=np.uint16)
            for view in views:
                self.assertFalse(validate_windows(*response_integrals(response, self.obstacle), self.sources, view[None]))

    def test_source_boundary_is_half_open_and_not_rounded_to_pixels(self):
        integrals = response_integrals(self.response, self.obstacle)
        view = np.asarray([[0, 0, 4]], dtype=np.uint16)
        self.assertTrue(validate_windows(*integrals, np.array([[4.0, 1.0]]), view))
        self.assertFalse(validate_windows(*integrals, np.array([[np.nextafter(4.0, 0), 1.0]]), view))
        self.assertTrue(validate_windows(*integrals, np.array([[4.25, 1.0]]), view))
        self.assertFalse(validate_windows(*integrals, np.array([[4.25, 1.0]]), np.asarray([[1, 0, 4]], dtype=np.uint16)))


if __name__ == "__main__":
    unittest.main()
