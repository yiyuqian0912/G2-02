"""Focused regression checks for the mathematical examples."""
import itertools
import random
import unittest

import verify as v


class ProbabilityTests(unittest.TestCase):
    def test_strip_formula_from_actual_grid_adjacency(self):
        count = 2048
        for h in (.125, 1., 3.):
            for ratio in (0., .13, -.13, .61, -.61, 1., -1., 1.7):
                d = ratio*h
                observed = sum(v.strip_bridge(d, h, (i+.5)*h/count)
                               for i in range(count))/count
                self.assertAlmostEqual(observed, v.strip_probability(d, h), delta=2/count)

    def test_uniform_gap_average_and_rate(self):
        self.assertEqual(v.uniform_gap_probability(1., 1.), .5)
        self.assertEqual(v.uniform_gap_probability(2., 1.), .75)
        for h in (1/2, 1/4, 1/8, 1/16):
            self.assertEqual(v.uniform_gap_probability(h, 1.)/h, .5)
        for function, args in ((v.strip_probability, (0, 0)),
                               (v.uniform_gap_probability, (1, 0))):
            with self.assertRaises(ValueError):
                function(*args)

    def test_crossing_persists_at_many_scales(self):
        rng = random.Random(418)
        for h in (v.RADIUS/4, v.RADIUS/16, v.RADIUS/128, v.RADIUS/1024):
            for _ in range(80):
                phase = (rng.random(), rng.random())
                edges = v.bad_edges(v.affine_label, h, phase, (-3*h, 3*h, -3*h, 3*h))
                self.assertTrue(edges)
                self.assertTrue(all(a != b and a.bit_count() == b.bit_count()
                                    for a, b in (e["labels"] for e in edges)))

    def test_parallel_resolved_strip_has_no_bad_edges(self):
        # Embed a resolved strip in 2D rather than using the slice helper.
        def label(point):
            x, _ = point
            return int(x < 0) | (int(x > .5) << 1)
        for phase in ((.13, .27), (.73, .41), (.29, .83)):
            self.assertFalse(v.bad_edges(label, .25, phase, (-1, 1, -1, 1)))


class GeometryTests(unittest.TestCase):
    def test_closed_rectangle_segment_intersection(self):
        box = (0., 1., 0., 1.)
        self.assertTrue(v.segment_hits_rectangle((-1, .5), (2, .5), box))
        self.assertTrue(v.segment_hits_rectangle((-1, -1), (0, 0), box))
        self.assertTrue(v.segment_hits_rectangle((.5, .5), (.5, .5), box))
        self.assertFalse(v.segment_hits_rectangle((-1, 2), (2, 2), box))
        self.assertFalse(v.segment_hits_rectangle((-1, -1), (-.1, -.1), box))

    def test_actual_blockers_realize_crossing_and_perturbations(self):
        rng = random.Random(107)
        source_cases = [v.NOMINAL_SOURCES]
        # Include corners of the full source-box product, not just random draws.
        for signs in itertools.product((-1, 1), repeat=4):
            source_cases.append(tuple((x+signs[2*i]*v.ETA, y+signs[2*i+1]*v.ETA)
                                      for i, (x, y) in enumerate(v.NOMINAL_SOURCES)))
        for sources in source_cases:
            lines = v.crossing_lines(sources)
            m1, b1, m2, b2 = lines
            self.assertTrue(.9 < m2-m1 < 1.1)
            cx = (b2-b1)/(m1-m2)
            cy = m1*cx+b1
            self.assertLess(max(abs(cx), abs(cy)), v.RADIUS/4)
            for _ in range(120):
                point = (rng.uniform(-v.RADIUS, v.RADIUS), rng.uniform(-v.RADIUS, v.RADIUS))
                self.assertEqual(v.physical_label(point, sources), v.affine_label(point, lines))

        # The local-coordinate construction also fits the upstream world and
        # a permitted size-64 observation window after this affine transform.
        transform = lambda p: (800+128*p[0], 800+128*p[1])
        sources = tuple(transform(p) for p in v.NOMINAL_SOURCES)
        blockers = tuple((800+128*xlo, 800+128*xhi, 800+128*ylo, 800+128*yhi)
                         for xlo, xhi, ylo, yhi in v.BLOCKERS)
        self.assertTrue(all(0 <= c <= 1024 for point in sources for c in point))
        self.assertTrue(all(not (768 < x < 832 and 768 < y < 832) for x, y in sources))
        for _ in range(120):
            point = (rng.uniform(-v.RADIUS, v.RADIUS), rng.uniform(-v.RADIUS, v.RADIUS))
            self.assertEqual(v.physical_label(transform(point), sources, blockers), v.affine_label(point))

    def test_obstacle_safe_edge_rejects_thin_unseen_blocker(self):
        def label(point):
            return 1 if point[0] < 0 else 2
        bounds, phase = (-1., 1., -.1, .1), (.5, .05)
        self.assertTrue(v.bad_edges(label, 1., phase, bounds))
        self.assertFalse(v.bad_edges(label, 1., phase, bounds,
                                     blockers=((-0.1, 0.1, -1., 1.),)))


class PeelingTests(unittest.TestCase):
    def test_compatible_full_mask_decomposition(self):
        a, b = (1, 1, 0), (0, 1, 1)
        target = v.sum_masks((a, b))
        result = v.ideal_peel((a, b), target, 3)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(v.sum_masks(result["selected"]), target)
        self.assertEqual(len(result["selected"]), 2)

    def test_missing_singleton_and_mixed_anchor(self):
        self.assertEqual(v.ideal_peel(((1, 1, 0),), (2, 2, 0), 3)["status"], "no_singleton")
        self.assertEqual(v.ideal_peel(((1, 0), (0, 1)), (1, 1), 2)["status"], "no_covering_mask")
        # Adding a physically admissible union mask changes compatibility.
        result = v.ideal_peel(((1, 0), (0, 1), (1, 1)), (1, 1), 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["selected"], [(1, 1)])

    def test_identical_masks_do_not_identify_source_positions(self):
        result = v.ideal_peel(((1, 0), (1, 0)), (1, 0), 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["selected"]), 1)

    def test_pure_anchor_can_admit_multiple_full_masks(self):
        a, b, alternative = (1, 1, 0), (0, 1, 1), (1, 0, 1)
        target = v.sum_masks((a, b))
        self.assertEqual(v.ideal_peel((a, b, alternative), target, 3)["status"],
                         "nonidentifiable_mask")

    def test_nonnegative_local_choice_can_be_a_dead_end(self):
        wrong, b, c = (1, 1, 0), (0, 1, 1), (1, 0, 0)
        target = (1, 1, 1)
        self.assertEqual(v.sum_masks((b, c)), target)
        after = tuple(r-m for r, m in zip(target, wrong))
        self.assertTrue(all(r >= 0 for r in after))
        self.assertEqual(v.ideal_peel((wrong, b, c), after, 3)["status"], "no_covering_mask")

    def test_full_mask_and_one_hop_can_differ(self):
        a, b = (1, 1, 1, 1, 1), (0, 1, 0, 0, 0)
        target = v.sum_masks((a, b))
        self.assertEqual(v.response_components(target, 5, 1), [(0,), (1,), (2, 3, 4)])
        after_local = v.one_hop_subtract(target, a, 0, 5)
        self.assertEqual(after_local, (0, 1, 1, 1, 1))
        self.assertNotEqual(after_local, b)
        self.assertEqual(v.ideal_peel((a, b), target, 5)["status"], "complete")
        self.assertEqual(v.ideal_peel((a, b), after_local, 5)["status"], "no_covering_mask")

    def test_one_hop_can_equal_entire_source(self):
        a, b = (1, 1, 1, 1, 0), (0, 1, 1, 1, 1)
        target = v.sum_masks((a, b))
        self.assertEqual(v.one_hop_subtract(target, a, 0, 5), b)

    def test_components_use_four_neighbors_not_diagonals(self):
        self.assertEqual(v.response_components((1, 0, 0, 1), 2, 2), [(0,), (1,), (2,), (3,)])


if __name__ == "__main__":
    unittest.main()
