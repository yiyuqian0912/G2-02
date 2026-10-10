"""Focused joint-space checks; truth/reference APIs are forbidden in generation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from rind_phase1_multi.data import Phase1MultiDataset
from rind_phase1_multi.diagnostics import inspect_teacher, source_space_projections
from rind_phase1_multi.physics import JointResponseEvaluator, REFERENCE_ATOL
from rind_phase1_multi.search import (CandidateDomain, CandidateEvaluation, NoCandidatePositionsError,
                                     canonical_sources, evaluate_joint_support, make_joint_support,
                                     sample_joint_support, uniform_source_grid)
from rind_phase1_multi.teacher import (IncompleteSearchError, NoValidCandidatesError, generate_joint_teacher,
                                      load_teacher_record, save_teacher_record, target_from_search,
                                      teacher_probabilities)
from test_multi_data import fixture


class MockEvaluator:
    def __init__(self, *, cost=None, positions=None):
        self.cost = cost or (lambda rows: abs(float(rows[:, 2].sum()) - .9))
        self.positions = positions

    def positions_valid(self, xy):
        return np.ones(len(xy), dtype=bool) if self.positions is None else self.positions(xy)

    def evaluate(self, sources):
        return CandidateEvaluation(True, self.cost(sources))


class LabelsForbidden:
    """Expose only observation, geometry and forward rendering to the teacher."""

    def __init__(self, dataset):
        self.dataset, self.root, self.manifest = dataset, dataset.root, dataset.manifest

    def get_observation(self, *args):
        return self.dataset.get_observation(*args)

    def get_region(self, *args):
        region = self.dataset.get_region(*args)
        return {"response": region["response"], "obstacle": region["obstacle"]}

    def get_scene(self, *args):
        scene = self.dataset.get_scene(*args)
        return {key: scene[key] for key in ("obstacle_types", "obstacle_params", "obstacle_count")}

    def rerender(self, *args, **kwargs):
        return self.dataset.rerender(*args, **kwargs)

    def get_candidates(self, *_):
        raise AssertionError("Reference witnesses must not enter proposals/targets")

    get_source_params = get_channels = get_candidates


class JointTeacherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rind-joint-test-")
        self.addCleanup(self.temp.cleanup)
        self.ds = Phase1MultiDataset(fixture(Path(self.temp.name) / "fixture"))

    def generate(self, **kwargs):
        options = dict(spacing=16, temperature=.05, source_counts=[1, 2, 3, 4],
                       samples_per_count=16, count_prior=[.25] * 4, seed=17, backend="reference")
        options.update(kwargs)
        return generate_joint_teacher(LabelsForbidden(self.ds), 1, 0, **options)

    def test_world_domain_edges_and_no_parent_prior(self):
        domain = CandidateDomain((16, 16, 16), 64)
        points = [[16, 16], [31.99, 31.99], [32, 16], [16, 32], [56, 56], [64, 64], [-1, 0], [np.nan, 0]]
        np.testing.assert_array_equal(domain.contains(points), [False, False, True, True, True, True, False, False])
        terminal = CandidateDomain((48, 48, 16), 64)
        np.testing.assert_array_equal(terminal.contains([[64, 64], [64, 47]]), [False, True])
        self.assertFalse(domain.metadata()["quadtree_prior"])
        with self.assertRaises(ValueError):
            CandidateDomain((60, 0, 16), 64)

    def test_deterministic_uniform_grid_order_and_geometry_mask(self):
        domain = CandidateDomain((16, 16, 16), 64)
        evaluator = MockEvaluator(positions=lambda xy: xy[:, 0] < 48)
        a = uniform_source_grid(domain, evaluator, spacing=16)
        b = uniform_source_grid(domain, evaluator, spacing=16)
        np.testing.assert_array_equal(a.positions, b.positions)
        self.assertEqual(len(a.positions), 15)
        self.assertTrue(domain.contains(a.positions).all())
        self.assertTrue(np.all(a.positions[a.valid, 0] < 48))
        self.assertEqual(a.positions[:4].tolist(), [[8, 8], [24, 8], [40, 8], [56, 8]])
        np.testing.assert_array_equal(a.grid_index[np.argsort(a.grid_index[:, 0] * 4 + a.grid_index[:, 1])], a.grid_index)

    def test_seed_reproducibility_count_substreams_and_strength_sampling(self):
        grid = uniform_source_grid(CandidateDomain((16, 16, 16), 64), MockEvaluator(), spacing=16)
        options = dict(source_counts=[1, 3], samples_per_count=12, count_prior=[.4, .6], seed=41)
        a = sample_joint_support(grid, **options)
        b = sample_joint_support(grid, **options)
        np.testing.assert_array_equal(a.sources, b.sources)
        c = sample_joint_support(grid, source_counts=[3], samples_per_count=12, seed=41)
        np.testing.assert_array_equal(a.sources[a.source_counts == 3, :3], c.sources)
        d = sample_joint_support(grid, **{**options, "seed": 42})
        self.assertFalse(np.array_equal(a.sources, d.sources))
        for i in range(len(a)):
            rows = a.source_set(i)
            self.assertTrue(grid.domain.contains(rows[:, :2]).all())
            self.assertTrue(np.all((rows[:, 2] >= 0) & (rows[:, 2] < 1)))
            np.testing.assert_array_equal(rows, canonical_sources(rows))

    def test_unknown_count_requires_explicit_prior_and_finite_settings(self):
        grid = uniform_source_grid(CandidateDomain((16, 16, 16), 64), MockEvaluator(), spacing=16)
        with self.assertRaisesRegex(ValueError, "count_prior"):
            sample_joint_support(grid, source_counts=[1, 2], samples_per_count=8, seed=0)
        for prior in ([.2, .2], [-.1, 1.1], [0, 1], [np.nan, .5]):
            with self.assertRaises(ValueError):
                sample_joint_support(grid, source_counts=[1, 2], samples_per_count=8, count_prior=prior, seed=0)
        for sizes in (0, 2.5, [2, 0], [2, 3, 4]):
            with self.assertRaises(ValueError):
                sample_joint_support(grid, source_counts=[1, 2], samples_per_count=sizes, count_prior=[.5, .5], seed=0)
        for spacing in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                uniform_source_grid(grid.domain, MockEvaluator(), spacing=spacing)

    def test_unequal_samples_do_not_change_declared_count_prior_at_equal_cost(self):
        grid = uniform_source_grid(CandidateDomain((16, 16, 16), 64), MockEvaluator(), spacing=16)
        support = sample_joint_support(grid, source_counts=[1, 4], samples_per_count=[3, 37],
                                       count_prior=[.7, .3], seed=8)
        result = evaluate_joint_support(support, MockEvaluator(cost=lambda _: .25))
        q = target_from_search(result, temperature=.01)
        self.assertAlmostEqual(q[support.source_counts == 1].sum(), .7)
        self.assertAlmostEqual(q[support.source_counts == 4].sum(), .3)

    def test_permutation_dedup_is_stable_and_preserves_sampling_mass(self):
        first = np.array([[8, 8, .2], [56, 56, .6]])
        second = np.array([[40, 8, .7]])
        split = np.array([[8, 8, .1], [8, 8, .1], [56, 56, .6]])
        support = make_joint_support([first, second, first[::-1], split], base_mass=[.2, .1, .3, .4])
        self.assertEqual(support.source_counts.tolist(), [2, 1, 3])
        self.assertEqual(support.multiplicity.tolist(), [2, 1, 1])
        np.testing.assert_allclose(support.base_mass, [.5, .1, .4])
        np.testing.assert_array_equal(support.source_set(0), first)
        self.assertNotEqual(support.source_set(0).shape, support.source_set(2).shape)

    def test_physical_cost_alignment_and_probability_favors_joint_response(self):
        sets = [[[8, 8, .9]], [[8, 8, .4], [56, 56, .5]], [[40, 8, .1]]]
        support = make_joint_support(sets, base_mass=[1, 1, 1])
        result = evaluate_joint_support(support, MockEvaluator())
        np.testing.assert_allclose(result.physical_cost, [0, 0, .8], atol=1e-15)
        q = target_from_search(result, temperature=.05)
        self.assertAlmostEqual(q[0], q[1])
        self.assertGreater(q[1], q[2])
        self.assertAlmostEqual(q.sum(), 1)

    def test_invalid_entries_and_extreme_temperature(self):
        q = teacher_probabilities([np.nan, 1e300, 0, 0], np.array([False, True, True, True]),
                                  temperature=1e-300, base_mass=[1, 1, .25, .75])
        np.testing.assert_array_equal(q[:2], [0, 0])
        np.testing.assert_allclose(q, [0, 0, .25, .75], rtol=0, atol=2e-16)
        with self.assertRaises(NoValidCandidatesError):
            teacher_probabilities([np.inf], np.array([False]), temperature=.1, base_mass=[1])
        with self.assertRaises(NoValidCandidatesError):
            teacher_probabilities([0], np.array([True]), temperature=.1, base_mass=[0])
        with self.assertRaises(ValueError):
            teacher_probabilities([-1], np.array([True]), temperature=.1, base_mass=[1])
        for tau in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                teacher_probabilities([0], np.array([True]), temperature=tau, base_mass=[1])

    def test_budget_stops_are_explicit_and_never_normalized(self):
        completed = self.generate(max_evaluations=np.int64(64))
        self.assertEqual(completed["metadata"]["settings"]["max_evaluations"], 64)
        with self.assertRaises(IncompleteSearchError):
            self.generate(max_evaluations=3)
        with self.assertRaises(IncompleteSearchError):
            self.generate(max_source_renders=0)
        support = make_joint_support([[[8, 8, .1]], [[40, 8, .5]]], base_mass=[.5, .5])
        result = evaluate_joint_support(support, MockEvaluator(), max_evaluations=1)
        self.assertFalse(result.complete)
        self.assertEqual(result.evaluated.tolist(), [True, False])
        self.assertEqual(result.num_joint_evaluations, 1)
        with self.assertRaises(IncompleteSearchError):
            target_from_search(result, temperature=.1)

    def test_visibility_and_joint_cost_cache_counts_real_unique_work(self):
        evaluator = JointResponseEvaluator(self.ds, 1, 0, backend="reference")
        first = np.array([[8, 8, .2], [56, 56, .8]])
        result = evaluator.evaluate(first)
        fresh = self.ds.rerender(1, evaluator.observation["window"], first)
        expected = np.abs(fresh - evaluator.observed_response).mean()
        self.assertAlmostEqual(result.physical_cost, expected)
        self.assertEqual(result.num_source_renders, 2)
        reverse = evaluator.evaluate(first[::-1])
        self.assertEqual(reverse.new_joint_evaluations, 0)
        self.assertEqual(reverse.num_source_renders, 0)
        changed = evaluator.evaluate([[8, 8, .4], [56, 56, 1]])
        self.assertLessEqual(changed.physical_cost, REFERENCE_ATOL)
        self.assertEqual(changed.num_source_renders, 0)
        self.assertEqual(evaluator.num_source_renders, 2)
        self.assertEqual(evaluator.visibility_cache_bytes, 2 * fresh.size // 8)
        support = make_joint_support([first, [[8, 8, .4], [56, 56, 1]]], base_mass=[.5, .5])
        reused = evaluate_joint_support(support, evaluator, max_evaluations=0)
        self.assertTrue(reused.complete)
        self.assertEqual(reused.num_joint_evaluations, 0)
        self.assertEqual(reused.num_cache_hits, 2)

    def test_source_bounds_window_exclusion_and_zero_strength(self):
        evaluator = JointResponseEvaluator(self.ds, 1, 0, backend="reference")
        x, y, side = evaluator.domain.window
        for rows in ([[x + .5, y + .5, .5]], [[8, 8, 1.1]], [[np.nan, 8, .5]], [[65, 8, .5]]):
            result = evaluator.evaluate(rows)
            self.assertFalse(result.valid)
            self.assertEqual(result.num_source_renders, 0)
            self.assertEqual(result.physical_cost, np.inf)
        zero = evaluator.evaluate([[8, 8, 0]])
        self.assertTrue(zero.valid)
        self.assertEqual(zero.num_source_renders, 0)
        self.assertGreater(zero.physical_cost, 1)

    def test_teacher_is_label_free_reproducible_and_preserves_full_source_sets(self):
        first, second = self.generate(), self.generate()
        for key in ("sources", "source_counts", "physical_cost", "teacher_prob", "base_mass"):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["metadata"]["config_id"], second["metadata"]["config_id"])
        self.assertFalse(first["metadata"]["reference_solutions_used_as_targets"])
        self.assertTrue(first["evaluated"].all() and first["valid"].all())
        self.assertEqual(first["source_counts"].tolist(), [k for k in range(1, 5) for _ in range(16)])
        self.assertAlmostEqual(first["teacher_prob"].sum(), 1)
        self.assertLess(first["metadata"]["num_source_renders"], first["metadata"]["num_joint_evaluations"])
        # The generation cap is not silently imposed on inverse support.
        bigger = self.generate(source_counts=[5], count_prior=None, samples_per_count=3)
        self.assertEqual(bigger["sources"].shape, (3, 5, 3))

    def test_portable_record_and_incomplete_or_corrupt_record_rejection(self):
        record = self.generate()
        path = save_teacher_record(record, Path(self.temp.name) / "teacher.npz")
        with np.load(path, allow_pickle=False) as archive:
            self.assertNotEqual(archive["sources"].dtype, object)
        loaded = load_teacher_record(path)
        np.testing.assert_array_equal(loaded["sources"], record["sources"])
        np.testing.assert_array_equal(loaded["teacher_prob"], record["teacher_prob"])
        self.assertEqual(loaded["metadata"], record["metadata"])
        loaded["position_grid_index"][0, 1] += 1
        with self.assertRaisesRegex(ValueError, "grid ordering"):
            save_teacher_record(loaded, path)
        record["teacher_prob"][0] = .9
        with self.assertRaises(ValueError):
            save_teacher_record(record, path)
        record = self.generate()
        record["evaluated"][0] = False
        with self.assertRaises(IncompleteSearchError):
            save_teacher_record(record, path)

    def test_projection_preserves_colocated_count_intensity_and_joint_cost(self):
        record = self.generate()
        a, b = record["position_xy"][[0, 1]]
        record.update(sources=np.array([[list(a) + [.2], list(a) + [.5]],
                                       [list(b) + [.8], [0, 0, 0]]]),
                      source_counts=np.array([2, 1]), teacher_prob=np.array([.3, .7]),
                      valid=np.array([True, True]), physical_cost=np.array([.4, .1]))
        maps = source_space_projections(record)
        ra, ca = record["position_grid_index"][0]
        rb, cb = record["position_grid_index"][1]
        self.assertAlmostEqual(maps["presence"][ra, ca], .3)
        self.assertAlmostEqual(maps["expected_count"][ra, ca], .6)
        self.assertAlmostEqual(maps["expected_intensity"][ra, ca], .21)
        self.assertAlmostEqual(np.nansum(maps["expected_count"]), 1.3)
        self.assertAlmostEqual(np.nansum(maps["expected_intensity"]), .77)
        self.assertAlmostEqual(maps["minimum_joint_cost"][rb, cb], .1)

    def test_references_are_separate_hard_checks_not_teacher_support(self):
        record = self.generate()
        before = record["sources"].copy()
        report = inspect_teacher(self.ds, record, low_cost_threshold=.1, backend="reference")
        self.assertEqual(report["references_checked"], 6)
        self.assertLessEqual(report["max_reference_cost"], REFERENCE_ATOL)
        np.testing.assert_array_equal(record["sources"], before)
        self.assertFalse(report["reference_solutions_used_as_targets"])
        self.assertFalse(report["position_coverage_is_joint_coverage"])

    def test_empty_geometry_support_fails_explicitly(self):
        grid = uniform_source_grid(CandidateDomain((0, 0, 32), 64),
                                   MockEvaluator(positions=lambda xy: np.zeros(len(xy), dtype=bool)), spacing=16)
        with self.assertRaises(NoCandidatePositionsError):
            sample_joint_support(grid, source_counts=[2], samples_per_count=8, seed=0)


if __name__ == "__main__":
    unittest.main()
