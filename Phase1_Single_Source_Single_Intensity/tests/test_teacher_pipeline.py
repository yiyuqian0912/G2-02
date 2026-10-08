"""Focused domain, ordering, completion, and Gibbs-target regression checks."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rind_phase1.diagnostics import zero_components
from rind_phase1.physics import CandidateEvaluation, PhysicalEvaluator, response_disagreement
from rind_phase1.search import CandidateDomain, deduplicate_candidates, uniform_grid, uniform_search
from rind_phase1.teacher import (IncompleteSearchError, NoValidCandidatesError,
                                 generate_uniform_teacher, load_teacher_record,
                                 save_teacher_record, target_from_search, teacher_probabilities)


class SyntheticEvaluator:
    def __init__(self):
        self.cache = {}

    def evaluate(self, xy):
        key = tuple(xy)
        if key in self.cache:
            valid, cost = self.cache[key]
            return CandidateEvaluation(valid, cost, cache_hit=True)
        valid = xy[0] < 3
        cost = float(xy[0] + 8 * xy[1]) if valid else np.inf
        self.cache[key] = (valid, cost)
        return CandidateEvaluation(valid, cost, int(valid))


class LabelFreeDataset:
    """Has geometry/rendering but no true source or reference solutions."""
    root = Path('/tmp/rind-label-free-fixture')
    manifest = {"global_size": 8, "schema_version": 3, "fixture": "no source labels"}

    def get_observation(self, scene_id, view_id):
        return {"scene_id": scene_id, "view_id": view_id, "window": np.array([0, 0, 2]),
                "response": np.zeros((2, 2), dtype=np.float32), "obstacle": np.zeros((2, 2), dtype=bool)}

    def get_scene(self, scene_id):
        return {"obstacle_types": np.empty(0, dtype=np.uint8), "obstacle_params": np.empty((0, 32))}

    def rerender(self, scene_id, window, xy, **kwargs):
        return np.full((2, 2), int(xy[0] >= 3), dtype=np.float32)

    def get_candidates(self, *args):
        raise AssertionError("Reference witnesses must not construct the teacher")


class TeacherPipelineTests(unittest.TestCase):
    def setUp(self):
        self.parent = CandidateDomain((0, 0, 2), 8, True)

    def test_parent_is_aligned_twice_window_size(self):
        self.assertEqual(CandidateDomain((6, 2, 2), 8, True).bounds, (4, 0, 8, 4))
        self.assertEqual(CandidateDomain((2, 0, 2), 8, True).bounds, (0, 0, 4, 4))

    def test_prior_and_ablation_have_distinct_domains(self):
        wide = CandidateDomain((0, 0, 2), 8)
        self.assertTrue(wide.contains([7, 7]))
        self.assertFalse(self.parent.contains([7, 7]))
        self.assertNotEqual(wide.name, self.parent.name)

    def test_window_exclusion_and_split_edges(self):
        self.assertFalse(self.parent.contains([0.5, 0.5]))
        self.assertTrue(self.parent.contains([2, 0.5]))
        self.assertFalse(self.parent.contains([4, 0.5]))
        self.assertFalse(self.parent.contains([-1, 2]))
        self.assertFalse(self.parent.contains([np.nan, 2]))

    def test_terminal_world_boundary_ownership(self):
        domain = CandidateDomain((6, 6, 2), 8, True)
        self.assertTrue(domain.contains([8, 4.5]))
        self.assertFalse(domain.contains([8, 8]))

    def test_non_quadtree_windows_rejected_for_parent_prior(self):
        with self.assertRaises(ValueError):
            CandidateDomain((1, 0, 2), 8, True)
        with self.assertRaises(ValueError):
            CandidateDomain((0, 0, 8), 8, True)

    def test_deterministic_grid_and_row_major_order(self):
        a, b = uniform_grid(self.parent, 1), uniform_grid(self.parent, 1)
        np.testing.assert_array_equal(a.candidate_xy, b.candidate_xy)
        np.testing.assert_array_equal(a.candidate_xy[:4], [[2.5, .5], [3.5, .5], [2.5, 1.5], [3.5, 1.5]])
        self.assertEqual(len(a.candidate_xy), 12)
        self.assertTrue(self.parent.contains(a.candidate_xy).all())
        np.testing.assert_array_equal(deduplicate_candidates(a.candidate_xy), a.candidate_xy)

    def test_prior_support_is_subset_of_same_global_lattice(self):
        parent = uniform_grid(self.parent, 1.3)
        wide = uniform_grid(CandidateDomain((0, 0, 2), 8, False), 1.3)
        np.testing.assert_array_equal(parent.candidate_xy, wide.candidate_xy[self.parent.contains(wide.candidate_xy)])

    def test_deduplication_preserves_first_occurrence(self):
        np.testing.assert_array_equal(deduplicate_candidates([[3, 2], [1, 4], [3, 2], [0, 1], [1, 4]]),
                                      [[3, 2], [1, 4], [0, 1]])

    def test_invalid_spacing_rejected(self):
        for spacing in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                uniform_grid(self.parent, spacing)

    def test_costs_and_validity_follow_candidate_order(self):
        result = uniform_search(self.parent, SyntheticEvaluator(), spacing=1)
        self.assertTrue(result.complete)
        for xy, valid, cost in zip(result.grid.candidate_xy, result.valid, result.physical_cost):
            self.assertEqual(valid, xy[0] < 3)
            self.assertEqual(cost, float(xy[0] + 8 * xy[1]) if valid else np.inf)
        self.assertEqual(result.num_physics_evaluations, int(result.valid.sum()))

    def test_cached_evaluations_do_not_count_as_new_physics(self):
        evaluator = SyntheticEvaluator()
        first = uniform_search(self.parent, evaluator, spacing=1)
        second = uniform_search(self.parent, evaluator, spacing=1)
        np.testing.assert_array_equal(first.physical_cost, second.physical_cost)
        self.assertEqual(second.num_physics_evaluations, 0)
        self.assertEqual(second.num_cache_hits, len(second.grid.candidate_xy))

    def test_insufficient_budget_is_incomplete_and_not_normalized(self):
        result = uniform_search(self.parent, SyntheticEvaluator(), spacing=1, max_evaluations=1)
        self.assertFalse(result.complete)
        self.assertEqual(result.num_physics_evaluations, 1)
        self.assertTrue(np.isnan(result.physical_cost[~result.evaluated]).all())
        with self.assertRaises(IncompleteSearchError):
            target_from_search(result, temperature=.1)

    def test_teacher_preserves_order_and_excludes_invalid_entries(self):
        q = teacher_probabilities([2, np.nan, 0, 1], [True, False, True, True], temperature=.5)
        self.assertEqual(q[1], 0)
        self.assertAlmostEqual(q.sum(), 1)
        self.assertGreater(q[2], q[3])
        self.assertGreater(q[3], q[0])

    def test_equal_modes_remain_soft_and_unweighted(self):
        q = teacher_probabilities([0, 0, 0], [True] * 3, temperature=.001)
        np.testing.assert_allclose(q, np.full(3, 1 / 3))

    def test_small_temperature_and_large_costs_are_stable(self):
        q = teacher_probabilities([0, 1e100, 1e300, np.inf], [True, True, True, False], temperature=1e-300)
        np.testing.assert_array_equal(q, [1, 0, 0, 0])
        shifted = teacher_probabilities([1000, 1001, 1002], [True] * 3, temperature=.01)
        original = teacher_probabilities([0, 1, 2], [True] * 3, temperature=.01)
        np.testing.assert_allclose(shifted, original)

    def test_empty_valid_set_and_empty_grid_fail_explicitly(self):
        for costs, valid in (([np.inf], [False]), ([], np.empty(0, dtype=bool))):
            with self.assertRaises(NoValidCandidatesError):
                teacher_probabilities(costs, valid, temperature=.1)
        result = uniform_search(self.parent, SyntheticEvaluator(), spacing=100)
        with self.assertRaises(NoValidCandidatesError):
            target_from_search(result, temperature=.1)

    def test_invalid_valid_cost_or_temperature_rejected(self):
        for cost in (np.nan, np.inf, -1):
            with self.assertRaises(ValueError):
                teacher_probabilities([cost], [True], temperature=.1)
        for tau in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                teacher_probabilities([0], [True], temperature=tau)

    def test_missing_evaluated_entries_rejected_even_if_complete_flag_true(self):
        result = uniform_search(self.parent, SyntheticEvaluator(), spacing=1)
        with self.assertRaises(IncompleteSearchError):
            target_from_search(replace(result, evaluated=np.zeros_like(result.evaluated)), temperature=.1)

    def test_binary_cost_is_normalized_disagreement(self):
        self.assertEqual(response_disagreement([[0, 1], [0, 1]], [[0, 0], [0, 1]]), .25)
        self.assertEqual(response_disagreement([[1]], [[1]]), 0)
        with self.assertRaises(ValueError):
            response_disagreement([[0, 0]], [[0]])

    def test_physical_cache_and_invalid_geometry(self):
        evaluator = PhysicalEvaluator(LabelFreeDataset(), 0, 0)
        first, second = evaluator.evaluate([2.5, .5]), evaluator.evaluate([2.5, .5])
        self.assertEqual(first.physical_cost, 0)
        self.assertEqual(first.num_physics_evaluations, 1)
        self.assertEqual(second.num_physics_evaluations, 0)
        self.assertTrue(second.cache_hit)
        invalid = evaluator.evaluate([-1, 0])
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.num_physics_evaluations, 0)

    def test_teacher_can_be_built_without_any_source_labels_or_references(self):
        record = generate_uniform_teacher(LabelFreeDataset(), 0, 0, spacing=1, temperature=.1)
        expected = uniform_grid(CandidateDomain((0, 0, 2), 8), 1)
        np.testing.assert_array_equal(record['candidate_xy'], expected.candidate_xy)
        self.assertEqual(len(record['candidate_xy']), 60)
        self.assertTrue(np.any(np.all(record['candidate_xy'] == [7.5, 7.5], axis=1)))
        self.assertEqual(record['metadata']['settings']['support']['candidate_domain'], 'world_minus_window')
        self.assertFalse(record['metadata']['reference_solutions_used_as_targets'])
        self.assertGreater(np.count_nonzero(record['teacher_prob'] > 0), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = save_teacher_record(record, Path(directory) / 'teacher.npz')
            loaded = load_teacher_record(path)
            np.testing.assert_array_equal(record['candidate_xy'], loaded['candidate_xy'])
            np.testing.assert_array_equal(record['teacher_prob'], loaded['teacher_prob'])
            self.assertEqual(record['metadata'], loaded['metadata'])

    def test_sampled_components_have_declared_four_neighbor_connectivity(self):
        grid = uniform_grid(self.parent, 1)
        mask = np.zeros(grid.shape, dtype=bool)
        mask[0, 2] = mask[1, 3] = True
        self.assertEqual(len(zero_components(mask, grid)), 2)
        mask[0, 3] = True
        components = zero_components(mask, grid)
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]['sample_count'], 3)
        self.assertEqual(components[0]['approximate_area'], 3)


if __name__ == '__main__':
    unittest.main()
