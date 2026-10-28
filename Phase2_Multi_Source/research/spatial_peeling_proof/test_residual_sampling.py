"""Independent finite-enumeration and boundary-case checks for decoupling."""
from fractions import Fraction
import itertools
import math
import unittest

import verify_residual_sampling as v


class ExactDecouplingTests(unittest.TestCase):
    def test_all_overlapping_layer_combinations_reconstruct(self):
        layers = (v.A, v.B)
        target = v.add_masks(layers)
        classes = v.validate_decomposition(target, v.MASKS, layers)
        self.assertEqual(target[1], 2)
        self.assertEqual(len(list(itertools.product(*classes))), 4)
        for indices in itertools.product(*classes):
            self.assertEqual(v.add_masks(v.MASKS[i] for i in indices), target)

    def test_unphysical_or_uncovered_layers_are_not_validated(self):
        with self.assertRaisesRegex(ValueError, "no exact"):
            v.validate_decomposition((1, 0, 0, 0), v.MASKS, ((1, 0, 0, 0),))
        with self.assertRaisesRegex(ValueError, "unit-source"):
            v.validate_decomposition((2, 2, 0, 0), v.MASKS, ((2, 2, 0, 0),))
        with self.assertRaisesRegex(ValueError, "sum"):
            v.validate_decomposition((1, 1, 1, 0), v.MASKS, (v.A, v.B))
        # Physical no-blocker model: every exterior source mask is all ones.
        with self.assertRaisesRegex(ValueError, "no exact"):
            v.validate_decomposition((1, 1, 1, 1), ((1, 1, 1, 1),)*2,
                                     ((1, 0, 0, 0), (0, 1, 1, 1)))

    def test_decoupled_family_can_exclude_other_compatible_decompositions(self):
        masks = ((1, 1, 0), (0, 0, 1), (1, 0, 0), (0, 1, 1))
        target = (1, 1, 1)
        classes = v.validate_decomposition(target, masks, masks[:2])
        self.assertEqual(v.add_masks(masks[2:]), target)
        self.assertFalse(v.family_hit((2, 3), classes, unordered=True))

    def test_observed_zero_screen_preserves_exact_supports(self):
        target = v.add_masks((v.A, v.B))
        classes = v.supports(v.MASKS, (v.A, v.B))
        retained = {i for i, mask in enumerate(v.MASKS)
                    if not any(value and response == 0 for value, response in zip(mask, target))}
        self.assertTrue(all(c <= retained for c in classes))

    def test_repeated_masks_are_valid_under_with_replacement_model(self):
        classes = v.validate_decomposition(v.add_masks((v.A, v.A)), v.MASKS, (v.A, v.A))
        self.assertTrue(v.family_hit((0, 0), classes))


class SamplingProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.classes = v.supports(v.MASKS, (v.A, v.B))

    def test_labeled_and_unordered_against_enumeration(self):
        self.assertEqual(v.joint_family_probability(self.classes, v.MU), Fraction(1, 16))
        self.assertEqual(v.joint_family_probability(self.classes, v.MU, True), Fraction(1, 8))
        for layers in ((v.A, v.B), (v.A, v.A, v.B)):
            classes = v.supports(v.MASKS, layers)
            for unordered in (False, True):
                self.assertEqual(v.joint_family_probability(classes, v.MU, unordered),
                                 v.enumerate_joint_probability(classes, v.MU, unordered))

    def test_nonuniform_proposals_and_duplicate_factor(self):
        mu = (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6))
        classes = (frozenset({0}), frozenset({0}), frozenset({1}))
        self.assertEqual(v.joint_family_probability(classes, mu, True), Fraction(1, 4))
        self.assertEqual(v.enumerate_joint_probability(classes, mu, True), Fraction(1, 4))

    def test_confidence_budget_is_minimal_and_handles_probability_one(self):
        p, eta = Fraction(1, 16), .05
        n = v.confidence_trials(p, eta)
        self.assertLessEqual(float(1-p)**n, eta)
        self.assertGreater(float(1-p)**(n-1), eta)
        self.assertEqual(v.confidence_trials(1, eta), 1)
        with self.assertRaises(ValueError):
            v.confidence_trials(0, eta)

    def test_shared_pool_is_not_independent_streams(self):
        self.assertEqual(v.shared_pool_probability(self.classes, v.MU, 1), 0)
        self.assertEqual(v.independent_budget_success(self.classes, v.MU, (1, 1)), Fraction(1, 16))
        for budget in (0, 1, 2, 3):
            self.assertEqual(v.shared_pool_probability(self.classes, v.MU, budget),
                             v.enumerate_shared_probability(self.classes, v.MU, budget))

    def test_shared_pool_failure_union_bound(self):
        for budget in (1, 2, 5, 20):
            failure = 1-v.shared_pool_probability(self.classes, v.MU, budget)
            bound = sum((1-v.support_mass(c, v.MU))**budget for c in self.classes)
            self.assertLessEqual(failure, bound)

    def test_without_replacement_formula_and_complete_grid(self):
        for budget in range(len(v.MASKS)+1):
            pools = list(itertools.combinations(range(len(v.MASKS)), budget))
            empirical = Fraction(sum(v.pool_covers(p, self.classes) for p in pools), len(pools))
            self.assertEqual(v.finite_pool_probability(self.classes, len(v.MASKS), budget), empirical)
        self.assertEqual(v.finite_pool_probability(self.classes, len(v.MASKS), len(v.MASKS)), 1)

    def test_zero_mass_support_cannot_be_hit(self):
        mu = (Fraction(0), Fraction(1))
        classes = (frozenset({0}),)
        self.assertEqual(v.joint_family_probability(classes, mu), 0)
        self.assertEqual(v.shared_pool_probability(classes, mu, 10), 0)
        # Additive rejection effort still has quadratic growth for equally
        # likely disjoint classes under one common proposal law.
        for m in range(1, 6):
            p = Fraction(1, m)
            self.assertEqual(sum(1/p for _ in range(m)), m*m)

    def test_normalized_max_is_not_a_joint_sampling_law(self):
        # Two point masses combine to (1/2,1/2); iid draws omit a layer half the time.
        mu = (Fraction(1, 2), Fraction(1, 2))
        classes = (frozenset({0}), frozenset({1}))
        self.assertEqual(v.enumerate_joint_probability(classes, mu, True), Fraction(1, 2))
        self.assertEqual(sum(1 for _ in itertools.product(*classes)), 1)


class ApproximationAndCostTests(unittest.TestCase):
    def test_error_bound_for_every_tuple_and_nonzero_defect(self):
        for target in (v.add_masks((v.A, v.B)), (1, 2, 1, 1)):
            for indices in itertools.product(range(len(v.MASKS)), repeat=2):
                result = v.error_budget(target, (v.A, v.B), [v.MASKS[i] for i in indices])
                self.assertLessEqual(result["reconstruction_mae"], result["upper_bound"]+1e-12)

    def test_dependent_layer_errors_can_cancel(self):
        target = v.add_masks((v.A, v.B))
        # Deterministically coupled swapped layers have positive layer errors and zero final error.
        result = v.error_budget(target, (v.A, v.B), (v.B, v.A))
        self.assertGreater(sum(result["layer_errors"]), 0)
        self.assertEqual(result["reconstruction_mae"], 0)

    def test_softmax_leakage_and_tail_bound(self):
        costs = [v.mae(v.A, mask) for mask in v.MASKS]
        c = v.supports(v.MASKS, (v.A,))[0]
        p = float(v.support_mass(c, v.MU))
        for tau in (.05, .2, 1.):
            q = v.gibbs(costs, v.MU, tau)
            self.assertAlmostEqual(sum(q), 1.)
            self.assertGreater(sum(mass for i, mass in enumerate(q) if i not in c), 0)
            for epsilon in (.1, .3, .6):
                tail = sum(mass for mass, error in zip(q, costs) if error > epsilon)
                self.assertLessEqual(tail, min(1., (1-p)/p*math.exp(-epsilon/tau))+1e-12)

    def test_weighted_cost_to_raw_mae_conversion(self):
        weights, alpha, beta = (1., 4., 2., 1.), 1.7, .2
        for mask in v.MASKS:
            weighted = sum(w*abs(x-y) for w, x, y in zip(weights, mask, v.A))/sum(weights)
            nonnegative_edge_cost = .35
            total = alpha*weighted+beta*nonnegative_edge_cost
            bound = sum(weights)/(len(weights)*min(weights)*alpha)*total
            self.assertLessEqual(v.mae(mask, v.A), bound)

    def test_full_window_zero_pixels_remain_in_mae_denominator(self):
        self.assertEqual(v.mae((1, 0), (0, 0)), .5)
        self.assertEqual(v.mae((1, 0, 0, 0), (0, 0, 0, 0)), .25)

    def test_shared_render_and_layer_score_counters(self):
        costs, counts = v.shared_pass(v.MASKS, (v.A, v.B))
        self.assertEqual(counts["unique_position_render_events"], 8)
        self.assertEqual(counts["layer_score_evaluations"], 16)
        self.assertEqual(counts["score_pixel_comparisons"], 64)
        self.assertEqual(counts["joint_tuple_evaluations"], 0)
        self.assertEqual(len(list(itertools.product(range(8), repeat=2))), 64)
        self.assertEqual(len(costs), 2)


if __name__ == "__main__":
    unittest.main()
