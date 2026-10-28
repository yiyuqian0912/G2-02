#!/usr/bin/env python3
"""Exact finite-dictionary checks and seeded sampling for the decoupling proof.

Standard library only. Dictionaries are declared admissible sampled masks;
this script does not infer physical realizability from arbitrary binary arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import random

SEED = 20261027
A = (1, 1, 0, 0)
B = (0, 1, 1, 0)
MASKS = (A, A, B, B, (1, 0, 1, 0), (0, 0, 0, 1), (0, 0, 0, 0), (1, 1, 1, 0))
MU = (Fraction(1, 8),) * len(MASKS)


def add_masks(masks):
    masks = tuple(tuple(mask) for mask in masks)
    if not masks or not masks[0] or any(len(m) != len(masks[0]) for m in masks):
        raise ValueError("aligned nonempty masks required")
    return tuple(sum(values) for values in zip(*masks))


def mae(a, b):
    if not a or len(a) != len(b):
        raise ValueError("aligned nonempty arrays required")
    return sum(abs(x-y) for x, y in zip(a, b)) / len(a)


def supports(masks, layers):
    return tuple(frozenset(i for i, mask in enumerate(masks) if tuple(mask) == tuple(layer))
                 for layer in layers)


def validate_decomposition(response, masks, layers):
    if not masks or not layers or not response:
        raise ValueError("nonempty dictionary, layers, and response required")
    if any(len(mask) != len(response) or any(v not in (0, 1) for v in mask) for mask in masks):
        raise ValueError("candidate masks must be aligned and binary")
    if any(len(layer) != len(response) or not any(layer)
           or any(v not in (0, 1) for v in layer) for layer in layers):
        raise ValueError("layers must be aligned, nonzero unit-source masks")
    if tuple(response) != add_masks(layers):
        raise ValueError("layer sum does not equal the observation")
    classes = supports(masks, layers)
    if any(not c for c in classes):
        raise ValueError("a layer has no exact candidate-grid support")
    return classes


def validate_measure(mu):
    if not mu or any(p < 0 for p in mu) or sum(mu) != 1:
        raise ValueError("normalized nonnegative proposal masses required")


def class_groups(classes):
    counts = Counter(frozenset(c) for c in classes)
    groups = tuple(counts)
    if any(a & b for a, b in itertools.combinations(groups, 2)):
        raise ValueError("different exact mask classes must have disjoint supports")
    return groups, counts


def support_mass(c, mu):
    return sum((mu[i] for i in c), Fraction(0))


def family_hit(indices, classes, unordered=False):
    if len(indices) != len(classes):
        return False
    if not unordered:
        return all(i in c for i, c in zip(indices, classes))
    groups, counts = class_groups(classes)
    actual = Counter()
    for index in indices:
        matched = next((c for c in groups if index in c), None)
        if matched is None:
            return False
        actual[matched] += 1
    return actual == counts


def joint_family_probability(classes, mu, unordered=False):
    validate_measure(mu)
    if not unordered:
        return math.prod(support_mass(c, mu) for c in classes)
    groups, counts = class_groups(classes)
    multiplier = math.factorial(len(classes))
    for count in counts.values():
        multiplier //= math.factorial(count)
    return multiplier * math.prod(support_mass(c, mu)**counts[c] for c in groups)


def enumerate_joint_probability(classes, mu, unordered=False):
    return sum((math.prod(mu[i] for i in indices)
                for indices in itertools.product(range(len(mu)), repeat=len(classes))
                if family_hit(indices, classes, unordered)), Fraction(0))


def independent_budget_success(classes, mu, budgets):
    if len(classes) != len(budgets) or any(b < 0 for b in budgets):
        raise ValueError("one nonnegative budget per layer required")
    return math.prod(1-(1-support_mass(c, mu))**b for c, b in zip(classes, budgets))


def confidence_trials(probability, failure_probability):
    if not 0 < probability <= 1 or not 0 < failure_probability < 1:
        raise ValueError("positive hit probability and a failure target in (0,1) required")
    if probability == 1:
        return 1
    return math.ceil(math.log(failure_probability) / math.log1p(-float(probability)))


def shared_pool_probability(classes, mu, budget):
    """Inclusion-exclusion for iid shared draws; accepts budget zero too."""
    if budget < 0:
        raise ValueError("nonnegative pool size required")
    validate_measure(mu)
    groups, _ = class_groups(classes)
    total = Fraction(0)
    for size in range(len(groups)+1):
        for subset in itertools.combinations(groups, size):
            mass = sum((support_mass(c, mu) for c in subset), Fraction(0))
            total += (-1)**size * (1-mass)**budget
    return total


def pool_covers(pool, classes):
    return all(any(i in c for i in pool) for c in classes)


def enumerate_shared_probability(classes, mu, budget):
    return sum((math.prod(mu[i] for i in pool)
                for pool in itertools.product(range(len(mu)), repeat=budget)
                if pool_covers(pool, classes)), Fraction(0))


def choose(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def finite_pool_probability(classes, grid_size, budget):
    if not 0 <= budget <= grid_size:
        raise ValueError("pool size must be between zero and grid size")
    groups, _ = class_groups(classes)
    numerator = 0
    for size in range(len(groups)+1):
        for subset in itertools.combinations(groups, size):
            excluded = sum(len(c) for c in subset)
            numerator += (-1)**size * choose(grid_size-excluded, budget)
    return Fraction(numerator, choose(grid_size, budget))


def error_budget(response, layers, sampled):
    if len(layers) != len(sampled):
        raise ValueError("one sampled mask per target layer required")
    delta = mae(response, add_masks(layers))
    errors = [mae(layer, mask) for layer, mask in zip(layers, sampled)]
    return {"reconstruction_mae": mae(response, add_masks(sampled)),
            "decomposition_defect": delta, "layer_errors": errors,
            "upper_bound": delta+sum(errors)}


def gibbs(costs, mu, temperature):
    if temperature <= 0 or len(costs) != len(mu):
        raise ValueError("aligned costs and positive temperature required")
    validate_measure(mu)
    active = [c for c, mass in zip(costs, mu) if mass > 0]
    if not active or not all(math.isfinite(c) and c >= 0 for c in active):
        raise ValueError("active costs must be finite and nonnegative")
    offset = min(active)
    weights = [float(mass)*math.exp(-(c-offset)/temperature) if mass > 0 else 0.
               for c, mass in zip(costs, mu)]
    normalizer = sum(weights)
    return [w/normalizer for w in weights]


def shared_pass(masks, layers):
    """Instrument the candidate-outer-loop approach without rendering physics."""
    renders = scores = pixel_comparisons = 0
    costs = [[] for _ in layers]
    for mask in masks:
        renders += 1  # One render event for this declared position.
        for t, layer in enumerate(layers):
            costs[t].append(mae(mask, layer))
            scores += 1
            pixel_comparisons += len(mask)
    return costs, {"unique_position_render_events": renders,
                   "layer_score_evaluations": scores, "score_pixel_comparisons": pixel_comparisons,
                   "joint_tuple_evaluations": 0}


def rejection_experiment(classes, mu, trials, rng):
    weights = tuple(map(float, mu))
    draws = 0
    for _ in range(trials):
        for c in classes:
            if support_mass(c, mu) == 0:
                raise ValueError("rejection cannot hit a zero-mass support")
            while True:
                draws += 1
                if rng.choices(range(len(mu)), weights=weights)[0] in c:
                    break
    return draws/trials


def as_record(value):
    return {"fraction": str(value), "value": float(value)}


def run_checks(seed=SEED):
    rng = random.Random(seed)
    layers = (A, B)
    response = add_masks(layers)
    classes = validate_decomposition(response, MASKS, layers)
    report = {"seed": seed, "scope": "finite declared-mask dictionaries; not a dataset rate",
              "implementation_commit": "77c03a031b923953197184ba16cf6dd6e55c5e48"}

    combinations = list(itertools.product(*classes))
    errors = [mae(response, add_masks(MASKS[i] for i in indices)) for indices in combinations]
    report["exact_decoupling"] = {
        "candidate_positions": len(MASKS), "pixels": len(A), "layers": len(layers),
        "supports": [sorted(c) for c in classes], "overlap_pixels": [1],
        "support_tuples_checked": len(combinations), "maximum_reconstruction_mae": max(errors),
        "passed": max(errors) == 0,
    }

    joint_records = []
    for case_layers in (layers, (A, A, B)):
        case_classes = supports(MASKS, case_layers)
        for unordered in (False, True):
            formula = joint_family_probability(case_classes, MU, unordered)
            enumeration = enumerate_joint_probability(case_classes, MU, unordered)
            joint_records.append({"layer_count": len(case_layers), "unordered": unordered,
                                  "formula": as_record(formula), "enumeration": as_record(enumeration),
                                  "passed": formula == enumeration})
    report["joint_target_probabilities"] = {"cases": joint_records,
                                             "passed": all(c["passed"] for c in joint_records)}

    pool_cases = []
    for budget in (1, 2, 3):
        exact = shared_pool_probability(classes, MU, budget)
        enumerated = enumerate_shared_probability(classes, MU, budget)
        union_failure = sum((1-support_mass(c, MU))**budget for c in classes)
        wrong_independent = independent_budget_success(classes, MU, (budget, budget))
        pool_cases.append({"budget": budget, "exact_shared_success": as_record(exact),
                           "enumerated_success": as_record(enumerated),
                           "independent_stream_success_not_shared": as_record(wrong_independent),
                           "union_failure_upper_bound": float(min(1, union_failure)),
                           "passed": exact == enumerated and 1-exact <= union_failure})
    without_replacement = []
    for budget in (1, 2, 4, 8):
        pools = list(itertools.combinations(range(len(MASKS)), budget))
        enumerated = Fraction(sum(pool_covers(pool, classes) for pool in pools), len(pools))
        exact = finite_pool_probability(classes, len(MASKS), budget)
        without_replacement.append({"budget": budget, "formula": as_record(exact),
                                    "enumeration": as_record(enumerated), "passed": exact == enumerated})
    report["shared_pool_coverage"] = {"iid_cases": pool_cases,
                                      "without_replacement": without_replacement,
                                      "passed": all(c["passed"] for c in pool_cases+without_replacement)}

    checks, worst_violation, smallest_slack = 0, 0., math.inf
    for target in (response, (1, 2, 1, 1)):
        for indices in itertools.product(range(len(MASKS)), repeat=2):
            result = error_budget(target, layers, [MASKS[i] for i in indices])
            checks += 1
            worst_violation = max(worst_violation, result["reconstruction_mae"]-result["upper_bound"])
            smallest_slack = min(smallest_slack, result["upper_bound"]-result["reconstruction_mae"])
    cancellation = error_budget(response, layers, (B, A))
    report["approximate_error_budget"] = {
        "tuples_checked_including_nonzero_decomposition_defect": checks,
        "maximum_bound_violation": worst_violation,
        "swapped_layer_cancellation": cancellation,
        "passed": worst_violation <= 1e-12 and cancellation["reconstruction_mae"] == 0
                  and sum(cancellation["layer_errors"]) > 0,
    }

    temperature, threshold = .3, .25
    costs, counts = shared_pass(MASKS, layers)
    qs = [gibbs(row, MU, temperature) for row in costs]
    tails = [sum(q for q, c in zip(qt, row) if c > threshold) for qt, row in zip(qs, costs)]
    tail_bounds = [min(1., float((1-support_mass(c, MU))/support_mass(c, MU))*
                       math.exp(-threshold/temperature)) for c in classes]
    means = [sum(q*c for q, c in zip(qt, row)) for qt, row in zip(qs, costs)]
    expected_global, global_tail, global_exact = 0., 0., 0.
    for i, j in itertools.product(range(len(MASKS)), repeat=2):
        mass = qs[0][i]*qs[1][j]
        error = mae(response, add_masks((MASKS[i], MASKS[j])))
        expected_global += mass*error
        global_tail += mass*(error > 2*threshold)
        global_exact += mass*(error == 0)
    off_support = [sum(q for i, q in enumerate(qt) if i not in c) for qt, c in zip(qs, classes)]
    report["finite_temperature"] = {
        "temperature": temperature, "threshold_per_layer": threshold,
        "off_exact_support_mass": off_support, "layer_mean_errors": means,
        "layer_tail_probabilities": tails, "proved_layer_tail_bounds": tail_bounds,
        "expected_global_mae": expected_global, "expected_mae_upper_bound": sum(means),
        "global_error_tail": global_tail, "union_tail_upper_bound": min(1., sum(tails)),
        "global_exact_probability_by_enumeration": global_exact,
        "all_selected_layers_exact_probability": math.prod(1-x for x in off_support),
        "passed": all(x > 0 for x in off_support)
                  and all(t <= b+1e-12 for t, b in zip(tails, tail_bounds))
                  and expected_global <= sum(means)+1e-12 and global_tail <= sum(tails)+1e-12,
    }

    trials = 20000
    expected_draws = sum(1/support_mass(c, MU) for c in classes)
    measured_draws = rejection_experiment(classes, MU, trials, rng)
    joint_trials = 40000
    hits = {False: 0, True: 0}
    for _ in range(joint_trials):
        indices = rng.choices(range(len(MU)), weights=tuple(map(float, MU)), k=len(layers))
        for unordered in hits:
            hits[unordered] += family_hit(indices, classes, unordered)
    mc_cases = [{"unordered": flag, "empirical": hit/joint_trials,
                 "exact": float(joint_family_probability(classes, MU, flag))}
                for flag, hit in hits.items()]
    confidence_budget = confidence_trials(joint_family_probability(classes, MU), .05)
    report["seeded_sampling"] = {
        "separate_rejection_runs": trials, "expected_position_draws": float(expected_draws),
        "mean_position_draws": measured_draws, "joint_trials": joint_trials,
        "joint_cases": mc_cases, "labeled_trials_for_95_percent_success": confidence_budget,
        "passed": abs(measured_draws-float(expected_draws)) < .15
                  and all(abs(c["empirical"]-c["exact"]) < .01 for c in mc_cases),
    }

    report["operation_accounting"] = {
        "shared_pass": counts,
        "cached_joint_baseline": {"unique_position_render_events": len(MASKS),
                                  "labeled_tuple_evaluations": len(MASKS)**len(layers),
                                  "unordered_multisets": math.comb(len(MASKS)+len(layers)-1, len(layers))},
        "peeling_cost": "T_peel is excluded from these synthetic counters and remains explicit in the theorem",
        "passed": counts["unique_position_render_events"] == 8 and counts["layer_score_evaluations"] == 16
                  and counts["score_pixel_comparisons"] == 64,
    }

    import verify as spatial
    local_a, local_b = (1, 1, 1, 1, 1), (0, 1, 0, 0, 0)
    local_response = spatial.sum_masks((local_a, local_b))
    local = spatial.one_hop_subtract(local_response, local_a, 0, 5)
    mixed = spatial.ideal_peel(((1, 0), (0, 1)), (1, 1), 2)
    rejected = {}
    for name, target, test_layers in (
            ("uncovered_on_grid_binary_layer", (1, 0, 0, 0), ((1, 0, 0, 0),)),
            ("nonunit_effective_layer", (2, 2, 0, 0), ((2, 2, 0, 0),)),
            ("incorrect_layer_sum", (1, 1, 1, 0), layers)):
        try:
            validate_decomposition(target, MASKS, test_layers)
        except ValueError as exc:
            rejected[name] = str(exc)
    # In a blocker-free scene every exterior source illuminates the entire
    # window. These nonconstant binary layers cannot be physical source masks,
    # even though their sum is a valid all-ones observation.
    try:
        validate_decomposition((1, 1, 1, 1), ((1, 1, 1, 1),)*2,
                               ((1, 0, 0, 0), (0, 1, 1, 1)))
    except ValueError as exc:
        rejected["physically_impossible_layers_in_blocker_free_scene"] = str(exc)
    point_layers = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    peaks = [max(row[i] for row in point_layers) for i in range(2)]
    summary = tuple(p/sum(peaks) for p in peaks)
    summary_success = enumerate_joint_probability((frozenset({0}), frozenset({1})), summary, True)
    report["failure_cases"] = {
        "rejected_decompositions": rejected, "local_subtraction_residual": local,
        "correct_remaining_mask": local_b, "mixed_anchor_status": mixed["status"],
        "normalized_max_then_two_iid_draws_exact_probability": float(summary_success),
        "one_draw_per_original_exact_layer_probability": 1.,
        "passed": len(rejected) == 4 and local != local_b and mixed["status"] == "no_covering_mask"
                  and summary_success == Fraction(1, 2),
    }
    report["all_passed"] = all(group["passed"] for group in report.values()
                               if isinstance(group, dict) and "passed" in group)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("residual_sampling_results.json"))
    args = parser.parse_args()
    report = run_checks(args.seed)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"all_passed": report["all_passed"], "output": str(args.output.resolve()),
                      "verification_groups": sum(isinstance(v, dict) and "passed" in v for v in report.values()),
                      "exact_reconstruction_mae": report["exact_decoupling"]["maximum_reconstruction_mae"],
                      "expected_separate_draws": report["seeded_sampling"]["expected_position_draws"],
                      "measured_separate_draws": report["seeded_sampling"]["mean_position_draws"]}, indent=2))
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
