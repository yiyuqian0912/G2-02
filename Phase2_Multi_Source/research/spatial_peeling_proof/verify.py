#!/usr/bin/env python3
"""Reproducible diagnostics for PROOF.md; standard library only.

These experiments verify examples and numerical identities. They are not a
substitute for the proofs or an estimate of the real dataset's failure rate.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

SEED = 20261027
RADIUS = 1 / 8
ETA = 1e-4
BLOCKERS = ((-2., -7/4, -9/4, -2.), (-2., -7/4, -7/2, -13/4))
NOMINAL_SOURCES = ((-3., -3.), (-3., -6.))


def strip_probability(d: float, h: float) -> float:
    if h <= 0:
        raise ValueError("pixel spacing must be positive")
    return max(0., 1. - abs(d) / h)


def uniform_gap_probability(h: float, half_width: float) -> float:
    if h <= 0 or half_width <= 0:
        raise ValueError("spacing and distribution half-width must be positive")
    return h / (2 * half_width) if h <= half_width else 1 - half_width / (2 * h)


def grid_indices(lo: float, hi: float, offset: float, h: float) -> range:
    """Indices strictly inside an open interval."""
    return range(math.floor((lo-offset)/h)+1, math.ceil((hi-offset)/h))


def strip_bridge(d: float, h: float, offset: float) -> bool:
    """Detect a genuine adjacent singleton-label swap on a sampled slice."""
    lo, hi = min(0., d)-2*h, max(0., d)+2*h
    labels = []
    for k in grid_indices(lo, hi, offset, h):
        x = offset + k*h
        labels.append(int(x < 0) | (int(x > d) << 1))
    return any(a != b and a.bit_count() == b.bit_count()
               for a, b in zip(labels, labels[1:]))


def segment_hits_rectangle(a, b, rect) -> bool:
    """Closed-segment/closed-rectangle intersection by slab clipping."""
    xlo, xhi, ylo, yhi = rect
    enter, leave = 0., 1.
    for p, q, lo, hi in ((a[0], b[0], xlo, xhi), (a[1], b[1], ylo, yhi)):
        delta = q-p
        if delta == 0:
            if not lo <= p <= hi:
                return False
            continue
        t0, t1 = sorted(((lo-p)/delta, (hi-p)/delta))
        enter, leave = max(enter, t0), min(leave, t1)
        if enter > leave:
            return False
    return True


def physical_label(point, sources=NOMINAL_SOURCES, blockers=BLOCKERS) -> int:
    return sum((1 << i) for i, source in enumerate(sources)
               if not any(segment_hits_rectangle(source, point, rect) for rect in blockers))


def crossing_lines(sources=NOMINAL_SOURCES):
    s1, s2 = sources
    m1 = (s1[1]+2)/(s1[0]+2)
    m2 = (s2[1]+7/2)/(s2[0]+7/4)
    return m1, 2*(m1-1), m2, 7/4*(m2-2)


def affine_label(point, lines=(1., 0., 2., 0.)) -> int:
    x, y = point
    m1, b1, m2, b2 = lines
    return int(y > m1*x+b1) | (int(y < m2*x+b2) << 1)


def bad_edges(label_function, h, phase, bounds, blockers=()):
    """Return bad edges; positive axis directions count each edge once.

    phase is a pair of unit-square coordinates, multiplied by h here.
    Any optional blockers are used both to remove pixels and to reject edges.
    """
    ux, uy = phase[0]*h, phase[1]*h
    xmin, xmax, ymin, ymax = bounds
    positions, labels = {}, {}
    for k in grid_indices(xmin, xmax, ux, h):
        for l in grid_indices(ymin, ymax, uy, h):
            point = (ux+k*h, uy+l*h)
            if any(segment_hits_rectangle(point, point, rect) for rect in blockers):
                continue
            positions[k, l] = point
            labels[k, l] = label_function(point)
    result = []
    for key, a in labels.items():
        k, l = key
        for direction in ((1, 0), (0, 1)):
            nxt = (k+direction[0], l+direction[1])
            if nxt not in labels:
                continue
            b = labels[nxt]
            if a == b or a.bit_count() != b.bit_count():
                continue
            if any(segment_hits_rectangle(positions[key], positions[nxt], rect)
                   for rect in blockers):
                continue
            result.append({"origin": positions[key], "end": positions[nxt],
                           "direction": direction, "labels": (a, b)})
    return result


def neighbors(index, width, height):
    x, y = index % width, index // width
    if x:
        yield index-1
    if x+1 < width:
        yield index+1
    if y:
        yield index-width
    if y+1 < height:
        yield index+width


def response_components(response, width, height):
    if width*height != len(response):
        raise ValueError("array dimensions do not match")
    unseen, components = set(range(len(response))), []
    for start in range(len(response)):
        if start not in unseen:
            continue
        unseen.remove(start)
        component, stack = [], [start]
        while stack:
            p = stack.pop()
            component.append(p)
            for q in neighbors(p, width, height):
                if q in unseen and response[q] == response[start]:
                    unseen.remove(q)
                    stack.append(q)
        components.append(tuple(sorted(component)))
    return components


def sum_masks(masks):
    return tuple(map(sum, zip(*masks)))


def admissible_masks(dictionary, residual, anchor):
    return [mask for mask in dictionary
            if all(mask[p] == 1 for p in anchor)
            and all(v <= r for v, r in zip(mask, residual))]


def ideal_peel(dictionary, response, width, height=1):
    """Implement Theorem 7's finite-dictionary procedure, with failure states.

    It does not assume the theorem's conditions hold for arbitrary inputs.
    Dictionary entries represent physical masks supplied by the caller, rather
    than masks invented from the image's connected components.
    """
    dictionary = list(dict.fromkeys(tuple(mask) for mask in dictionary))
    if width*height != len(response) or any(len(m) != len(response) for m in dictionary):
        raise ValueError("array dimensions do not match")
    if any(v not in (0, 1) for mask in dictionary for v in mask):
        raise ValueError("source masks must be binary")
    if any(not any(mask) for mask in dictionary):
        raise ValueError("invisible masks must be excluded")
    if any(r < 0 or int(r) != r for r in response):
        raise ValueError("unit-strength residual must be a nonnegative integer array")
    residual, selected, trace = tuple(response), [], []
    while any(residual):
        components = response_components(residual, width, height)
        positive = [c for c in components if residual[c[0]] > 0]
        anchor = min(positive, key=lambda c: (residual[c[0]], c[0]))
        minimum = residual[anchor[0]]
        if minimum != 1:
            return {"status": "no_singleton", "residual": residual,
                    "selected": selected, "trace": trace}
        feasible = admissible_masks(dictionary, residual, anchor)
        if len(feasible) != 1:
            return {"status": "no_covering_mask" if not feasible else "nonidentifiable_mask",
                    "residual": residual, "selected": selected, "trace": trace}
        chosen = feasible[0]
        updated = tuple(r-v for r, v in zip(residual, chosen))
        trace.append({"anchor": anchor, "mask": chosen, "before": residual, "after": updated})
        selected.append(chosen)
        residual = updated
    return {"status": "complete", "residual": residual, "selected": selected, "trace": trace}


def one_hop_subtract(response, source_mask, anchor_pixel, width, height=1):
    """Teacher-style anchor + uniformly lit adjacent-region subtraction.

    Uses known candidate visibility to choose neighbors. Fails for candidates
    that only partially illuminate an adjacent equal-response component.
    """
    components = response_components(response, width, height)
    region_at = {p: i for i, component in enumerate(components) for p in component}
    anchor_id = region_at[anchor_pixel]
    anchor = components[anchor_id]
    if not all(source_mask[p] for p in anchor):
        raise ValueError("candidate does not cover the anchor")
    adjacent = {region_at[q] for p in anchor for q in neighbors(p, width, height)
                if region_at[q] != anchor_id}
    selected = set(anchor)
    for rid in adjacent:
        component = components[rid]
        lit = sum(source_mask[p] for p in component)
        if lit not in (0, len(component)):
            raise ValueError("candidate partially illuminates a neighboring region")
        if lit:
            selected.update(component)
    return tuple(r-int(p in selected) for p, r in enumerate(response))


def run_checks(seed=SEED):
    rng = random.Random(seed)
    report = {
        "seed": seed,
        "scope": "synthetic examples and probability identities; not a dataset failure rate",
        "upstream_unit_branch_commit": "77c03a031b923953197184ba16cf6dd6e55c5e48",
    }

    phase_count = 4096
    cases = []
    for d in (0., .17, -.17, .47, -.47, .93, -.93, 1., -1., 1.6, -1.6):
        measured = sum(strip_bridge(d, 1., (j+.5)/phase_count)
                       for j in range(phase_count))/phase_count
        expected = strip_probability(d, 1.)
        cases.append({"d": d, "h": 1., "exact": expected, "phase_quadrature": measured,
                      "error": abs(measured-expected)})
    report["fixed_strip"] = {"phase_count": phase_count, "cases": cases,
                             "passed": all(c["error"] <= 2/phase_count for c in cases)}

    nquad, nrandom, half_width = 20000, 30000, 1.
    averages = []
    for h in (.1, .5, 1., 2.):
        expected = uniform_gap_probability(h, half_width)
        quadrature = sum(strip_probability(-half_width+(j+.5)*2*half_width/nquad, h)
                         for j in range(nquad))/nquad
        hits = sum(strip_bridge(rng.uniform(-half_width, half_width), h, rng.random()*h)
                   for _ in range(nrandom))
        empirical = hits/nrandom
        averages.append({"h": h, "L": half_width, "exact": expected,
                         "quadrature": quadrature, "joint_scene_phase_monte_carlo": empirical,
                         "quadrature_error": abs(quadrature-expected),
                         "monte_carlo_error": abs(empirical-expected)})
    report["uniform_signed_gap"] = {
        "quadrature_points": nquad, "joint_draws_per_spacing": nrandom, "cases": averages,
        "passed": all(c["quadrature_error"] < 1e-7 and c["monte_carlo_error"] < .015
                      for c in averages)}

    crossing = []
    nphases = 512
    for h in (RADIUS/4, RADIUS/8, RADIUS/16, RADIUS/32):
        counts = [len(bad_edges(affine_label, h, (rng.random(), rng.random()),
                                (-3*h, 3*h, -3*h, 3*h))) for _ in range(nphases)]
        crossing.append({"h": h, "phases": nphases, "false_connection_fraction":
                         sum(c > 0 for c in counts)/nphases,
                         "min_bad_edges": min(counts), "max_bad_edges": max(counts)})
    report["fixed_scene_crossing"] = {
        "continuous_shared_segment": False, "theorem_phase_probability": 1., "cases": crossing,
        "passed": all(c["false_connection_fraction"] == 1. for c in crossing)}

    source_scenes, points_per_scene, phases_per_scene = 24, 96, 32
    visibility_mismatches, perturbed_hits, perturbed_total = 0, 0, 0
    min_gap, max_gap, max_crossing_coordinate = math.inf, -math.inf, 0.
    for _ in range(source_scenes):
        sources = tuple((x+rng.uniform(-ETA, ETA), y+rng.uniform(-ETA, ETA))
                        for x, y in NOMINAL_SOURCES)
        lines = crossing_lines(sources)
        m1, b1, m2, b2 = lines
        min_gap, max_gap = min(min_gap, m2-m1), max(max_gap, m2-m1)
        cx = (b2-b1)/(m1-m2)
        cy = m1*cx+b1
        max_crossing_coordinate = max(max_crossing_coordinate, abs(cx), abs(cy))
        for _ in range(points_per_scene):
            point = (rng.uniform(-RADIUS, RADIUS), rng.uniform(-RADIUS, RADIUS))
            visibility_mismatches += physical_label(point, sources) != affine_label(point, lines)
        for h in (RADIUS/16, RADIUS/64):
            bounds = (cx-4*h, cx+4*h, cy-4*h, cy+4*h)
            for _ in range(phases_per_scene):
                phase = (rng.random(), rng.random())
                physical_edges = bad_edges(lambda p: physical_label(p, sources), h, phase, bounds,
                                           blockers=BLOCKERS)
                affine_edges = bad_edges(lambda p: affine_label(p, lines), h, phase, bounds)
                visibility_mismatches += physical_edges != affine_edges
                perturbed_hits += bool(physical_edges)
                perturbed_total += 1
    report["physical_random_source_crossing"] = {
        "source_distribution": "independent uniform squares, half-width 1e-4, around (-3,-3),(-3,-6)",
        "source_scene_draws": source_scenes, "point_comparisons": source_scenes*points_per_scene,
        "phase_trials_across_two_spacings": perturbed_total,
        "segment_renderer_vs_affine_mismatches": visibility_mismatches,
        "false_connection_fraction": perturbed_hits/perturbed_total,
        "observed_slope_gap_range": [min_gap, max_gap],
        "max_abs_intersection_coordinate": max_crossing_coordinate,
        "proved_probability_lower_bound": 1/128,
        "interpretation": "finite synthetic diagnostics, not proof of the universal lower bound",
        "passed": visibility_mismatches == 0 and perturbed_hits/perturbed_total >= 1/128,
    }

    a, b = (1, 1, 0), (0, 1, 1)
    success = ideal_peel((a, b), sum_masks((a, b)), 3)
    no_singleton = ideal_peel(((1, 1, 0),), (2, 2, 0), 3)
    mixed = ideal_peel(((1, 0), (0, 1)), (1, 1), 2)
    ambiguous = ideal_peel(((1, 1, 0), (1, 0, 1), (0, 1, 1)), (1, 2, 1), 3)
    full_a, full_b = (1, 1, 1, 1, 1), (0, 1, 0, 0, 0)
    response = sum_masks((full_a, full_b))
    local = one_hop_subtract(response, full_a, 0, 5)
    full = tuple(r-v for r, v in zip(response, full_a))
    local_completion = ideal_peel((full_a, full_b), local, 5)
    report["finite_dictionary_peeling"] = {
        "success": success, "no_singleton": no_singleton,
        "mixed_anchor": mixed, "nonidentifiable_anchor": ambiguous,
        "full_subtraction_residual": full, "one_hop_residual": local,
        "completion_after_incorrect_local_step": local_completion["status"],
        "passed": (success["status"] == "complete" and len(success["selected"]) == 2
                   and no_singleton["status"] == "no_singleton"
                   and mixed["status"] == "no_covering_mask"
                   and ambiguous["status"] == "nonidentifiable_mask"
                   and full == full_b and local != full
                   and local_completion["status"] == "no_covering_mask"),
    }
    report["all_passed"] = all(v["passed"] for v in report.values()
                               if isinstance(v, dict) and "passed" in v)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    report = run_checks(args.seed)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"all_passed": report["all_passed"], "output": str(args.output.resolve()),
                      "fixed_strip_max_error": max(c["error"] for c in report["fixed_strip"]["cases"]),
                      "crossing_fractions": [c["false_connection_fraction"]
                                             for c in report["fixed_scene_crossing"]["cases"]],
                      "physical_visibility_mismatches": report["physical_random_source_crossing"]
                                                            ["segment_renderer_vs_affine_mismatches"]},
                     indent=2))
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
