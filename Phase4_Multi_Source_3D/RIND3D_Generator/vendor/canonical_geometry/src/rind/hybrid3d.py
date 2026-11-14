"""Certified top-level union of exact 3D polyhedra and ellipsoids.

This is intentionally limited to the pilot's expression shape. An ellipsoid
inside INTERSECTION or DIFFERENCE is not silently approximated or accepted.
Ambiguous union boundaries/box coverage raise RuntimeError, which the public
facade converts to UNRESOLVED rather than guessing a scientific label.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import lcm
from typing import Any, Iterable

from .ellipsoid3d import ExactEllipsoid3D


@dataclass(frozen=True, slots=True)
class HybridWitness:
    kind: str
    index: int
    payload: Any

    @property
    def approximate(self) -> list[float]:
        if self.kind == "polyhedron":
            return list(self.payload.approximate)
        return [float(value) for value in self.payload]

    @property
    def homogeneous(self) -> list[str]:
        if self.kind == "polyhedron":
            return list(self.payload.homogeneous)
        denominator = lcm(*(value.denominator for value in self.payload))
        return [*(str(value.numerator * (denominator // value.denominator))
                  for value in self.payload), str(denominator)]


class HybridGeometry3D:
    def __init__(self, polyhedron: Any, ellipsoids: Iterable[ExactEllipsoid3D],
                 *, qfnra_rlimit: int = 1_000_000):
        self.polyhedron = polyhedron
        self.ellipsoids = tuple(item for item in ellipsoids
                                if item.survives_domain_clip)
        if qfnra_rlimit <= 0:
            raise ValueError("QF_NRA resource limit must be positive")
        self.qfnra_rlimit = qfnra_rlimit

    def is_empty(self) -> bool:
        return self.polyhedron.is_empty() and not self.ellipsoids

    def is_full(self) -> bool | None:
        if self.polyhedron.is_full() or any(
            item.classify_box([0, 0, 0], [1, 1, 1]) == 1
            for item in self.ellipsoids
        ):
            return True
        if self.is_empty() or (self.polyhedron.classify_box([0, 0, 0],
                                                            [1, 1, 1]) == 0 and
                               all(item.classify_box([0, 0, 0], [1, 1, 1]) == 0
                                   for item in self.ellipsoids)):
            return False
        return None

    def classify(self, point: Iterable[float]) -> int:
        p = list(point)
        poly = self.polyhedron.classify(p)
        if poly == 1:
            return 1
        ball_classes = [item.classify_point(p) for item in self.ellipsoids]
        if 1 in ball_classes:
            return 1
        incidences = (poly == 2) + sum(value == 2 for value in ball_classes)
        if incidences == 0:
            return 0
        if incidences == 1:
            return 2
        raise RuntimeError("union_boundary_unproved")

    def segment_hit(self, start: Iterable[float], end: Iterable[float]) -> bool:
        a, b = list(start), list(end)
        if self.polyhedron.segment_hit(a, b):
            return True
        return any(item.segment_hit(a, b) for item in self.ellipsoids)

    def classify_box(self, lower: Iterable[float], upper: Iterable[float]) -> int:
        lo, hi = list(lower), list(upper)
        poly = self.polyhedron.classify_box(lo, hi)
        balls = [item.classify_box(lo, hi) for item in self.ellipsoids]
        if poly == 1 or 1 in balls:
            return 1
        if poly == 0 and all(value == 0 for value in balls):
            return 0
        # A piece's exact MIXED result proves that the query box intersects S.
        # A separate exact FREE point establishes that the union does not cover
        # it. Without that second fact, return UNRESOLVED rather than MIXED.
        for choices in product((0, .25, .5, .75, 1), repeat=3):
            point = [lo[i] + (hi[i] - lo[i]) * choices[i] for i in range(3)]
            try:
                if self.classify(point) == 0:
                    return 2
            except RuntimeError:
                continue
        raise RuntimeError("union_box_complement_unproved")

    def clearance(self, point: Iterable[float], radius: float):
        p = list(point)
        decision, witness = self.polyhedron.clearance(p, radius)
        if decision == 0:
            if witness is None or not self.polyhedron.verify_clearance_witness(
                    p, radius, witness):
                raise RuntimeError("polyhedral_clearance_witness_unproved")
            return 0, HybridWitness("polyhedron", 0, witness)
        unresolved = False
        for index, ellipsoid in enumerate(self.ellipsoids):
            result = ellipsoid.clearance(p, radius, rlimit=self.qfnra_rlimit)
            if result.decision == "LE":
                if result.witness is None or not ellipsoid.verify_witness(
                        p, radius, result.witness):
                    raise RuntimeError("ellipsoid_clearance_witness_unproved")
                return 0, HybridWitness("ellipsoid", index, result.witness)
            if result.decision == "UNRESOLVED":
                unresolved = True
        if unresolved:
            raise RuntimeError("ellipsoid_clearance_unproved")
        return 1, None

    def verify_clearance_witness(self, point: Iterable[float], radius: float,
                                 witness: HybridWitness) -> bool:
        p = list(point)
        if witness.kind == "polyhedron":
            return bool(self.polyhedron.verify_clearance_witness(
                p, radius, witness.payload))
        if witness.kind == "ellipsoid" and 0 <= witness.index < len(self.ellipsoids):
            return self.ellipsoids[witness.index].verify_witness(
                p, radius, witness.payload)
        return False

    def verify_homogeneous_witness(self, point: Iterable[float], radius: float,
                                   homogeneous: Iterable[str]) -> bool:
        entries = tuple(int(value) for value in homogeneous)
        if len(entries) != 4 or entries[3] <= 0:
            return False
        if self.polyhedron.verify_homogeneous_witness(
                list(point), radius, [str(value) for value in entries]):
            return True
        q = tuple(Fraction(entries[i], entries[3]) for i in range(3))
        return any(item.verify_witness(point, radius, q)
                   for item in self.ellipsoids)
