"""Exact rational 3D ellipsoid primitives and a bounded QF_NRA clearance gate.

This module deliberately does not claim to solve mixed polyhedral/ellipsoid
CSG. It is the independently testable leaf backend for that later integration.
All inputs are interpreted as the exact rationals represented by binary64.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
from typing import Iterable

import z3


Q = Fraction
Vector = tuple[Q, Q, Q]


def _dyadic(value: object) -> Q:
    if isinstance(value, Q):
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("ellipsoid parameters must be finite binary64")
    return Q.from_float(number)


def _vector(values: Iterable[object]) -> Vector:
    parts = tuple(_dyadic(value) for value in values)
    if len(parts) != 3:
        raise ValueError("expected exactly three coordinates")
    return parts  # type: ignore[return-value]


def _inverse3(matrix: tuple[tuple[Q, Q, Q], ...]) -> tuple[tuple[Q, Q, Q], ...]:
    rows = [list(matrix[i]) + [Q(i == j) for j in range(3)] for i in range(3)]
    for pivot in range(3):
        chosen = next((row for row in range(pivot, 3) if rows[row][pivot]), None)
        if chosen is None:
            raise ValueError("singular affine ellipsoid transform")
        rows[pivot], rows[chosen] = rows[chosen], rows[pivot]
        divisor = rows[pivot][pivot]
        rows[pivot] = [value / divisor for value in rows[pivot]]
        for row in range(3):
            if row == pivot:
                continue
            factor = rows[row][pivot]
            rows[row] = [value - factor * base
                         for value, base in zip(rows[row], rows[pivot], strict=True)]
    return tuple(tuple(row[3:]) for row in rows)  # type: ignore[return-value]


def _solve(matrix: list[list[Q]], rhs: list[Q]) -> list[Q]:
    n = len(rhs)
    rows = [matrix[i][:] + [rhs[i]] for i in range(n)]
    for pivot in range(n):
        chosen = next((row for row in range(pivot, n) if rows[row][pivot]), None)
        if chosen is None:
            raise AssertionError("positive-definite principal minor was singular")
        rows[pivot], rows[chosen] = rows[chosen], rows[pivot]
        divisor = rows[pivot][pivot]
        rows[pivot] = [value / divisor for value in rows[pivot]]
        for row in range(n):
            if row == pivot:
                continue
            factor = rows[row][pivot]
            rows[row] = [value - factor * base
                         for value, base in zip(rows[row], rows[pivot], strict=True)]
    return [row[-1] for row in rows]


def _fraction_from_z3(value: z3.RatNumRef) -> Q:
    return Q(value.numerator_as_long(), value.denominator_as_long())


@dataclass(frozen=True, slots=True)
class EllipsoidClearance:
    decision: str  # LE, GT, or UNRESOLVED
    witness: Vector | None
    solver_status: str
    rlimit_count: int

    @property
    def durable_witness(self) -> tuple[tuple[str, str], ...] | None:
        if self.witness is None:
            return None
        return tuple((str(value.numerator), str(value.denominator))
                     for value in self.witness)


class ExactEllipsoid3D:
    def __init__(self, center: Iterable[object], matrix: Iterable[Iterable[object]]):
        self.center = _vector(center)
        rows = tuple(_vector(row) for row in matrix)
        if len(rows) != 3:
            raise ValueError("expected a 3x3 affine matrix")
        self.matrix = rows
        self.inverse = _inverse3(rows)
        self.quadratic = tuple(tuple(sum(self.inverse[k][i] * self.inverse[k][j]
                                          for k in range(3))
                                     for j in range(3)) for i in range(3))
        minimum, _ = self.minimum_on_box((Q(0), Q(0), Q(0)),
                                          (Q(1), Q(1), Q(1)))
        # A zero-volume contact with Ω disappears under relative regularization.
        self.survives_domain_clip = minimum < 1

    def value(self, point: Iterable[object] | Vector) -> Q:
        p = _vector(point)
        delta = tuple(p[i] - self.center[i] for i in range(3))
        return sum(delta[i] * self.quadratic[i][j] * delta[j]
                   for i in range(3) for j in range(3))

    def minimum_on_box(self, lower: Vector, upper: Vector) -> tuple[Q, Vector]:
        if any(lower[i] > upper[i] for i in range(3)):
            raise ValueError("reversed query box")
        best: tuple[Q, Vector] | None = None
        for status in product((-1, 0, 1), repeat=3):
            point = [Q(0)] * 3
            free = [i for i in range(3) if status[i] == 0]
            fixed = [i for i in range(3) if status[i] != 0]
            for i in fixed:
                point[i] = lower[i] if status[i] == -1 else upper[i]
            if free:
                matrix = [[self.quadratic[i][j] for j in free] for i in free]
                rhs = [-sum(self.quadratic[i][j] *
                            (point[j] - self.center[j]) for j in fixed)
                       for i in free]
                displacement = _solve(matrix, rhs)
                for index, value in zip(free, displacement, strict=True):
                    point[index] = self.center[index] + value
            if any(point[i] < lower[i] or point[i] > upper[i] for i in range(3)):
                continue
            candidate = tuple(point)
            score = self.value(candidate)
            if best is None or score < best[0]:
                best = score, candidate  # type: ignore[assignment]
        if best is None:
            raise AssertionError("compact box had no quadratic minimum")
        return best

    @staticmethod
    def _domain(point: Vector) -> bool:
        return all(0 <= value <= 1 for value in point)

    def classify_point(self, point: Iterable[object]) -> int:
        p = _vector(point)
        if not self._domain(p):
            raise ValueError("point is outside Ω")
        if not self.survives_domain_clip:
            return 0
        value = self.value(p)
        if value > 1:
            return 0
        if value < 1:
            return 1
        # Equality at a domain vertex can still be interior relative to Ω.
        if all(coordinate in (0, 1) for coordinate in p):
            delta = tuple(p[i] - self.center[i] for i in range(3))
            gradient = [sum(self.quadratic[i][j] * delta[j]
                            for j in range(3)) for i in range(3)]
            if all(gradient[i] * (1 if p[i] == 0 else -1) < 0
                   for i in range(3)):
                return 1
        return 2

    def segment_hit(self, start: Iterable[object], end: Iterable[object]) -> bool:
        a, b = _vector(start), _vector(end)
        if not self._domain(a) or not self._domain(b):
            raise ValueError("segment endpoint is outside Ω")
        if not self.survives_domain_clip:
            return False
        direction = tuple(b[i] - a[i] for i in range(3))
        delta = tuple(a[i] - self.center[i] for i in range(3))
        aa = sum(direction[i] * self.quadratic[i][j] * direction[j]
                 for i in range(3) for j in range(3))
        bb = 2 * sum(direction[i] * self.quadratic[i][j] * delta[j]
                     for i in range(3) for j in range(3))
        t = Q(0) if aa == 0 else max(Q(0), min(Q(1), -bb / (2 * aa)))
        q = tuple(a[i] + t * direction[i] for i in range(3))
        return self.value(q) <= 1

    def classify_box(self, lower: Iterable[object], upper: Iterable[object]) -> int:
        lo, hi = _vector(lower), _vector(upper)
        if not self._domain(lo) or not self._domain(hi):
            raise ValueError("query box is outside Ω")
        if not self.survives_domain_clip:
            return 0
        minimum, _ = self.minimum_on_box(lo, hi)
        if minimum > 1:
            return 0
        if all(self.value(corner) <= 1 for corner in product(*zip(lo, hi))):
            return 1
        return 2

    def _z3_value(self, q: tuple[z3.ArithRef, ...], ctx: z3.Context):
        def constant(value: Q):
            return z3.Q(value.numerator, value.denominator, ctx=ctx)

        delta = [q[i] - constant(self.center[i]) for i in range(3)]
        return sum(constant(self.quadratic[i][j]) * delta[i] * delta[j]
                   for i in range(3) for j in range(3))

    def _rational_witness(self, model: z3.ModelRef,
                          variables: tuple[z3.ArithRef, ...],
                          point: Vector, radius_squared: Q) -> Vector | None:
        values = [model.eval(variable, model_completion=True)
                  for variable in variables]
        for precision in (12, 24, 48, 96, 192):
            choices: list[tuple[Q, ...]] = []
            for value in values:
                if z3.is_rational_value(value):
                    choices.append((_fraction_from_z3(value),))
                elif isinstance(value, z3.AlgebraicNumRef):
                    upper = _fraction_from_z3(value.approx(precision))
                    step = Q(1, 10**precision)
                    choices.append((upper, upper - step, upper - 2 * step))
                else:
                    return None
            for trial in product(*choices):
                candidate = tuple(trial)
                if (self._domain(candidate) and self.value(candidate) <= 1 and
                    sum((candidate[i] - point[i]) ** 2 for i in range(3)) <= radius_squared):
                    return candidate  # type: ignore[return-value]
        return None

    def clearance(self, point: Iterable[object], radius: object,
                  *, rlimit: int = 1_000_000) -> EllipsoidClearance:
        p, r = _vector(point), _dyadic(radius)
        if not self._domain(p) or r < 0:
            raise ValueError("invalid clearance request")
        if rlimit <= 0:
            raise ValueError("rlimit must be positive")
        if not self.survives_domain_clip:
            return EllipsoidClearance("GT", None, "analytic_empty", 0)
        if self.value(p) <= 1:
            return EllipsoidClearance("LE", p, "analytic_member", 0)
        if r == 0:
            return EllipsoidClearance("GT", None, "analytic_zero_radius", 0)
        context = z3.Context()
        q = tuple(z3.Real(f"q{i}", ctx=context) for i in range(3))
        solver = z3.Tactic("qfnra-nlsat", ctx=context).solver()
        solver.set(rlimit=rlimit, seed=0, randomize=False, shuffle_vars=False)

        def constant(value: Q):
            return z3.Q(value.numerator, value.denominator, ctx=context)

        for coordinate in q:
            solver.add(constant(Q(0)) <= coordinate, coordinate <= constant(Q(1)))
        solver.add(self._z3_value(q, context) <= constant(Q(1)))
        solver.add(sum((q[i] - constant(p[i])) ** 2 for i in range(3))
                   <= constant(r * r))
        answer = solver.check()
        try:
            work = int(solver.statistics().get_key_value("rlimit count"))
        except (KeyError, z3.Z3Exception):
            work = 0
        if answer == z3.unsat:
            return EllipsoidClearance("GT", None, "unsat_qfnra_nlsat", work)
        if answer != z3.sat:
            return EllipsoidClearance("UNRESOLVED", None,
                                      f"unknown:{solver.reason_unknown()}", work)
        witness = self._rational_witness(solver.model(), q, p, r * r)
        if witness is None:
            return EllipsoidClearance("UNRESOLVED", None,
                                      "sat_without_verified_rational_witness", work)
        return EllipsoidClearance("LE", witness, "sat_verified_rational_witness", work)

    def verify_witness(self, point: Iterable[object], radius: object,
                       witness: Iterable[Q]) -> bool:
        p, r = _vector(point), _dyadic(radius)
        q = tuple(witness)
        if len(q) != 3 or r < 0:
            return False
        return (self._domain(q) and self.survives_domain_clip and
                self.value(q) <= 1 and
                sum((q[i] - p[i]) ** 2 for i in range(3)) <= r * r)
