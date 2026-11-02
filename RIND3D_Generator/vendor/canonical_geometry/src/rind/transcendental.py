"""Conservative Arb predicates for local transcendental primitive fields.

This module is deliberately separate from GeometryView while the algebraic 3D
pilot is being generated. It never publishes a READY capability by itself.
Every float64 input is first interpreted as its exact dyadic rational value;
an indeterminate Arb enclosure stays UNKNOWN rather than receiving a label.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math

from flint import arb, ctx, fmpq
import numpy as np

from .contracts import CodecID, Leaf
from .geometry import BoxClass, Clearance, Membership, SegmentDecision


PRECISION_LADDER = (64, 128, 256, 512, 1024)


@dataclass(slots=True)
class ProofTrace:
    """Deterministic counts for one query item, not elapsed-time estimates."""

    precision_bits: int = 0
    subdivisions: int = 0
    branches: int = 0
    backend_work_units: int = 0  # Number of Arb precision passes.

    def as_dict(self) -> dict[str, int]:
        return {"precision_bits": self.precision_bits,
                "subdivisions": self.subdivisions,
                "branches": self.branches,
                "backend_work_units": self.backend_work_units}


_ACTIVE_TRACE: ContextVar[ProofTrace | None] = ContextVar(
    "rind_transcendental_proof_trace", default=None)


@contextmanager
def capture_work():
    trace = ProofTrace()
    token = _ACTIVE_TRACE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE_TRACE.reset(token)


@contextmanager
def _precision(bits: int):
    trace = _ACTIVE_TRACE.get()
    if trace is not None:
        trace.precision_bits = max(trace.precision_bits, bits)
        trace.backend_work_units += 1
    with ctx.workprec(bits):
        yield


def count_subdivision() -> None:
    trace = _ACTIVE_TRACE.get()
    if trace is not None:
        trace.subdivisions += 1


def count_branch() -> None:
    trace = _ACTIVE_TRACE.get()
    if trace is not None:
        trace.branches += 1


def count_native_work(work: dict[str, int]) -> None:
    trace = _ACTIVE_TRACE.get()
    if trace is not None:
        trace.precision_bits = max(trace.precision_bits,
                                   int(work.get("precision_bits", 0)))
        trace.subdivisions += int(work.get("subdivisions", 0))
        trace.branches += int(work.get("branches", 0))
        trace.backend_work_units += int(work.get("backend_work_units", 0))


def _q(value: float) -> Fraction:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("transcendental predicate input must be finite")
    return Fraction.from_float(value)


def _ball(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def _range(low: Fraction, high: Fraction) -> arb:
    if low > high:
        raise ValueError("reversed interval")
    return _ball(low).union(_ball(high))


def _upper_fraction(value: arb) -> Fraction:
    """Extract an outward Arb upper endpoint as an exact dyadic rational."""
    mantissa, exponent = value.upper().man_exp()
    m, e = int(mantissa), int(exponent)
    return Fraction(m << e) if e >= 0 else Fraction(m, 1 << -e)


def _point(values: object, dimension: int) -> tuple[Fraction, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (dimension,):
        raise ValueError(f"expected a {dimension}-coordinate point")
    return tuple(_q(value) for value in array)


class CosineFieldPredicate:
    """Strict-positive cosine-field solid with conservative region bounds."""

    def __init__(self, leaf: Leaf,
                 ladder: tuple[int, ...] = PRECISION_LADDER):
        if leaf.codec != CodecID.COSINE_FIELD:
            raise ValueError("expected cosine_field_v1")
        self.dimension = leaf.dimension
        self.ladder = ladder
        fields = {field.name: field.as_array() for field in leaf.fields}
        frequency = fields["frequency"]
        grouped: dict[tuple[tuple[Fraction, ...], Fraction], Fraction] = {}
        for amplitude, vector, phase in zip(fields["amplitude"], frequency,
                                             fields["phase"], strict=True):
            key = (tuple(_q(value) for value in vector), _q(phase))
            grouped[key] = grouped.get(key, Fraction(0)) + _q(amplitude)
        bias = -_q(fields["threshold"])
        terms = []
        for (vector, phase), coefficient in sorted(grouped.items()):
            if not coefficient:
                continue
            if phase == 0 and all(value == 0 for value in vector):
                bias += coefficient  # cos(0) is exactly one.
            else:
                terms.append((coefficient, vector, phase))
        self.terms = tuple(terms)
        self.bias = bias

    def _score(self, region: tuple[tuple[Fraction, Fraction], ...]) -> arb:
        total = _ball(self.bias)
        for amplitude, vector, phase in self.terms:
            low = high = phase
            for (coordinate_low, coordinate_high), coefficient in zip(
                    region, vector, strict=True):
                a = coefficient * coordinate_low
                b = coefficient * coordinate_high
                low += min(a, b)
                high += max(a, b)
            total += _ball(amplitude) * _range(low, high).cos()
        return total

    def classify_point(self, values: object) -> Membership:
        point = _point(values, self.dimension)
        return self._classify_exact_point(point)

    def _classify_exact_point(self, point: tuple[Fraction, ...]) -> Membership:
        # Identical terms cancel exactly before Arb evaluation. An exact
        # constant zero is outside cl{f > threshold}, not an equality remnant.
        if not self.terms:
            return Membership.INTERIOR if self.bias > 0 else Membership.FREE
        angles: dict[Fraction, Fraction] = {}
        gradient_terms: dict[Fraction, list[Fraction]] = {}
        for amplitude, vector, phase in self.terms:
            angle = phase + sum((w*x for w, x in zip(vector, point, strict=True)),
                                Fraction(0))
            angles[angle] = angles.get(angle, Fraction(0)) + amplitude
            row = gradient_terms.setdefault(angle, [Fraction(0)] * self.dimension)
            for axis, coefficient in enumerate(vector):
                row[axis] -= amplitude * coefficient
        exact_bias = self.bias + angles.pop(Fraction(0), Fraction(0))
        residual = tuple((angle, coefficient) for angle, coefficient in angles.items()
                         if coefficient)
        if not residual:
            if exact_bias > 0:
                return Membership.INTERIOR
            if exact_bias < 0:
                return Membership.FREE
            # Equality is exact. A certified nonzero gradient gives a regular
            # boundary for points strictly inside Ω; domain faces need a
            # separate relative-topology proof.
            if all(0 < coordinate < 1 for coordinate in point):
                for bits in self.ladder:
                    with _precision(bits):
                        for axis in range(self.dimension):
                            derivative = sum(
                                (_ball(weights[axis]) * _ball(angle).sin()
                                 for angle, weights in gradient_terms.items()),
                                arb(0))
                            if derivative > 0 or derivative < 0:
                                return Membership.BOUNDARY
            return Membership.UNKNOWN
        for bits in self.ladder:
            with _precision(bits):
                score = _ball(exact_bias)
                for angle, coefficient in residual:
                    score += _ball(coefficient) * _ball(angle).cos()
                if score > 0:
                    return Membership.INTERIOR
                if score < 0:
                    return Membership.FREE
        return Membership.UNKNOWN

    def classify_box(self, lower: object, upper: object) -> BoxClass:
        return self._classify_exact_box(_point(lower, self.dimension),
                                        _point(upper, self.dimension))

    def _classify_exact_box(self, lo: tuple[Fraction, ...],
                            hi: tuple[Fraction, ...]) -> BoxClass:
        if any(a > b for a, b in zip(lo, hi, strict=True)):
            raise ValueError("reversed box")
        if not self.terms:
            return BoxClass.SOLID if self.bias > 0 else BoxClass.FREE
        region = tuple(zip(lo, hi, strict=True))
        for bits in self.ladder:
            with _precision(bits):
                score = self._score(region)
                if score > 0:
                    return BoxClass.SOLID
                if score < 0:
                    return BoxClass.FREE
        corners = product(*zip(lo, hi, strict=True))
        labels = {self._classify_exact_point(point) for point in corners}
        if Membership.INTERIOR in labels and Membership.FREE in labels:
            return BoxClass.MIXED
        return BoxClass.UNKNOWN

    def clearance_decide(self, point: object, radius: float) -> Clearance:
        world = _point(point, self.dimension)
        if not all(0 <= coordinate <= 1 for coordinate in world):
            raise ValueError("clearance query must be inside the scene domain")
        r = _q(radius)
        if r < 0:
            raise ValueError("negative clearance radius")
        if self._classify_exact_point(world) == Membership.INTERIOR:
            return Clearance.LE
        lo = tuple(value-r for value in world)
        hi = tuple(value+r for value in world)
        if self._classify_exact_box(lo, hi) == BoxClass.FREE:
            return Clearance.GT
        return Clearance.UNRESOLVED

    def segment_intersect(self, start: object, stop: object,
                          *, max_subdivisions: int = 1024) -> SegmentDecision:
        a, b = _point(start, self.dimension), _point(stop, self.dimension)
        if max_subdivisions < 0:
            raise ValueError("negative subdivision budget")
        if not self.terms:
            return SegmentDecision.HIT if self.bias > 0 else SegmentDecision.MISS
        stack = [(Fraction(0), Fraction(1))]
        splits = 0
        while stack:
            left, right = stack.pop()
            region = tuple((min(x + (y-x)*left, x + (y-x)*right),
                            max(x + (y-x)*left, x + (y-x)*right))
                           for x, y in zip(a, b, strict=True))
            proven_outside = False
            for bits in self.ladder:
                with _precision(bits):
                    score = self._score(region)
                    if score > 0:
                        return SegmentDecision.HIT
                    if score < 0:
                        proven_outside = True
                        break
            if proven_outside:
                continue
            midpoint = (left + right) / 2
            sample = tuple(x + (y-x)*midpoint for x, y in zip(a, b, strict=True))
            if self._classify_exact_point(sample) in (Membership.INTERIOR,
                                                      Membership.BOUNDARY):
                return SegmentDecision.HIT
            if splits >= max_subdivisions:
                return SegmentDecision.UNKNOWN
            splits += 1
            count_subdivision()
            stack.append((midpoint, right))
            stack.append((left, midpoint))
        return SegmentDecision.MISS


class RadialFourierPredicate:
    """Conservative local point, box and segment queries for radial leaves."""

    def __init__(self, leaf: Leaf,
                 ladder: tuple[int, ...] = PRECISION_LADDER):
        if leaf.codec not in (CodecID.RADIAL_FOURIER_2D,
                              CodecID.RADIAL_FOURIER_SPHERE):
            raise ValueError("expected a radial Fourier codec")
        self.dimension = leaf.dimension
        self.codec = leaf.codec
        self.ladder = ladder
        self.a = tuple(_q(value) for value in leaf.field("coeff_a").as_array())
        self.b = tuple(_q(value) for value in leaf.field("coeff_b").as_array())
        self.amplitude_bound = sum((abs(value) for value in (*self.a, *self.b)),
                                   Fraction(0))
        if self.codec == CodecID.RADIAL_FOURIER_SPHERE:
            self.frequency = tuple(tuple(_q(value) for value in row)
                                   for row in leaf.field("frequency").as_array())
        else:
            self.frequency = ()
        self._radius_bound = self._compute_outer_radius_bound()

    def _outer_radius_bound(self) -> Fraction | None:
        return self._radius_bound

    def _compute_outer_radius_bound(self) -> Fraction | None:
        """Certify a finite outer radius from the stored spectrum."""
        if not any(self.a) and not any(self.b):
            return Fraction(1)
        log_bound = self.amplitude_bound
        if log_bound > 64:
            return None  # Avoid constructing an enormous loose bound.
        if self.dimension == 2:
            # |f'(theta)| <= sum_j j(|a_j|+|b_j|). The rational interval
            # [0,7] covers a full turn. Center samples plus L*h/2 bound
            # every angle without trusting a floating-point maximum.
            samples = 128
            lipschitz = sum((index * (abs(a) + abs(b)) for index, (a, b) in
                             enumerate(zip(self.a, self.b, strict=True), 1)),
                            Fraction(0))
            sample_upper = None
            with _precision(128):
                for index in range(samples):
                    angle = _ball(Fraction(7 * (2*index + 1), 2*samples))
                    value = arb(0)
                    for harmonic, (a, b) in enumerate(
                            zip(self.a, self.b, strict=True), 1):
                        phase = harmonic * angle
                        value += _ball(a) * phase.cos() + _ball(b) * phase.sin()
                    bound = _upper_fraction(value)
                    sample_upper = bound if sample_upper is None else max(sample_upper, bound)
            log_bound = min(log_bound, sample_upper + lipschitz * Fraction(7, 2*samples))
        with _precision(128):
            return _upper_fraction(_ball(log_bound).exp())

    def _score(self, region: tuple[tuple[Fraction, Fraction], ...]) -> arb | None:
        coordinates = tuple(_range(low, high) for low, high in region)
        # Squaring a generic interval crossing zero can include negatives;
        # use exact rational extrema of each squared coordinate instead.
        squared = arb(0)
        for low, high in region:
            minimum = (Fraction(0) if low <= 0 <= high else
                       min(low*low, high*high))
            maximum = max(low*low, high*high)
            squared += _range(minimum, maximum)
        if not squared > 0:
            return None  # Division by radius is not yet certified safe.
        radius = squared.sqrt()
        unit = tuple(value / radius for value in coordinates)
        score = arb(0)
        if self.dimension == 2:
            cosine, sine = unit
            for a, b in zip(self.a, self.b, strict=True):
                score += _ball(a) * cosine + _ball(b) * sine
                cosine, sine = (cosine * unit[0] - sine * unit[1],
                                sine * unit[0] + cosine * unit[1])
        else:
            for a, b, frequency in zip(self.a, self.b,
                                        self.frequency, strict=True):
                angle = sum((_ball(w) * u for w, u in
                             zip(frequency, unit, strict=True)), arb(0))
                score += _ball(a) * angle.cos() + _ball(b) * angle.sin()
        return score - radius.log()

    def classify_local_point(self, values: object) -> Membership:
        return self._classify_exact_point(_point(values, self.dimension))

    def _classify_exact_point(self, vector: tuple[Fraction, ...]) -> Membership:
        squared = sum((value * value for value in vector), Fraction(0))
        if squared == 0:
            return Membership.INTERIOR
        outer = self._outer_radius_bound()
        if outer is not None and squared > outer * outer:
            return Membership.FREE
        if not any(self.a) and not any(self.b):
            if squared < 1:
                return Membership.INTERIOR
            if squared > 1:
                return Membership.FREE
            return Membership.BOUNDARY
        for bits in self.ladder:
            with _precision(bits):
                score = self._score(tuple((value, value) for value in vector))
                if score is None:
                    continue
                if score > 0:
                    return Membership.INTERIOR
                if score < 0:
                    return Membership.FREE
        return Membership.UNKNOWN

    def classify_local_box(self, lower: object, upper: object) -> BoxClass:
        return self._classify_exact_box(_point(lower, self.dimension),
                                        _point(upper, self.dimension))

    def _classify_exact_box(self, lo: tuple[Fraction, ...],
                            hi: tuple[Fraction, ...]) -> BoxClass:
        if any(a > b for a, b in zip(lo, hi, strict=True)):
            raise ValueError("reversed box")
        nearest = sum((min(abs(a), abs(b)) ** 2 if a * b > 0 else 0
                       for a, b in zip(lo, hi, strict=True)), Fraction(0))
        outer = self._outer_radius_bound()
        if outer is not None and nearest > outer * outer:
            return BoxClass.FREE
        if not any(self.a) and not any(self.b):
            farthest = sum((max(a*a, b*b) for a, b in zip(lo, hi, strict=True)),
                           Fraction(0))
            if farthest <= 1:
                return BoxClass.SOLID
            if nearest > 1:
                return BoxClass.FREE
            return BoxClass.MIXED
        region = tuple(zip(lo, hi, strict=True))
        for bits in self.ladder:
            with _precision(bits):
                score = self._score(region)
                if score is None:
                    continue
                if score > 0:
                    return BoxClass.SOLID
                if score < 0:
                    return BoxClass.FREE
        labels = {self._classify_exact_point(point)
                  for point in product(*zip(lo, hi, strict=True))}
        if Membership.INTERIOR in labels and Membership.FREE in labels:
            return BoxClass.MIXED
        return BoxClass.UNKNOWN

    def segment_intersect_local(self, start: object, stop: object,
                                *, max_subdivisions: int = 1024) -> SegmentDecision:
        return self._segment_exact(_point(start, self.dimension),
                                   _point(stop, self.dimension),
                                   max_subdivisions=max_subdivisions)

    def _segment_exact(self, a: tuple[Fraction, ...], b: tuple[Fraction, ...],
                       *, max_subdivisions: int) -> SegmentDecision:
        if max_subdivisions < 0:
            raise ValueError("negative subdivision budget")
        direction = tuple(y - x for x, y in zip(a, b, strict=True))
        # The origin belongs to every radial solid; prove its presence on the
        # segment with exact rational parameter relations, not coordinatewise
        # floating-point rounding.
        origins = {-x / delta for x, delta in zip(a, direction, strict=True)
                   if delta != 0}
        if (all(delta != 0 or x == 0 for x, delta in zip(a, direction, strict=True))
                and len(origins) == 1 and 0 <= next(iter(origins)) <= 1):
            return SegmentDecision.HIT
        quadratic = sum((delta*delta for delta in direction), Fraction(0))
        linear_half = sum((x*delta for x, delta in zip(a, direction, strict=True)),
                          Fraction(0))
        t = (max(Fraction(0), min(Fraction(1), -linear_half / quadratic))
             if quadratic else Fraction(0))
        minimum = sum(((x + delta*t) ** 2 for x, delta in
                       zip(a, direction, strict=True)), Fraction(0))
        outer = self._outer_radius_bound()
        if outer is not None and minimum > outer * outer:
            return SegmentDecision.MISS
        if not any(self.a) and not any(self.b):
            return SegmentDecision.HIT if minimum <= 1 else SegmentDecision.MISS
        stack = [(Fraction(0), Fraction(1))]
        splits = 0
        while stack:
            left, right = stack.pop()
            region = tuple((min(x + delta*left, x + delta*right),
                            max(x + delta*left, x + delta*right))
                           for x, delta in zip(a, direction, strict=True))
            outside = False
            for bits in self.ladder:
                with _precision(bits):
                    score = self._score(region)
                    if score is None:
                        continue
                    if score > 0:
                        return SegmentDecision.HIT
                    if score < 0:
                        outside = True
                        break
            if outside:
                continue
            midpoint = (left + right) / 2
            sample = tuple(x + delta*midpoint for x, delta in
                           zip(a, direction, strict=True))
            if self._classify_exact_point(sample) in (Membership.INTERIOR,
                                                       Membership.BOUNDARY):
                return SegmentDecision.HIT
            if splits >= max_subdivisions:
                return SegmentDecision.UNKNOWN
            splits += 1
            count_subdivision()
            stack.append((midpoint, right))
            stack.append((left, midpoint))
        return SegmentDecision.MISS


def _inverse(matrix: tuple[tuple[Fraction, ...], ...]) -> tuple[tuple[Fraction, ...], ...]:
    n = len(matrix)
    augmented = [list(row) + [Fraction(int(i == j)) for j in range(n)]
                 for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = next((row for row in range(column, n)
                      if augmented[row][column] != 0), None)
        if pivot is None:
            raise ValueError("singular radial affine transform")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [a - factor*b for a, b in
                              zip(augmented[row], augmented[column], strict=True)]
    return tuple(tuple(row[n:]) for row in augmented)


class AffineRadialPredicate:
    """Unclipped radial leaf with exact affine coordinate conversion.

    The eventual GeometryView integration must apply Ω clipping and CSG; this
    primitive alone does not claim final-scene semantics.
    """

    def __init__(self, leaf: Leaf,
                 ladder: tuple[int, ...] = PRECISION_LADDER):
        self.radial = RadialFourierPredicate(leaf, ladder)
        self.dimension = leaf.dimension
        self.center = tuple(_q(value) for value in leaf.field("center").as_array())
        matrix = tuple(tuple(_q(value) for value in row)
                       for row in leaf.field("matrix").as_array())
        self.matrix = matrix
        self.inverse = _inverse(matrix)

    def _local(self, world: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
        shifted = tuple(value-center for value, center in
                        zip(world, self.center, strict=True))
        return tuple(sum((coefficient*value for coefficient, value in
                          zip(row, shifted, strict=True)), Fraction(0))
                     for row in self.inverse)

    def classify_point(self, world: object) -> Membership:
        return self.radial._classify_exact_point(
            self._local(_point(world, self.dimension)))

    def _local_box_bounds(self, lo: tuple[Fraction, ...],
                          hi: tuple[Fraction, ...]
                          ) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
        lows, highs = [], []
        for row in self.inverse:
            interval = [(coefficient*(a-center), coefficient*(b-center))
                        for coefficient, a, b, center in
                        zip(row, lo, hi, self.center, strict=True)]
            lows.append(sum((min(a, b) for a, b in interval), Fraction(0)))
            highs.append(sum((max(a, b) for a, b in interval), Fraction(0)))
        return tuple(lows), tuple(highs)

    def _classify_exact_world_box(self, lo: tuple[Fraction, ...],
                                  hi: tuple[Fraction, ...]) -> BoxClass:
        outer = self.radial._outer_radius_bound()
        if outer is not None:
            # The local solid lies in the radius-outer ball. Its affine image
            # lies in this exact world-axis box; this often proves separation
            # where inverse-transforming a world box loses correlations.
            for axis, row in enumerate(self.matrix):
                extent = outer * sum((abs(value) for value in row), Fraction(0))
                if hi[axis] < self.center[axis] - extent or \
                        lo[axis] > self.center[axis] + extent:
                    return BoxClass.FREE
        local_lo, local_hi = self._local_box_bounds(lo, hi)
        result = self.radial._classify_exact_box(local_lo, local_hi)
        if result in (BoxClass.SOLID, BoxClass.FREE):
            return result
        labels = {self.radial._classify_exact_point(self._local(corner))
                  for corner in product(*zip(lo, hi, strict=True))}
        if Membership.INTERIOR in labels and Membership.FREE in labels:
            return BoxClass.MIXED
        return BoxClass.UNKNOWN

    def classify_box(self, lower: object, upper: object) -> BoxClass:
        lo, hi = _point(lower, self.dimension), _point(upper, self.dimension)
        if any(a > b for a, b in zip(lo, hi, strict=True)):
            raise ValueError("reversed box")
        return self._classify_exact_world_box(lo, hi)

    def segment_intersect(self, start: object, stop: object,
                          *, max_subdivisions: int = 1024) -> SegmentDecision:
        a = self._local(_point(start, self.dimension))
        b = self._local(_point(stop, self.dimension))
        return self.radial._segment_exact(a, b,
                                          max_subdivisions=max_subdivisions)

    def clearance_decide(self, point: object, radius: float) -> Clearance:
        world = _point(point, self.dimension)
        if not all(0 <= coordinate <= 1 for coordinate in world):
            raise ValueError("clearance query must be inside the scene domain")
        r = _q(radius)
        if r < 0:
            raise ValueError("negative clearance radius")
        membership = self.radial._classify_exact_point(self._local(world))
        # An ambient boundary point on an Ω face may be an exterior-only
        # tangency removed by relative regularization. Only an interior
        # boundary point is automatically a witness for the final solid.
        if (membership == Membership.INTERIOR or
                (membership == Membership.BOUNDARY and
                 all(0 < coordinate < 1 for coordinate in world))):
            return Clearance.LE
        if (all(0 <= coordinate <= 1 for coordinate in self.center) and
            sum(((a-b)**2 for a, b in zip(world, self.center, strict=True)),
                Fraction(0)) <= r*r):
            return Clearance.LE
        lo = tuple(value-r for value in world)
        hi = tuple(value+r for value in world)
        if self._classify_exact_world_box(lo, hi) == BoxClass.FREE:
            return Clearance.GT
        return Clearance.UNRESOLVED
