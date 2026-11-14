"""Certified additive response terms for the v3 decay codecs.

Binary64 input parameters denote exact dyadic rationals. Arb encloses only
the transcendental evaluation; a term is emitted after both interval endpoints
round to the same binary64 value. Finite budgets return unresolved explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

from flint import arb, ctx, fmpq
import numpy as np

from .contracts import Physics, Scene
from .geometry import (GeometryView, Membership, QueryStatus, SegmentDecision,
                       _points)


PRECISION_LADDER = (128, 256, 512, 1024)


def _dyadic(value: float) -> Fraction:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("decay input is not finite")
    return Fraction.from_float(value)


def _ball(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def _distance_squared(a: np.ndarray, b: np.ndarray) -> Fraction:
    return sum((_dyadic(float(x)) - _dyadic(float(y))) ** 2
               for x, y in zip(a, b, strict=True))


def _exact_rounded_sum(terms: list[float]) -> float:
    """Sum already rounded finite terms as exact dyadics, then round once."""
    total = sum((_dyadic(term) for term in terms), Fraction(0))
    answer = float(total)
    if not math.isfinite(answer):
        raise OverflowError("additive response is outside binary64 range")
    return answer


def _endpoint_slope(a: Fraction, b: Fraction, ha: Fraction,
                    hb: Fraction) -> Fraction:
    candidate = ((2 * ha + hb) * a - ha * b) / (ha + hb)
    if candidate * a <= 0:
        return Fraction(0)
    if a * b < 0 and abs(candidate) > 3 * abs(a):
        return 3 * a
    return candidate


@dataclass(frozen=True, slots=True)
class LogPchipLaw:
    knots: tuple[Fraction, ...]
    values: tuple[Fraction, ...]
    slopes: tuple[Fraction, ...]

    @classmethod
    def from_physics(cls, physics: Physics, dimension: int) -> "LogPchipLaw":
        if physics.codec_id != 2 or dimension not in (2, 3):
            raise ValueError("expected log_pchip_v1 physics and dimension 2 or 3")
        fields = {field.name: field.as_array() for field in physics.fields}
        knots = tuple(_dyadic(x) for x in fields["knots"])
        values = tuple(_dyadic(x) for x in fields["log_values"])
        if len(knots) != 9 or len(values) != 9 or knots[0] != 0 or values[0] != 0:
            raise ValueError("invalid PCHIP shape or origin")
        if any(knots[i] >= knots[i + 1] or values[i] <= values[i + 1]
               for i in range(8)):
            raise ValueError("PCHIP knots and log values must be strictly monotone")
        expected_end = math.nextafter(math.sqrt(dimension), math.inf)
        if knots[-1] != _dyadic(expected_end) or knots[-1] ** 2 < dimension:
            raise ValueError("PCHIP range must cover the full domain diameter")
        h = tuple(knots[i + 1] - knots[i] for i in range(8))
        secants = tuple((values[i + 1] - values[i]) / h[i] for i in range(8))
        slopes = [_endpoint_slope(secants[0], secants[1], h[0], h[1])]
        for i in range(1, 8):
            left, right = secants[i - 1], secants[i]
            if left * right <= 0:
                slopes.append(Fraction(0))
            else:
                w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
                slopes.append((w1 + w2) / (w1 / left + w2 / right))
        slopes.append(_endpoint_slope(secants[-1], secants[-2], h[-1], h[-2]))
        result = cls(knots, values, tuple(slopes))
        result._certify_monotone()
        return result

    def _coefficients(self, i: int) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        h = self.knots[i + 1] - self.knots[i]
        delta = self.values[i + 1] - self.values[i]
        return (self.values[i], h * self.slopes[i],
                3 * delta - h * (2 * self.slopes[i] + self.slopes[i + 1]),
                -2 * delta + h * (self.slopes[i] + self.slopes[i + 1]))

    def _certify_monotone(self) -> None:
        for i in range(8):
            _, c1, c2, c3 = self._coefficients(i)
            # Derivative is 3*c3*t^2 + 2*c2*t + c1 on t in [0,1].
            candidates = [c1, 3 * c3 + 2 * c2 + c1]
            if c3 < 0:
                vertex = -c2 / (3 * c3)
                if 0 < vertex < 1:
                    candidates.append(3 * c3 * vertex ** 2 +
                                      2 * c2 * vertex + c1)
            if max(candidates) > 0:
                raise ValueError("stored PCHIP law fails exact monotonicity proof")

    def interval(self, squared_radius: Fraction) -> int:
        if squared_radius < 0 or squared_radius > self.knots[-1] ** 2:
            raise ValueError("distance outside PCHIP domain")
        for i in range(8):
            if squared_radius <= self.knots[i + 1] ** 2:
                return i
        raise AssertionError("PCHIP interval selection failed")


def certified_term(physics: Physics, dimension: int, source: np.ndarray,
                   target: np.ndarray, strength: float,
                   *, ladder: tuple[int, ...] = PRECISION_LADDER,
                   pchip: LogPchipLaw | None = None) -> tuple[float | None, int]:
    """Return one correctly rounded strength*decay term and bits used."""
    if physics.codec_id not in (1, 2):
        raise ValueError("additive term requires exponential or log-PCHIP physics")
    squared = _distance_squared(source, target)
    gain = _dyadic(strength)
    if gain <= 0:
        raise ValueError("driver strength must be positive")
    if squared == 0:
        return float(gain), 0
    law = pchip if physics.codec_id == 2 else None
    if physics.codec_id == 2 and law is None:
        law = LogPchipLaw.from_physics(physics, dimension)
    index = law.interval(squared) if law is not None else None
    tau = (None if law is not None else
           _dyadic(next(field for field in physics.fields if field.name == "tau").as_array()))
    if tau is not None and tau <= 0:
        raise ValueError("exponential tau must be positive")
    for bits in ladder:
        if bits < 64:
            raise ValueError("precision ladder entries must be at least 64 bits")
        with ctx.workprec(bits):
            radius = _ball(squared).sqrt()
            if law is None:
                exponent = -radius / _ball(tau)
            else:
                assert index is not None
                h = law.knots[index + 1] - law.knots[index]
                t = (radius - _ball(law.knots[index])) / _ball(h)
                c0, c1, c2, c3 = law._coefficients(index)
                exponent = ((_ball(c3) * t + _ball(c2)) * t +
                            _ball(c1)) * t + _ball(c0)
            enclosure = _ball(gain) * exponent.exp()
            lower, upper = float(enclosure.lower()), float(enclosure.upper())
        if lower == upper and math.isfinite(lower):
            return lower, bits
    return None, ladder[-1] if ladder else 0


@dataclass(frozen=True, slots=True)
class IntensityResult:
    status: np.ndarray
    value: np.ndarray
    valid: np.ndarray
    reason_code: tuple[str, ...]
    precision_bits: np.ndarray


class AdditiveOracle:
    """Private additive oracle using certified terms and exact-rounded sum."""

    def __init__(self, scene: Scene, geometry_view: GeometryView,
                 *, ladder: tuple[int, ...] = PRECISION_LADDER):
        if scene.geometry_id != geometry_view.geometry_id or scene.dimension != geometry_view.dimension:
            raise ValueError("scene/geometry identity mismatch")
        if scene.physics.codec_id not in (1, 2):
            raise ValueError("additive oracle requires an additive physics codec")
        self.scene, self.view, self.ladder = scene, geometry_view, ladder
        self.law = (LogPchipLaw.from_physics(scene.physics, scene.dimension)
                    if scene.physics.codec_id == 2 else None)

    def query_response(self, points: object) -> IntensityResult:
        try:
            coordinates = _points(points, self.scene.dimension)
        except ValueError:
            return IntensityResult(np.array([QueryStatus.INVALID_INPUT], dtype=np.uint8),
                                   np.zeros(1), np.array([False]),
                                   ("shape_or_dtype",), np.zeros(1, dtype=np.uint32))
        membership = self.view.classify_points(coordinates)
        positions = self.scene.driver_position.as_array()
        strengths = self.scene.driver_strength.as_array()
        statuses, values, reasons, bits_used = [], [], [], []
        for i, point in enumerate(coordinates):
            if membership.value[i] == Membership.INTERIOR:
                statuses.append(QueryStatus.IN_STRUCTURE)
                values.append(0.0)
                reasons.append("query_in_structure")
                bits_used.append(0)
                continue
            if membership.value[i] == Membership.BOUNDARY:
                statuses.append(QueryStatus.ON_BOUNDARY)
                values.append(0.0)
                reasons.append("query_on_boundary")
                bits_used.append(0)
                continue
            if membership.status[i] != QueryStatus.OK:
                statuses.append(int(membership.status[i]))
                values.append(0.0)
                reasons.append(membership.reason_code[i])
                bits_used.append(0)
                continue
            terms, maximum_bits, failure = [], 0, None
            for source, strength in zip(positions, strengths, strict=True):
                path = self.view.segment_intersect(source, point)
                if path.status[0] != QueryStatus.OK:
                    failure = (int(path.status[0]), "driver_path_unproved")
                    break
                if path.value[0] == SegmentDecision.HIT:
                    continue
                term, bits = certified_term(self.scene.physics, self.scene.dimension,
                                            source, point, float(strength),
                                            ladder=self.ladder, pchip=self.law)
                maximum_bits = max(maximum_bits, bits)
                if term is None:
                    failure = (QueryStatus.UNRESOLVED, "decay_rounding_unproved")
                    break
                terms.append(term)
            if failure is None:
                try:
                    answer = _exact_rounded_sum(terms)
                except OverflowError:
                    failure = (QueryStatus.UNRESOLVED, "sum_outside_binary64")
            if failure is None:
                statuses.append(QueryStatus.OK)
                values.append(answer)
                reasons.append("")
            else:
                statuses.append(failure[0])
                values.append(0.0)
                reasons.append(failure[1])
            bits_used.append(maximum_bits)
        status_array = np.asarray(statuses, dtype=np.uint8)
        return IntensityResult(status_array, np.asarray(values, dtype=np.float64),
                               status_array == QueryStatus.OK, tuple(reasons),
                               np.asarray(bits_used, dtype=np.uint32))
