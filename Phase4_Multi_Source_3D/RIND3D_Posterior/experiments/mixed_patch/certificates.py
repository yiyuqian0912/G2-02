"""Exact visibility certificates on Q (sources) × W (receiver centres).

Coordinates are numerators on one common positive lattice denominator. Integers
in this module are Python integers, including every coefficient of a body.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product, combinations
from math import factorial, gcd, lcm


def integer_row(row):
    """Positive rescaling of an exact canonical row, with no float round trip."""
    scale = lcm(*(c.denominator for c in row))
    v = [int(c * scale) for c in row]
    div = gcd(*v)
    if not div:
        raise ValueError("empty plane")
    return tuple(c // div for c in v)


def value(row, point, denominator):
    return sum(a * x for a, x in zip(row[:3], point)) + row[3] * denominator


def corners(box):
    return tuple(product(*zip(*box)))


def determinant_range(f_s, b_s, f_x, b_x):
    # Separately affine in the source and receiver coordinates. The maximum
    # and minimum on a product of boxes are attained at pairs of vertices.
    vals = [fs * bx - bs * fx for fs, bs in zip(f_s, b_s)
            for fx, bx in zip(f_x, b_x)]
    return min(vals), max(vals)


def body_chart(body, sv, rv, denom):
    ss = [[value(row, p, denom) for p in sv] for row in body]
    rr = [[value(row, p, denom) for p in rv] for row in body]
    front = [i for i, v in enumerate(ss) if min(v) > 0]
    back = [i for i, v in enumerate(ss) if max(v) < 0]
    if any(min(rr[i]) > 0 for i in front):
        return "miss", None
    uncertain = []
    for i in front:
        for j in back:
            low, high = determinant_range(ss[i], ss[j], rr[i], rr[j])
            if low > 0:
                return "miss", None
            if high > 0:
                uncertain.append((i, j))
    if (len(front) + len(back) != len(body) or not front
            or any(max(rr[i]) > 0 for i in front)):
        return "unknown", None
    if not uncertain:
        return "hit", None
    if len(uncertain) == 1:
        i, j = uncertain[0]
        if max(rr[i]) < 0:  # denominator strictly negative for every receiver
            return "pencil", (i, j)
    return "unknown", None


def mixed_chart(bodies, source_box, receiver_box, denom):
    sv, rv = corners(source_box), corners(receiver_box)
    possible = []
    for body_id, body in enumerate(bodies):
        state, pair = body_chart(body, sv, rv, denom)
        if state == "hit":
            return "constant_hit", None
        if state != "miss":
            possible.append((state, body_id, pair))
    if not possible:
        return "constant_miss", None
    if len(possible) == 1 and possible[0][0] == "pencil":
        _, c, (i, j) = possible[0]
        return "pencil", (c, i, j)
    return "unknown", None


@dataclass(frozen=True)
class Band:
    front: tuple
    back: tuple
    lower: Fraction
    upper: Fraction
    active_witness: tuple
    inactive_witness: tuple

    def accepts(self, point, denom):
        a = value(self.front, point, denom)
        if a <= 0:
            raise ValueError("outside the certified positive-denominator chart")
        b = value(self.back, point, denom)
        return (self.lower.numerator * a < b * self.lower.denominator and
                b * self.upper.denominator <= self.upper.numerator * a)

    def fraction_of_box(self, box, denom):
        """Exact fraction satisfying the chart, conditional on its validity over box."""
        if min(value(self.front, s, denom) for s in corners(box)) <= 0:
            raise ValueError("source denominator changes sign")
        if self.lower >= self.upper:
            return Fraction(0)
        def cdf(t):
            row = tuple(t.denominator * b - t.numerator * a for a, b
                        in zip(self.front, self.back))
            return halfspace_fraction(row, box, denom)
        return cdf(self.upper) - cdf(self.lower)


def compile_band(front, back, observations, denom):
    """Retain extremizers of the pencil for both FREE classes, exactly."""
    best = {}
    for p, active in observations:
        f = value(front, p, denom)
        if f >= 0:
            raise ValueError("receiver denominator is not strictly negative")
        ratio = Fraction(value(back, p, denom), f)
        if active not in best or (ratio > best[active][0] if active else ratio < best[active][0]):
            best[active] = ratio, p
    if set(best) != {False, True}:
        raise ValueError("expected both FREE response classes")
    return Band(front, back, best[True][0], best[False][0], best[True][1], best[False][1])


def halfspace_fraction(row, box, denom):
    """Relative volume of row <= 0 in a full-dimensional axis-aligned box.

    Reflect axes so every coefficient is nonnegative, then integrate a clipped
    simplex by inclusion/exclusion (at most eight terms).
    """
    lo, hi = box
    if any(b <= a for a, b in zip(lo, hi)):
        raise ValueError("zero-volume box")
    base = row[3] * denom
    widths = []
    for a, l, h in zip(row[:3], lo, hi):
        base += min(a * l, a * h)
        if a:
            widths.append(abs(a) * (h - l))
    if base > 0:
        return Fraction(0)
    if base + sum(widths) <= 0:
        return Fraction(1)
    d = len(widths)
    scale = factorial(d)
    for w in widths:
        scale *= w
    total = sum((-1)**k * max(0, -base - sum(sub))**d for k in range(d+1)
                for sub in combinations(widths, k))
    return Fraction(total, scale)


def segment_hits(body, source, receiver, denom):
    """Independent reference: clip t in [0,1] against each closed body plane."""
    low, high = Fraction(0), Fraction(1)
    for plane in body:
        a = value(plane, source, denom)
        b = value(plane, receiver, denom)
        if a > 0 and b > 0:
            return False
        if a > 0:
            low = max(low, Fraction(a, a - b))
        elif b > 0:
            high = min(high, Fraction(-a, b - a))
        if low > high:
            return False
    return True
