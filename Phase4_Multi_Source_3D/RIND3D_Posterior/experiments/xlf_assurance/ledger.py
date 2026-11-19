"""Executable specification of online volume accounting. Standard library only.

Regions are rational axis boxes. The fixed classify predicate is a trusted input
to this model. The scheduler can request an assessment but cannot supply its answer.
The wire format here is for a reference experiment, not a dataset format.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
from math import isfinite, prod
from struct import Struct

N = 128
M = N**3
CAP = Q(1, M)
TICKS = 2**52
RECORD = Struct("<Id")


@dataclass(frozen=True)
class Box:
    lo: tuple
    hi: tuple

    def __post_init__(self):
        if len(self.lo) != 3 or len(self.hi) != 3:
            raise ValueError("three axes required")
        if not all(isinstance(v, (int, Q)) for v in self.lo + self.hi):
            raise ValueError("use exact rational coordinates")
        if not all(0 <= a < b <= 1 for a, b in zip(self.lo, self.hi)):
            raise ValueError("empty or non-world box")

    @property
    def volume(self):
        return prod(Q(b - a) for a, b in zip(self.lo, self.hi))

    def contains(self, other):
        return all(a <= c and d <= b for a, b, c, d in
                   zip(self.lo, self.hi, other.lo, other.hi))

    def overlap(self, other):
        return prod(max(Q(0), min(b, d) - max(a, c)) for a, b, c, d in
                    zip(self.lo, self.hi, other.lo, other.hi))

    def vertices(self):
        return product(*zip(self.lo, self.hi))


WORLD = Box((0, 0, 0), (1, 1, 1))


def grid_box(cid):
    if not 0 <= cid < M:
        raise ValueError("invalid cell")
    indices = (cid // N**2, cid // N % N, cid % N)
    return Box(tuple(Q(i, N) for i in indices), tuple(Q(i + 1, N) for i in indices))


def containing_cell(box):
    x, y, z = (int(N * v) for v in box.lo)
    cid = (x * N + y) * N + z
    return cid if grid_box(cid).contains(box) else None


class Ledger:
    """Single owner, private state, unique binary paths. Copy-on-write reference."""
    def __init__(self, classifier):
        self._classifier = classifier
        self._leaves = {1: (WORLD, "R")}

    def _open(self, key):
        box, status = self._leaves[key]
        if status != "R":
            raise ValueError("region already settled")
        return box

    def split(self, key, axis, cut):
        box = self._open(key)
        if axis not in (0, 1, 2) or not isinstance(cut, (int, Q)):
            raise ValueError("invalid cut")
        if not box.lo[axis] < cut < box.hi[axis]:
            raise ValueError("cut must be interior")
        lo, hi = list(box.lo), list(box.hi)
        lo[axis] = hi[axis] = cut
        first, second = Box(box.lo, tuple(hi)), Box(tuple(lo), box.hi)
        new = self._leaves.copy()
        del new[key]
        new[2 * key], new[2 * key + 1] = (first, "R"), (second, "R")
        self._leaves = new
        return 2 * key, 2 * key + 1

    def assess(self, key):
        box = self._open(key)
        answer = self._classifier(box)
        if answer not in ("IN", "OUT", "R"):
            raise ValueError("unknown proof outcome")
        if answer == "IN" and containing_cell(box) is None:
            return "R"  # require grid cuts before taking an IN credit
        self._leaves[key] = (box, answer)
        return answer

    def credits(self):
        lower, outside = {}, []
        for box, status in self._leaves.values():
            if status == "IN":
                cid = containing_cell(box)
                lower[cid] = lower.get(cid, Q(0)) + box.volume
            elif status == "OUT":
                outside.append(box)
        return lower, outside

    def certify(self, tolerance):
        if not isinstance(tolerance, (int, Q)) or not 0 <= tolerance <= 1:
            raise ValueError("exact tolerance required")
        lower, outside = self.credits()
        if not lower:
            return "ABSTAIN", None, None
        payload = encode(lower)
        bound = bound_on_bytes(payload, lower, outside)
        if bound > tolerance:
            return "ABSTAIN", None, bound
        return "CERTIFIED", payload, bound


def encode(lower):
    total = sum(lower.values(), Q(0))
    if total <= 0 or any(v <= 0 for v in lower.values()):
        raise ValueError("positive lower mass required")
    quotas = {c: TICKS * Q(m) / total for c, m in lower.items()}
    ticks = {c: int(x) for c, x in quotas.items()}
    order = sorted(ticks, key=lambda c: (-(quotas[c] - ticks[c]), c))
    for c in order[:TICKS - sum(ticks.values())]:
        ticks[c] += 1
    return b"".join(RECORD.pack(c, n / TICKS) for c, n in sorted(ticks.items()) if n)


def decode(payload):
    if len(payload) % RECORD.size or len(payload) // RECORD.size > M:
        raise ValueError("invalid record length")
    probabilities = {}
    previous = -1
    for cid, value in RECORD.iter_unpack(payload):
        if not previous < cid < M or not isfinite(value) or not 0 < value <= 1:
            raise ValueError("invalid probability record")
        probabilities[cid] = Q(value)
        previous = cid
    if sum(probabilities.values(), Q(0)) != 1:
        raise ValueError("payload is not exactly a probability")
    return probabilities


def bound_on_bytes(payload, lower, outside):
    """Credits are internal, disjoint and proven. Return the exact upper error."""
    p = decode(payload)
    if sum(lower.values(), Q(0)) <= 0:
        raise ValueError("posterior positivity not established")
    out_total = sum((b.volume for b in outside), Q(0))
    common = Q(0)
    for cid, probability in p.items():
        l = lower.get(cid, Q(0))
        if l:
            cell = grid_box(cid)
            out_here = sum((b.overlap(cell) for b in outside), Q(0))
            upper_other = 1 - CAP - (out_total - out_here)
            if upper_other < 0:
                raise ValueError("inconsistent credits")
            common += min(probability, l / (l + upper_other))
    return 1 - common
