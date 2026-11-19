"""Property tests of the accounting core: exact encoding and the TV bound on decoded bytes.

Random exact instances (standard library only). For each instance a hidden "true" cell
mass vector v is drawn; credits are generated so that the kernel hypotheses hold
(lower <= v per cell, IN uppers >= IN masses, unknown pieces cover the rest); then
  * decode(encode(lower)) is an exact probability vector, within one tick per cell;
  * kernel.wire_bound >= the exact theorem value and <= it + (#cells)/2^96;
  * the true TV(p, v/|v|) never exceeds wire_bound (soundness of theorem + code).
"""
from fractions import Fraction
import random

import pytest

import inputs  # noqa: F401
from kernel import wire_bound, COMMON
from ledger import TICKS, decode, encode

CAP = 1 << 139  # one 128^3 cell in units of 2^-160


def exact_bound(p, lower, upper_in, unknown_total, unknown_in):
    total = sum(upper_in.values()) + unknown_total
    common = Fraction(0)
    for c, pc in p.items():
        l = lower.get(c, 0)
        if l:
            outside = total - upper_in.get(c, 0) - unknown_in.get(c, 0)
            common += min(pc, Fraction(l, l + outside))
    return 1 - common


def instance(rng):
    m = rng.randint(1, 12)
    cells = rng.sample(range(128 ** 3), m)
    v = {c: rng.randint(1, CAP) for c in cells}                  # true feasible mass per cell
    lower = {c: rng.randint(0, v[c]) if rng.random() < .8 else v[c] for c in cells}
    if not any(lower.values()):
        lower[cells[0]] = v[cells[0]]
    upper_in = {c: lower[c] + rng.randint(0, 3) for c in cells if lower[c]}  # IN masses + rounding
    # unknown pieces: cover every unproved part; some single-cell, some spanning cells
    unknown_total, unknown_in = 0, {}
    for c in cells:
        rest = v[c] - lower[c]
        if rest:
            extra = rng.randint(0, rest)
            unknown_total += rest + extra
            if rng.random() < .5:
                unknown_in[c] = unknown_in.get(c, 0) + rest + extra
    unknown_total += rng.randint(0, CAP)  # possible mass in cells with no feasible volume
    return v, {c: l for c, l in lower.items() if l}, upper_in, unknown_total, unknown_in


@pytest.mark.parametrize("seed", range(300))
def test_wire_bound_properties(seed):
    rng = random.Random(seed)
    v, lower, upper_in, ut, ui = instance(rng)
    payload = encode(lower)
    p = decode(payload)
    assert sum(p.values()) == 1
    L = sum(lower.values())
    for c, l in lower.items():
        assert abs(p.get(c, 0) - Fraction(l, L)) < Fraction(1, TICKS)
    b = wire_bound(payload, lower, upper_in, ut, ui)
    e = exact_bound(p, lower, upper_in, ut, ui)
    assert e <= b <= e + Fraction(len(p), COMMON)
    V = sum(v.values())
    true_tv = sum(abs(p.get(c, 0) - Fraction(v.get(c, 0), V)) for c in set(p) | set(v)) / 2
    assert true_tv <= b


def test_rejects_tampered_or_invalid_payloads():
    lower = {5: 3, 9: 1}
    payload = bytearray(encode(lower))
    payload[-1] ^= 0x01
    with pytest.raises(ValueError):
        decode(bytes(payload))          # no longer exactly normalized
    with pytest.raises(ValueError):
        wire_bound(encode(lower), {}, {}, 0, {})  # no positive feasible mass


@pytest.mark.parametrize("seed", range(200))
def test_integer_encoder_matches_reference(seed):
    from cbc_kernel import encode_int
    rng = random.Random(10_000 + seed)
    m = rng.randint(1, 40)
    lower = {c: rng.choice((1, rng.randint(1, 1 << 160), rng.randint(1, 1 << 20))) for c in rng.sample(range(128 ** 3), m)}
    if rng.random() < 0.3:  # equal masses exercise the tie-breaking by cell id
        v = rng.randint(1, 1 << 100)
        lower = {c: v for c in lower}
    assert encode_int(lower) == encode(lower)
