"""Rigorous total-variation bound of a stored label against the exact posterior, in exact rational arithmetic.

Target: q(c) = v_c / V,  v_c = Vol(F n c),  V = sum v_c > 0. Certificate: rationals lo_c <= v_c <= hi_c + p_c with
sum_c p_c <= P (pool of undetermined mass whose cell is not resolved). With Lt = sum lo_c > 0 and the reference
estimate qt = lo / Lt:
    TV(qt, q) <= sup_{0 <= e <= x} sum_c e_c w_c / (Lt + E),   x_c = hi_c - lo_c, w_c = 1 - lo_c / Lt, E = sum e_c
(proof: |Lt e_c - lo_c E| <= e_c (Lt - lo_c) + lo_c (E - e_c), summed), and pool mass contributes like a cell with
w = 1 (the largest possible weight). The supremum of the linear-fractional form over the box is attained by adding
cells in decreasing w while w exceeds the running value (exact greedy). For a stored label qh:
    TV(qh, q) <= TV(qh, qt) + TV(qt, q),
with TV(qh, qt) computed exactly from the stored binary64 values. Positivity Lt > 0 is required: it proves
V > 0, i.e. that the posterior is well defined; otherwise no bound is issued."""
from fractions import Fraction as Q

import numpy as np


def tv_bound_exact(cells, lo, hi, pool=0.0):
    """(bound (Fraction), Lt (Fraction)). lo, hi: binary64 arrays (exact rationals), pool >= 0."""
    L = [Q(float(v)) for v in lo]; H = [Q(float(v)) for v in hi]
    if any(h < l for l, h in zip(L, H)):
        raise ValueError("hi < lo")
    Lt = sum(L, Q(0))
    if Lt <= 0:
        return None, Lt
    items = [(1 - l / Lt, h - l) for l, h in zip(L, H) if h > l]
    if pool > 0:
        items.append((Q(1), Q(float(pool))))
    items.sort(key=lambda t: -t[0])
    num = Q(0); den = Lt; best = Q(0)
    for w, x in items:
        if w <= best:
            break
        num += x * w; den += x
        best = max(best, num / den)
    return min(best, Q(1)), Lt


def label_from_bounds(cells, lo):
    """The released estimate qh = binary64 rounding of lo_c / Lt (cells with lo_c > 0)."""
    L = [Q(float(v)) for v in lo]; Lt = sum(L, Q(0))
    m = np.array([l > 0 for l in L], bool)
    return np.asarray(cells)[m], np.array([float(l / Lt) for l, k in zip(L, m) if k])


def tv_of_stored(cells_h, qh, cells, lo, hi, pool=0.0):
    """Certified TV(qh, q) for a stored label (cells_h, qh) and the verifier's enclosures. Returns dict."""
    b, Lt = tv_bound_exact(cells, lo, hi, pool)
    if b is None:
        return dict(status="undefined", reason="no certified positive feasible volume (Lt = 0)", tv=None)
    ref = {int(c): Q(float(v)) / Lt for c, v in zip(cells, lo) if v > 0}
    st = {int(c): Q(float(v)) for c, v in zip(cells_h, qh)}
    d = sum((abs(st.get(c, Q(0)) - ref.get(c, Q(0))) for c in set(ref) | set(st)), Q(0)) / 2
    tot = b + d
    t = float(tot)
    if Q(t) < tot:
        t = float(np.nextafter(t, np.inf))
    return dict(status="certified", tv=t, tv_reference=float(b), tv_rounding=float(d), Lt=float(Lt), Lt_exact=Lt)
