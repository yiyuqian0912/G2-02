"""Trusted receiver screens with inherited obligations (integer arithmetic only).

Both functions classify FREE receivers against per-body receiver-space regions of one
source polytope P, encoded by raster.encoded as directed fixed-point integer rows:

  group 3c   : guaranteed shadow of body c  (rows rounded up  -> a subset of the exact region)
  group 3c+1 : possible shadow of body c    (rows rounded down -> a superset of the exact region)
  group 3c+2 : anchor-source shadow (search hint only; never used for a decision)

A receiver lies in a group region iff every row of the group is <= 0 at its homogeneous
integer coordinates (2(o+i)+1, ..., 2N); all such coordinates are positive, which is what
makes coefficient-wise directed rounding sound. Range: |coef| <= 2^45 and coordinates
<= 2^11 (enforced by the kernel), so every row value has magnitude < 2^58 (int64-safe).

Decisions (for every source s in P, up to Lebesgue-null sets):
  active receiver in some guaranteed shadow          -> contradiction (P infeasible)
  inactive receiver outside every possible shadow    -> contradiction
  active receiver outside every possible shadow      -> proved for P
  inactive receiver inside some guaranteed shadow    -> proved for P
  otherwise                                          -> ambiguous (obligation remains)

A receiver proved on P stays proved on every subset of P (both regions are monotone in
P), so a child polytope inherits the parent's residual ambiguous list.
"""
import numpy as np
from numba import njit

ROW_LIMIT = 1 << 45
COORD_LIMIT = 1 << 11


def check_ranges(rows, origin, side, denom):
    if rows.size and int(np.abs(rows).max()) > ROW_LIMIT:
        raise ArithmeticError("screen row exceeds the verified fixed-point range")
    if denom > COORD_LIMIT or any(2 * (o + side) + 1 > COORD_LIMIT + 1 for o in origin):
        raise ArithmeticError("receiver coordinates exceed the verified range")


@njit(cache=True)
def scan_collect(cls, origin, denom, rows, off, skipped, cap):
    """Full-window scanline screen. Returns (n_ambiguous or -1, ids[:min(n,cap)], wrong, boundary)."""
    L = cls.shape[0]
    ids = np.empty(cap, np.int64)
    wrong_out = np.empty(cap, np.uint8)
    bnd_out = np.empty(cap, np.uint8)
    diff = np.zeros((3, L + 1), np.int32)
    counts = np.empty(3, np.int32)
    n = 0
    for z in range(L):
        wz = 2 * (origin[2] + z) + 1
        for y in range(L):
            wy = 2 * (origin[1] + y) + 1
            diff[:] = 0
            for group in range(len(off) - 1):
                lo, hi = 0, L - 1
                for i in range(off[group], off[group + 1]):
                    a, b, c, d = rows[i]
                    v = a * (2 * origin[0] + 1) + b * wy + c * wz + d * denom
                    if a > 0:
                        hi = min(hi, (-v) // (2 * a))
                    elif a < 0:
                        lo = max(lo, -((-v) // (-2 * a)))
                    elif v > 0:
                        hi = -1
                    if lo > hi:
                        break
                if lo <= hi:
                    diff[group % 3, lo] += 1
                    diff[group % 3, hi + 1] -= 1
            counts[:] = 0
            for x in range(L):
                for j in range(3):
                    counts[j] += diff[j, x]
                act = cls[z, y, x]
                if act == 2:
                    continue
                if (act == 1 and counts[0] > 0) or (act == 0 and counts[1] == 0):
                    return -1, ids[:0], wrong_out[:0], bnd_out[:0]
                if (act == 1 and counts[1] == 0) or (act == 0 and counts[0] > 0):
                    continue
                idx = (z * L + y) * L + x
                if len(skipped):
                    pos = np.searchsorted(skipped, idx)
                    if pos < len(skipped) and skipped[pos] == idx:
                        continue
                if n < cap:
                    ids[n] = idx
                    wrong_out[n] = (act == 1) == (counts[2] > 0)
                    bnd_out[n] = ((x > 0 and cls[z, y, x - 1] == 1 - act) or (x + 1 < L and cls[z, y, x + 1] == 1 - act)
                                  or (y > 0 and cls[z, y - 1, x] == 1 - act) or (y + 1 < L and cls[z, y + 1, x] == 1 - act)
                                  or (z > 0 and cls[z - 1, y, x] == 1 - act) or (z + 1 < L and cls[z + 1, y, x] == 1 - act))
                n += 1
    m = min(n, cap)
    return n, ids[:m], wrong_out[:m], bnd_out[:m]


@njit(cache=True)
def scan_ids(cls, ids_in, origin, denom, rows, off):
    """Screen only the listed receivers (an inherited obligation set, sorted ascending).

    Receivers are grouped by scanline (z, y); group intervals are computed once per
    scanline exactly as in scan_collect, so a decision is identical to the full scan.
    """
    L = cls.shape[0]
    k = len(ids_in)
    ng = len(off) - 1
    ids = np.empty(k, np.int64)
    wrong_out = np.empty(k, np.uint8)
    bnd_out = np.empty(k, np.uint8)
    glo = np.empty(ng, np.int64)
    ghi = np.empty(ng, np.int64)
    n = 0
    t = 0
    prev = -1
    while t < k:
        idx = ids_in[t]
        if idx <= prev or idx < 0 or idx >= L * L * L:
            return -3, ids[:0], wrong_out[:0], bnd_out[:0]  # unsorted or invalid list
        prev = idx
        line = idx // L
        z = line // L
        y = line % L
        wy = 2 * (origin[1] + y) + 1
        wz = 2 * (origin[2] + z) + 1
        for group in range(ng):
            lo, hi = 0, L - 1
            for i in range(off[group], off[group + 1]):
                a, b, c, d = rows[i]
                v = a * (2 * origin[0] + 1) + b * wy + c * wz + d * denom
                if a > 0:
                    hi = min(hi, (-v) // (2 * a))
                elif a < 0:
                    lo = max(lo, -((-v) // (-2 * a)))
                elif v > 0:
                    hi = -1
                if lo > hi:
                    break
            glo[group] = lo
            ghi[group] = hi
        while t < k and ids_in[t] // L == line:
            idx = ids_in[t]
            if t > 0 and idx <= ids_in[t - 1]:
                return -3, ids[:0], wrong_out[:0], bnd_out[:0]
            x = idx % L
            t += 1
            act = cls[z, y, x]
            if act == 2:
                return -2, ids[:0], wrong_out[:0], bnd_out[:0]  # never an obligation
            c0 = 0
            c1 = 0
            c2 = 0
            for group in range(ng):
                if glo[group] <= x and x <= ghi[group]:
                    g = group % 3
                    if g == 0:
                        c0 += 1
                    elif g == 1:
                        c1 += 1
                    else:
                        c2 += 1
            if (act == 1 and c0 > 0) or (act == 0 and c1 == 0):
                return -1, ids[:0], wrong_out[:0], bnd_out[:0]
            if (act == 1 and c1 == 0) or (act == 0 and c0 > 0):
                continue
            ids[n] = idx
            wrong_out[n] = (act == 1) == (c2 > 0)
            bnd_out[n] = ((x > 0 and cls[z, y, x - 1] == 1 - act) or (x + 1 < L and cls[z, y, x + 1] == 1 - act)
                          or (y > 0 and cls[z, y - 1, x] == 1 - act) or (y + 1 < L and cls[z, y + 1, x] == 1 - act)
                          or (z > 0 and cls[z - 1, y, x] == 1 - act) or (z + 1 < L and cls[z + 1, y, x] == 1 - act))
            n += 1
    return n, ids[:n], wrong_out[:n], bnd_out[:n]
