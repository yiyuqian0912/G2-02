"""Receiver-convexity proposals (untrusted search helper).

For a piece P and an obstacle K, let R_K be the dark receivers that are ambiguous on P and
whose only possible blocker is K. A source s reproduces all of R_K iff R_K lies in the shadow
S_K(s), which is convex in the receiver; hence iff the extreme points of conv(R_K) do
(XLF_CBC_DESIGN.md, section 4e). Enforcing only those extreme receivers, which is an exact
intersection with the cones C_K(r) and does not fragment the piece, therefore settles all of
R_K. Measured at L = 256: millions of receivers, 50-80 hull vertices.

Nothing here is trusted. The kernel enforces each proposed receiver exactly (it checks the
class and uses its own candidate bodies), and the next screen re-proves every receiver.
"""
import numpy as np
from numba import njit
from scipy.spatial import ConvexHull, QhullError

from screen2 import _intervals


@njit(cache=True)
def scan_extremes(cls, origin, denom, rows, off, skipped):
    """Per body c and scanline (z, y): min and max x of the ambiguous dark receivers whose only
    possible blocker is c (same interval arithmetic as screen2.scan_collect). -1 if none."""
    L = cls.shape[0]
    ng = len(off) - 1
    nb = ng // 3
    lo = np.full((nb, L * L), -1, np.int32)
    hi = np.full((nb, L * L), -1, np.int32)
    count = np.zeros(nb, np.int64)
    glo = np.empty(ng, np.int64)
    ghi = np.empty(ng, np.int64)
    for z in range(L):
        wz = 2 * (origin[2] + z) + 1
        for y in range(L):
            wy = 2 * (origin[1] + y) + 1
            _intervals(origin, denom, wy, wz, rows, off, L, glo, ghi)
            line = z * L + y
            for x in range(L):
                if cls[z, y, x] != 0:
                    continue
                g0 = 0
                single = -1
                npos = 0
                for c in range(nb):
                    if glo[3 * c] <= x and x <= ghi[3 * c]:
                        g0 += 1
                    if glo[3 * c + 1] <= x and x <= ghi[3 * c + 1]:
                        npos += 1
                        single = c
                if g0 > 0 or npos != 1:
                    continue  # proved dark, contradiction, or several possible blockers
                if len(skipped):
                    idx = (z * L + y) * L + x
                    pos = np.searchsorted(skipped, idx)
                    if pos < len(skipped) and skipped[pos] == idx:
                        continue
                if lo[single, line] < 0:
                    lo[single, line] = x
                hi[single, line] = x
                count[single] += 1
    return lo, hi, count


def hull_ids(L, xs, ys, zs):
    """Flat receiver ids of the convex-hull vertices of the given lattice points."""
    pts = np.column_stack((xs, ys, zs)).astype(float)
    if len(pts) < 5:
        return (zs.astype(np.int64) * L + ys) * L + xs
    try:
        v = ConvexHull(pts).vertices
    except QhullError:  # flat point set: every point is a proposal
        v = np.arange(len(pts))
    return (zs[v].astype(np.int64) * L + ys[v]) * L + xs[v]


def from_scan(L, lo, hi, count, minimum):
    """Hull receivers per body from scan_extremes output (bodies with >= minimum receivers)."""
    out = {}
    for c in np.nonzero(count >= minimum)[0]:
        lines = np.nonzero(lo[c] >= 0)[0]
        z, y = np.divmod(lines, L)
        xs = np.r_[lo[c, lines], hi[c, lines]]
        out[int(c)] = hull_ids(L, xs, np.r_[y, y], np.r_[z, z])
    return out


def from_list(L, ids, masks, dark, minimum):
    """Hull receivers per body from a stored obligation list with candidate masks."""
    out = {}
    single = (masks != 0) & ((masks & (masks - np.uint64(1))) == 0) & dark
    if single.sum() < minimum:
        return out
    m = masks[single]
    sel_ids = ids[single]
    for bit in np.unique(m):
        I = sel_ids[m == bit]
        if len(I) < minimum:
            continue
        z, yx = np.divmod(I, L * L)
        y, x = np.divmod(yx, L)
        line = z * L + y
        order = np.lexsort((x, line))
        lz = line[order]
        first = np.r_[True, lz[1:] != lz[:-1]]
        last = np.r_[lz[1:] != lz[:-1], True]
        k = np.unique(np.r_[order[first], order[last]])
        out[int(bit).bit_length() - 1] = hull_ids(L, x[k], y[k], z[k])
    return out
