"""Direct response interfaces (contract v3 window eligibility).

Definition (contract rind3d-100k-contract-v3, `eligibility.rule = "direct_interface"`). Channel j of window W has a
**direct FREE–FREE response interface** iff there are two 6-adjacent receivers u, v of W (|u - v|_1 = 1 voxel) with
    u FREE, v FREE, R_j(u) != R_j(v),
where R_j is the binary response of source j (1 = response-active: the closed segment from the source to the
receiver misses the closed solid; legacy name "lit"). m(W) = number of such channels. The v1/v2 rule ("mixed": some
active FREE and some inactive FREE receiver anywhere in W) is weaker: a mixed channel can have every active/inactive
contact through SOLID (a SOLID-separated window), which v3 excludes from the primary population.

Everything here is exact on the stored bits; nothing is estimated.
* `interface_axes(lit, solid)`: per-axis counts of interface edges of full crops bool [Z, Y, X].
* `TileScan`: the same predicate accumulated over 64^3 tiles rendered in any order, including the edges that cross
  tile faces, so a window can be decided without assembling the full crop and with an early stop.
* `lattice_path` / bisection (used by `local_eval`): a constructive proof of an interface from exact point queries.
"""
import numpy as np

TILE = 64
RULES = ("mixed", "direct_interface")


def interface_axes(lit, solid):
    """Number of direct FREE–FREE interface edges along Z, Y, X of bool crops [Z, Y, X] (SOLID receivers unlit)."""
    lit = np.asarray(lit, bool); free = ~np.asarray(solid, bool)
    out = []
    for ax in range(3):
        a = [slice(None)] * 3; b = [slice(None)] * 3
        a[ax] = slice(1, None); b[ax] = slice(None, -1)
        a = tuple(a); b = tuple(b)
        out.append(int(np.count_nonzero((lit[a] ^ lit[b]) & free[a] & free[b])))
    return out


def has_interface(lit, solid):
    return sum(interface_axes(lit, solid)) > 0


def is_mixed(lit, solid):
    """The v1/v2 rule: an active FREE and an inactive FREE receiver exist."""
    free = ~np.asarray(solid, bool); lit = np.asarray(lit, bool)
    return bool(np.any(lit & free) and np.any(~lit & free))


def channel_class(lit, solid):
    """'interface' | 'mixed_no_interface' (SOLID-separated) | 'all_active' | 'all_inactive' | 'no_free'."""
    free = ~np.asarray(solid, bool); lit = np.asarray(lit, bool)
    if not free.any():
        return "no_free"
    if has_interface(lit, solid):
        return "interface"
    a = bool(np.any(lit & free)); i = bool(np.any(~lit & free))
    return "mixed_no_interface" if (a and i) else ("all_active" if a else "all_inactive")


class TileScan:
    """Exact interface / mixed / FREE accumulation over the 64^3 tiles of an L^3 window (L a multiple of 64).
    Tiles are added as (tile offset (a, b, c) in tile units along x, y, z; lit bool [64,64,64] ZYX; solid bool ZYX).
    Interface edges inside a tile are tested at once; an edge crossing a tile face is tested as soon as both tiles are
    present. `found` becomes True at the first interface edge; `complete` when every tile and face was examined."""

    def __init__(self, L):
        self.n = max(L // TILE, 1)
        self.faces = {}                       # offset -> {axis: (lo face (lit, free), hi face (lit, free))}
        self.found = False; self.nl = 0; self.nd = 0; self.ns = 0; self.tiles = 0

    def add(self, off, lit, solid, counts=None):
        """lit / solid: bool [64,64,64] ZYX arrays, or a Python bool for a uniform tile (then counts = (nl, nd, ns)
        must be given). A uniform lit or uniform FREE tile has constant faces and no interior interface edge."""
        self.tiles += 1
        uni_l = isinstance(lit, (bool, np.bool_)); uni_s = isinstance(solid, (bool, np.bool_))
        if counts is None:
            lit = np.asarray(lit, bool); solid = np.asarray(solid, bool)
            nl = int(np.count_nonzero(lit)); ns = int(np.count_nonzero(solid)); counts = (nl, lit.size - nl - ns, ns)
        self.nl += int(counts[0]); self.nd += int(counts[1]); self.ns += int(counts[2])
        free = (not solid) if uni_s else ~np.asarray(solid, bool)
        if not self.found and not uni_l and has_interface(lit, np.zeros_like(lit) if uni_s and not solid else
                                                          (np.ones_like(lit) if uni_s else solid)):
            self.found = True
        # faces in ZYX array axes: offset axis x -> array axis 2, y -> 1, z -> 0
        f = {}
        for t, arr_ax in ((0, 2), (1, 1), (2, 0)):
            lo = [slice(None)] * 3; hi = [slice(None)] * 3; lo[arr_ax] = 0; hi[arr_ax] = -1
            fl = (lambda s: np.bool_(lit)) if uni_l else (lambda s: lit[s])
            ff = (lambda s: np.bool_(free)) if uni_s else (lambda s: free[s])
            f[t] = ((fl(tuple(lo)), ff(tuple(lo))), (fl(tuple(hi)), ff(tuple(hi))))
        off = tuple(int(v) for v in off)
        self.faces[off] = f
        if not self.found:
            for t in range(3):
                nb = list(off); nb[t] -= 1                       # neighbour below: its hi face meets my lo face
                g = self.faces.get(tuple(nb))
                if g is not None and self._cross(g[t][1], f[t][0]):
                    self.found = True; break
                nb = list(off); nb[t] += 1                       # neighbour above: my hi face meets its lo face
                g = self.faces.get(tuple(nb))
                if g is not None and self._cross(f[t][1], g[t][0]):
                    self.found = True; break

    @staticmethod
    def _cross(a, b):
        (la, fa), (lb, fb) = a, b
        return bool(np.any((la ^ lb) & fa & fb))

    @property
    def complete(self):
        return self.tiles == self.n ** 3

    @property
    def mixed(self):
        """True/False when decided (an interface implies mixed); None while incomplete and not yet proven."""
        if self.nl > 0 and self.nd > 0:
            return True
        return False if self.complete else None


def lattice_path(a, b, order=(0, 1, 2)):
    """6-connected lattice path from voxel a to voxel b (inclusive), moving axis by axis in `order`."""
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    pts = [a.copy()]; cur = a.copy()
    for ax in order:
        step = 1 if b[ax] > cur[ax] else -1
        while cur[ax] != b[ax]:
            cur = cur.copy(); cur[ax] += step; pts.append(cur)
    return np.array(pts, np.int64)


def bisect_interface(query, path, lit_end_first):
    """Constructive interface proof on a lattice path whose two ends are FREE with different responses.
    query(vox (n, 3) xyz) -> (lit bool, solid bool) is an EXACT point predicate. Invariant: the two current ends are
    FREE with different responses; returns the adjacent pair (u, v) found, or None if a SOLID voxel was met (then the
    path proves nothing)."""
    lo, hi = 0, len(path) - 1
    r_lo = bool(lit_end_first)
    n_q = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lit, sol = query(path[mid:mid + 1]); n_q += 1
        if bool(sol[0]):
            return None, n_q
        if bool(lit[0]) == r_lo:
            lo = mid
        else:
            hi = mid
    return (path[lo], path[hi]), n_q
