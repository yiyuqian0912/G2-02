"""Window helpers for FastLabel (production copy of experiments/fastlabel/fastwin.py, v3 helpers).
All outputs are identical to the originals (class tensor = SceneObs.cls; boundary mask, sentinels, point items and
top items = rind3d_posterior.tpl.window.Window); verified in RIND3D_Posterior/experiments/fastlabel/test_fastwin2.py."""
import numpy as np
from numba import njit
from rind3d_posterior.tpl.window import Window

class LazyWindow(Window):
    def __init__(self, cls, origin, N):
        cls = np.ascontiguousarray(cls, np.uint8); L = cls.shape[0]; q = int(round(np.log2(L)))
        if (1 << q) != L or cls.shape != (L, L, L):
            raise ValueError("window side must be a power of two")
        self.cls, self.origin, self.N, self.L, self.q = cls, np.asarray(origin, np.int64), int(N), L, q
        self._pyr = None

    def _build(self):
        Window.__init__(self, self.cls, self.origin, self.N)
        self._pyr = True

    def top_items(self):
        if self._pyr is None:
            self._build()
        return Window.top_items(self)


@njit(cache=True)
def _boundary(cls, out):
    """Identical to Window.boundary_mask: FREE voxels with a 6-neighbour of the other FREE class."""
    L = cls.shape[0]
    for x in range(L):
        for y in range(L):
            for z in range(L):
                c = cls[x, y, z]
                if c == 2:
                    continue
                o = 1 - c
                if (x + 1 < L and cls[x + 1, y, z] == o) or (y + 1 < L and cls[x, y + 1, z] == o) or (z + 1 < L and cls[x, y, z + 1] == o):
                    out[x, y, z] = True
                    if x + 1 < L and cls[x + 1, y, z] == o:
                        out[x + 1, y, z] = True
                    if y + 1 < L and cls[x, y + 1, z] == o:
                        out[x, y + 1, z] = True
                    if z + 1 < L and cls[x, y, z + 1] == o:
                        out[x, y, z + 1] = True
    return out


@njit(cache=True)
def _top_items(cls, q):
    """Identical to Window.top_items (pure pyramid: 0 dark, 1 lit, 2 solid, 3 mixed; LIFO traversal, same order)."""
    L = cls.shape[0]
    levels = []
    cur = cls.copy()
    levels.append(cur)
    for l in range(q):
        n = cur.shape[0] // 2
        nxt = np.empty((n, n, n), np.uint8)
        for bx in range(n):
            for by in range(n):
                for bz in range(n):
                    c = cur[2 * bx, 2 * by, 2 * bz]
                    if c != 3:
                        for dx in range(2):
                            for dy in range(2):
                                for dz in range(2):
                                    if cur[2 * bx + dx, 2 * by + dy, 2 * bz + dz] != c:
                                        c = 3
                    nxt[bx, by, bz] = c
        levels.append(nxt)
        cur = nxt
    cap = 1 << 20
    codes = np.empty(cap, np.int64); kls = np.empty(cap, np.int8); nout = 0
    st = np.empty((8 * (q + 1) + 8, 4), np.int64); top = 0
    st[0, 0] = q; st[0, 1] = 0; st[0, 2] = 0; st[0, 3] = 0; top = 1
    while top > 0:
        top -= 1
        l = st[top, 0]; bx = st[top, 1]; by = st[top, 2]; bz = st[top, 3]
        k = levels[l][bx, by, bz]
        if k == 2:
            continue
        if k == 3:
            for dx in range(2):
                for dy in range(2):
                    for dz in range(2):
                        st[top, 0] = l - 1; st[top, 1] = 2 * bx + dx; st[top, 2] = 2 * by + dy; st[top, 3] = 2 * bz + dz
                        top += 1
            continue
        if nout == codes.shape[0]:
            codes = np.concatenate((codes, np.empty(codes.shape[0], np.int64)))
            kls = np.concatenate((kls, np.empty(kls.shape[0], np.int8)))
        codes[nout] = (l << 48) | (bx << 32) | (by << 16) | bz
        kls[nout] = k
        nout += 1
    return codes[:nout].copy(), kls[:nout].copy()


class LazyWindow2(LazyWindow):
    """LazyWindow with numba boundary mask (cached) and numba top_items; outputs identical to Window."""
    def boundary_mask(self):
        if getattr(self, "_bd", None) is None:
            self._bd = _boundary(self.cls, np.zeros(self.cls.shape, np.bool_))
        return self._bd

    def top_items(self):
        if getattr(self, "_items", None) is None:
            self._items = _top_items(self.cls, self.q)
        return self._items


@njit(cache=True)
def _cls_from_bits_blocked(lit, sol, L, out):
    """Same values as _cls_from_bits; loop (z-block of 64, y, x, z): the 64 source rows of a (z-block, y) stay in L1
    while every x writes one full 64-byte line of out[x, y, z0:z0+64]."""
    LL = L * L
    Z = 64 if L >= 64 else L
    for z0 in range(0, L, Z):
        for y in range(L):
            for x in range(L):
                sh = 7 - (x & 7)
                base = y * L + x
                for z in range(z0, z0 + Z):
                    i = z * LL + base
                    if (sol[i >> 3] >> sh) & 1:
                        out[x, y, z] = 2
                    else:
                        out[x, y, z] = (lit[i >> 3] >> sh) & 1
    return out


def fast_cls3(S, w, s):
    L = int(S.W[w]["L"]); dz = S.dz
    def raw(j):
        o, n = S._pos[(w, j)]
        return np.frombuffer(dz.decompress(S.blob[o:o + n]), np.uint8)
    return _cls_from_bits_blocked(raw(s), raw(-1), L, np.empty((L, L, L), np.uint8))
