"""Faster per-label setup: class tensor by integer arithmetic; block pyramid built lazily (only the exact phase 2 needs it)."""
import numpy as np
import zstandard
from numba import njit
from rind3d_posterior.tpl.window import Window


@njit(cache=True)
def _zyx_to_xyz(a, out):
    """out[x, y, z] = a[z, y, x], cache-blocked."""
    L = a.shape[0]; B = 16
    for z0 in range(0, L, B):
        for x0 in range(0, L, B):
            for y in range(L):
                for z in range(z0, min(z0 + B, L)):
                    for x in range(x0, min(x0 + B, L)):
                        out[x, y, z] = a[z, y, x]
    return out


def fast_cls(S, w, s):
    """cls[x, y, z] in {0 dark FREE, 1 lit FREE, 2 SOLID}, identical to SceneObs.cls."""
    L = int(S.W[w]["L"]); dz = S.dz
    def bits(j):
        o, n = S._pos[(w, j)]
        return np.unpackbits(np.frombuffer(dz.decompress(S.blob[o:o + n]), np.uint8), count=L ** 3)
    sol = bits(-1); lit = bits(s)
    c = (lit & (1 - sol)) + 2 * sol                    # uint8: SOLID wins (stored lit is never set on SOLID)
    return _zyx_to_xyz(c.reshape(L, L, L), np.empty((L, L, L), np.uint8))


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


# ---------------------------------------------------------------------------------------------------- v3 additions
@njit(cache=True)
def _cls_from_bits(lit, sol, L, out):
    """out[x, y, z] = 2 if SOLID else lit bit, from the stored big-endian packed ZYX crops. Sequential writes; the
    strided bit reads stay in cache (2 x L^3 / 8 bytes)."""
    LL = L * L
    for x in range(L):
        sh = 7 - (x & 7)
        for y in range(L):
            base = y * L + x
            for z in range(L):
                i = z * LL + base
                if (sol[i >> 3] >> sh) & 1:
                    out[x, y, z] = 2
                else:
                    out[x, y, z] = (lit[i >> 3] >> sh) & 1
    return out


def fast_cls2(S, w, s):
    """Identical to fast_cls / SceneObs.cls, without unpacking to bytes."""
    L = int(S.W[w]["L"]); dz = S.dz
    def raw(j):
        o, n = S._pos[(w, j)]
        return np.frombuffer(dz.decompress(S.blob[o:o + n]), np.uint8)
    return _cls_from_bits(raw(s), raw(-1), L, np.empty((L, L, L), np.uint8))


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
