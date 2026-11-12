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
