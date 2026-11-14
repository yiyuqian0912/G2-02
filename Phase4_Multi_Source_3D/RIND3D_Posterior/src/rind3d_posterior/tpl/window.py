"""Observation windows at an arbitrary physical resolution N (1024 for RIND3D v0.1, 128 for Stage A).

Receivers are voxel centres (o + v + 1/2) / N.  cls[x, y, z]: 0 dark FREE, 1 lit FREE, 2 SOLID.
A *pure* pyramid is used: a block is an item only if all its voxels have the same class (0 or 1), so block
constraints are exact (no SOLID voxel inside a dark or lit block).
"""
import numpy as np

DARK, LIT, SOLID, MIXED = 0, 1, 2, 3


def enc(l, bx, by, bz):
    return (np.int64(l) << 48) | (np.int64(bx) << 32) | (np.int64(by) << 16) | np.int64(bz)


class Window:
    def __init__(self, cls, origin, N):
        cls = np.ascontiguousarray(cls, np.uint8)
        L = cls.shape[0]
        q = int(round(np.log2(L)))
        if (1 << q) != L or cls.shape != (L, L, L):
            raise ValueError('window side must be a power of two')
        self.cls, self.origin, self.N, self.L, self.q = cls, np.asarray(origin, np.int64), int(N), L, q
        codes = []
        d = cls == DARK
        li = cls == LIT
        s = cls == SOLID
        for l in range(q + 1):
            code = np.full(d.shape, MIXED, np.uint8)
            code[d] = DARK
            code[li] = LIT
            code[s] = SOLID
            codes.append(code.reshape(-1))
            if l < q:
                n = d.shape[0] // 2
                d = d.reshape(n, 2, n, 2, n, 2).all(axis=(1, 3, 5))
                li = li.reshape(n, 2, n, 2, n, 2).all(axis=(1, 3, 5))
                s = s.reshape(n, 2, n, 2, n, 2).all(axis=(1, 3, 5))
        self.pyr = np.concatenate(codes)
        self.off = np.zeros(q + 2, np.int64)
        for l in range(q + 1):
            self.off[l + 1] = self.off[l] + codes[l].size

    def code_at(self, l, bx, by, bz):
        n = 1 << (self.q - l)
        return int(self.pyr[self.off[l] + (bx * n + by) * n + bz])

    def top_items(self):
        """Maximal pure non-SOLID blocks (codes, classes)."""
        out_c, out_k = [], []
        stack = [(self.q, 0, 0, 0)]
        while stack:
            l, bx, by, bz = stack.pop()
            k = self.code_at(l, bx, by, bz)
            if k == SOLID:
                continue
            if k == MIXED:
                for dx in range(2):
                    for dy in range(2):
                        for dz in range(2):
                            stack.append((l - 1, 2 * bx + dx, 2 * by + dy, 2 * bz + dz))
                continue
            out_c.append(enc(l, bx, by, bz))
            out_k.append(k)
        return np.array(out_c, np.int64), np.array(out_k, np.int8)

    def counts(self):
        return dict(dark=int((self.cls == DARK).sum()), lit=int((self.cls == LIT).sum()),
                    solid=int((self.cls == SOLID).sum()))

    def boundary_mask(self):
        lit = self.cls == LIT
        dark = self.cls == DARK
        bd = np.zeros_like(lit)
        for ax in range(3):
            n = lit.shape[ax]
            a = [slice(None)] * 3; b = [slice(None)] * 3
            a[ax] = slice(0, n - 1); b[ax] = slice(1, n)
            a = tuple(a); b = tuple(b)
            x = (lit[a] & dark[b]) | (dark[a] & lit[b])
            bd[a] |= x
            bd[b] |= x
        return bd

    def sentinels(self, n_boundary=512, n_class=64, seed=0):
        """Receiver points (n,3) and lit flags used only to *refute* source boxes in the coarse stage."""
        rng = np.random.default_rng(seed)
        lit = self.cls == LIT
        dark = self.cls == DARK
        bd = self.boundary_mask()
        picks = []
        for mask, n in ((bd, n_boundary), (lit & ~bd, n_class), (dark & ~bd, n_class)):
            idx = np.flatnonzero(mask.reshape(-1))
            if len(idx) and n > 0:
                picks.append(rng.choice(idx, min(n, len(idx)), replace=False))
        idx = np.concatenate(picks) if picks else np.zeros(0, np.int64)
        v = np.stack(np.unravel_index(idx, self.cls.shape), 1)
        X = (self.origin + v + 0.5) / self.N
        return np.ascontiguousarray(X), (self.cls.reshape(-1)[idx] == LIT).astype(np.int8)

    def receivers(self):
        """All FREE receivers (m,3) and observations (1 lit / 0 dark)."""
        g = np.stack(np.meshgrid(*[np.arange(self.L)] * 3, indexing='ij'), -1).reshape(-1, 3)
        c = self.cls.reshape(-1)
        free = c != SOLID
        X = (self.origin + g[free] + 0.5) / self.N
        return np.ascontiguousarray(X), (c[free] == LIT).astype(np.int8)

    def box(self):
        return self.origin / self.N, (self.origin + self.L) / self.N
