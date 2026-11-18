"""The authoritative voxel solid G_V (union of closed occupied cells) and its exact queries.

Storage: sparse 64^3 bricks (`bricks`: key (bi, bj, bk) -> packed bits, or the FULL marker). Reconstructible exactly
from the generative metadata and the versioned rule (`build_world`); `digest()` hashes the occupied set.
Queries:
* `solid_crop(o, L)`: the window's SOLID observation bool [Z, Y, X] (identical to occupancy: a cell centre lies in
  G_V iff its own cell is occupied, since centres are interior to their cell and cells only share boundaries);
* `dense_bits()`: the 1024^3 packed bitset (128 MiB) used by the traversal kernels (materialized on demand);
* `legal_source(s, rho)`: s in the open cube with distance > rho from every occupied cell (exact);
* visibility: rind3d_voxel.visibility (closed segment against closed cells)."""
import hashlib
import time
from fractions import Fraction

import numpy as np

from .voxelize import Polytope, Ball, voxelize_body

BRICK = 64
FULL = b"F"


class VoxelWorld:
    def __init__(self, N=1024, rule="supercover"):
        self.N = int(N); self.rule = rule; self.bricks = {}; self._dense = None; self.stats = {}

    # ---------------------------------------------------------------- construction
    def add_cells(self, cells):
        cells = np.asarray(cells, np.int64).reshape(-1, 3)
        if not len(cells):
            return
        key = cells // BRICK; loc = cells % BRICK
        order = np.lexsort((key[:, 2], key[:, 1], key[:, 0])); key = key[order]; loc = loc[order]
        cut = np.flatnonzero(np.any(np.diff(key, axis=0) != 0, axis=1)) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(key)]):
            kk = tuple(int(v) for v in key[a])
            cur = self.bricks.get(kk)
            if cur == FULL:
                continue
            arr = np.zeros(BRICK ** 3, bool) if cur is None else np.unpackbits(np.frombuffer(cur, np.uint8), count=BRICK ** 3).astype(bool)
            l = loc[a:b]; arr[(l[:, 0] * BRICK + l[:, 1]) * BRICK + l[:, 2]] = True
            self.bricks[kk] = FULL if arr.all() else np.packbits(arr).tobytes()
        self._dense = None

    def n_occupied(self):
        n = 0
        for v in self.bricks.values():
            n += BRICK ** 3 if v == FULL else int(np.unpackbits(np.frombuffer(v, np.uint8)).sum())
        return n

    def nbytes_sparse(self):
        return sum(1 if v == FULL else len(v) for v in self.bricks.values()) + 24 * len(self.bricks)

    def digest(self):
        h = hashlib.sha256(f"{self.N}|{self.rule}".encode())
        for k in sorted(self.bricks):
            v = self.bricks[k]; h.update(np.array(k, np.int64).tobytes()); h.update(v if v == FULL else v)
        return h.hexdigest()

    # ---------------------------------------------------------------- queries
    def occupied(self, cells):
        cells = np.asarray(cells, np.int64).reshape(-1, 3)
        ok = np.all((cells >= 0) & (cells < self.N), axis=1)
        out = np.zeros(len(cells), bool)
        D = self.dense_bits()
        idx = (cells[ok, 0] * self.N + cells[ok, 1]) * self.N + cells[ok, 2]
        out[ok] = (D[idx >> 3] >> (7 - (idx & 7)).astype(np.uint8)) & 1 == 1
        return out

    def dense_bits(self):
        """Packed bitset over i (x) major, then j, then k (z): bit index (i*N + j)*N + k, big-endian within a byte."""
        if self._dense is None:
            t = time.perf_counter(); N = self.N
            D = np.zeros(N ** 3 // 8, np.uint8)
            g = np.stack(np.meshgrid(*[np.arange(BRICK)] * 3, indexing="ij"), -1).reshape(-1, 3)
            for k, v in self.bricks.items():
                arr = np.ones(BRICK ** 3, bool) if v == FULL else np.unpackbits(np.frombuffer(v, np.uint8), count=BRICK ** 3).astype(bool)
                c = g[arr] + np.array(k) * BRICK
                idx = (c[:, 0] * N + c[:, 1]) * N + c[:, 2]
                np.bitwise_or.at(D, idx >> 3, (1 << (7 - (idx & 7))).astype(np.uint8))
            self._dense = D; self.stats["dense_s"] = time.perf_counter() - t
        return self._dense

    def solid_crop(self, o, L):
        o = np.asarray(o, np.int64)
        g = np.stack(np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing="ij"), -1).reshape(-1, 3)   # z, y, x
        cells = o[None] + g[:, ::-1]
        return self.occupied(cells).reshape(L, L, L)

    def legal_source(self, s, rho=2.0 ** -26):
        s = np.asarray(s, float)
        if not np.all((s > 0) & (s < 1)):
            return False
        N = self.N
        cand = [sorted({int(np.floor((s[a] - rho) * N)), int(np.floor((s[a] + rho) * N))}) for a in range(3)]
        sq = [Fraction(float(v)) for v in s]; rq = Fraction(rho)
        import itertools
        for c in itertools.product(*cand):
            if any(v < 0 or v >= N for v in c) or not self.occupied(np.array(c))[0]:
                continue
            d2 = sum((min(max(sq[a], Fraction(c[a], N)), Fraction(c[a] + 1, N)) - sq[a]) ** 2 for a in range(3))
            if d2 <= rq * rq:
                return False
        return True


def bodies_from_context(context, balls=None):
    out = [Polytope(c.rows) for c in context.components] if context is not None else []
    if balls is not None:
        out += [Ball(b[:3], b[3]) for b in np.asarray(balls).reshape(-1, 4)]
    return out


def build_world(bodies, N=1024, rule="supercover"):
    """Deterministic: the union of every body's occupied cells under `rule` (order-independent)."""
    W = VoxelWorld(N, rule); t = time.perf_counter(); st = {}
    for b in bodies:
        W.add_cells(voxelize_body(b, N, rule, stats=st))
    W.stats.update(st, voxelize_s=time.perf_counter() - t)
    return W
