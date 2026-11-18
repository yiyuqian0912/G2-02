"""Exact visibility in the voxel world: the closed segment [s, x] against the union of closed occupied cells.

Contact convention (closed semantics, as Model A): touching an occupied cell's face, edge or vertex blocks. The
receiver x is a cell centre, so only its own cell contains it. A legal source is at distance > rho from every
occupied cell.

* `touched_cells_exact(s, x, N)`: the exact set of closed cells the closed segment meets, in rational arithmetic.
  This is the independent reference: crossing parameters of every lattice plane, the open pieces between them (one
  open cell each, or two when the segment runs inside a lattice plane), and every crossing point (all 2^m cells
  around it when it lies on m lattice planes at once).
* `free_exact(world, s, x)`: no touched cell occupied.
* `free_fast(world, s, X)`: vectorized Amanatides-Woo traversal in float64. A ray is handed to the exact reference
  when two crossing parameters coincide within a relative 1e-9 (an edge or vertex crossing), or when it starts on a
  lattice plane or runs exactly along one. Decisions therefore equal the exact rule."""
import itertools
import math
from fractions import Fraction

import numpy as np


def touched_cells_exact(s, x, N):
    sq = [Fraction(float(v)) for v in s]; xq = [Fraction(float(v)) for v in x]
    d = [xq[a] - sq[a] for a in range(3)]
    T = {Fraction(0), Fraction(1)}
    for a in range(3):
        if d[a] == 0:
            continue
        lo, hi = sorted((sq[a] * N, xq[a] * N))
        for k in range(int(np.ceil(float(lo))) - 1, int(np.floor(float(hi))) + 2):
            t = (Fraction(k, N) - sq[a]) / d[a]
            if 0 <= t <= 1:
                T.add(t)
    T = sorted(T)
    cells = set()

    def at(t):
        p = [sq[a] + t * d[a] for a in range(3)]
        cand = []
        for a in range(3):
            v = p[a] * N
            if v.denominator == 1:
                m = int(v); cand.append([c for c in (m - 1, m) if 0 <= c < N])
            else:
                cand.append([math.floor(v)])
        return cand
    for t in T:
        cells.update(itertools.product(*at(t)))
    for t0, t1 in zip(T[:-1], T[1:]):
        cells.update(itertools.product(*at((t0 + t1) / 2)))
    return cells


def free_exact(world, s, x):
    """Response-active: the closed segment touches no occupied cell. A receiver in an occupied cell (SOLID) touches its
    own cell, so it is inactive, as in Model A."""
    cells = np.array(sorted(touched_cells_exact(s, x, world.N)), np.int64)
    return not world.occupied(cells).any()


def free_fast(world, s, X, max_steps=None, tie_rel=1e-9):
    """Vectorized exact visibility of receivers X (n, 3) world coords from source s. Returns bool (n,)."""
    N = world.N; D = world.dense_bits()
    s = np.asarray(s, float); X = np.asarray(X, float)
    n = len(X); d = X - s[None]
    cur = np.floor(s * N).astype(np.int64)
    flag = np.zeros(n, bool)
    if np.any(np.abs(s * N - np.round(s * N)) < 1e-9):            # source on a lattice plane: exact
        flag[:] = True
    end = np.floor(X * N).astype(np.int64)
    step = np.sign(d).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        nxt = np.where(step > 0, (cur[None] + 1) / N, cur[None] / N)
        tmax = np.where(step != 0, (nxt - s[None]) / d, np.inf)
        tdel = np.where(step != 0, (1.0 / N) / np.abs(d), np.inf)
    c = np.broadcast_to(cur, (n, 3)).copy()
    free = np.ones(n, bool); active = ~flag
    # the source's own cell is free for a legal source; walk until the receiver's cell
    steps = 0; max_steps = max_steps or 3 * N + 3
    occ = lambda cc: ((D[((cc[:, 0] * N + cc[:, 1]) * N + cc[:, 2]) >> 3] >> (7 - (((cc[:, 0] * N + cc[:, 1]) * N + cc[:, 2]) & 7)).astype(np.uint8)) & 1) == 1
    while active.any() and steps < max_steps:
        steps += 1
        ia = np.flatnonzero(active)
        tm = tmax[ia]
        o = np.argsort(tm, axis=1)
        t1 = np.take_along_axis(tm, o[:, :1], 1)[:, 0]; t2 = np.take_along_axis(tm, o[:, 1:2], 1)[:, 0]
        tie = np.isfinite(t2) & (t2 - t1 <= tie_rel * np.maximum(t1, 1e-30)) & (t1 <= 1)
        if tie.any():
            flag[ia[tie]] = True; active[ia[tie]] = False
        ok = ~tie
        ia = ia[ok]; ax = o[ok, 0]; t1 = t1[ok]
        done = t1 > 1                                               # the segment ends inside the current cell
        active[ia[done]] = False
        ia = ia[~done]; ax = ax[~done]
        if not len(ia):
            continue
        c[ia, ax] += step[ia, ax]
        tmax[ia, ax] += tdel[ia, ax]
        reached = np.all(c[ia] == end[ia], axis=1)
        inside = np.all((c[ia] >= 0) & (c[ia] < N), axis=1)
        hit = np.zeros(len(ia), bool); hit[inside] = occ(c[ia][inside])
        hit &= ~reached
        free[ia[hit]] = False; active[ia[hit]] = False
        active[ia[reached]] = False
    for i in np.flatnonzero(flag):
        free[i] = free_exact(world, s, X[i])
    free &= ~world.occupied(end)                     # SOLID receivers are response-inactive by convention (as in Model A)
    world.stats["fast_exact_fallbacks"] = world.stats.get("fast_exact_fallbacks", 0) + int(flag.sum())
    return free
