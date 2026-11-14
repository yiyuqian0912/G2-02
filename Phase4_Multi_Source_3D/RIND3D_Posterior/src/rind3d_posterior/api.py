"""Public API of the labeller."""
import numpy as np

from .rpl.bound import tv_bound  # noqa: F401
from .tpl.geom import Geom
from .tpl.window import Window

ENGINE_VERSION = "tpe-b4/p2-cpu+g2-gpu-pass-v3.1"   # v3.1: record-buffer guard + exact rerun (results identical whenever the v3 buffer sufficed)
CELL_DEPTH = 7          # 128^3 source-space output cells


def label_window(P, K, cls, origin, N, engine="cpu", gpu_version=3, gpu_f32=2, gpu_cap=1 << 22, d1=4, gpu_min=1024, gpu_prefix=64, **kw):
    """Certified label of one window.
    P[c, i, :] = (a, b, c, d) rows of convex body c (a.x + d <= 0 inside), K[c] row counts, cls[x, y, z] in
    {0 dark FREE, 1 lit FREE, 2 SOLID} on the N^3 receiver grid, origin = integer window corner.
    Returns dict(cell, lo, hi, kind, tv_bound, stats, ...) where records (cell, lo, hi) bound the uniform volume of
    F_W inside 128^3 cell `cell` (kind 0 certified IN, 2 undetermined)."""
    g = Geom(np.asarray(P, float), np.asarray(K))
    w = Window(np.asarray(cls, np.uint8), np.asarray(origin, np.int64), int(N))
    if engine == "cpu":
        from .tpl import run_p2
        return run_p2.run_window(g, w, d1=d1, **kw)
    from .tpl import run_g1
    return run_g1.run_window(g, w, d1=d1, use_gpu=True, gpu_version=gpu_version, gpu_f32=gpu_f32, gpu_cap=gpu_cap, gpu_min=gpu_min, gpu_prefix=gpu_prefix, **kw)


def cell_bounds(rec):
    """Aggregate records to per-cell lower/upper volumes: returns (cells, L, U), cells sorted."""
    cell = np.asarray(rec["cell"]); lo = np.asarray(rec["lo"]); hi = np.asarray(rec["hi"])
    cells, inv = np.unique(cell, return_inverse=True)
    L = np.bincount(inv, weights=lo, minlength=len(cells)); U = np.bincount(inv, weights=hi, minlength=len(cells))
    return cells, L, U


def point_estimate(rec):
    """The label distribution q = L / sum(L) over 128^3 cells; TV(q, true posterior) <= tv_bound(L, U)."""
    cells, L, U = cell_bounds(rec)
    s = L.sum()
    return cells, (L / s if s > 0 else L), float(tv_bound(L, U))
