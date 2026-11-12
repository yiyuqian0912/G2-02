"""Clearance shell: the legal source domain excludes {s : d(s, solid) <= 2^-26}.  The engine certifies volumes of
F_W without that constraint; here the excluded part is bounded exactly from above per IN polytope:

    vol(P ∩ shell)  <=  sum_c vol(P ∩ C_c^eps),   C_c^eps = {s : f_i(s) <= eps for all faces i}  (unit normals),

since d(s, C) <= eps implies f_i(s) <= eps for every face.  P ∩ C_c^eps is a polytope, computed with the engine's
exact clipper (tpl/poly.py); empty unless P comes within eps of C.  The certified per-cell interval becomes
[L_c - shell_c, U_c] and the TV bound is recomputed (rpl.bound).  Re-runs the frozen engine b3 to obtain the IN
polytopes."""
import sys, os, json
import numpy as np
from scipy.spatial import ConvexHull
sys.path.insert(0, os.path.dirname(__file__))
from rpl.data import Scene, scene_ids
from rpl.bound import tv_bound
from tpl.geom import Geom
from tpl.window import Window
from tpl.run_b3 import run_window
from tpl.run import cell_intervals
from tpl.verify import polytopes

EPS = 2.0 ** -26


def shell_volume(V, P, K, eps=EPS):
    """Exact (float) upper bound via the engine's own clipper: build the polytope as a box clipped by its hull
    facets, then clip by {f_i <= eps} of each nearby component (as engine_v2._shell_volume)."""
    from tpl.poly import set_box, clip, volume, copy_poly
    c0 = V.mean(0); sc = max(float(np.max(V.max(0) - V.min(0))), 1e-300)
    try:
        h = ConvexHull((V - c0) / sc)
    except Exception:
        # degenerate hull: if it is near any component, charge its (zero-ish) bounding-box volume (conservative)
        near = any(not np.any((V @ P[c, :K[c], :3].T + P[c, :K[c], 3]).min(0) > eps) for c in range(P.shape[0]))
        return float(np.prod(V.max(0) - V.min(0))) if near else 0.0
    E = h.equations
    A = E[:, :3]; b = E[:, 3] * sc - A @ c0
    tot = 0.0
    for c in range(P.shape[0]):
        rows = P[c, :K[c]]
        f = V @ rows[:, :3].T + rows[:, 3]
        if np.any(f.min(0) > eps):
            continue
        Vb = np.zeros((160, 3)); Fc = np.zeros((72, 48), np.int32); Fn = np.zeros(72, np.int32)
        lo = V.min(0) - 1e-9; hi = V.max(0) + 1e-9
        nv, nf = set_box(Vb, Fc, Fn, lo, hi)
        st = 0
        for a_, b_ in zip(A, b):
            st, nv, nf = clip(Vb, Fc, Fn, nv, nf, a_[0], a_[1], a_[2], b_)
            if st != 0:
                break
        if st == 1:
            continue
        for i in range(K[c]):
            if st != 0:
                break
            st, nv, nf = clip(Vb, Fc, Fn, nv, nf, rows[i, 0], rows[i, 1], rows[i, 2], rows[i, 3] - eps)
        if st == 1:
            continue
        if st == 2:
            tot += ConvexHull(V).volume          # overflow: charge the whole polytope (conservative)
            continue
        tot += volume(Vb, Fc, Fn, nf)
    return tot


def check(pref, wid):
    sid = [s for s in scene_ids() if s.startswith(pref)][0]
    sc = Scene(sid); g = Geom.from_scene(sc)
    cls, o, L = sc.window_classes(wid)
    r = run_window(g, Window(cls, o, 1024), d1=4)
    polys = polytopes(r, 0)
    shell = {}
    n_near = 0
    for cell, V, vol in polys:
        s = shell_volume(V, g.P, g.K) if len(V) >= 4 else 0.0
        if s > 0:
            n_near += 1
            shell[cell] = shell.get(cell, 0.0) + min(s, vol)
    cells, Lc, Uc = cell_intervals(r)
    sh = np.array([shell.get(int(c), 0.0) for c in cells])
    L2 = np.maximum(Lc - sh, 0.0)
    return dict(window=f'{pref}/{wid}', polytopes=len(polys), polytopes_touching_shell=n_near,
                shell_volume=float(sh.sum()), volume=float(Lc.sum()), shell_fraction=float(sh.sum() / Lc.sum()),
                tv_bound_without_shell=float(tv_bound(Lc, Uc)), tv_bound_with_shell=float(tv_bound(L2, Uc)))


if __name__ == '__main__':
    tasks = json.load(open(sys.argv[1]))
    tasks = tasks['tasks'] if isinstance(tasks, dict) else tasks
    out = []
    for s, w in tasks:
        res = check(s[:8], w)
        out.append(res)
        print(json.dumps(res), flush=True)
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
