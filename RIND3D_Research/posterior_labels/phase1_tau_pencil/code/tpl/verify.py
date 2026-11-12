"""Independent verification of an engine result by brute force (every FREE receiver of the window tested with the
float Cyrus-Beck segment predicate, rpl.synth.feasible_points; no tau coordinates, no polytopes, no octree).

* soundness:     points sampled uniformly inside emitted IN polytopes must all be feasible;
* completeness:  feasible points found by uniform sampling of a box around the support (and of the whole world)
                 must lie inside an emitted polytope;
* volume:        the Monte Carlo estimate of the feasible volume in that box must match the certified volume.
"""
import numpy as np
from scipy.spatial import Delaunay

from rpl.synth import feasible_points


def polytopes(res, kind=0):
    out = []
    for i in np.flatnonzero(res['kind'] == kind):
        if res['voff'][i] < 0:
            continue
        out.append((int(res['cell'][i]), res['vpool'][res['voff'][i]:res['voff'][i] + res['vn'][i]], float(res['hi'][i])))
    return out


def _normalized(Vt):
    c = Vt.mean(0)
    sc = max(float(np.max(Vt.max(0) - Vt.min(0))), 1e-300)
    return c, sc


def sample_in_hull(Vt, n, rng):
    c, sc = _normalized(Vt)
    d = Delaunay((Vt - c) / sc)
    simp = d.simplices
    P = Vt[simp]
    vol = np.abs(np.linalg.det(P[:, 1:] - P[:, :1])) / 6
    k = rng.choice(len(simp), n, p=vol / vol.sum())
    w = rng.dirichlet(np.ones(4), n)
    return np.einsum('nk,nkd->nd', w, P[k])


def inside_any(X, hulls, tol=1e-12):
    ins = np.zeros(len(X), bool)
    for Vt in hulls:
        lo = Vt.min(0) - 1e-9; hi = Vt.max(0) + 1e-9
        m = np.all((X >= lo) & (X <= hi), 1) & ~ins
        if not m.any():
            continue
        c, sc = _normalized(Vt)
        try:
            d = Delaunay((Vt - c) / sc)
        except Exception:
            continue
        ins[np.flatnonzero(m)[d.find_simplex((X[m] - c) / sc, tol=tol / sc) >= 0]] = True
    return ins


def verify(res, geom, win, n_sound=20000, n_box=200000, n_world=200000, inflate=1.0, seed=0, max_polys=3000):
    rng = np.random.default_rng(seed)
    X, obs = win.receivers()
    # random receiver order: infeasible points are rejected after few receivers (same result, less time)
    perm = rng.permutation(len(X)); X = np.ascontiguousarray(X[perm]); obs = np.ascontiguousarray(obs[perm])
    wlo, whi = win.box()
    P, K = geom.P, geom.K
    polys = polytopes(res, 0)
    out = dict(n_polytopes=len(polys))
    if not polys:
        return out
    vols = np.array([p[2] for p in polys])
    # soundness
    idx = rng.choice(len(polys), min(len(polys), max_polys), replace=False, p=None)
    per = max(1, n_sound // len(idx))
    chunks = []
    bad = 0
    for i in idx:
        if len(polys[i][1]) < 4:
            bad += 1
            continue
        try:
            chunks.append(sample_in_hull(polys[i][1], per, rng))
        except Exception:
            bad += 1
    out['sound_polytopes_skipped'] = bad
    S = np.concatenate(chunks) if chunks else np.zeros((0, 3))
    f = feasible_points(S, X, obs, P, K, wlo, whi)
    out.update(sound_points=int(len(S)), sound_feasible=int((f == 1).sum()), sound_infeasible=int((f == 0).sum()),
               sound_undecided=int((f == -1).sum()))
    # completeness + volume in an inflated box around the support
    allv = np.concatenate([p[1] for p in polys])
    lo = allv.min(0); hi = allv.max(0); ext = np.maximum(hi - lo, 1e-9)
    blo = np.maximum(lo - inflate * ext, 0); bhi = np.minimum(hi + inflate * ext, 1)
    B = blo + rng.random((n_box, 3)) * (bhi - blo)
    fb = feasible_points(B, X, obs, P, K, wlo, whi)
    hulls = [p[1] for p in polys]
    feas = B[fb == 1]
    ins = inside_any(feas, hulls) if len(feas) else np.zeros(0, bool)
    vbox = float(np.prod(bhi - blo))
    k = int((fb == 1).sum())
    out.update(box_volume=vbox, box_points=n_box, box_feasible=k, box_feasible_outside_polytopes=int((~ins).sum()),
               box_undecided=int((fb == -1).sum()), mc_volume=vbox * k / n_box,
               mc_volume_se=vbox * np.sqrt(max(k, 1)) / n_box, certified_volume=float(vols.sum()))
    # whole world
    Wd = rng.random((n_world, 3))
    fw = feasible_points(Wd, X, obs, P, K, wlo, whi)
    fe = Wd[fw == 1]
    insw = inside_any(fe, hulls) if len(fe) else np.zeros(0, bool)
    out.update(world_points=n_world, world_feasible=int(len(fe)), world_feasible_outside=int((~insw).sum()),
               world_undecided=int((fw == -1).sum()))
    return out


def verify_local(res, geom, win, n_polys=400, per=50, grow=0.5, seed=0):
    """Completeness near the support boundary: points uniform in each sampled polytope's AABB grown by `grow` x its
    extent per side; every brute-force feasible point must lie in some emitted polytope (IN region)."""
    rng = np.random.default_rng(seed)
    X, obs = win.receivers()
    wlo, whi = win.box()
    polys = polytopes(res, 0)
    if not polys:
        return dict(local_points=0)
    hulls = [p[1] for p in polys]
    lo_all = np.array([h.min(0) for h in hulls]); hi_all = np.array([h.max(0) for h in hulls])
    idx = rng.choice(len(polys), min(n_polys, len(polys)), replace=False)
    pts = []
    for i in idx:
        lo, hi = lo_all[i], hi_all[i]
        e = np.maximum(hi - lo, 1e-12)
        a = np.maximum(lo - grow * e, 0); b = np.minimum(hi + grow * e, 1)
        pts.append(a + rng.random((per, 3)) * (b - a))
    S = np.concatenate(pts)
    f = feasible_points(S, X, obs, geom.P, geom.K, wlo, whi)
    feas = S[f == 1]
    miss = int((~inside_any(feas, hulls)).sum()) if len(feas) else 0
    return dict(local_points=int(len(S)), local_feasible=int(len(feas)), local_infeasible=int((f == 0).sum()),
                local_undecided=int((f == -1).sum()), local_feasible_outside=int(miss))


def verify_scaled(res, geom, win, n_polys=300, per=40, scale=2.0, seed=0):
    """Soundness + completeness near the boundary of thin polytopes: points uniform in each sampled IN polytope
    scaled by `scale` about its vertex centroid (1/scale^3 of them fall inside the original polytope).  Points inside
    an emitted polytope must be feasible; feasible points must lie inside some emitted polytope."""
    rng = np.random.default_rng(seed)
    X, obs = win.receivers()
    wlo, whi = win.box()
    polys = polytopes(res, 0)
    if not polys:
        return dict(scaled_points=0)
    hulls = [p[1] for p in polys]
    lo_all = np.array([h.min(0) for h in hulls]); hi_all = np.array([h.max(0) for h in hulls])
    idx = rng.choice(len(polys), min(n_polys, len(polys)), replace=False)
    pts = []
    for i in idx:
        V = hulls[i]
        if len(V) < 4:
            continue
        c = V.mean(0)
        try:
            pts.append(c + scale * (sample_in_hull(V, per, rng) - c))
        except Exception:
            continue
    if not pts:
        return dict(scaled_points=0)
    S = np.clip(np.concatenate(pts), 1e-12, 1 - 1e-12)
    f = feasible_points(S, X, obs, geom.P, geom.K, wlo, whi)
    inside = inside_any(S, hulls)
    return dict(scaled_points=int(len(S)), inside=int(inside.sum()), inside_infeasible=int((inside & (f == 0)).sum()),
                outside=int((~inside).sum()), outside_feasible=int((~inside & (f == 1)).sum()),
                feasible=int((f == 1).sum()), undecided=int((f == -1).sum()))
