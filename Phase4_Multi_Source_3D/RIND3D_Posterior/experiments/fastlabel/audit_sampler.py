"""Audit of sample_in_B: is the sampling law Uniform(B)? Per piece, compare the engine volume hi (used to choose
pieces and for q_B) with the volume of the tetrahedralisation used to sample inside the piece (convex hull of the
dumped vertices), and check that sampled points lie in the piece's vertex bounding box. A deterministic bound on
TV(sampling law, Uniform(B)) is sum_i |hi_i / H - t_i / T'| / 2 + (in-piece misallocation), where t_i = tetra volume;
here we report the per-piece relative discrepancy |t_i - hi_i| / hi_i and the mass-weighted TV of the piece choice."""
import json, sys
import numpy as np
sys.path[:0] = [".", "../twophase"]
import twophase as T, adaptive3 as A
from fastwin import fast_cls, LazyWindow
from stat_tier import _tetrahedra
out = []
for spec in sys.argv[1:]:
    sc, w, s = spec.split(":"); w = int(w); s = int(s)
    scn = A.scene_cache(sc); S = scn["S"]; geom = scn["geom"]
    win = LazyWindow(fast_cls(S, w, s), np.asarray(S.origin(w), np.int64), 1024)
    roots, _ = T.stage1(geom, win); xc, xk = T.point_items(win, 1024, 512, 0)
    res = T.stage2(geom, win, xc, xk, roots)
    hi = res["hi"]; voff, vn, vp = res["voff"], res["vn"], res["vpool"]
    rel = []; tv = []; npc = len(hi)
    idx = np.arange(npc) if npc <= 20000 else np.random.default_rng(0).choice(npc, 20000, replace=False)
    tsum = np.zeros(len(idx))
    for j, i in enumerate(idx):
        V = vp[voff[i]:voff[i] + vn[i]]; ctr = V.mean(0); sc_ = max(float(np.max(np.abs(V - ctr))), 1e-300)
        Tt = _tetrahedra((V - ctr) / sc_)
        vol = np.abs(np.einsum("ij,ij->i", Tt[:, 1] - Tt[:, 0], np.cross(Tt[:, 2] - Tt[:, 0], Tt[:, 3] - Tt[:, 0]))).sum() / 6 * sc_ ** 3
        tsum[j] = vol
    h = hi[idx]; ok = h > 0
    r = np.abs(tsum[ok] - h[ok]) / h[ok]
    tvc = 0.5 * np.sum(np.abs(h / h.sum() - tsum / tsum.sum()))
    rec = dict(spec=spec, pieces=int(npc), audited=int(len(idx)), rel_max=float(r.max()), rel_p99=float(np.quantile(r, 0.99)),
               rel_mean=float(r.mean()), mass_weighted_rel=float(np.sum(np.abs(tsum - h)) / h.sum()), tv_piece_choice=float(tvc))
    print(json.dumps(rec), flush=True); out.append(rec)
