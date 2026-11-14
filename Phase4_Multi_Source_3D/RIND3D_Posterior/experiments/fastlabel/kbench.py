"""Kernel-only timing of verify.cu on realistic certification batches (points from a refined B)."""
import sys, time, json, ctypes as C
import numpy as np
sys.path[:0] = [".", "../twophase"]
import twophase as T, adaptive3 as A, gpuverify as GV
from fastwin import fast_cls2, LazyWindow2
from sampler3 import BSampler
from fastbf import build_filter2, items_from_vox2
res_all = []
for spec in sys.argv[1:]:
    sc, w, s = spec.split(":"); w = int(w); s = int(s)
    scn = A.scene_cache(sc); S = scn["S"]; geom = scn["geom"]
    cls = fast_cls2(S, w, s); win = LazyWindow2(cls, np.asarray(S.origin(w), np.int64), 1024)
    roots, _ = T.stage1(geom, win); xc, xk = T.point_items(win, 256, 128, 0)
    res = T.stage2(geom, win, xc, xk, roots)
    V = GV.WindowVerifier(A.process_factory(), scn["ctx"], S, w, s, cls)
    rng = np.random.default_rng(1)
    # one refinement round so the batch looks like a certification round
    pts = BSampler(res, rng, {}).draw(256); viol = [g[1][:64] for g in V.test_batch(pts) if not g[0] and g[1] is not None]
    if viol:
        nc, nk = items_from_vox2(win, np.unique(np.concatenate(viol), axis=0)[:2048])
        res = T.stage2(geom, win, np.concatenate([xc, nc]), np.concatenate([xk, nk]), roots, bf=build_filter2(res))
    pts = BSampler(res, rng, {}).draw(1024)
    rt = V.rt
    V.test_batch(pts)                                       # warm
    k0 = rt.timings["kernel_seconds"]
    ev = [C.c_void_p(), C.c_void_p()]
    for e in ev: rt.ck("cuEventCreate", C.byref(e), 0)
    rt.ck("cuEventRecord", ev[0], None)
    t = time.perf_counter(); out = V.test_batch(pts); wall = time.perf_counter() - t
    rt.ck("cuEventRecord", ev[1], None); rt.ck("cuEventSynchronize", ev[1])
    ms = C.c_float(); rt.ck("cuEventElapsedTime", C.byref(ms), ev[0], ev[1])
    rec = dict(spec=spec, L=win.L, bodies=V.G.nb, rows_per_src=V.G.rows_per_src, bricks=len(V.bricks), n=len(pts),
               passes=int(sum(o[0] for o in out)), gpu_ms=round(ms.value, 2), gpu_us_per_pt=round(1e3 * ms.value / len(pts), 1),
               wall_ms=round(1e3 * wall, 1))
    print(json.dumps(rec), flush=True)
    V.close()
