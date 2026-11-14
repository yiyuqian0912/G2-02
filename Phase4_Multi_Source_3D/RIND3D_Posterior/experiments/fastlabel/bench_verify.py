"""Time the batched GPU verifier against the current per-point tester (adaptive2.Tester + culling), uncontended."""
import json, sys, time
import numpy as np
sys.path[:0] = [".", "../twophase"]
import twophase as T, adaptive2 as A, gpuverify as GV
from fastwin import fast_cls, LazyWindow
from stat_tier import sample_in_B
from rind3d_1k.physical import POLICY
for spec in sys.argv[2:]:
    sc, w, s = spec.split(":"); w = int(w); s = int(s)
    scn = A.scene_cache(sc); S = scn["S"]; geom = scn["geom"]
    cls = fast_cls(S, w, s); win = LazyWindow(cls, np.asarray(S.origin(w), np.int64), 1024)
    roots, _ = T.stage1(geom, win); xc, xk = T.point_items(win, 1024, 512, 0)
    res = T.stage2(geom, win, xc, xk, roots)
    n = int(sys.argv[1])
    pts, _ = sample_in_B(res, n, np.random.default_rng(3))
    tester = A.Tester(scn["bundle"], POLICY, S.crop(w, s), S.crop(w, -1), S.origin(w)); tester.set_cull(geom)
    tester.test(pts[0], tester.need_tiles_point(pts[0]))
    t = time.perf_counter(); old = [tester.test(p, tester.need_tiles_point(p))[0] for p in pts]; t_old = time.perf_counter() - t
    t = time.perf_counter(); V = GV.WindowVerifier(A.process_factory(), tester.ctx, S, w, s, cls); t_init = time.perf_counter() - t
    V.test_batch(pts[:4])
    V.stats.update(gpu_s=0.0, legal_s=0.0)
    t = time.perf_counter(); new = [g[0] for g in V.test_batch(pts)]; t_new = time.perf_counter() - t
    print(json.dumps(dict(spec=spec, L=win.L, n=n, agree=old == new, passes=int(sum(new)), old_ms_per_pt=round(1e3 * t_old / n, 2),
                          new_ms_per_pt=round(1e3 * t_new / n, 3), gpu_ms_per_pt=round(1e3 * V.stats["gpu_s"] / n, 3),
                          legal_ms_per_pt=round(1e3 * V.stats["legal_s"] / n, 3), init_ms=round(1e3 * t_init, 1),
                          speedup=round(t_old / t_new, 1))), flush=True)
    V.close()
