"""Profile the current exact point test (uncontended): per-point CPU components and GPU kernel time."""
import sys, time, json, numpy as np
sys.path[:0] = [".", "../twophase"]
import twophase as T, adaptive2 as A, fastplanes, gpurender
from fastwin import fast_cls, LazyWindow
from stat_tier import sample_in_B
from rind3d_1k.physical import POLICY, legal_source
sc, w, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); npts = int(sys.argv[4]) if len(sys.argv) > 4 else 64
scn = A.scene_cache(sc); S = scn["S"]; geom = scn["geom"]
win = LazyWindow(fast_cls(S, w, s), np.asarray(S.origin(w), np.int64), 1024)
roots, _ = T.stage1(geom, win); xc, xk = T.point_items(win, 1024, 512, 0)
res = T.stage2(geom, win, xc, xk, roots)
tester = A.Tester(scn["bundle"], POLICY, S.crop(w, s), S.crop(w, -1), S.origin(w)); tester.set_cull(geom)
pts, _ = sample_in_B(res, npts, np.random.default_rng(5))
tester.test(pts[0], tester.need_tiles_point(pts[0]))          # warm
rt = tester.fac.runtime
acc = dict(legal=0., cull=0., bind=0., planes=0., encode=0., gpu_section=0., kernel=0., post=0.); ntiles = 0; nfail = 0
for p in pts:
    t = time.perf_counter(); need = tester.need_tiles_point(p); acc["cull"] += time.perf_counter() - t
    t = time.perf_counter(); legal_source(tester.ctx, p); acc["legal"] += time.perf_counter() - t
    t = time.perf_counter(); cpu = tester.ctx.bind(p); acc["bind"] += time.perf_counter() - t
    t = time.perf_counter(); planes = [fastplanes.shadow_planes_fast(b, cpu.source) for b in tester.ctx.bodies]; acc["planes"] += time.perf_counter() - t
    t = time.perf_counter(); rows, offs = fastplanes.encode_fast(planes); acc["encode"] += time.perf_counter() - t
    idx = np.flatnonzero(need); ntiles += len(idx)
    k0 = rt.timings["kernel_seconds"]
    t = time.perf_counter(); lit, sol = gpurender.render_tiles(tester.dg, tester.ctx, p, tester.origins[idx]) if len(idx) else (None, None); acc["gpu_section"] += time.perf_counter() - t
    acc["kernel"] += rt.timings["kernel_seconds"] - k0
    t = time.perf_counter()
    if len(idx):
        nfail += bool(np.any(lit != tester.expL[idx]))
    acc["post"] += time.perf_counter() - t
n = len(pts)
nrows = len(rows)
print(json.dumps(dict(sc=sc, w=w, s=s, L=win.L, bodies=len(tester.ctx.bodies), shadow_rows=nrows, geom_rows=int(sum(len(b) for b in tester.ctx.bodies)),
                      tiles_per_point=ntiles / n, tiles_total=len(tester.origins), fail_frac=nfail / n,
                      per_point_ms={k: round(1e3 * v / n, 3) for k, v in acc.items()},
                      kernel_ms_per_tile=round(1e3 * acc["kernel"] / max(ntiles, 1), 3))))
