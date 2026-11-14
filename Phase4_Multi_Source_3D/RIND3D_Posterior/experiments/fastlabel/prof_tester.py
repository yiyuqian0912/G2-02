"""Profile one exact point test (bind, legality, kernels, D2H, Python post-processing) on an L256 window."""
import sys, time, numpy as np
sys.path[:0] = [".", "../twophase"]
import twophase as T, adaptive2 as A, fastplanes
from rind3d.geometry import load_geometry
from rind3d_1k.physical import POLICY, legal_source
sc, w, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
S = T.SceneObs(T.R / sc)
tester = A.Tester(load_geometry(T.R / sc / "privileged" / "geometry.npz"), POLICY, S.crop(w, s), S.crop(w, -1), S.origin(w))
src0 = np.array([float.fromhex(v) for v in __import__("json").load(open(T.R / sc / "privileged" / "sources.json"))["sources_hex"][s]])
rng = np.random.default_rng(0)
pts = [src0 + rng.normal(0, 1e-7, 3) for _ in range(40)]           # near the true source: feasible-ish, full renders
tester.test(pts[0])
rt = tester.fac.runtime; rt.timings.update({k: 0.0 for k in rt.timings})
t0 = time.perf_counter(); tb = 0; tl = 0; tr = 0
for p in pts:
    t = time.perf_counter(); legal_source(tester.ctx, p); tl += time.perf_counter() - t
    t = time.perf_counter(); src = tester.fac.source(tester.bundle, p, "cuda_certified", POLICY); tb += time.perf_counter() - t
    t = time.perf_counter()
    for i in range(0, len(tester.origins), 128):
        src.packed_batch(tester.origins[i:i + 128], 64, 1024)
    tr += time.perf_counter() - t
    src.close()
n = len(pts); tot = time.perf_counter() - t0
print(f"per point (ms): legal {1e3*tl/n:.2f} bind {1e3*tb/n:.2f} packed_batch {1e3*tr/n:.2f} (64 tiles) total {1e3*tot/n:.2f}")
print("runtime timings per point (ms):", {k: round(1e3 * v / n, 2) for k, v in rt.timings.items()})
