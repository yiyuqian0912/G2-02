"""Independent validation of a statistical-tier label: rebuild B exactly as adaptive2 did (same seed), then estimate
p = vol(F)/vol(B) with N fresh exact point tests (validation only; not part of any certificate)."""
import sys, json, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "../twophase")
import adaptive2 as A
import twophase as T
from stat_tier import sample_in_B

sc, w, s, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
# rerun the portfolio with a hook capturing the final B
captured = {}
orig = T.stage2
def hook(*a, **k):
    r = orig(*a, **k); captured["last"] = r; return r
T.stage2 = hook
out = A.run(sc, w, s)
res = captured["last"]
print("path", out["path"], "B volume", res["U_total"], "records", len(res["cell"]), flush=True)
from rind3d.geometry import load_geometry
from rind3d_1k.physical import POLICY
S = T.SceneObs(T.R / sc)
tester = A.Tester(load_geometry(T.R / sc / "privileged" / "geometry.npz"), POLICY, S.crop(w, s), S.crop(w, -1), S.origin(w))
rng = np.random.default_rng(987654)
pts, why = sample_in_B(res, N, rng)
ok = 0; t = time.perf_counter()
for p in pts:
    good, _ = tester.test(p)
    ok += good
from scipy.stats import beta
print(json.dumps(dict(sc=sc, w=w, s=s, N=N, feasible=ok, p_hat=ok / N, p_lo99=float(beta.ppf(0.005, ok, N - ok + 1)),
                      p_hi99=float(beta.ppf(0.995, ok + 1, N - ok)) if ok < N else 1.0, seconds=time.perf_counter() - t)))
