import sys, json, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "RIND3D_Dataset1K"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from pathlib import Path
import numpy as np
from rind3d_1k.labels import SceneObs
from rind3d_posterior import label_window
R = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes")
sc, w, s, d1, nb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
S = SceneObs(R / sc); cls = S.cls(w, s)
ref = np.load(R / sc / "labels" / "tasks" / f"w{w:03d}_s{s}.npz")
t0 = time.perf_counter()
r = label_window(S.P, S.K, cls, S.origin(w), 1024, engine="cpu", nq_cap=1 << 25, ne_cap=1 << 26, d1=d1, n_boundary=nb, max_tests=nb + 128)
wall = time.perf_counter() - t0
same = (len(r["cell"]) == len(ref["cell"])) and np.array_equal(np.sort(r["cell"]), np.sort(ref["cell"]))
def masses(c, lo, hi):
    u, inv = np.unique(c, return_inverse=True); return dict(zip(u.tolist(), np.bincount(inv, weights=0.5 * (lo + hi)).tolist()))
a = masses(r["cell"], r["lo"], r["hi"]); b = masses(ref["cell"], ref["lo"], ref["hi"])
sa, sb = sum(a.values()), sum(b.values())
tv = 0.5 * sum(abs(a.get(k, 0) / sa - b.get(k, 0) / sb) for k in set(a) | set(b))
print(json.dumps(dict(sc=sc, w=w, s=s, d1=d1, nb=nb, wall=wall, s1=r["seconds"]["stage1"], s2=r["seconds"]["stage2"],
                      cand=r["stage1"]["candidates"], tests=r["stage1"]["tests"], evals=r["stats"]["entry_evals"],
                      nodes=r["stats"]["nodes"], tv_bound=r["tv_bound"], tv_vs_ref=tv)))
