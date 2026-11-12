import sys, json, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "RIND3D_Dataset1K"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from pathlib import Path
import numpy as np
from rind3d_1k.labels import SceneObs
from rind3d_posterior import label_window
R = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes")
sc, w, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
S = SceneObs(R / sc); cls = S.cls(w, s)
print("cls counts dark/lit/solid", np.bincount(cls.ravel(), minlength=3))
t0 = time.perf_counter()
r = label_window(S.P, S.K, cls, S.origin(w), 1024, engine="cpu", nq_cap=1 << 25, ne_cap=1 << 26)
print("wall", time.perf_counter() - t0, "seconds", r["seconds"], "stage1", r["stage1"])
st = r["stats"]; print({k: v for k, v in st.items() if v})
de, dn = np.array(r["dep_evals"]), np.array(r["dep_nodes"])
tot = de.sum()
for d in np.nonzero(dn)[0]:
    print(f"depth {d:2d} nodes {dn[d]:6d} evals {de[d]:.3e} ({de[d]/tot:5.1%}) evals/node {de[d]/max(dn[d],1):.3g}")
ce = np.array(r["cell_evals"]); cn = np.array(r["cell_nodes"])
o = np.argsort(-ce)
print("roots", len(ce), "top roots evals share", [round(float(x), 3) for x in np.cumsum(ce[o[:10]]) / ce.sum()])
print("tv", r["tv_bound"], "records", len(r["cell"]))
