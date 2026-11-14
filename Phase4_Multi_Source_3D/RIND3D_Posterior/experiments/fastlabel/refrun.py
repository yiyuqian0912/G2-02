"""Reference label (production engine v3.1, tier 1) for a random task; stores the label for TV checks."""
import os as _os; _P4 = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../../..'))   # Phase4 root
import sys, json, time
from pathlib import Path
sys.path[:0] = [_os.path.join(_P4, "RIND3D_Dataset1K"), _os.path.join(_P4, "RIND3D_Posterior/src")]
import numpy as np
from rind3d_1k.labels import SceneObs
from rind3d_posterior import label_window
R = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes")
out = Path(__file__).resolve().parent / "refs"; out.mkdir(exist_ok=True)
sc, w, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
S = SceneObs(R / sc); cls = S.cls(w, s)
c0 = time.process_time(); t0 = time.perf_counter()
r = label_window(S.P, S.K, cls, S.origin(w), 1024, engine="cpu", nq_cap=1 << 25, ne_cap=1 << 26)
cpu = time.process_time() - c0; wall = time.perf_counter() - t0
np.savez_compressed(out / f"{sc}_w{w:03d}_s{s}.npz", cell=r["cell"], lo=r["lo"], hi=r["hi"], kind=r["kind"], tv_bound=r["tv_bound"])
print(json.dumps(dict(sc=sc, w=w, s=s, L=int(cls.shape[0]), cpu=cpu, wall=wall, tv_bound=float(r["tv_bound"]), records=len(r["cell"]),
                      vol=float(np.sum(r["lo"])), evals=int(r["stats"]["entry_evals"]))), flush=True)
