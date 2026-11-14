"""E11: run rind3d_1k.fastlabel.label_task on one (scene, window, source) and print a JSON summary line.
The label is NOT written anywhere (label_task returns it in memory). Uses the GPU verifier.
Usage: python run_task.py <scene_dir> <window> <source> <L> <type>"""
import os as _os; _P4 = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../../..'))   # Phase4 root
import json, sys, time
import numpy as np
sys.path.insert(0, _os.path.join(_P4, "RIND3D_Dataset1K")); sys.path.insert(0, _os.path.join(_P4, "RIND3D_Posterior/src"))
from rind3d_1k.fastlabel import label_task
sc, w, s, L, typ = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
t0 = time.time()
try:
    lab = label_task(sc, w, s)
    m = lab["meta"]
    q = np.asarray(lab["lo"], float); q = q / q.sum() if q.sum() > 0 else q
    out = dict(scene=sc.rsplit("/", 1)[-1], window=w, source=s, L=L, type=typ, status=lab["status"], path=m.get("path"),
               tv_bound=float(lab["tv_bound"]), n_cells=m.get("n_cells"), wall=time.time() - t0,
               vol_label=float(np.sum(lab["hi"])), entropy_bits=float(-(q[q > 0] * np.log2(q[q > 0])).sum()),
               rounds=[{k: r.get(k) for k in ("round", "items", "records", "B", "tested", "K")} for r in m.get("rounds", [])],
               certificate={k: v for k, v in (m.get("certificate") or {}).items() if k in ("kind", "n", "K", "p_lower", "round", "stage", "tv_bound")})
except Exception as e:
    out = dict(scene=sc.rsplit("/", 1)[-1], window=w, source=s, L=L, type=typ, error=repr(e)[:300], wall=time.time() - t0)
print(json.dumps(out, default=float), flush=True)
