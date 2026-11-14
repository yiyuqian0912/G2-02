"""E1: phase-1 cost and outer-bound tightness vs receiver-subset size (phase 1 only)."""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "twophase"))
import twophase as T
import numpy as np

sc, w, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
nbd, nin = int(sys.argv[4]), int(sys.argv[5])
S = T.SceneObs(T.R / sc); cls = S.cls(w, s)
c0 = time.process_time()
geom = T.Geom(np.asarray(S.P, float), np.asarray(S.K)); win = T.Window(np.asarray(cls, np.uint8), np.asarray(S.origin(w), np.int64), 1024)
t_setup = time.process_time() - c0
ref = np.load(T.R / sc / "labels" / "tasks" / f"w{w:03d}_s{s}.npz"); refL = float(ref["lo"].sum())
c1 = time.process_time(); roots1, ts1 = T.stage1(geom, win); t_s1 = time.process_time() - c1
c2 = time.process_time(); codes, kls = win.top_items(); t_items = time.process_time() - c2
c3 = time.process_time(); sc_, sk_ = T.point_items(win, nbd, nin); t_pts = time.process_time() - c3
r1 = T.stage2(geom, win, sc_, sk_, roots1)
refm = T.cell_masses(ref["cell"], ref["lo"], ref["hi"]); bm = T.cell_masses(r1["cell"], r1["lo"], r1["hi"])
print(json.dumps(dict(sc=sc, w=w, s=s, L=int(win.L), nbd=nbd, nin=nin, setup=t_setup, s1=t_s1, items=t_items, pts=t_pts,
                      p1=r1["seconds"], p1_evals=r1["stats"]["entry_evals"], p1_nodes=r1["stats"]["nodes"], rec=len(r1["cell"]),
                      undet=r1["stats"]["undet"], B=r1["U_total"], BL=r1["L_total"], ratio=refL / r1["U_total"] if r1["U_total"] > 0 else None,
                      tv_qB_qF=T.tv(bm, refm), cells_B=len(bm), cells_F=len(refm))), flush=True)
