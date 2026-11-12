"""Run CEPL v2 on one window to completion and save the label + diagnostics."""
import sys, time, json, numpy as np
sys.path.insert(0, 'code')
from rpl.data import Scene, scene_ids, splits
import importlib, os
from rpl.engine2 import scene_geometry
run_window = importlib.import_module(os.environ.get("RPL_ENGINE", "rpl.engine3")).run_window
sid = sys.argv[1]; wid = int(sys.argv[2]); kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
out = sys.argv[4] if len(sys.argv) > 4 else f'outputs/label_{sid[:8]}_{wid}'
sc = Scene(sid); geom = scene_geometry(sc)
cls, o, L = sc.window_classes(wid)
t = time.time(); res = run_window(geom, cls, o, **kw); dt = time.time() - t
st = res['stats']; Lm = float(res['lo'].sum()); U = float(res['hi'].sum()) - Lm
cells, inv = np.unique(res['cell'], return_inverse=True)
lo_c = np.bincount(inv, res['lo']); hi_c = np.bincount(inv, res['hi'])
tc = (int(sc.truth[0]*128)*128 + int(sc.truth[1]*128))*128 + int(sc.truth[2]*128)
summary = dict(scene=sid, window=wid, origin=list(o), side=L, seconds=dt, stats=st, L=Lm, U=U,
               tv_bound=U / (Lm + U) if Lm + U > 0 else None, n_cells_pos=int((lo_c > 0).sum()),
               n_cells_any=int((hi_c > 0).sum()), truth_cell=tc,
               truth_cell_lo=float(lo_c[cells == tc].sum()), truth_cell_hi=float(hi_c[cells == tc].sum()), kw=kw,
               depth_stats=res['depth_stats'][res['depth_stats'][:, 0] > 0].tolist())
np.savez_compressed(out + '.npz', cells=cells, lo=lo_c, hi=hi_c)
json.dump(summary, open(out + '.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != 'depth_stats'}))
