import sys, time, json, numpy as np
sys.path.insert(0, 'code')
from rpl.data import Scene, scene_ids, splits
import importlib, os
from rpl.engine2 import scene_geometry
run_window = importlib.import_module(os.environ.get("RPL_ENGINE", "rpl.engine3")).run_window
sp = splits(); ids = [s for s in scene_ids() if sp.get(s) == 'train']
sid = ids[int(sys.argv[1])]; wid = int(sys.argv[2]); maxn = int(float(sys.argv[3])) if len(sys.argv) > 3 else 2_000_000
kw = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
sc = Scene(sid); geom = scene_geometry(sc)
cls, o, L = sc.window_classes(wid)
print('scene', sid[:8], 'window', wid, 'origin', o, 'L', L, 'components', len(geom[1]), 'lit', int((cls == 1).sum()), 'dark', int((cls == 0).sum()), flush=True)
t = time.time(); res = run_window(geom, cls, o, max_nodes=maxn, **kw); dt = time.time() - t
st = res['stats']; Lm = res['lo'].sum(); U = res['hi'].sum() - Lm
print(f'time {dt:.2f}s', json.dumps(st))
print(f'L={Lm:.4e} U-L={U:.4e} TVbound={U/(Lm+U) if Lm+U>0 else float("nan"):.4f} records={len(res["cell"])}')
tc = (int(sc.truth[0]*128)*128 + int(sc.truth[1]*128))*128 + int(sc.truth[2]*128)
cells, inv = np.unique(res['cell'], return_inverse=True)
lo_c = np.bincount(inv, res['lo']); hi_c = np.bincount(inv, res['hi'])
order = np.argsort(-hi_c)[:8]
print('top cells (cell, lo, hi, is_truth):', [(int(cells[i]), float(lo_c[i]), float(hi_c[i]), bool(cells[i] == tc)) for i in order])
print('depth nodes rec_tests block_tests out kept')
for d, row in enumerate(res['depth_stats']):
    if row[0]: print(d, *row.tolist())
