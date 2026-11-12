import sys, time, json, numpy as np
sys.path.insert(0, 'code')
from rpl.data import Scene, scene_ids, splits
from rpl.engine import run_window
sp = splits(); ids = [s for s in scene_ids() if sp.get(s) == 'train']
sid = ids[int(sys.argv[1])]; wid = int(sys.argv[2]); dmax = int(sys.argv[3]) if len(sys.argv) > 3 else 11
maxn = int(float(sys.argv[4])) if len(sys.argv) > 4 else 2_000_000
sc = Scene(sid); P, K, clo, chi = sc.geometry_arrays()
cls, o, L = sc.window_classes(wid)
print('scene', sid[:8], 'window', wid, 'origin', o, 'L', L, 'components', len(K), 'rows', K.tolist(), 'truth', sc.truth)
t = time.time()
res = run_window(P, K, clo, chi, cls, o, d_max=dmax, max_nodes=maxn)
dt = time.time() - t
st = res['stats']; Lm = res['lo'].sum(); Um = res['hi'].sum(); U = Um - Lm
print(f'time {dt:.2f}s', json.dumps(st))
print(f'L={Lm:.4e} U-L={U:.4e} TVbound={U/(Lm+U) if Lm+U>0 else float("nan"):.4f} records={len(res["cell"])} cells={len(np.unique(res["cell"]))}')
truth_cell = (int(sc.truth[0]*128)*128 + int(sc.truth[1]*128))*128 + int(sc.truth[2]*128)
m = res['cell'] == truth_cell
print('truth cell mass lo/hi', res['lo'][m].sum(), res['hi'][m].sum())
print('depth nodes rec_tests block_tests out kept')
for d, row in enumerate(res['depth_stats']):
    if row[0]: print(d, *row.tolist())
