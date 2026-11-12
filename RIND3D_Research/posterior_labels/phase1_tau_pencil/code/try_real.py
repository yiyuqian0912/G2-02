import sys, time, json, numpy as np
sys.path.insert(0, 'code')
from rpl.data import Scene, scene_ids
from tpl.geom import Geom
from tpl.window import Window
import os
_E = os.environ.get("TPL_ENGINE", "3")
if _E == "1":
    from tpl.run_v1 import run_window, cell_of, cell_intervals
elif _E == "2":
    from tpl.run2 import run_window, cell_of, cell_intervals
else:
    from tpl.run3 import run_window, cell_of, cell_intervals
pref, wid = sys.argv[1], int(sys.argv[2])
kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
sid = [s for s in scene_ids() if s.startswith(pref)][0]
sc = Scene(sid)
geom = Geom.from_scene(sc)
cls, o, L = sc.window_classes(wid)
w = Window(cls, o, 1024)
t = time.perf_counter(); cpu0 = time.process_time()
r = run_window(geom, w, **kw); cpu = time.process_time() - cpu0
print('  CPU seconds %.2f' % cpu)
cells, Lc, Uc = cell_intervals(r)
tc = cell_of(sc.truth)
m = cells == tc
print(sid[:8], wid, 'L', L, 'comps', geom.C, w.counts(), 'items', r['n_items'], 'kw', kw)
print('  stage1', r['stage1'], 'sec', {k: round(v, 3) for k, v in r['seconds'].items()})
print('  stats', {k: v for k, v in r['stats'].items() if v})
print('  L %.4e U %.4e undet %.3e cells_cert %d possible %d TVbound %.5f truth-cell mass L/sum %.4f' % (
    r['L_total'], r['U_total'], r['undet_total'], r['n_cells_certified'], r['n_cells_possible'], r['tv_bound'],
    Lc[m].sum() / max(Lc.sum(), 1e-300)))
ev = r['cell_evals']; nd = r['cell_nodes']
o_ = np.argsort(-ev)
print('  evals per cell: median %d mean %.0f p90 %d max %d; top-10 share %.3f; nodes>1 cells %d' % (
    np.median(ev), ev.mean(), np.percentile(ev, 90), ev.max(), ev[o_[:10]].sum() / ev.sum(), (nd > 1).sum()))
print('  top cells (evals, nodes):', [(int(ev[i]), int(nd[i])) for i in o_[:10]])
if 'face_by_comp' in r:
    print('  face splits by comp', r['face_by_comp'], 'survivors passed', r['face_surv'], 'K', geom.K.tolist())
    print('  comp AABB extents', np.round(geom.chi - geom.clo, 3).tolist())
