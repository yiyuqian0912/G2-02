"""Exact posterior given a stratified receiver bank (other voxels masked to 'no constraint') vs the full window."""
import sys, json, time, numpy as np
sys.path.insert(0, 'code')
from rpl.data import Scene
from rpl.engine2 import scene_geometry
from rpl.engine3 import run_window
from rpl.bound import tv_bound
sid = sys.argv[1]; wid = int(sys.argv[2]); n = int(sys.argv[3]); seed = 0
sc = Scene(sid); geom = scene_geometry(sc); cls, o, L = sc.window_classes(wid)
rng = np.random.default_rng(seed)
lit = np.flatnonzero(cls.reshape(-1) == 1); dark = np.flatnonzero(cls.reshape(-1) == 0)
keep = np.concatenate([rng.choice(lit, min(n // 2, len(lit)), replace=False), rng.choice(dark, min(n // 2, len(dark)), replace=False)])
c2 = np.full(cls.size, 2, np.uint8); c2[keep] = cls.reshape(-1)[keep]; c2 = c2.reshape(cls.shape)
t = time.time(); r = run_window(geom, c2, o, d_full=10, d_max=13, max_terms=1024, max_nodes=3_000_000, n_boundary=min(512, n), n_class=0); dt = time.time() - t
cells, inv = np.unique(r['cell'], return_inverse=True); lo = np.bincount(inv, r['lo']); hi = np.bincount(inv, r['hi'])
tr = sc.truth; tc = (int(tr[0]*128)*128 + int(tr[1]*128))*128 + int(tr[2]*128)
q = lo / lo.sum() if lo.sum() > 0 else lo
res = dict(scene=sid[:8], window=wid, bank=int(len(keep)), seconds=dt, complete=r['stats']['budget_unresolved'] == 0,
           L=float(lo.sum()), U=float((hi - lo).sum()), tv_bound=tv_bound(lo, hi), cells_certified=int((lo > 0).sum()),
           bank_mass_in_truth_cell=float(q[cells == tc].sum()), nodes=r['stats']['nodes'])
print(json.dumps(res)); json.dump(res, open(f'outputs/bank_{sid[:8]}_{wid}_{n}.json', 'w'), indent=1)
