"""Depth-resolved evaluation counts (instrumented copy engine_p2 = b4 + counters): where is the work?"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tpl.geom import Geom
from tpl.window import Window
from tpl.run_p2 import run_window
bundle, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for name in sys.argv[3:]:
    z = np.load(os.path.join(bundle, name + '.npz'))
    g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    c0 = time.process_time(); r = run_window(g, w, d1=4); c1 = time.process_time()
    de = np.array(r['dep_evals']); dn = np.array(r['dep_nodes'])
    rec = dict(name=name, side=int(w.L), cpu=c1 - c0, tv=r['tv_bound'], n_roots=r['stage1']['candidates'], n_items=r['n_items'],
               evals_total=int(de.sum()), evals_depth0=int(de[0]), evals_depth_le3=int(de[:4].sum()),
               nodes_depth0=int(dn[0]), dep_evals=de.tolist(), dep_nodes=dn.tolist(), stats=r['stats'])
    json.dump(rec, open(os.path.join(out, name + '.json'), 'w'), indent=1)
    print(name, 'side', w.L, 'cpu %.1f' % rec['cpu'], 'roots', rec['n_roots'], 'items', rec['n_items'],
          'evals depth0 share %.2f, depth<=3 %.2f' % (de[0] / de.sum(), de[:4].sum() / de.sum()),
          'max depth', int(np.flatnonzero(dn)[-1]), flush=True)
