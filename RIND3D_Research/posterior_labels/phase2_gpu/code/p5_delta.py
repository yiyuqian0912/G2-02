"""Differential timing of engine operations: CPU(op x2) - CPU(op x1) = time spent in that operation."""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tpl.geom import Geom
from tpl.window import Window
from tpl.run_p5 import run_window
out = sys.argv[1]
res = []
for f in sys.argv[2:]:
    z = np.load(f); g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    run_window(g, w, d1=4)
    def t(**kw):
        best = 1e9
        for _ in range(2):
            c = time.process_time(); r = run_window(g, w, d1=4, **kw); best = min(best, time.process_time() - c)
        return best, r
    base, r = t()
    d = dict(name=os.path.basename(f)[:-4], side=int(w.L), N=int(w.N), base=base, comp_evals=r['stats']['comp_evals'],
             entry_evals=r['stats']['entry_evals'], nodes=r['stats']['nodes'])
    for k in ('rep_eval', 'rep_hull', 'rep_copy', 'rep_ctx'):
        tt, _ = t(**{k: 2}); d['delta_' + k] = tt - base; d['share_' + k] = (tt - base) / base
    res.append(d); print(json.dumps(d), flush=True)
json.dump(res, open(out, 'w'), indent=1)
