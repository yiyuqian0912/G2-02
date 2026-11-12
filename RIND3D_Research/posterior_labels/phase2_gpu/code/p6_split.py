"""Wall split: evaluation passes (entry loop + clip application) vs. the rest of stage2, and pass-size histograms."""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tpl.geom import Geom
from tpl.window import Window
from tpl.run_p6 import run_window
out = sys.argv[1]; res = []
for f in sys.argv[2:]:
    z = np.load(f); g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    run_window(g, w, d1=4)
    c0 = time.perf_counter(); r = run_window(g, w, d1=4); c1 = time.perf_counter()
    nqh = np.array(r['nqh']); neh = np.array(r['neh'])
    d = dict(name=os.path.basename(f)[:-4], side=int(w.L), N=int(w.N), C=int(g.P.shape[0]) if hasattr(g, 'P') else -1,
             wall=c1 - c0, entry_loop=r['tim'][0], pass_total=r['tim'][1], passes=int(r['stats']['passes']),
             nodes=r['stats']['nodes'], entry_evals=r['stats']['entry_evals'], comp_evals=r['stats']['comp_evals'],
             nq_hist_log2={int(b): int(v) for b, v in enumerate(nqh) if v}, ne_hist_log2={int(b): int(v) for b, v in enumerate(neh) if v})
    # share of evaluated entries in passes whose queue has >= 2^b entries (weight by 2^b approx)
    res.append(d); print(json.dumps(d), flush=True)
json.dump(res, open(out, 'w'), indent=1)
