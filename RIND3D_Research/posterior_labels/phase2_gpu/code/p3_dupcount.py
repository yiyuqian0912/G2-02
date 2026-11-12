import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tpl.geom import Geom
from tpl.window import Window
from tpl.run_p3 import run_window
bundle, out = sys.argv[1], sys.argv[2]
res = []
for nm in sys.argv[3:]:
    z = np.load(os.path.join(bundle, nm + '.npz'))
    g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    r = run_window(g, w, d1=4, use_cache=False, count_dups=True)
    cs = r['cache_stats']
    rec = dict(name=nm, side=int(w.L), N=int(w.N), full=cs['full'], first=cs['first_full'], repeat=cs['repeat_full'],
               repeat_share=cs['repeat_full'] / max(cs['full'], 1), roots=r['stage1']['candidates'], items=r['n_items'])
    res.append(rec); print(json.dumps(rec), flush=True)
json.dump(res, open(out, 'w'), indent=1)
