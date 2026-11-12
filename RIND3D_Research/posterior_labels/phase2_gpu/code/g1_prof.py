import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tpl.geom import Geom
from tpl.window import Window
from tpl import run_g1
for f in sys.argv[3:]:
    z = np.load(f); g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    kw = dict(no_alloc_hull=False, use_gpu=True, gpu_min=int(sys.argv[1]), gpu_prefix=int(sys.argv[2]))
    run_g1.run_window(g, w, d1=4, **kw)
    t0 = time.perf_counter(); r = run_g1.run_window(g, w, d1=4, gpu_profile=True, **kw); t1 = time.perf_counter()
    print(os.path.basename(f), 'wall %.3f' % (t1 - t0), 'tt[pack,h2d,kern,d2h,compact,final]', [round(x, 3) for x in r['gpu']['tt']],
          'segs', r['gpu']['segments'], 'gpu_entries', r['gpu']['entries'], 'evals', r['stats']['entry_evals'], flush=True)
