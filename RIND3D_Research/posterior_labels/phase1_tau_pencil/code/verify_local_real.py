"""Boundary-local completeness + soundness on real 1024^3 windows (engine b3): points uniform in each sampled IN
polytope's AABB grown by 50% per side; brute force over every FREE receiver; feasible points must lie in the
emitted support.  Declared subset: argv[1] task file; per-window sample sizes scale with the receiver count."""
import sys, json, time
import numpy as np
sys.path.insert(0, 'code')
from rpl.data import Scene, scene_ids
from tpl.geom import Geom
from tpl.window import Window
from tpl.run_b3 import run_window
from tpl.verify import verify_local, verify_scaled
tasks = json.load(open(sys.argv[1])); tasks = tasks['tasks'] if isinstance(tasks, dict) else tasks
out = []
for sid, wid in tasks:
    sc = Scene(sid); g = Geom.from_scene(sc)
    cls, o, L = sc.window_classes(wid)
    w = Window(cls, o, 1024)
    r = run_window(g, w, d1=4)
    npol = 150 if L == 64 else (25 if L == 128 else 8)
    t = time.perf_counter()
    v = verify_local(r, g, w, n_polys=npol, per=40, seed=11) if L == 64 else {}
    v.update(verify_scaled(r, g, w, n_polys=npol, per=40, scale=2.0, seed=12))
    v.update(window=f'{sid[:8]}/{wid}', side=int(L), seconds=time.perf_counter() - t, tv_bound=r['tv_bound'])
    out.append(v)
    print(json.dumps(v), flush=True)
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
