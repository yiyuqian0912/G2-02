"""Equivalence + savings of the per-item cache (engine_p3): run each window with use_cache False and True; the
emitted records (cell, lo, hi, kind) and all engine statistics must be identical; report component-evaluation
counts and CPU seconds.  Usage: p3_cache_compare.py <bundle> <out.json> [names...] (default: all in INDEX)."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tpl.geom import Geom
from tpl.window import Window
from tpl.run_p4 import run_window


def one(z, use_cache):
    g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    c0 = time.process_time(); r = run_window(g, w, d1=4, use_gc=use_cache); c1 = time.process_time()
    return r, c1 - c0


def main(bundle, out, names):
    idx = json.load(open(os.path.join(bundle, 'INDEX.json')))['windows']
    names = names or [w['name'] for w in idx]
    z0 = np.load(os.path.join(bundle, names[0] + '.npz'))
    one(z0, False); one(z0, True)                   # JIT warm-up (not timed)
    res = []
    for nm in names:
        z = np.load(os.path.join(bundle, nm + '.npz'))
        ra, ta = one(z, False)
        rb, tb = one(z, True)
        same_rec = all(np.array_equal(ra[k], rb[k]) for k in ('cell', 'lo', 'hi', 'kind'))
        keys = [k for k in ra['stats'] if k not in ('comp_evals',)]
        same_stats = all(ra['stats'][k] == rb['stats'][k] for k in keys)
        ca = rb['gc_stats']
        full_a = ra['stats']['comp_evals']; full_b = ca['computed']
        rec = dict(name=nm, side=int(z['cls'].shape[0]), N=int(z['N']), identical_records=bool(same_rec),
                   identical_stats=bool(same_stats), tv_bound=rb['tv_bound'],
                   cpu_nocache=ta, cpu_cache=tb, speedup=ta / max(tb, 1e-9),
                   full_evals_nocache=full_a, full_evals_cache=full_b, cached_evals=ca['hits'],
                   full_eval_reduction=full_a / max(full_b, 1), nodes=rb['stats']['nodes'])
        res.append(rec)
        print(json.dumps(rec), flush=True)
        json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
