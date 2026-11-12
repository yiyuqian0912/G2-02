"""Exactness and cost of engine_g1 variants against the reference engine (engine_p2 = Phase-1 b4 + counters).
Variants: ref (p2), cpu_na (allocation-free hull test), gpu (exact GPU pass segments, FP64).  Records must be
identical (cell, lo, hi, kind); decision statistics must be identical; evaluation counters may differ for the GPU
variant only (it also evaluates entries after a refuter).  Usage: g1_compare.py out.json gpu_min gpu_prefix files..."""
import sys, os, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tpl.geom import Geom
from tpl.window import Window
from tpl import run_p2, run_g1

COUNTERS = {'entry_evals', 'comp_evals', 'ambig_evals', 'max_entries', 'expansions'}


def digest(r):
    h = hashlib.sha256()
    for k in ('cell', 'lo', 'hi', 'kind'):
        h.update(np.ascontiguousarray(r[k]).tobytes())
    return h.hexdigest()[:16]


def timed(fn, g, w, **kw):
    c0 = time.process_time(); t0 = time.perf_counter(); r = fn(g, w, d1=4, **kw)
    return r, time.process_time() - c0, time.perf_counter() - t0


def main(out, gmin, gpre, files):
    res = []
    z = np.load(files[0]); g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    run_p2.run_window(g, w, d1=4); run_g1.run_window(g, w, d1=4, no_alloc_hull=True)
    run_g1.run_window(g, w, d1=4, use_gpu=True, gpu_min=gmin, gpu_prefix=gpre, gpu_version=int(os.environ.get('GPU_VERSION', '2')), gpu_f32=int(os.environ.get('GPU_F32', '0')))
    for f in files:
        z = np.load(f); g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
        ra, ca, wa = timed(run_p2.run_window, g, w)
        if os.environ.get('SKIP_NA'):
            rb, cb, wb = ra, ca, wa
        else:
            rb, cb, wb = timed(run_g1.run_window, g, w, no_alloc_hull=True)
        rc, cc, wc = timed(run_g1.run_window, g, w, use_gpu=True, gpu_min=gmin, gpu_prefix=gpre,
                           gpu_version=int(os.environ.get('GPU_VERSION', '2')), gpu_f32=int(os.environ.get('GPU_F32', '0')))
        da, db, dc = digest(ra), digest(rb), digest(rc)
        sa, sb, sc = ra['stats'], rb['stats'], rc['stats']
        d = dict(name=os.path.basename(f)[:-4], side=int(w.L), N=int(w.N),
                 identical_cpu_na=bool(da == db and sa == sb),
                 identical_gpu_records=bool(da == dc),
                 identical_gpu_decision_stats=bool(all(sa[k] == sc[k] for k in sa if k not in COUNTERS)),
                 tv_bound=rc['tv_bound'], ref_cpu=ca, ref_wall=wa, na_cpu=cb, na_wall=wb, gpu_cpu=cc, gpu_wall=wc,
                 speedup_na=ca / cb, speedup_gpu_wall=wa / wc,
                 entry_evals_ref=sa['entry_evals'], entry_evals_gpu=sc['entry_evals'], gpu=rc.get('gpu'),
                 nodes=sa['nodes'])
        if not d['identical_gpu_decision_stats']:
            d['stat_diff'] = {k: (sa[k], sc[k]) for k in sa if sa[k] != sc[k] and k not in COUNTERS}
        res.append(d); print(json.dumps(d), flush=True)
        json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:])
