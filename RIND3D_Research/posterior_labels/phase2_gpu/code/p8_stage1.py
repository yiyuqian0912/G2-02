"""Stage-1 depth/sentinel sweep: d1 (source cells 2^d1 per axis tested with sentinel receivers before stage 2).
Records CPU seconds split (stage1/stage2), roots, TV bound, certified volume.  Usage: p8_stage1.py out k cfgs -- files
cfgs: comma list of d1:n_boundary:n_class:max_tests"""
import sys, os, json, time
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def task(a):
    f, cfg = a
    sys.path.insert(0, HERE)
    from tpl.geom import Geom
    from tpl.window import Window
    from tpl import run_p2
    d1, nb, nc, mt = (int(x) for x in cfg.split(':'))
    z = np.load(f); g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    c0 = time.process_time()
    r = run_p2.run_window(g, w, d1=d1, n_boundary=nb, n_class=nc, max_tests=mt)
    cpu = time.process_time() - c0
    t = r.get('timing', {})
    return dict(file=os.path.basename(f)[:-4], cfg=cfg, cpu=cpu, tv=float(r['tv_bound']), lo=float(r['lo'].sum()),
                hi=float(r['hi'].sum()), roots=int(r['stage1']['candidates']), s1=r['seconds'], stats_nodes=int(r['stats']['nodes']),
                evals=int(r['stats']['entry_evals']))


if __name__ == '__main__':
    out, k, cfgs = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(',')
    files = sys.argv[sys.argv.index('--') + 1:]
    tasks = [(f, c) for f in files for c in cfgs]
    with mp.get_context('spawn').Pool(k) as pool:
        res = []
        for r in pool.imap_unordered(task, tasks, chunksize=1):
            res.append(r); print(json.dumps(r), flush=True); json.dump(res, open(out, 'w'), indent=1)
