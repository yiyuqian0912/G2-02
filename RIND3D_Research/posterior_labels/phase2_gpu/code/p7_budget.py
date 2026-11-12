"""Direction F: certified early stopping by per-cell node budget.  For each window and budget B, run the reference
engine with max_nodes_cell = B (undecided regions are emitted as UNDET, i.e. counted in the upper masses only) and
record CPU seconds, the certified TV bound of the reported label (rpl.bound.tv_bound), and the UNDET volume share.
Usage: p7_budget.py out.json workers budgets(comma) -- files..."""
import sys, os, json, time
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def task(a):
    f, B = a
    sys.path.insert(0, HERE)
    from tpl.geom import Geom
    from tpl.window import Window
    from tpl import run_p2
    z = np.load(f); g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    c0 = time.process_time()
    r = run_p2.run_window(g, w, d1=4, max_nodes_cell=B)
    cpu = time.process_time() - c0
    lo = r['lo']; hi = r['hi']; kind = r['kind']
    und = float(hi[kind == 2].sum()); tot_hi = float(hi.sum()); tot_lo = float(lo.sum())
    return dict(file=os.path.basename(f)[:-4], side=int(w.L), N=int(w.N), budget=B, cpu=cpu, tv=float(r['tv_bound']),
                undet_share=und / max(tot_hi, 1e-300), vol_lo=tot_lo, vol_hi=tot_hi, nodes=int(r['stats']['nodes']),
                evals=int(r['stats']['entry_evals']), nrec=int(len(lo)))


def main():
    out, k, budgets = sys.argv[1], int(sys.argv[2]), [int(b) for b in sys.argv[3].split(',')]
    files = sys.argv[sys.argv.index('--') + 1:]
    tasks = [(f, B) for f in files for B in budgets]
    tasks.sort(key=lambda t: -t[1])
    ctx = mp.get_context('spawn')
    with ctx.Pool(k) as pool:
        res = []
        for r in pool.imap_unordered(task, tasks, chunksize=1):
            res.append(r); print(json.dumps(r), flush=True)
            json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
