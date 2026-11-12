"""Batch (engine v2): tau-pencil engine on real RIND3D v0.1 windows at 1024^3 (validation split only), with brute-force
verification.  One process per window (single-threaded numba), results as JSON + NPZ per window."""
import sys, os, json, time, argparse, multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def work(task):
    try:
        return _work(task)
    except Exception as e:
        import traceback
        sid, wid, out = task[0], task[1], task[2]
        f = os.path.join(out, f'{sid[:8]}_{wid}.err')
        open(f, 'w').write(traceback.format_exc())
        return f


def _work(task):
    sid, wid, out, kw, ver = task
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    from rpl.data import Scene
    from tpl.geom import Geom
    from tpl.window import Window
    from tpl.run_b3 import run_window, cell_of, cell_intervals
    from tpl.verify import verify
    f = os.path.join(out, f'{sid[:8]}_{wid}.json')
    if os.path.exists(f):
        return f
    sc = Scene(sid)
    geom = Geom.from_scene(sc)
    cls, o, L = sc.window_classes(wid)
    w = Window(cls, o, 1024)
    t0 = time.perf_counter(); c0 = time.process_time()
    r = run_window(geom, w, **kw)
    wall = time.perf_counter() - t0; cpu = time.process_time() - c0
    cells, Lc, Uc = cell_intervals(r)
    tc = cell_of(sc.truth)
    rec = dict(scene=sid, window=int(wid), side=int(L), origin=[int(v) for v in o], components=int(geom.C),
               counts=w.counts(), n_items=r['n_items'], kw=kw, wall_seconds=wall, cpu_seconds=cpu,
               seconds=r['seconds'], stage1=r['stage1'], stats=r['stats'],
               L_total=r['L_total'], U_total=r['U_total'], undet_total=r['undet_total'], tv_bound=r['tv_bound'],
               n_cells_certified=r['n_cells_certified'], n_cells_possible=r['n_cells_possible'],
               truth_cell=tc, truth_cell_L=float(Lc[cells == tc].sum()), truth_cell_U=float(Uc[cells == tc].sum()),
               n_records=int(len(r['cell'])))
    if ver:
        t1 = time.perf_counter()
        # brute-force cost per sound point grows with the window's receiver count: scale the sample down
        ns = max(200 if L <= 128 else 60, int(ver * (64 / L) ** 3))
        nb = 10000 if L <= 128 else 4000
        rec['verify'] = verify(r, geom, w, n_sound=ns, n_box=nb, n_world=nb, inflate=1.0)
        rec['verify_seconds'] = time.perf_counter() - t1
    import resource
    rec['maxrss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    np.savez_compressed(os.path.join(out, f'{sid[:8]}_{wid}.npz'), cells=cells, L=Lc, U=Uc)
    json.dump(rec, open(f + '.tmp', 'w'), indent=1)
    os.replace(f + '.tmp', f)
    return f


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True, help='JSON file with [[scene_id, window_id], ...]')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--kw', default='{"d1": 4}')
    ap.add_argument('--verify', type=int, default=5000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = json.load(open(a.tasks))
    if isinstance(tasks, dict):
        tasks = tasks['tasks']
    kw = json.loads(a.kw)
    json.dump(dict(args=vars(a), tasks=tasks), open(os.path.join(a.out, 'SAMPLE.json'), 'w'), indent=1)
    T = [(s, w, a.out, kw, a.verify) for s, w in tasks]
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for i, f in enumerate(pool.imap_unordered(work, T)):
            if f.endswith('.err'):
                print(f"{i+1}/{len(T)} ERROR {os.path.basename(f)}", flush=True)
                continue
            r = json.load(open(f))
            v = r.get('verify', {})
            print(f"{i+1}/{len(T)} {r['scene'][:8]}/{r['window']} side {r['side']} wall {r['wall_seconds']:.1f}s "
                  f"L {r['L_total']:.3e} undet {r['undet_total']:.2e} TV {r['tv_bound']:.4f} cells {r['n_cells_certified']} "
                  f"truthL {r['truth_cell_L']/max(r['L_total'],1e-300):.3f} sound {v.get('sound_infeasible')} "
                  f"miss {v.get('box_feasible_outside_polytopes')}/{v.get('world_feasible_outside')}", flush=True)
