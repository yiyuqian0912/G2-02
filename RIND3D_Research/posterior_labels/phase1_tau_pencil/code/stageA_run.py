"""Stage A comparison on the 128^3-receiver platform (outputs/stageA_windows): per window and method, the reported
TV upper bound, guarantee type, cost.  Methods:
  tpe        tau-pencil engine (deterministic certificate)
  cepl       first-study engine (rpl engine3 at N = 128; deterministic certificate), declared node budget
  cover_mc   stratified Monte Carlo inside the deterministic stage-1 cover (99% statistical certificate)
  prior_mc   Monte Carlo from the prior with sample splitting (99% statistical certificate)
The reference posterior for accuracy checks is the TPE label when TPE certifies TV = 0 (exact), else none."""
import sys, os, json, time, argparse, multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))


def tv(a_cells, a_p, b_cells, b_p):
    da = dict(zip(np.asarray(a_cells).tolist(), np.asarray(a_p).tolist()))
    db = dict(zip(np.asarray(b_cells).tolist(), np.asarray(b_p).tolist()))
    return 0.5 * sum(abs(da.get(k, 0) - db.get(k, 0)) for k in set(da) | set(db))


def work(task):
    name, out, methods, budgets = task
    os.environ.setdefault('NUMBA_NUM_THREADS', '1')
    from rpl.data import Scene
    from rpl.bound import tv_bound
    from tpl.geom import Geom
    from tpl.window import Window
    from tpl.run_b3 import run_window
    from tpl.run import cell_intervals, label, cell_of
    from tpl import stat
    f = os.path.join(out, name + '.json')
    if os.path.exists(f):
        return f
    z = np.load(os.path.join('outputs/stageA_windows', name + '.npz'))
    sid = str(z['scene'])
    geom = Geom.from_scene(Scene(sid))
    w = Window(z['cls'], z['origin'], int(z['N']))
    truth = z['truth']
    rec = dict(name=name, scene=sid, side=int(w.L), counts=w.counts(), components=int(geom.C), results={})
    ref = None
    if 'tpe' in methods:
        t0 = time.process_time()
        r = run_window(geom, w, d1=4)
        cpu = time.process_time() - t0
        cl, q = label(r)
        rec['results']['tpe'] = dict(guarantee='deterministic', confidence=1.0, tv_bound=r['tv_bound'], cpu_seconds=cpu,
                                     L_total=r['L_total'], undet_total=r['undet_total'], cells=int(len(cl)),
                                     truth_cell_mass=float(q[cl == cell_of(truth)].sum()) if len(cl) else 0.0,
                                     stage1=r['stage1'], stats=r['stats'])
        if r['tv_bound'] == 0.0:
            ref = (cl, q)
        cover = r['candidates']
        np.savez_compressed(os.path.join(out, name + '_tpe.npz'), cells=cl, q=q)
    if 'cepl' in methods:
        from rpl128.engine3 import run_window as cepl_run
        from rpl.topology import adjacency
        g = (geom.P, geom.K, geom.clo, geom.chi, geom.adj, geom.aoff)
        for nb in budgets.get('cepl', [20000]):
            t0 = time.process_time()
            rr = cepl_run(g, w.cls, w.origin, d_full=10, d_max=13, max_terms=1024, max_nodes=nb)
            cpu = time.process_time() - t0
            res = dict(cell=rr['cell'], lo=rr['lo'], hi=rr['hi'])
            cells, Lc, Uc = cell_intervals(res)
            tb = float(tv_bound(Lc, Uc)) if Lc.sum() > 0 else 1.0
            e = dict(guarantee='deterministic', confidence=1.0, tv_bound=tb, cpu_seconds=cpu, nodes_budget=nb,
                     L_total=float(Lc.sum()), undet_total=float((Uc - Lc).sum()), stats=rr['stats'])
            if ref is not None and Lc.sum() > 0:
                e['tv_vs_exact'] = tv(cells[Lc > 0], Lc[Lc > 0] / Lc.sum(), *ref)
            rec['results'][f'cepl_{nb}'] = e
    if 'cover_mc' in methods:
        f_ = 1 << (7 - 4)
        gg = np.stack(np.meshgrid(*[np.arange(f_)] * 3, indexing='ij'), -1).reshape(-1, 3)
        cells_xyz = (cover[:, None, 1:] * f_ + gg[None]).reshape(-1, 3)
        for npc in budgets.get('cover_mc', [64]):
            t0 = time.process_time()
            m = stat.cover_mc(geom, w, cells_xyz, npc, seed=1)
            cpu = time.process_time() - t0
            lab = m.pop('label')
            m.update(guarantee='statistical', confidence=0.99, tv_bound=m.pop('tv_cert'), cpu_seconds=cpu,
                     n_per_cell=npc)
            if ref is not None and len(lab[0]):
                m['tv_vs_exact'] = tv(*lab, *ref)
            rec['results'][f'cover_mc_{npc}'] = m
    if 'prior_mc' in methods:
        for n in budgets.get('prior_mc', [1 << 20]):
            t0 = time.process_time()
            m = stat.prior_mc(geom, w, n, seed=2)
            cpu = time.process_time() - t0
            lab = m.pop('label')
            m.update(guarantee='statistical', confidence=0.99, tv_bound=m.pop('tv_cert'), cpu_seconds=cpu)
            if ref is not None and len(lab[0]):
                m['tv_vs_exact'] = tv(*lab, *ref)
            rec['results'][f'prior_mc_{n}'] = m
    json.dump(rec, open(f + '.tmp', 'w'), indent=1)
    os.replace(f + '.tmp', f)
    return f


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs/stageA_results')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--methods', default='tpe,cepl,cover_mc,prior_mc')
    ap.add_argument('--budgets', default='{"cepl": [20000, 150000], "cover_mc": [16, 256], "prior_mc": [1048576, 16777216]}')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--names', default='', help='JSON file with {"names": [...]} (subset)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    idx = json.load(open('outputs/stageA_windows/INDEX.json'))['windows']
    names = [w['name'] for w in idx][: a.limit or None]
    if a.names:
        names = json.load(open(a.names))['names']
    json.dump(vars(a), open(os.path.join(a.out, 'ARGS.json'), 'w'), indent=1)
    T = [(n, a.out, a.methods.split(','), json.loads(a.budgets)) for n in names]
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for i, f in enumerate(pool.imap_unordered(work, T)):
            r = json.load(open(f))
            s = ' '.join(f"{k}:{v['tv_bound']:.3f}/{v['cpu_seconds']:.1f}s" for k, v in r['results'].items())
            print(f"{i+1}/{len(T)} {r['name']} side {r['side']} {s}", flush=True)
