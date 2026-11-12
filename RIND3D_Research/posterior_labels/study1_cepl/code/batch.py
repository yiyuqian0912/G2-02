"""Batch CEPL runs on a declared sample of real v0.1 windows (validation split only; test is never opened).

Sample: `--scenes` validation scenes drawn with a fixed seed; per scene `--per-scene` windows drawn stratified by
side.  Each window runs in its own worker process with a node budget; results are written per window as
<out>/<scene8>_<wid>.json (summary) and .npz (per-128^3-cell lower/upper volumes).
"""
import argparse, json, os, sys, time, traceback
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def window_task(args):
    sid, wid, kw, out = args
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    import importlib
    from rpl.data import Scene
    from rpl.engine2 import scene_geometry
    eng = importlib.import_module(kw.pop('engine', 'rpl.engine3'))
    path = os.path.join(out, f'{sid[:8]}_{wid}')
    if os.path.exists(path + '.json'):
        return json.load(open(path + '.json'))
    try:
        sc = Scene(sid)
        geom = scene_geometry(sc)
        cls, o, L = sc.window_classes(wid)
        t = time.time()
        res = eng.run_window(geom, cls, o, **kw)
        dt = time.time() - t
        st = res['stats']
        cells, inv = np.unique(res['cell'], return_inverse=True)
        lo = np.bincount(inv, res['lo']) if len(cells) else np.zeros(0)
        hi = np.bincount(inv, res['hi']) if len(cells) else np.zeros(0)
        Lm = float(lo.sum()); U = float((hi - lo).sum())
        complete = st['budget_unresolved'] == 0 and st['invalid'] == 0
        tc = (int(sc.truth[0] * 128) * 128 + int(sc.truth[1] * 128)) * 128 + int(sc.truth[2] * 128)
        q = lo / Lm if Lm > 0 else lo
        nz = q[q > 0]
        coarse = {}
        for cidx, m in zip(cells, q):
            ix, rem = divmod(int(cidx), 128 * 128); iy, iz = divmod(rem, 128)
            k = ((ix // 8) * 16 + iy // 8) * 16 + iz // 8
            coarse[k] = coarse.get(k, 0.0) + float(m)
        cq = np.array([v for v in coarse.values() if v > 0])
        tck = ((int(sc.truth[0] * 16)) * 16 + int(sc.truth[1] * 16)) * 16 + int(sc.truth[2] * 16)
        summary = dict(scene=sid, window=int(wid), origin=list(o), side=int(L), seconds=dt, stats=st,
                       complete=bool(complete), L=Lm, U=U, tv_bound=(U / (Lm + U)) if Lm + U > 0 else None,
                       n_fine_cells=int(len(nz)), fine_entropy=float(-(nz * np.log(nz)).sum()) if len(nz) else None,
                       truth_fine_mass=float(q[cells == tc].sum()) if Lm > 0 else None,
                       n_coarse_cells=int(len(cq)), coarse_entropy=float(-(cq * np.log(cq)).sum()) if len(cq) else None,
                       truth_coarse_mass=float(coarse.get(tck, 0.0)),
                       lit=int((cls == 1).sum()), dark=int((cls == 0).sum()), solid=int((cls == 2).sum()),
                       components=int(len(geom[1])), kw=kw)
        np.savez_compressed(path + '.npz', cells=cells.astype(np.int64), lo=lo, hi=hi)
        json.dump(summary, open(path + '.json', 'w'), indent=1)
        return summary
    except Exception as e:
        err = dict(scene=sid, window=int(wid), error=repr(e), trace=traceback.format_exc())
        json.dump(err, open(path + '.err.json', 'w'), indent=1)
        return err


def declared_sample(n_scenes, per_scene, sides, seed, split='val'):
    from rpl.data import scene_ids, splits
    from rpl import rindenv
    sp = splits()
    ids = sorted(s for s in scene_ids() if sp.get(s) == split)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ids, n_scenes, replace=False)
    tasks = []
    for sid in chosen:
        rec = np.fromfile(os.path.join(rindenv.SCENES, sid, 'window_records.bin'), '<u2').reshape(-1, 4)
        for side in sides:
            w = np.flatnonzero(rec[:, 3] == side)
            for wid in rng.choice(w, min(per_scene, len(w)), replace=False):
                tasks.append((str(sid), int(wid)))
    return tasks


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, default=10)
    ap.add_argument('--per-scene', type=int, default=2)
    ap.add_argument('--sides', default='64')
    ap.add_argument('--seed', type=int, default=20261111)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--kw', default='{}')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = declared_sample(a.scenes, a.per_scene, [int(s) for s in a.sides.split(',')], a.seed)
    json.dump(dict(args=vars(a), tasks=tasks), open(os.path.join(a.out, 'SAMPLE.json'), 'w'), indent=1)
    kw = json.loads(a.kw)
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(window_task, [(s, w, dict(kw), a.out) for s, w in tasks]):
            msg = {k: r.get(k) for k in ('scene', 'window', 'side', 'seconds', 'complete', 'tv_bound', 'n_fine_cells',
                                          'truth_fine_mass', 'error')}
            msg['scene'] = (msg['scene'] or '')[:8]
            print(json.dumps(msg), flush=True)
    print('total wall', time.time() - t0)
