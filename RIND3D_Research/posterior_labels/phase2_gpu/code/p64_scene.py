"""P64 scene-level batch: one scene, a fixed proposal budget of informative source-excluded windows, all labels.

Declared design (docs/TARGET_V03_SOURCE_EXCLUDED.md):
* scene: validation-split v0.1 geometry and its stored single source (the v0.3 source prior is the v0.1 driver rule);
* proposals: per physical side Lp in {64, 128, 256} (side L = Lp*N/1024 on the N^3 receiver grid), M uniform origins
  on the 128^3 lattice (o_N = o_128 * N/128), from a stream keyed by (seed, scene, Lp) only, so independent of the
  source;
* acceptance A_inf: the window has lit-FREE and dark-FREE receivers and the source lies outside the closed box
  (N = 128: the float renderer of Phase-1 Stage A, windows with an undecided receiver are skipped and counted;
  N = 1024: stored exact fields).  All accepted windows are kept (no quota), so lambda_W(s) does not depend on s and
  the label is exactly Uniform(F_W) on this declared informative population (not the natural distribution);
* labels: the exact engine (cpu: reference engine_p2; gpu: engine_g2 + GPU pass); records are cached per window.
Timing is reported per stage: scene preprocessing, selection (proposal rendering), labelling, certification
(TV bound, inside the engine), storage.
Usage: p64_scene.py out_dir scene_id N M engine workers [seed]"""
import sys, os, json, time, hashlib
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def render_N(P, K, s, origin, L, N):
    from rpl.predicates import render_points
    g = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing='ij'), -1).reshape(-1, 3)
    X = (np.asarray(origin) + g + 0.5) / N
    solid = np.zeros(len(X), bool)
    for c in range(P.shape[0]):
        solid |= np.all(X @ P[c, :K[c], :3].T + P[c, :K[c], 3] <= 0, axis=1)
    y = render_points(np.asarray(s, float), X, P, K)
    und = int((y[~solid] < 0).sum())
    cls = np.where(solid, 2, np.where(y == 1, 1, 0)).astype(np.uint8).reshape(L, L, L)
    return cls, und


def _init(engine, counter=None, kg=0):
    sys.path.insert(0, HERE)
    global ENG
    if engine == 'mix':
        with counter.get_lock():
            i = counter.value; counter.value += 1
        engine = 'gpu' if i < kg else 'cpu'
    ENG = engine


def _label(task):
    f, outf = task
    from tpl.geom import Geom
    from tpl.window import Window
    z = np.load(f); g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    c0 = time.process_time(); t0 = time.perf_counter()
    if ENG == 'cpu':
        from tpl import run_p2
        r = run_p2.run_window(g, w, d1=4)
    else:
        from tpl import run_g1
        r = run_g1.run_window(g, w, d1=4, use_gpu=True, gpu_min=1024, gpu_prefix=64)
    cpu = time.process_time() - c0; wall = time.perf_counter() - t0
    t1 = time.perf_counter()
    np.savez_compressed(outf, cell=r['cell'], lo=r['lo'], hi=r['hi'], kind=r['kind'], tv_bound=r['tv_bound'])
    h = hashlib.sha256()
    for k in ('cell', 'lo', 'hi', 'kind'):
        h.update(np.ascontiguousarray(r[k]).tobytes())
    return dict(engine=ENG, window=os.path.basename(f)[:-4], cpu=cpu, wall=wall, store_s=time.perf_counter() - t1,
                bytes=os.path.getsize(outf), tv=float(r['tv_bound']), nrec=int(len(r['cell'])),
                evals=int(r['stats']['entry_evals']), digest=h.hexdigest()[:16])


def main(out, sid, N, M, engine, workers, seed=20261111, kg=2):
    from rpl.data import Scene
    from tpl.geom import Geom
    os.makedirs(os.path.join(out, 'windows'), exist_ok=True); os.makedirs(os.path.join(out, 'labels'), exist_ok=True)
    T = {}
    t0 = time.perf_counter(); c0 = time.process_time()
    sc = Scene(sid)
    P, K, _, _ = sc.geometry_arrays()
    s = np.asarray(sc.truth, float)
    T['preprocess_wall'] = time.perf_counter() - t0
    # ---- selection: fixed proposal budget, source-independent origin streams
    t1 = time.perf_counter()
    sel = dict(proposals=0, contains_source=0, undecided=0, uninformative=0, accepted=0, by_side={})
    files = []
    key = int(sid[:8], 16)
    for Lp in (64, 128, 256):
        L = Lp * N // 1024
        L128 = Lp // 8
        rng = np.random.default_rng([seed, key, Lp])
        org = rng.integers(0, 128 - L128 + 1, size=(M, 3))
        acc = 0
        for j, o128 in enumerate(org):
            sel['proposals'] += 1
            o = tuple(int(v) * N // 128 for v in o128)
            lo = np.array(o) / N; hi = (np.array(o) + L) / N
            if np.all((s >= lo) & (s <= hi)):
                sel['contains_source'] += 1
                continue
            if N == 1024:
                cls = sc.region_classes(o, L); und = 0
            else:
                cls, und = render_N(P, K, s, o, L, N)
            if und:
                sel['undecided'] += 1
                continue
            if not ((cls == 0).any() and (cls == 1).any()):
                sel['uninformative'] += 1
                continue
            name = f'{sid[:8]}_L{Lp}_{j:03d}'
            f = os.path.join(out, 'windows', name + '.npz')
            np.savez_compressed(f, P=P, K=K, cls=cls, origin=np.array(o), N=N, truth=s)
            files.append(f); acc += 1
        sel['by_side'][Lp] = dict(proposals=M, accepted=acc)
        sel['accepted'] += acc
    T['selection_wall'] = time.perf_counter() - t1
    T['main_cpu_pre_sel'] = time.process_time() - c0
    # ---- labels (heaviest first)
    files.sort(key=lambda f: -os.path.getsize(f))
    tasks = [(f, os.path.join(out, 'labels', os.path.basename(f))) for f in files]
    t2 = time.perf_counter()
    ctx = mp.get_context('spawn')
    counter = ctx.Value('i', 0)
    with ctx.Pool(workers, initializer=_init, initargs=(engine, counter, kg)) as pool:
        list(pool.map(_label, [tasks[-1]] * workers, chunksize=1))   # warm-up (JIT, NVRTC), not timed
        t2 = time.perf_counter()
        res = list(pool.imap_unordered(_label, tasks, chunksize=1))
    T['label_wall'] = time.perf_counter() - t2
    T['label_cpu_sum'] = sum(r['cpu'] for r in res)
    T['label_task_wall_sum'] = sum(r['wall'] for r in res)
    T['storage_bytes'] = sum(r['bytes'] for r in res)
    tv = [r['tv'] for r in res]
    rep = dict(scene=sid, N=N, M_per_side=M, engine=engine, workers=workers, seed=seed, selection=sel, timing=T,
               n_labels=len(res), tv_max=max(tv) if tv else None, tv_mean=float(np.mean(tv)) if tv else None,
               n_tv_le_005=int(sum(t <= 0.05 for t in tv)), per_window=sorted(res, key=lambda r: r['window']),
               design=__doc__)
    json.dump(rep, open(os.path.join(out, 'P64_REPORT.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('per_window', 'design')}, indent=1))


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], int(a[3]), int(a[4]), a[5], int(a[6]), *(int(x) for x in a[7:9]))
