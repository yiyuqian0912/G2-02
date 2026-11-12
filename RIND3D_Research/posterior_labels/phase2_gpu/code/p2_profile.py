"""Phase 2, step 1: profile the frozen Phase-1 engine b4 on the declared 12-window bundle.

Restartable: one JSON per window in <out>; existing results are skipped.  Records hardware, CPU-seconds per stage,
engine counters, and derived per-evaluation costs.  Usage: python p2_profile.py <bundle_dir> <out_dir> [names...]"""
import sys, os, json, time, platform, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))


def hardware():
    info = dict(platform=platform.platform(), python=platform.python_version(), processor=platform.processor())
    try:
        cpu = open('/proc/cpuinfo').read()
        names = [l.split(':', 1)[1].strip() for l in cpu.splitlines() if l.startswith('model name')]
        info.update(cpu_model=names[0] if names else None, logical_cpus=len(names))
        mem = [l for l in open('/proc/meminfo').read().splitlines() if l.startswith('MemTotal')]
        info['mem_total'] = mem[0].split(':')[1].strip() if mem else None
    except Exception as e:  # noqa: BLE001
        info['cpu_error'] = repr(e)
    try:
        import subprocess
        info['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                             '--format=csv,noheader'], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        info['nvidia_smi'] = None
    import numba
    info['numba'] = numba.__version__; info['numpy'] = np.__version__
    return info


def main(bundle, out, names=None):
    os.makedirs(out, exist_ok=True)
    json.dump(hardware(), open(os.path.join(out, 'HARDWARE.json'), 'w'), indent=1)
    from tpl.geom import Geom
    from tpl.window import Window
    from tpl.run_b4 import run_window
    from tpl.run import cell_intervals
    idx = json.load(open(os.path.join(bundle, 'INDEX.json')))['windows']
    if names:
        idx = [w for w in idx if w['name'] in names]
    # JIT warm-up on the smallest window (compile time recorded separately, not charged to windows)
    t = time.perf_counter()
    z = np.load(os.path.join(bundle, idx[0]['name'] + '.npz'))
    g = Geom(z['P'], z['K'])
    run_window(g, Window(z['cls'], z['origin'], int(z['N'])), d1=4)
    json.dump(dict(warmup_wall_seconds=time.perf_counter() - t), open(os.path.join(out, 'WARMUP.json'), 'w'))
    for w in idx:
        f = os.path.join(out, w['name'] + '.json')
        if os.path.exists(f):
            continue
        z = np.load(os.path.join(bundle, w['name'] + '.npz'))
        t0 = time.perf_counter(); c0 = time.process_time()
        g = Geom(z['P'], z['K'])
        win = Window(z['cls'], z['origin'], int(z['N']))
        t1 = time.perf_counter(); c1 = time.process_time()
        r = run_window(g, win, d1=4)
        t2 = time.perf_counter(); c2 = time.process_time()
        cells, L, U = cell_intervals(r)
        st = r['stats']
        rec = dict(w, prep_cpu=c1 - c0, engine_cpu=c2 - c1, engine_wall=t2 - t1, seconds=r['seconds'], stats=st,
                   stage1=r['stage1'], n_items=r['n_items'], tv_bound=r['tv_bound'], L_total=r['L_total'],
                   n_cells=int((L > 0).sum()),
                   ns_per_comp_eval=1e9 * (c2 - c1) / max(st['comp_evals'], 1),
                   evals_per_node=st['entry_evals'] / max(st['nodes'], 1),
                   face_split_share=st['face_splits'] / max(st['nodes'], 1))
        json.dump(rec, open(f + '.tmp', 'w'), indent=1)
        os.replace(f + '.tmp', f)
        print(w['name'], 'side', w['side'], 'cpu %.1f (phase1 %.1f)' % (c2 - c1, w['phase1_cpu_seconds']),
              'tv', r['tv_bound'], 'evals %.2e comp %.2e nodes %d face %d' % (st['entry_evals'], st['comp_evals'], st['nodes'],
                                                                          st['face_splits']), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
