"""Machine-level throughput: k worker processes drain a fixed task list (window files).  mode=cpu uses the reference
engine (run_p2); mode=gpu uses the exact GPU-pass engine (run_g1, shared GPU, one CUDA context per worker).
Reports wall time, windows per hour, summed worker CPU seconds, and per-task record digests (to check exactness
across modes).  Usage: throughput.py out.json mode k [f32] -- files..."""
import sys, os, json, time, hashlib
import multiprocessing as mp
import numpy as np


def _init(mode, f32, counter=None, kg=0):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    global G
    if mode == 'mix':
        with counter.get_lock():
            idx = counter.value; counter.value += 1
        mode = 'gpu' if idx < kg else 'cpu'
    G = dict(mode=mode, f32=f32)
    if mode == 'gpu':
        pass   # blocking sync is done with cupy Events(block=True) inside gpu_pass2


def _task(f):
    from tpl.geom import Geom
    from tpl.window import Window
    z = np.load(f); g = Geom(z['P'], z['K']); w = Window(z['cls'], z['origin'], int(z['N']))
    c0 = time.process_time(); t0 = time.perf_counter()
    if G['mode'] == 'cpu':
        from tpl import run_p2
        r = run_p2.run_window(g, w, d1=4)
    else:
        from tpl import run_g1
        r = run_g1.run_window(g, w, d1=4, use_gpu=True, gpu_min=1024, gpu_prefix=64, gpu_version=int(os.environ.get('GPU_VERSION', '3')), gpu_f32=int(os.environ.get('GPU_F32', '2')), gpu_cap=int(os.environ.get('GPU_CAP', str(1 << 22))))
    h = hashlib.sha256()
    for k in ('cell', 'lo', 'hi', 'kind'):
        h.update(np.ascontiguousarray(r[k]).tobytes())
    return dict(mode=G['mode'], file=os.path.basename(f), cpu=time.process_time() - c0, wall=time.perf_counter() - t0,
                digest=h.hexdigest()[:16], tv=r['tv_bound'], pid=os.getpid(), gpu=r.get('gpu'))


def main():
    out, mode, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    f32 = len(sys.argv) > 4 and sys.argv[4] == 'f32'
    kg = int(sys.argv[5]) if mode == 'mix' else 0
    files = sys.argv[sys.argv.index('--') + 1:]
    ctx = mp.get_context('spawn')
    counter = ctx.Value('i', 0)
    with ctx.Pool(k, initializer=_init, initargs=(mode, f32, counter, kg)) as pool:
        # warm-up: JIT/NVRTC in every worker on the smallest task (not timed)
        list(pool.map(_task, [files[-1]] * k, chunksize=1))
        t0 = time.perf_counter()
        res = list(pool.imap_unordered(_task, files, chunksize=1))
        wall = time.perf_counter() - t0
    d = dict(mode=mode, k=k, kg=kg, f32=f32,
             by_mode={m: sum(1 for r in res if r['mode'] == m) for m in ('cpu', 'gpu')}, n=len(files), wall=wall, windows_per_hour=3600 * len(files) / wall,
             cpu_sum=sum(r['cpu'] for r in res), task_wall_sum=sum(r['wall'] for r in res), tasks=res)
    json.dump(d, open(out, 'w'), indent=1)
    print(json.dumps({k_: v for k_, v in d.items() if k_ != 'tasks'}))


if __name__ == '__main__':
    main()
