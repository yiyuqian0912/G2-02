"""Worker entry points for the supervised pool (`pool.WorkerPool`)."""
import time
import traceback

_FACTORY = None


def physical_task(t, engine, kw):
    """t = dict(root, slot, attempt, k, c_table); kw = contract. Renders locally on this worker's GPU, writes the scene
    container atomically if accepted, and returns the manifest record (never the bulk data)."""
    global _FACTORY
    from .physical import run_scene
    from .storage import scene_path, atomic_write
    try:
        if _FACTORY is None:
            from rind3d.rendering.backends import BackendFactory
            _FACTORY = BackendFactory(cuda_batch=int(kw.get("cuda_batch", 32)), geometry_cache=2)
        t0 = time.perf_counter()
        rec, payload = run_scene(kw, int(t["slot"]), int(t["k"]), t["c_table"], _FACTORY, int(t["attempt"]))
        if payload is not None:
            p = scene_path(t["root"], t["slot"])
            rec["sha256"] = atomic_write(p, payload); rec["path"] = str(p.relative_to(t["root"]))
        rec["wall"] = time.perf_counter() - t0
        return rec
    except Exception as e:
        return dict(slot=int(t["slot"]), attempt=int(t["attempt"]), status="failed", error=repr(e)[:500],
                    trace=traceback.format_exc()[-1500:])


def warmup_task(t, engine, kw):
    """Compile the numba engine caches and NVRTC once (keeps concurrent first-compiles off the critical path)."""
    t0 = time.perf_counter()
    import numpy as np
    from rind3d_posterior import label_window
    cls = np.ones((64, 64, 64), np.uint8); cls[20:30, 20:30, 20:30] = 0
    P = np.zeros((1, 6, 4)); K = np.array([6])
    for i, (ax, sg) in enumerate([(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]):
        P[0, i, ax] = sg; P[0, i, 3] = -0.51 if sg > 0 else 0.49     # cube [0.49, 0.51]^3
    try:
        label_window(P, K, cls, np.array([700, 700, 700]), 1024, engine=engine if engine in ("cpu", "gpu") else "cpu")
        if engine == "fastlabel":                 # FastLabel's own numba kernels (no GPU work here)
            from rind3d_posterior.fastlabel import engine as FE, gpuverify as GV
            from rind3d_posterior.fastlabel.window import cls_from_packed, LazyWindow2
            zyx = np.transpose(cls, (2, 1, 0))
            c2 = cls_from_packed(np.packbits((zyx == 1).ravel()), np.packbits((zyx == 2).ravel()), 64)
            win = LazyWindow2(c2, np.array([700, 700, 700]), 1024); win.boundary_mask(); win.top_items()
            GV.leaf_codes(c2, False); GV.leaf_codes(c2, True)
            geom = FE.Geom(P.astype(float), K); roots, _ = FE.stage1(geom, win)
            xc, xk = FE.point_items(win, 256, 128, 0); FE.stage2(geom, win, xc, xk, roots)
        return dict(ok=True, s=time.perf_counter() - t0)
    except Exception as e:
        return dict(ok=False, error=repr(e)[:300], s=time.perf_counter() - t0)
