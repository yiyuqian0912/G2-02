"""RIND3D-100K label tasks: one task per (scene, window, source) informative channel, using the validated tau-pencil
engine (`rind3d_posterior.label_window`) on the exact LOCAL response plus the canonical GLOBAL geometry rows.

**Target** (unchanged from 1K): Uniform(F_W^j) over D(G) minus the closed window box. It is reported as certified
interval cell masses on the 128^3 grid with a rigorous TV bound (tv_bound).

No global response field is read. The only inputs are the scene container's local crops and geometry rows."""
import os
import time
from pathlib import Path

import numpy as np

from .storage import SceneFile, scene_path, task_key

STATUS_CERT, STATUS_LOOSE, STATUS_UNCERT = "certified", "certified_loose", "uncertified"
_CACHE = {}


def cell_arrays(cell, lo, hi):
    """Per-cell intervals (the label proper): the engine's polytope pieces summed per 128^3 cell. Cell ids are stored
    delta-encoded (sorted). tv_bound depends only on per-cell sums, so it is recomputed here and must equal the
    engine's value."""
    from rind3d_posterior.rpl.bound import tv_bound
    u, inv = np.unique(np.asarray(cell, np.int64), return_inverse=True)
    L = np.bincount(inv, weights=np.asarray(lo, np.float64), minlength=len(u))
    U = np.bincount(inv, weights=np.asarray(hi, np.float64), minlength=len(u))
    tv = float(tv_bound(L, U)) if L.sum() > 0 else 1.0
    return dict(dcell=np.diff(u, prepend=0).astype(np.int32), lo=L, hi=U), int(len(u)), tv


def decode_cells(d):
    """(cells, lo, hi) per cell from a journal payload (delta-encoded per-cell format, or the raw piece format)."""
    if "dcell" in d:
        return np.cumsum(d["dcell"].astype(np.int64)), d["lo"], d["hi"]
    u, inv = np.unique(d["cell"].astype(np.int64), return_inverse=True)
    return u, np.bincount(inv, weights=d["lo"], minlength=len(u)), np.bincount(inv, weights=d["hi"], minlength=len(u))


def tasks_for_scene(root, slot, include_uninformative=False):
    S = SceneFile(scene_path(root, slot))
    out = []
    for r in S.W:
        for j in range(int(r["k"])):
            inf = bool((int(r["mask"]) >> j) & 1)
            if inf or include_uninformative:
                out.append((str(root), int(slot), int(r["window"]), j, int(r["L"]), inf))
    return out


def _fastlabel_task(root, slot, w, s, L, inf, key, kw):
    """FastLabel v3.4 (rind3d_posterior.fastlabel) on the scene container. Journal arrays are per-cell (dcell, lo, hi) as
    for the tau-pencil engine; the label status and tv_bound are the CERTIFICATE's (statistical labels: lo = hi =
    vol(B ∩ cell), status certified_stat, tv_bound = eps; never re-derived from lo/hi)."""
    import hashlib, json
    from rind3d_posterior.fastlabel import label_window_fast, window_from_100k, DEFAULTS, VERSION
    cfg = dict(kw.get("fastlabel") or {})
    if cfg.get("delta") is None and not cfg.get("n_tasks"):
        raise ValueError("fastlabel: n_tasks (planned label count of the campaign) must be set")
    c0 = time.process_time(); t0 = time.perf_counter()
    lab = label_window_fast(window_from_100k(scene_path(root, slot), w, s), **cfg)
    cpu = time.process_time() - c0; wall = time.perf_counter() - t0
    m = lab["meta"]; eff = dict(DEFAULTS); eff.update(cfg)
    u, inv = np.unique(np.asarray(lab["cell"], np.int64), return_inverse=True)
    Lc = np.bincount(inv, weights=np.asarray(lab["lo"], np.float64), minlength=len(u))
    Uc = np.bincount(inv, weights=np.asarray(lab["hi"], np.float64), minlength=len(u))
    arrays = dict(dcell=np.diff(u, prepend=0).astype(np.int32), lo=Lc, hi=Uc)
    meta = dict(key=key, slot=int(slot), window=int(w), source=int(s), L=int(L), informative=bool(inf),
                tv_bound=float(lab["tv_bound"]), status=lab["status"], engine="fastlabel", fallback=None,
                engine_version=VERSION, config_sha256=hashlib.sha256(json.dumps(eff, sort_keys=True, default=str).encode()).hexdigest(),
                path=m["path"], certificate=m.get("certificate"), cpu=cpu, wall=wall, n_records=int(len(lab["cell"])),
                n_cells=int(len(u)), vol_lo=float(Lc.sum()), vol_hi=float(Uc.sum()),
                rounds=[{k: x.get(k) for k in ("round", "items", "records", "tested", "K")} for x in m.get("rounds", [])],
                verifier={k: v for k, v in (m.get("verifier") or {}).items() if not k.endswith("_s")}, pid=os.getpid())
    if lab["status"] != "certified_stat":
        from rind3d_posterior.rpl.bound import tv_bound
        meta["tv_bound_cells"] = float(tv_bound(Lc, Uc)) if Lc.sum() > 0 else 1.0
    return dict(key=key, arrays=arrays, meta=meta)


def label_task(t, engine, kw):
    """Supervisor worker entry. t = (root, slot, window, source, L, informative). Returns dict(key, arrays, meta)
    or dict(key, error). GPU out-of-memory falls back to the exact CPU engine, recorded in meta.
    engine "fastlabel": FastLabel v3.4 (needs CUDA in the worker; never mixed with the CuPy engine in one process)."""
    root, slot, w, s, L, inf = t
    key = task_key(slot, w, s)
    try:
        if engine == "fastlabel":
            return _fastlabel_task(root, slot, w, s, L, inf, key, kw)
        S = _CACHE.get(slot)
        if S is None:
            _CACHE.clear(); S = _CACHE[slot] = SceneFile(scene_path(root, slot))
        from rind3d_posterior import label_window, ENGINE_VERSION
        cls = S.cls(w, s)
        tier = int(kw.get("tier", 1))
        a = dict(kw.get("tier%d" % tier, kw.get("all", {})))
        fallback = None
        c0 = time.process_time(); t0 = time.perf_counter()
        if engine == "gpu":
            try:
                r = label_window(S.P, S.K, cls, S.origin(w), int(S.meta["resolution"]), engine="gpu",
                                 gpu_min=1024, gpu_prefix=64, **a, **kw.get("gpu", {}))
            except Exception as e:                 # any GPU-side failure: the exact CPU reference engine instead
                if "OutOfMemory" in type(e).__name__ or "out of memory" in repr(e).lower():
                    fallback = "gpu_oom_cpu"
                elif any(x in (type(e).__module__ or "") + repr(e) for x in ("cupy", "CUDA", "cuda", "nvrtc", "NVRTC")):
                    fallback = "gpu_error_cpu: " + repr(e)[:160]
                else:
                    raise
                r = label_window(S.P, S.K, cls, S.origin(w), int(S.meta["resolution"]), engine="cpu", **a)
        else:
            r = label_window(S.P, S.K, cls, S.origin(w), int(S.meta["resolution"]), engine="cpu", **a)
        cpu = time.process_time() - c0; wall = time.perf_counter() - t0
        tv = float(r["tv_bound"])
        status = STATUS_CERT if tv <= 0.05 else (STATUS_LOOSE if tv <= 0.10 else STATUS_UNCERT)
        meta = dict(key=key, slot=int(slot), window=int(w), source=int(s), L=int(L), informative=bool(inf),
                    tv_bound=tv, status=status, engine=engine, fallback=fallback, engine_version=ENGINE_VERSION,
                    tier=tier, cpu=cpu, wall=wall, n_records=int(len(r["cell"])),
                    vol_lo=float(np.sum(r["lo"])), vol_hi=float(np.sum(r["hi"])),
                    entry_evals=int(r["stats"]["entry_evals"]), nodes=int(r["stats"]["nodes"]),
                    engine_stats={kk: int(r["stats"][kk]) for kk in ("overflow", "budget_undet", "max_entries", "cells_out",
                                                                      "passes", "record_reruns") if kk in r["stats"]},
                    params={kk: v for kk, v in a.items() if isinstance(v, (int, float, str))}, pid=os.getpid(),
                    gpu=(dict((kk, v) for kk, v in r["gpu"].items() if kk != "tt") if r.get("gpu") else None))
        arrays, n_cells, tv_cells = cell_arrays(r["cell"], r["lo"], r["hi"])
        meta.update(n_cells=n_cells, tv_bound_cells=tv_cells)
        return dict(key=key, arrays=arrays, meta=meta)
    except Exception as e:
        import traceback
        return dict(key=key, error=repr(e)[:400], trace=traceback.format_exc()[-1500:])
