"""Local-first certified labelling pathway (research prototype; LOCAL_INVERSE_GEOMETRY.md §6).

label_window_localfirst(task, select="local"|"random", **cfg) for one (scene, window, source channel):
1. Local (X only): direct interface edges, planar components, normal conditioning (`local`).
2. Outer bound: B = F(X') by the tau-pencil phase-2 engine on a receiver subset X' (B ⊇ F for ANY subset: every
   dropped receiver only removes a constraint). select="local" takes X' from the planar components; "random" is
   FastLabel v3.4's choice (256 random interface receivers + 128 interior FREE receivers), for a paired comparison.
3. Certificate (deterministic): the exact tau-pencil phase 2 on ALL receivers, restricted to B by the outer-bound
   filter (the same computation as FastLabel's exact fallback). Its tv_bound is the tau-pencil certificate (sound
   under the engine's float-margin predicate assumption A1, as for the reference engine). No statistics.
Exit: single-cell support (F ⊆ B ⊆ one cell, F ≠ ∅) gives TV 0, as in FastLabel.

The world geometry enters only in steps 2-3, through the same engine and predicates as the tau-pencil reference.
"""
import time
import tracemalloc

import numpy as np

from ..fastlabel import engine as T
from ..fastlabel.filters import build_filter2, items_from_vox2
from ..fastlabel.window import window_classes, LazyWindow2
from .local import interface_edges, plane_components, conditioning, select_receivers

DEFAULTS = dict(nbd=256, nin=128, s1_d1=4, s1_nb=512, s1_nc=64, s1_learn=32, s1_tests=640, seed=20261118)


def _b_stats(res):
    cells, L, U = np.unique(res["cell"]), None, None
    u, inv = np.unique(res["cell"], return_inverse=True)
    hi = np.bincount(inv, weights=res["hi"]) if len(u) else np.zeros(0)
    return dict(records=int(len(res["cell"])), cells=int(len(u)), volume=float(hi.sum()), undet=int(res["stats"]["undet"]))


def label_window_localfirst(task, select="local", trace_memory=False, **kw):
    cfg = dict(DEFAULTS); cfg.update(kw)
    t0 = time.perf_counter(); c0 = time.process_time()
    if trace_memory:
        tracemalloc.start()
    geom = T.Geom(np.asarray(task.P, float), np.asarray(task.K))
    cls = window_classes(task); win = LazyWindow2(cls, np.asarray(task.origin, np.int64), int(task.N))
    tt = {}
    t = time.perf_counter()
    lit, dark = interface_edges(cls)
    comps = plane_components(lit, dark, seed=cfg["seed"])
    cond = conditioning(comps)
    tt["local"] = time.perf_counter() - t
    t = time.perf_counter()
    roots, _ = T.stage1(geom, win, d1=cfg["s1_d1"], n_boundary=cfg["s1_nb"], n_class=cfg["s1_nc"], n_learn=cfg["s1_learn"],
                        max_tests=cfg["s1_tests"])
    tt["stage1"] = time.perf_counter() - t
    t = time.perf_counter()
    if select == "local":
        vox, _ = select_receivers(cls, cfg["nbd"], cfg["nin"], seed=cfg["seed"], comps=comps)
        xc, xk = items_from_vox2(win, vox)
    elif select == "random":
        sc, w, s = (int(v) for v in task.key)
        from ..fastlabel.pipeline import label_rng
        seed_items, _, _ = label_rng(20261112, sc, w, s)
        xc, xk = T.point_items(win, cfg["nbd"], cfg["nin"], seed_items)
    else:
        raise ValueError(select)
    res = T.stage2(geom, win, xc, xk, roots)
    tt["outer"] = time.perf_counter() - t
    B = _b_stats(res)
    out = dict(select=select, local=dict(n_edges=int(len(lit)), planes=[dict(n=c["n"], extent=c["extent"], major=c["major"],
                                                                           rms=c["rms"], normal=c["normal"].tolist()) for c in comps],
                                         conditioning=cond),
               items=int(len(xc)), B=B)
    cells = np.unique(res["cell"])
    if len(cells) == 1 and res["stats"]["undet"] == 0 and len(res["cell"]) > 0:
        lab = dict(cell=cells.astype(np.int32), lo=np.array([1.0]), hi=np.array([1.0]), kind=np.zeros(1, np.int8),
                   tv_bound=0.0, status="certified")
        out["path"] = "single_cell_support"
    else:
        t = time.perf_counter()
        codes, kls = win.top_items()
        final = T.stage2(geom, win, codes, kls, roots, bf=build_filter2(res))
        tt["exact_on_B"] = time.perf_counter() - t
        tv = float(final["tv_bound"])
        lab = dict(cell=np.asarray(final["cell"], np.int32), lo=final["lo"], hi=final["hi"], kind=np.asarray(final["kind"], np.int8),
                   tv_bound=tv, status="certified" if tv <= 0.05 else ("certified_loose" if tv <= 0.10 else "uncertified"))
        out["path"] = "exact_on_B"
        out["exact_records"] = int(len(final["cell"]))
    out["times"] = {k: round(v, 4) for k, v in tt.items()}
    out["wall"] = time.perf_counter() - t0; out["cpu"] = time.process_time() - c0
    if trace_memory:
        out["peak_py_mem_mb"] = tracemalloc.get_traced_memory()[1] / 2 ** 20; tracemalloc.stop()
    out["certificate"] = dict(kind="deterministic" if out["path"] == "exact_on_B" else "support", tv_bound=lab["tv_bound"],
                              engine="tau-pencil phase 2 on B (engine_p2x, outer-bound filter)")
    lab["meta"] = out
    return lab
