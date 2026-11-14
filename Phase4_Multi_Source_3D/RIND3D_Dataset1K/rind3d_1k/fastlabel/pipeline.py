"""1K wrapper of rind3d_posterior.fastlabel.pipeline: label_task(scene_dir, w, s) and the SceneObs-based scene_cache.
Everything else (DEFAULTS, VERSION, design2, kstar, label_rng, process_factory, GPU_LOCK, ...) is the core module."""
import time

from rind3d_posterior.fastlabel import pipeline as _core
from rind3d_posterior.fastlabel.pipeline import (VERSION, N_DATASET, STAT_KIND, DEFAULTS, set_gpu_lock,  # noqa: F401
                                                 process_factory, scene_context, task_context, kstar, design2, cp_lower,
                                                 label_rng, label_window_fast)
from rind3d_posterior.fastlabel.inputs import window_from_1k, POLICY  # noqa: F401


def __getattr__(name):                   # live core state (GPU_LOCK is set by set_gpu_lock in the worker initialiser)
    return getattr(_core, name)


def scene_cache(scene_dir):
    """dict(S=SceneObs, geom, bundle, ctx) of a 1K scene directory (the per-scene objects shared with label_task)."""
    from rind3d_1k.labels import SceneObs
    from rind3d.geometry import load_geometry
    from pathlib import Path
    key = str(scene_dir)
    v = _core._SCENES.get(key)
    if v is None:
        S = SceneObs(scene_dir)
        v = scene_context(key, S.P, S.K, load_geometry(Path(scene_dir) / "privileged" / "geometry.npz"), POLICY)
    if "S" not in v:
        v["S"] = SceneObs(scene_dir)
    return v


def label_task(scene_dir, w, s, **kw):
    """Label one (1K scene, window, source channel). Returns dict(cell, lo, hi, kind, tv_bound, status, meta)."""
    t0 = time.perf_counter(); c0 = time.process_time()
    task = window_from_1k(scene_dir, w, s)
    ta = time.perf_counter() - t0; ca = time.process_time() - c0
    lab = label_window_fast(task, **kw)
    lab["meta"]["wall"] += ta; lab["meta"]["cpu"] += ca      # wall / cpu include reading the scene, as before
    return lab
