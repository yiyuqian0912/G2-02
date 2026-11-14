"""1K entry points of FastLabel v3.4 (compatibility layer).

The implementation lives in rind3d_posterior.fastlabel (scene-format independent core). This package keeps the 1K API
unchanged: label_task(scene_dir, window, source, **cfg) = label_window_fast(window_from_1k(scene_dir, window, source)),
and the submodules pipeline / gpuverify / window / engine / filters / sampler / fastplanes re-export the core with the
1K-specific signatures (SceneObs-based scene_cache, fast_cls3, WindowVerifier(fac, ctx, S, w, s, cls))."""
from .pipeline import VERSION, DEFAULTS, label_task, set_gpu_lock, design2, STAT_KIND

__all__ = ["VERSION", "DEFAULTS", "label_task", "set_gpu_lock", "design2", "STAT_KIND"]
