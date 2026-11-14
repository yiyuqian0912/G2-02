"""FastLabel v3.4: certified fast posterior labels for RIND3D windows (core package, scene-format independent).

Public API:
* `label_window_fast(task, **cfg)`: label one task, an `inputs.FastLabelWindow`; returns dict(cell, lo, hi, kind,
  tv_bound, status, meta) with status "certified" (deterministic TV bound <= 0.05) or "certified_stat" (TV <= eps with
  the stated statistical confidence; the certificate is in meta);
* adapters `window_from_1k(scene_dir, w, s)` and `window_from_100k(container, w, s)`;
* `DEFAULTS` (production configuration; its sha256 is recorded per label), `VERSION`, `STAT_KIND`, `design2`,
  `set_gpu_lock` (cross-process GPU semaphore initialiser);
* `gpuverify.WindowVerifier`: the exact GPU verifier of candidate sources (validated by Gate A, `gate_a.py`).

Research record, benchmarks and audit: RIND3D_Posterior/experiments/fastlabel/ (REPORT.md, CERTIFICATE_AUDIT.md,
GPUVERIFY.md). Needs CUDA (the rind3d cuda_certified backend) and, for the adapters, zstandard.
The 1K entry points (rind3d_1k.fastlabel, label_task(scene_dir, w, s)) are compatibility wrappers of this package."""
from .pipeline import VERSION, DEFAULTS, STAT_KIND, label_window_fast, set_gpu_lock, design2
from .inputs import FastLabelWindow, POLICY, legal_source, window_from_1k, window_from_100k

__all__ = ["VERSION", "DEFAULTS", "STAT_KIND", "label_window_fast", "set_gpu_lock", "design2", "FastLabelWindow", "POLICY",
           "legal_source", "window_from_1k", "window_from_100k"]
