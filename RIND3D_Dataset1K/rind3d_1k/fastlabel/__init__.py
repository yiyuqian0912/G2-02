"""FastLabel v3.4: certified fast posterior labels for the RIND3D 1K (production package).

Research record, benchmarks and audit: RIND3D_Posterior/experiments/fastlabel/ (REPORT.md, CERTIFICATE_AUDIT.md,
GPUVERIFY.md). Entry point: label_task(scene_dir, window, source, **cfg). Needs CUDA (cuda_certified backend)."""
from .pipeline import VERSION, DEFAULTS, label_task, set_gpu_lock, design2, STAT_KIND

__all__ = ["VERSION", "DEFAULTS", "label_task", "set_gpu_lock", "design2", "STAT_KIND"]
