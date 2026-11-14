"""rind3d_posterior: certified single-source global posterior labels for RIND3D windows (tau-pencil engine).

Target: Uniform(F_W) over the source-excluded legal domain, F_W = {s in D(G) minus the closed window box : s
reproduces every FREE receiver of W} (closed-solid, closed-segment visibility), reported as per-128^3-cell volume
intervals [lo, hi] with a deterministic TV bound (`tv_bound`).  Engines: 'cpu' (reference, Phase-1 b4 + counters)
and 'gpu' (exact GPU evaluation pass; bit-identical records on all tested windows).  See docs/METHOD.md."""
from .api import label_window, cell_bounds, point_estimate, ENGINE_VERSION  # noqa: F401
