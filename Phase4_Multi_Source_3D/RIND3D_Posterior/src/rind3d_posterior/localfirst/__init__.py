"""Local-first research labeller (docs/LOCAL_INVERSE_GEOMETRY.md, Phase4_Multi_Source_3D/docs/).

* `local`: X-only analysis of the direct response interface: interface edges, planar components, the conditioning of
  their normals and the *heuristic* local band (never used for pruning: local-only pruning is not certifiable,
  LOCAL_INVERSE_GEOMETRY.md Theorem 3).
* `pipeline.label_window_localfirst`: certified pathway. Receivers chosen from the local interface geometry define an
  outer bound B = F(X') (sound for any receiver subset), then the exact tau-pencil phase 2 restricted to B gives the
  same deterministic certificate as the tau-pencil reference.
"""
from .local import interface_edges, plane_components, conditioning, local_band_cells, select_receivers  # noqa: F401
