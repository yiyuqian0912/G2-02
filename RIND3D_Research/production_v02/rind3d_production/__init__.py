"""RIND3D production-readiness package (research-to-production transition, 2026-11).

Layers (kept separate on purpose):
1. geometry generators      -> rind3d.lite_generators (certified convex-union families, unchanged)
2. canonical geometry       -> rind CanonicalBundle (binary64, exact interpretation; the physical truth)
3. query acceleration       -> certified native/CUDA convex predicates (exact), optional triangulated
                               mesh + BVH (an approximation of the canonical geometry, never the truth)

The certified physics code of the producer checkout (G2-02-3d/RIND3D_Generator) is used read-only;
this package adds production-format storage, profiles, observation policy, jobs, validation and loaders.
"""
