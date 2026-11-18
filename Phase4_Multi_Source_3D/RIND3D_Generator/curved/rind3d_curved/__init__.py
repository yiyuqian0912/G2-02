"""RIND3D curved occluders, research extension (spheres first; docs/CURVED_OCCLUDERS.md).

Exact closed balls B(c, r) = {x : |x - c|^2 <= r^2} with binary64 centre and radius (exact dyadic rationals), composed
with the certified convex-polyhedral engine by union: a receiver is SOLID iff it is in a polytope or a ball, and is
response-active iff the closed segment to the driver source misses every polytope and every ball. Semantics are the
generator's closed-solid closed-segment policy; tangency counts as a hit.

Not a mesh approximation: every decision is exact, either from directed-rounding interval arithmetic (CPU numpy or
CUDA) that decides the sign, or from exact rational arithmetic where the interval is inconclusive."""
from .spheres import (ball_contains_exact, segment_hits_ball_exact, clearance_exact, BallSet)  # noqa: F401
