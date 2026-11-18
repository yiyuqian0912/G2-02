"""Curved-occluder scenes for the 100K local evaluation (research; rind3d_curved, docs/CURVED_OCCLUDERS.md).

`CurvedLocalScene` has the LocalScene interface (probe, informative, interface, render via `_packed`) for a solid
that is the union of the certified convex polytopes (optional) and exact closed balls:
    SOLID(x)  = SOLID_poly(x) or x in some ball;
    active(x) = active_poly(x) and the closed segment [s, x] meets no ball (and x not SOLID).
Polytope bits come from the unchanged certified CUDA/CPU backends; ball bits from rind3d_curved (directed-rounding
CUDA intervals, exact rational fallback). The all-active hull certificate is disabled (it ignores balls)."""
import time

import numpy as np

from .local_eval import LocalScene, TILE


class CurvedLocalScene(LocalScene):
    def __init__(self, factory, bundle, balls, sources, policy, N=1024, backend="cuda_certified", batch=32):
        from rind3d_curved.spheres import BallSet
        from rind3d_curved.cuda_balls import BallTiles
        self.N = int(N); self.k = len(sources); self.batch = int(batch)
        self.sources = np.asarray(sources, float)
        self.src = [factory.source(bundle, s, backend, policy) for s in sources] if bundle is not None else []
        self.cpu = [getattr(s, "cpu", s) for s in self.src]
        self.balls = BallSet(balls[:, :3], balls[:, 3]) if len(balls) else None
        self.ball_array = np.asarray(balls, float)
        self.btiles = BallTiles(balls) if len(balls) else None
        self.stats = dict(probes=0, probe_s=0.0, tiles=0, render_s=0.0, rendered_channels=0, decided_by_probes=0, decided_by_render=0,
                          contains_source=0, ball_s=0.0)
        self._solid_cache = {}
        self.hull = None

    def enable_hull(self, context):
        """All-active certificate for the composite solid: the polytope certificate (HullCert, when polytopes exist)
        and, for every ball, an exact separating plane between the ball and conv({s} + receiver box)."""
        from .local_eval import HullCert
        self.hull = _CurvedHull(HullCert(context, self.N) if context is not None else None, self.ball_array, self.N)

    def probe(self, j, vox):
        t = time.perf_counter()
        pts = (np.asarray(vox, np.float64) + 0.5) / self.N
        if self.src:
            yy, mm, _ = self.cpu[j].native.points(np.ascontiguousarray(pts, dtype="<f8"), False)
            if np.any(yy < 0) or np.any(mm < 0):
                raise RuntimeError("exact point query did not certify")
            lit = yy.astype(bool); sol = mm.astype(bool)
        else:
            lit = np.ones(len(pts), bool); sol = np.zeros(len(pts), bool)
        if self.balls is not None:
            sb = self.balls.contains(pts); bb = self.balls.blocked(self.sources[j], pts)
            sol = sol | sb; lit = lit & ~bb & ~sol
        self.stats["probes"] += len(vox); self.stats["probe_s"] += time.perf_counter() - t
        return lit, sol

    def _packed(self, j, origins, side=TILE):
        if side != TILE:
            raise ValueError("curved scenes use 64^3 tiles")
        poly = super()._packed(j, origins, side) if self.src else None
        out = []
        t = time.perf_counter()
        bt = self.btiles.tiles(self.sources[j], origins, TILE, self.N) if self.btiles is not None else None
        for i, o in enumerate(np.asarray(origins)):
            if poly is not None:
                pl, ps = poly[i]
                lit = np.unpackbits(np.frombuffer(pl, np.uint8), bitorder="little").astype(bool).reshape(TILE, TILE, TILE)
                sol = np.unpackbits(np.frombuffer(ps, np.uint8), bitorder="little").astype(bool).reshape(TILE, TILE, TILE)
            else:
                lit = np.ones((TILE,) * 3, bool); sol = np.zeros((TILE,) * 3, bool)
            if bt is not None:
                blk, sb = bt[i]
                sol = sol | sb; lit = lit & ~blk & ~sol
            out.append((np.packbits(lit.ravel(), bitorder="little").tobytes(), np.packbits(sol.ravel(), bitorder="little").tobytes()))
        self.stats["ball_s"] += time.perf_counter() - t
        return out

    def close(self):
        for s in self.src:
            s.close()


class _CurvedHull:
    """Exact all-active certificate for polytopes + balls (see CurvedLocalScene.enable_hull)."""

    def __init__(self, poly_hull, balls, N):
        self.poly = poly_hull; self.balls = np.asarray(balls, float); self.N = int(N); self.calls = 0; self.certified = 0

    def all_lit(self, s, o, L):
        from rind3d_curved.spheres import ball_misses_hull_exact
        self.calls += 1
        if self.poly is not None and not self.poly.all_lit(s, o, L):
            return False
        lo = (np.asarray(o, float) + 0.5) / self.N; hi = (np.asarray(o, float) + L - 0.5) / self.N
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        pts = np.r_[np.asarray(s, float)[None], corners]
        if not all(ball_misses_hull_exact(b[:3], b[3], pts) for b in self.balls):
            return False
        self.certified += 1
        return True
