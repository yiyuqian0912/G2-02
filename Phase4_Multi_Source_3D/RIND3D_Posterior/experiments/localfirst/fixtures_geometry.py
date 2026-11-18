"""Small exact configurations (docs/LOCALFIRST_E123.md §4.1): the dimension and extent of the exact feasible set F and of
the certified interface region R (Proposition A', all free interface edges) around the true source.

usage: fixtures_geometry.py > out.json
Fixtures (tests/test_certregion.py): a rotated box, a 7-degree knife-edge wedge, two near-aligned boxes, and a source
1e-9 from a facet plane's extension. For each, points of F (exact predicate, multi-scale around the true source plus
bisection to the boundary) and points of R (same proposals, membership in R) give principal extents: the square
roots of the covariance eigenvalues of the accepted points within radius 0.05 of the source. A plane-like set has one
small extent, a line-like set two, a point-like set three."""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "tests")); sys.path.insert(0, str(HERE.parents[3] / "RIND3D_Dataset100K"))


def main():
    from rind3d.rendering.backends import BackendFactory
    from rind3d_1k.physical import geometry_rows
    import test_certregion as TC
    from rind3d_posterior.localfirst.certregion import world_edges, admissible_edges, in_region_batch, MARGIN
    from rind3d_posterior.localfirst.local import interface_edges, edge_free_mask, plane_components, conditioning
    fac = BackendFactory(cuda_allowed=False); N = TC.N; out = {}
    for name in ("box", "knife", "two_boxes", "coplanar"):
        leaves, s, _, L = TC._scene(name); ctx = TC._ctx(fac, leaves); o = TC._place_window(ctx, s, L)
        cls, X, g = TC._window(ctx, s, o, L)
        lit, dark = interface_edges(cls); free = edge_free_mask([c.rows for c in ctx.components], o, N, lit, dark)
        comps = plane_components(lit, dark, seed=20261118); cond = conditioning(comps)
        Lp = (o + lit[free] + .5) / N; Kp = (o + dark[free] + .5) / N
        P, K = geometry_rows(ctx); EA, EB, Fr, _ = world_edges(P, K); off, idx = admissible_edges(Fr, Lp, Kp)
        rng = np.random.default_rng(0)
        Q = np.concatenate([s + rng.normal(0, sc, (1500, 3)) for sc in (1e-5, 1e-4, 1e-3, 1e-2, 3e-2)])
        Q = Q[np.all((Q > 0) & (Q < 1), 1)]
        inF = TC._feasible(ctx, cls, X, Q, o, L)
        inR = in_region_batch(Q, EA, EB, Lp, Kp, off, idx, MARGIN) & ~np.all((Q >= o / N) & (Q <= (o + L) / N), 1)

        def extents(Pts):
            Pts = Pts[np.linalg.norm(Pts - s, axis=1) < 0.05]
            if len(Pts) < 5:
                return dict(n=int(len(Pts)))
            w = np.linalg.eigvalsh(np.cov((Pts - s).T))
            return dict(n=int(len(Pts)), extents=np.sqrt(np.maximum(w, 0)).tolist())
        out[name] = dict(edges=int(len(lit)), free=int(free.sum()), components=len([c for c in comps if c["n"] >= 24]),
                         rank=cond["rank"], sigma=cond["sigma"], proposals=int(len(Q)), F=extents(Q[inF]), R=extents(Q[inR]),
                         F_in_R=bool(np.all(inR[inF])), share_R_of_proposals=float(inR.mean()), share_F_of_proposals=float(inF.mean()))
        print(name, json.dumps(out[name]), flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
