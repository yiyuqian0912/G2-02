"""Stage B1b: grazing rays against directly triangulated CSG meshes, judged by canonical certified truth.

For one fresh scene per family (benchmark seed): the true source, silhouette edges of the float32 component
meshes as seen from it, and receivers p = s + lam (q + g u - s) displaced by a signed gap g = +-10^U(-12,-5)
from a point q of the edge (u points away from the component: g > 0 grazes outside, g < 0 passes inside).
Every ray is judged by the certified canonical predicate A (binary64 source, exact rational geometry).
Reported per |gap| decade: Embree and Warp disagreement with A, common-mode disagreement (both wrong), the
translation-probe flag, and the hybrid result (A on flagged rays, Embree elsewhere).
Mesh-vs-canonical disagreement at tiny gaps is expected: the mesh is a float32 perturbation of the canonical
solid (vertex rounding ~1e-8) and the Embree source is float32; this quantifies it.
"""
import json
import time

import numpy as np

from rind3d.rendering.backends import BackendFactory

from . import scenes as S
from .bench_backends import SEED
from .env import POLICY, ROOT, mesh_engines
from .triangulate import triangulate

OUT = ROOT/"outputs"/"benchmarks"


def silhouettes(V, F, s):
    tri = V.astype(np.float64)[F]
    nrm = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    facing = np.einsum("ij,ij->i", nrm, s - tri.mean(axis=1)) > 0
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    own = np.tile(np.arange(len(F)), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks, os_ = key[order], own[order]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    i = np.flatnonzero(same)
    sil = i[facing[os_[i]] != facing[os_[i+1]]]
    return ks, os_, sil, tri


def main(per_scene=4000):
    engines, _ = mesh_engines()
    from gaussian_mesh_dataset.labels import perpendiculars
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 7]))
    rows = []
    with BackendFactory(geometry_cache=2) as factory:
        for i, family in enumerate(S.FAMILIES):
            bundle, _ = S.geometry(SEED, 200+i, family)
            ctx = factory.context(bundle, POLICY)
            truth, _, _ = S.sample_sources(ctx, SEED, 200+i, 1)
            s = truth[0]
            mesh = triangulate(bundle)
            V32, F = mesh["vertices32"], mesh["faces"]
            em, wa = engines.Embree(V32, F), engines.Warp(V32, F)
            ks, os_, sil, tri = silhouettes(V32, F, s)
            if not len(sil):
                continue
            P, G = [], []
            for j in rng.choice(sil, per_scene, replace=True):
                a, b = V32[ks[j, 0]].astype(np.float64), V32[ks[j, 1]].astype(np.float64)
                q = a + rng.uniform(.2, .8)*(b-a)
                side = np.cross(q - s, b-a)
                side /= np.linalg.norm(side)
                u = side if np.dot(side, tri[os_[j]].mean(axis=0)-q) < 0 else -side
                g = (1 if rng.random() < .5 else -1)*10.0**rng.uniform(-12, -5)
                p = s + rng.uniform(1.05, 1.8)*(q + g*u - s)
                if np.all((p > 1e-3) & (p < 1-1e-3)) and not ctx.contains(p[None])[0]:
                    P.append(p)
                    G.append(g)
            P, G = np.array(P), np.array(G)
            m = ctx.bind(s)
            A = m.native.points(np.ascontiguousarray(P), False)[0].astype(np.int8)      # 1 = lit
            s32, P32 = s.astype(np.float32), P.astype(np.float32)
            oe, ow = em.occluded(s32, P32), wa.occluded(s32, P32)
            t1, t2 = perpendiculars(P32 - s32)
            unstable = np.zeros(len(P32), bool)
            for v in (t1, -t1, t2, -t2):
                unstable |= em.segments(s32 + np.float32(2.0**-20)*v, P32 + np.float32(2.0**-20)*v) != oe
            flag = unstable | (oe != ow)
            hybrid = np.where(flag, A == 1, ~oe)
            litA = A == 1
            for lo, hi in ((-12, -9), (-9, -7), (-7, -6), (-6, -5)):
                sel = (np.log10(np.abs(G)) >= lo) & (np.log10(np.abs(G)) < hi)
                if not sel.any():
                    continue
                rows.append({"family": family, "gap_decades": [lo, hi], "rays": int(sel.sum()),
                             "embree_wrong": int(((~oe) != litA)[sel].sum()), "warp_wrong": int(((~ow) != litA)[sel].sum()),
                             "common_mode_wrong": int((((~oe) != litA) & ((~ow) != litA))[sel].sum()),
                             "flagged": int(flag[sel].sum()), "hybrid_wrong": int((hybrid != litA)[sel].sum()),
                             "unknown_A": int((A[sel] < 0).sum())})
            print(family, len(P), "embree wrong", int(((~oe) != litA).sum()), "hybrid wrong", int((hybrid != litA).sum()), flush=True)
    tot = {k: sum(r[k] for r in rows) for k in ("rays", "embree_wrong", "warp_wrong", "common_mode_wrong", "flagged", "hybrid_wrong", "unknown_A")}
    by_gap = {}
    for r in rows:
        k = f"1e{r['gap_decades'][0]}..1e{r['gap_decades'][1]}"
        d = by_gap.setdefault(k, {x: 0 for x in tot})
        for x in tot:
            d[x] += r[x]
    (OUT/"grazing-csg-mesh.json").write_text(json.dumps({"seed": SEED, "rows": rows, "totals": tot, "by_gap": by_gap}, indent=1))
    print(json.dumps({"totals": tot, "by_gap": by_gap}, indent=1))


if __name__ == "__main__":
    main()
