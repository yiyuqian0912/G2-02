"""TeacherPhysics: the stable dataset-side physics interface for privileged Teacher research.

It exposes the canonical-geometry oracle, not a model. Every answer is certified (exact closed-segment
semantics) unless the caller explicitly asks for the screened hybrid mode, which is labelled as such.
  ref()                                      geometry/contract/oracle identities and the legal-source domain
  legal(sources)                             certified source legality (open world, FREE, clearance > 2^-26)
  visibility(sources, receivers)             int8 [S, N] lit (1) / shadow (0) for FREE receivers, -1 for SOLID
  response(sources, strengths, receivers)    combined additive response [S-set] and per-source channels
  window(sources, origin, L, backend)        dense per-source lit + SOLID for an L^3 window (certified raster)
  free_mask(origin, L)                       stored SOLID mask -> FREE receivers of a window
  receiver_bank(origin, L, design, k, seed)  sampled FREE receivers with exact inclusion probabilities/weights
Receivers are world coordinates (float64); window receivers are voxel centres (index + 1/2)/1024.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rind3d.geometry import load_geometry
from rind3d.lite_acceptance import verify_geometry_proof
from rind3d.pack import RINDPack
from rind3d.rendering.context import GeometryContext

from . import scenes as S
from .env import POLICY


def window_centres(origin, L, n=1024):
    a = np.arange(L)+.5
    z, y, x = np.meshgrid(a, a, a, indexing="ij")
    return np.column_stack([(origin[0]+x.ravel())/n, (origin[1]+y.ravel())/n, (origin[2]+z.ravel())/n])


class TeacherPhysics:
    def __init__(self, scene_dir, *, threads=16, factory=None):
        self.dir = Path(scene_dir)
        self.bundle = load_geometry(self.dir/"public"/"geometry.npz")
        self.context = GeometryContext(self.bundle, POLICY, source_cache_size=256)
        verify_geometry_proof(self.bundle, self.context.geometry_proof())
        self.threads = int(threads)
        self.factory = factory

    def ref(self):
        return {"geometry_id": self.context.geometry_id, "policy_sha256": self.context.policy_hash,
                "oracle": "certified native convex predicates (cpu_native_certified); window raster cpu or cuda_certified",
                "legal_source_domain": "open world (0,1)^3, outside the closed solid, clearance > 2^-26 (certified)",
                "source_prior": "dataset sources: independent uniform proposals in (0,1)^3 conditioned on legality"}

    def legal(self, sources):
        p = np.asarray(sources, np.float64).reshape(-1, 3)
        return np.array([S.legal_source(self.context, x) is not None for x in p])

    def visibility(self, sources, receivers):
        src = np.asarray(sources, np.float64).reshape(-1, 3)
        rec = np.ascontiguousarray(np.asarray(receivers, np.float64).reshape(-1, 3))
        if not self.legal(src).all():
            raise ValueError("illegal source in query (use legal() first)")
        solid = self.context.contains(rec)

        def one(s):
            y = self.context.bind(s).native.points(rec, False)[0].astype(np.int8)
            if np.any(y < 0):
                raise RuntimeError("UNKNOWN visibility cannot be a label")
            y[solid] = -1
            return y
        with ThreadPoolExecutor(self.threads) as ex:
            return np.stack(list(ex.map(one, src)))

    def response(self, sources, strengths, receivers):
        vis = self.visibility(sources, receivers)
        lit = vis == 1
        total = np.zeros(vis.shape[1])
        for k, a in enumerate(np.asarray(strengths, np.float64)):
            total = total + np.where(lit[k], float(a), 0.0)
        return total, vis

    def window(self, sources, origin, L, backend="cpu_native_certified"):
        out, solid = [], None
        for s in np.asarray(sources, np.float64).reshape(-1, 3):
            if backend == "cuda_certified":
                from .bench_materialization import render_window
                r = self.factory.source(self.bundle, s, "cuda_certified", POLICY)
                y = render_window(r, tuple(origin), L)
                r.close()
                m = None
            else:
                m_ = self.context.bind(s)
                y = np.zeros((L, L, L), bool)
                m = np.zeros((L, L, L), bool)
                for oz in range(0, L, 64):
                    for oy in range(0, L, 64):
                        for ox in range(0, L, 64):
                            b, mb, _ = m_.render([origin[0]+ox, origin[1]+oy, origin[2]+oz], min(64, L), 1024)
                            y[oz:oz+64, oy:oy+64, ox:ox+64] = b
                            m[oz:oz+64, oy:oy+64, ox:ox+64] = mb
            out.append(y)
            solid = m if m is not None else solid
        return np.stack(out), solid

    def free_mask(self, origin, L):
        with RINDPack(self.dir/"public"/"solid.rindb") as m:
            return ~m.read_window(*origin, L)

    def receiver_bank(self, origin, L, design="uniform", k=256, seed=0, response=None):
        """Sampled FREE receivers with exact first-order inclusion probabilities.

        uniform:    simple random sample without replacement of k FREE receivers; pi = k/FREE; weight = FREE/k
                    (Horvitz-Thompson: sum_i w_i c_i is unbiased for the full-window FREE total of any cost c).
        stratified: (Teacher-only: depends on the observed response) proportional allocation over response
                    classes (distinct values), at least one receiver per class; pi_i = k_h/N_h in class h.
        """
        free = np.flatnonzero(self.free_mask(origin, L).ravel())
        rng = np.random.default_rng(seed)
        if design == "uniform":
            take = np.sort(rng.choice(free, min(k, len(free)), replace=False))
            pi = np.full(len(take), len(take)/len(free))
        elif design == "stratified":
            vals = np.asarray(response).ravel()[free]
            classes = np.unique(vals)
            take, pi = [], []
            for c in classes:
                members = free[vals == c]
                kh = max(1, int(round(k*len(members)/len(free))))
                kh = min(kh, len(members))
                sel = rng.choice(members, kh, replace=False)
                take.append(sel)
                pi.append(np.full(kh, kh/len(members)))
            order = np.argsort(np.concatenate(take))
            take, pi = np.concatenate(take)[order], np.concatenate(pi)[order]
        else:
            raise ValueError(design)
        return {"local_indices": take, "world_xyz": window_centres(origin, L)[take], "inclusion_probability": pi,
                "weight": 1/pi, "design": design, "k": int(len(take)), "free": int(len(free)), "seed": int(seed)}
