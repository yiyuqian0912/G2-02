"""TeacherPhysics v0.3: the privileged dataset-side physics interface (fixes of the v0.2 proposal).

Changes against rind3d_production.teacher (v0.2 is left unchanged):
  * works on both layouts (accepted v0.2 canary, restricted v0.1 1K) through scenes.SceneRef;
  * CUDA window raster owns its BackendFactory (lazy, created on first use, closed by close()) and returns the
    certified SOLID mask from the same raster (v0.2 needed an external factory and returned SOLID=None);
  * stratified receiver banks allocate exactly k receivers (largest-remainder proportional allocation with at
    least one per stratum; v0.2 rounding could return k +/- #strata); a census is used when k >= FREE;
  * windows without FREE receivers return an explicit empty bank / NaN mismatch with status "no_free"
    (strict=True raises NoFreeReceivers), never a silent zero error;
  * multi-source additive semantics: candidate sets are unordered; the response is sum_k a_k lit_k evaluated in
    binary64 in the order given, compared with the observation to an absolute tolerance of MA_TOL (sums of the
    same strengths in another order may differ in the last ulp);
  * clear, typed error messages (IllegalSource lists the offending indices and reasons).
Everything here is privileged (geometry, stored per-source channels, labels): never call it from Student code.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rind3d.geometry import load_geometry
from rind3d.lite_acceptance import DRIVER_RADIUS, verify_geometry_proof
from rind3d.pack import RINDPack
from rind3d.rendering.context import GeometryContext

from rind3d_production.env import POLICY
from rind3d_production.production import combined_window

MA_TOL = 1e-12


class NoFreeReceivers(ValueError):
    pass


class IllegalSource(ValueError):
    def __init__(self, bad, reasons):
        super().__init__(f"illegal candidate source(s) at indices {list(bad)}: {[reasons[i] for i in bad]}")
        self.bad, self.reasons = list(bad), reasons


def window_centres(origin, L, n=1024):
    a = np.arange(L)+.5
    z, y, x = np.meshgrid(a, a, a, indexing="ij")
    return np.column_stack([(origin[0]+x.ravel())/n, (origin[1]+y.ravel())/n, (origin[2]+z.ravel())/n])


def in_closed_box(points, origin, L, n=1024):
    p = np.asarray(points, np.float64).reshape(-1, 3)
    lo = np.asarray(origin, np.float64)/n
    return np.all((p >= lo) & (p <= lo + L/n), axis=1)


class TeacherPhysics:
    def __init__(self, scene, *, threads=16):
        self.scene = scene
        self.bundle = load_geometry(scene.geometry)
        self.context = GeometryContext(self.bundle, POLICY, source_cache_size=256)
        verify_geometry_proof(self.bundle, self.context.geometry_proof())
        if self.context.geometry_id != scene.geometry_id:
            raise ValueError("scene/geometry identity conflict")
        self.threads = int(threads)
        self._factory = None
        self._packs = None

    # ------------------------------------------------------------------ identities and legality
    def ref(self):
        return {"geometry_id": self.context.geometry_id, "policy_sha256": self.context.policy_hash,
                "release": self.scene.release, "profile": self.scene.profile, "K": self.scene.K,
                "oracle": "certified native convex predicates (cpu_native_certified); window raster cpu or cuda_certified",
                "legal_source_domain": "open world (0,1)^3, outside the closed solid, certified clearance > 2^-26",
                "task_domains": {"global": "legal domain", "src-excluded": "legal domain minus the CLOSED window box"},
                "source_prior": ("uniform on the legal domain (uniform proposals conditioned on legality)" if self.scene.K == 1 else
                                 "K i.i.d. uniform legal sources conditioned on pairwise separation >= 1/32; strengths i.i.d. U[0.1,1]")}

    def legality(self, sources, *, window=None):
        """Per candidate: (valid, reason). window=(origin, L) applies the src-excluded task domain (closed box)."""
        p = np.asarray(sources, np.float64).reshape(-1, 3)
        out = []
        inside = in_closed_box(p, *window) if window is not None else np.zeros(len(p), bool)
        for i, x in enumerate(p):
            if not np.all((x > 0) & (x < 1)):
                out.append((False, "outside_open_world"))
            elif inside[i]:
                out.append((False, "inside_window_box"))
            elif self.context.contains(x[None])[0]:
                out.append((False, "solid_or_boundary"))
            else:
                proof = self.context.clearance(x, DRIVER_RADIUS)
                out.append((True, "certified_valid") if proof["status"] == "GT" else
                           (False, "insufficient_clearance" if proof["status"] == "LE" else "unresolved_clearance"))
        return out

    def legal(self, sources, *, window=None):
        return np.array([v for v, _ in self.legality(sources, window=window)])

    def _require_legal(self, src):
        leg = self.legality(src)
        bad = [i for i, (v, _) in enumerate(leg) if not v]
        if bad:
            raise IllegalSource(bad, [r for _, r in leg])

    # ------------------------------------------------------------------ point queries
    def visibility(self, sources, receivers):
        """int8 [S, N]: 1 lit, 0 shadow (FREE receivers), -1 SOLID receiver."""
        src = np.asarray(sources, np.float64).reshape(-1, 3)
        rec = np.ascontiguousarray(np.asarray(receivers, np.float64).reshape(-1, 3))
        self._require_legal(src)
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
        """Additive response sum_k a_k lit_k (binary64, order given) and per-source visibility."""
        src = np.asarray(sources, np.float64).reshape(-1, 3)
        a = np.asarray(strengths, np.float64).reshape(-1)
        if len(a) != len(src):
            raise ValueError(f"{len(src)} sources but {len(a)} strengths")
        vis = self.visibility(src, receivers)
        total = np.zeros(vis.shape[1])
        for k in range(len(src)):
            total = total + np.where(vis[k] == 1, a[k], 0.0)
        return total, vis

    # ------------------------------------------------------------------ dense windows
    def _cuda(self):
        if self._factory is None:
            from rind3d.rendering.backends import BackendFactory
            self._factory = BackendFactory(geometry_cache=1, cuda_batch=32)
        return self._factory

    def window(self, sources, origin, L, backend="cpu_native_certified"):
        """Certified per-source lit [S,L,L,L] and SOLID [L,L,L] of an L^3 window (L multiple of 64)."""
        src = np.asarray(sources, np.float64).reshape(-1, 3)
        self._require_legal(src)
        if L % 64 or min(origin) < 0 or max(origin) + L > 1024:
            raise ValueError("window must be inside the world with L a multiple of 64")
        tiles = [(origin[0]+64*i, origin[1]+64*j, origin[2]+64*k) for k in range(L//64) for j in range(L//64) for i in range(L//64)]
        lit = np.zeros((len(src), L, L, L), bool)
        solid = None
        for si, s in enumerate(src):
            m = np.zeros((L, L, L), bool)
            if backend == "cuda_certified":
                r = self._cuda().source(self.bundle, s, "cuda_certified", POLICY)
                try:
                    for pos in range(0, len(tiles), 32):
                        chunk = tiles[pos:pos+32]
                        for (x, y, z), (resp, sol) in zip(chunk, r.packed_batch([list(c) for c in chunk], 64, 1024)):
                            sl = (slice(z-origin[2], z-origin[2]+64), slice(y-origin[1], y-origin[1]+64), slice(x-origin[0], x-origin[0]+64))
                            lit[si][sl] = np.unpackbits(np.frombuffer(resp, np.uint8), bitorder="little").reshape(64, 64, 64)
                            m[sl] = np.unpackbits(np.frombuffer(sol, np.uint8), bitorder="little").reshape(64, 64, 64)
                finally:
                    r.close()
            elif backend == "cpu_native_certified":
                b = self.context.bind(s)
                for (x, y, z) in tiles:
                    sl = (slice(z-origin[2], z-origin[2]+64), slice(y-origin[1], y-origin[1]+64), slice(x-origin[0], x-origin[0]+64))
                    lit[si][sl], m[sl], _ = b.render([x, y, z], 64, 1024)
            else:
                raise ValueError(f"unknown backend {backend!r} (cpu_native_certified | cuda_certified)")
            if solid is not None and not np.array_equal(solid, m):
                raise AssertionError("SOLID differs between sources")
            solid = m
        return lit, solid

    def combined(self, lit, strengths):
        total = np.zeros(lit.shape[1:])
        for k, a in enumerate(np.asarray(strengths, np.float64)):
            total = total + np.where(lit[k], float(a), 0.0)
        return total

    # ------------------------------------------------------------------ stored fields
    def _stored(self):
        if self._packs is None:
            self._packs = ([RINDPack(p, cache_bricks=64) for p in self.scene.responses],
                           RINDPack(self.scene.solid[0], cache_bricks=64))
        return self._packs

    def free_mask(self, origin, L):
        return ~self._stored()[1].read_window(*origin, L)

    def observation(self, origin, L):
        """The local observation reconstructed from the stored certified fields (MA: combined, local palette)."""
        packs, solid = self._stored()
        if self.scene.K == 1:
            return packs[0].read_window(*origin, L), solid.read_window(*origin, L)
        _, a = self.scene.privileged_labels()
        pal, codes = combined_window(packs, a, solid, tuple(origin), L)
        return pal[codes], solid.read_window(*origin, L)

    # ------------------------------------------------------------------ receiver banks and mismatch
    def receiver_bank(self, origin, L, design="uniform", k=256, seed=0, response=None, strict=False):
        """Sampled FREE receivers with exact first-order inclusion probabilities pi and HT weights 1/pi.

        uniform     simple random sample without replacement; pi = k/FREE.  Response independent: allowed as a
                    Student-side sparse view.
        stratified  Teacher-only (strata = distinct observed values, i.e. response dependent): exactly k receivers,
                    largest-remainder proportional allocation with >= 1 per stratum, pi_h = k_h/N_h.
                    Requires k >= number of strata.
        If k >= FREE the bank is a census (pi = 1).  No FREE receiver: empty bank, status "no_free".
        """
        free = np.flatnonzero(self.free_mask(origin, L).ravel())
        N = len(free)
        empty = {"local_indices": np.zeros(0, np.int64), "world_xyz": np.zeros((0, 3)), "inclusion_probability": np.zeros(0),
                 "weight": np.zeros(0), "stratum": np.zeros(0, np.int64), "design": design, "k": 0, "free": 0,
                 "seed": int(seed), "status": "no_free"}
        if N == 0:
            if strict:
                raise NoFreeReceivers(f"window origin={tuple(origin)} L={L} has no FREE receiver: mismatch is undefined")
            return empty
        rng = np.random.default_rng(seed)
        strata = np.zeros(N, np.int64)
        if k >= N:
            take, pi, status = free, np.ones(N), "census"
            if design == "stratified":
                vals = np.asarray(response).ravel()[free]
                strata = np.unique(vals, return_inverse=True)[1]
        elif design == "uniform":
            take = np.sort(rng.choice(free, k, replace=False))
            pi, status = np.full(k, k/N), "sampled"
            strata = np.zeros(k, np.int64)
        elif design == "stratified":
            if response is None:
                raise ValueError("stratified design needs the observed response (Teacher-only)")
            vals = np.asarray(response).ravel()[free]
            classes, inv = np.unique(vals, return_inverse=True)
            H = len(classes)
            if k < H:
                raise ValueError(f"stratified bank needs k >= number of strata ({H}); got k={k}")
            Nh = np.bincount(inv, minlength=H)
            alloc = np.ones(H, np.int64)
            rest = k - H
            share = rest*Nh/N
            extra = np.floor(share).astype(np.int64)
            extra = np.minimum(extra, Nh-1)
            left = rest - extra.sum()
            order = np.argsort(-(share-np.floor(share)), kind="stable")
            while left > 0:                                  # largest remainders first, capacity N_h respected
                progressed = False
                for h in order:
                    if left == 0:
                        break
                    if alloc[h]+extra[h] < Nh[h]:
                        extra[h] += 1
                        left -= 1
                        progressed = True
                if not progressed:
                    raise AssertionError("stratified allocation exceeds stratum capacity")
            alloc += extra
            if alloc.sum() != k:
                raise AssertionError("stratified allocation does not sum to k")
            take, pi, st = [], [], []
            for h in range(H):
                members = free[inv == h]
                sel = rng.choice(members, alloc[h], replace=False)
                take.append(sel)
                pi.append(np.full(alloc[h], alloc[h]/Nh[h]))
                st.append(np.full(alloc[h], h))
            take, pi, strata = (np.concatenate(v) for v in (take, pi, st))
            o = np.argsort(take)
            take, pi, strata = take[o], pi[o], strata[o]
            status = "sampled"
        else:
            raise ValueError(f"unknown design {design!r} (uniform | stratified)")
        z, y, x = np.unravel_index(take, (L, L, L))
        xyz = np.column_stack([(origin[0]+x+.5)/1024, (origin[1]+y+.5)/1024, (origin[2]+z+.5)/1024])
        return {"local_indices": take, "world_xyz": xyz, "inclusion_probability": pi, "weight": 1/pi, "stratum": strata,
                "design": design, "k": int(len(take)), "free": int(N), "seed": int(seed), "status": status}

    def full_mismatch(self, sources, strengths, origin, L, observed, backend="cpu_native_certified"):
        """Number and fraction of FREE receivers whose predicted response differs from the observation."""
        lit, solid = self.window(sources, origin, L, backend)
        free = ~solid
        nf = int(free.sum())
        pred = lit[0] if len(lit) == 1 and np.asarray(observed).dtype == bool else self.combined(lit, strengths)
        if nf == 0:
            return {"status": "no_free", "free": 0, "mismatch_count": None, "mismatch_fraction": float("nan")}
        diff = (pred != observed) if pred.dtype == bool else (np.abs(pred - observed) > MA_TOL)
        c = int((diff & free).sum())
        return {"status": "ok", "free": nf, "mismatch_count": c, "mismatch_fraction": c/nf}

    def sampled_mismatch(self, sources, strengths, bank, observed):
        """Horvitz-Thompson estimate of the FREE mismatch count (and fraction) from a receiver bank."""
        if bank["status"] == "no_free":
            return {"status": "no_free", "estimate_count": None, "estimate_fraction": float("nan")}
        val, _ = self.response(sources, strengths, bank["world_xyz"])
        obs = np.asarray(observed).ravel()[bank["local_indices"]].astype(np.float64)
        c = (np.abs(val - obs) > MA_TOL).astype(float)
        est = float((bank["weight"]*c).sum())
        return {"status": "ok", "estimate_count": est, "estimate_fraction": est/bank["free"], "k": bank["k"]}

    def close(self):
        if self._factory is not None:
            self._factory.close()
            self._factory = None
        if self._packs is not None:
            for p in self._packs[0]:
                p.close()
            self._packs[1].close()
            self._packs = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
