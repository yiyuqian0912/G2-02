"""Adaptive constraint generation with a statistical certificate (prototype).

**Loop** (round r = 1..R):
1. B_r = F(X'_r) ∩ B_{r-1}: the exact TPE engine with the receiver subset X'_r (point items), restricted to B_{r-1} by
   the outer-bound filter (engine_p2x). B_r ⊇ F always.
2. If B_r is concentrated (few pieces), run the exact phase 2 on B_r (cheap) and return an exact label with a
   deterministic certificate.
3. Draw n uniform points in B_r and test each exactly (legality, exclusion, certified CUDA render of the whole window).
   - **All pass:** the label is q_{B_r}, certified TV <= eps (statistical).
   - **A failure:** the failing render names the receivers it violates; up to M of them are added to X'_{r+1}.
4. After R rounds: exact phase 2 on B_R (deterministic fallback).

**Discovery and certification are separated.** Discovery rounds of m fresh samples never certify. Their failures only
add constraints. The first discovery round without failures triggers ONE certification test with n fresh samples:
certify iff the one-sided Clopper-Pearson lower bound for p = vol(F)/vol(B) at level 1 - delta is >= 1 - eps.
**Guarantee.** At most C certification attempts per label, each at level delta / C (Bonferroni):
P(certificate issued and p < 1 - eps) <= delta. A failed attempt's violations feed discovery. After C failed attempts,
the exact phase 2 runs on B.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "twophase")); sys.path.insert(0, str(HERE))
import numpy as np
import twophase as T
from stat_tier import sample_in_B


class Tester:
    """Exact membership test of candidate sources; on failure returns the mismatching receivers (window x, y, z)."""

    def __init__(self, bundle, policy, lit_zyx, solid_zyx, origin, N=1024, max_report=64):
        from rind3d.rendering.backends import BackendFactory
        self.fac = BackendFactory(cuda_batch=32); self.bundle = bundle; self.policy = policy
        self.ctx = self.fac.context(bundle, policy)
        self.o = np.asarray(origin, np.int64); self.N = int(N); self.L = lit_zyx.shape[0]; self.M = max_report
        n = max(self.L // 64, 1)
        self.offs = [(a, b, c) for c in range(n) for b in range(n) for a in range(n)]
        self.origins = np.array([self.o + np.array(t) * 64 for t in self.offs], np.int32)
        self.exp = []
        for (a, b, c) in self.offs:
            sl = (slice(c * 64, (c + 1) * 64), slice(b * 64, (b + 1) * 64), slice(a * 64, (a + 1) * 64))
            self.exp.append((np.packbits(lit_zyx[sl].ravel(), bitorder="little"), np.packbits(solid_zyx[sl].ravel(), bitorder="little")))
        self.stats = dict(points=0, bind_s=0.0, render_s=0.0, legal_s=0.0, fail_legal=0, fail_box=0, fail_obs=0, tiles=0)
        self.rng = np.random.default_rng(12345)

    def test(self, s):
        from rind3d_1k.physical import legal_source
        self.stats["points"] += 1
        lo = self.o / self.N; hi = (self.o + self.L) / self.N
        if np.all((s >= lo) & (s <= hi)):
            self.stats["fail_box"] += 1; return False, None
        t = time.perf_counter(); ok = legal_source(self.ctx, s) is not None; self.stats["legal_s"] += time.perf_counter() - t
        if not ok:
            self.stats["fail_legal"] += 1; return False, None
        t = time.perf_counter(); src = self.fac.source(self.bundle, s, "cuda_certified", self.policy); self.stats["bind_s"] += time.perf_counter() - t
        t = time.perf_counter()
        try:
            bad = []
            for i in range(0, len(self.origins), 32):
                out = src.packed_batch(self.origins[i:i + 32], 64, self.N); self.stats["tiles"] += len(out)
                for j, (pl, ps) in enumerate(out):
                    el, es = self.exp[i + j]
                    got = np.frombuffer(pl, np.uint8)
                    if not np.array_equal(got, el):
                        diff = np.flatnonzero(np.unpackbits(got ^ el, bitorder="little"))
                        z, y, x = np.unravel_index(diff, (64, 64, 64))
                        a, b, c = self.offs[i + j]
                        bad.append(np.stack([x + a * 64, y + b * 64, z + c * 64], 1))
            if not bad:
                return True, None
            vox = np.concatenate(bad)                  # every receiver this source gets wrong (all tiles)
            self.stats["fail_obs"] += 1
            if len(vox) > self.M:
                vox = vox[self.rng.choice(len(vox), self.M, replace=False)]
            return False, vox
        finally:
            self.stats["render_s"] += time.perf_counter() - t
            src.close()


def items_from_vox(win, vox):
    c = win.cls
    keep = [v for v in vox if c[v[0], v[1], v[2]] != 2]
    codes = np.array([T.enc(0, int(a), int(b), int(d)) for a, b, d in keep], np.int64)
    return codes, np.array([c[a, b, d] for a, b, d in keep], np.int8)


def cp_lower(succ, n, alpha):
    """One-sided Clopper-Pearson lower confidence bound (level 1 - alpha) for a binomial success probability."""
    from scipy.stats import beta
    return 0.0 if succ == 0 else float(beta.ppf(alpha, succ, n - succ + 1))


def cert_n(eps, alpha):
    """Smallest n with (1 - eps)^n <= alpha: all-pass certification at level 1 - alpha."""
    import math
    return int(math.ceil(math.log(alpha) / math.log(1 - eps)))


def max_failures(n, eps, alpha):
    k = -1
    while k + 1 < n and cp_lower(n - (k + 1), n, alpha) >= 1 - eps:
        k += 1
    return k


def run(sc, w, s, nbd=2048, nin=1024, n=None, C=3, R=8, m_disc=32, eps=0.05, delta=0.01, small_records=64, M=64, seed=0, p1_budget=5_000_000):
    from rind3d.geometry import load_geometry
    from rind3d_1k.physical import POLICY
    out = dict(sc=sc, w=w, s=s, nbd=nbd, nin=nin, n=n, R=R)
    tt = {}
    t0 = time.perf_counter(); c0 = time.process_time()
    from fastwin import fast_cls, LazyWindow
    S = T.SceneObs(T.R / sc); cls = fast_cls(S, w, s)
    geom = T.Geom(np.asarray(S.P, float), np.asarray(S.K)); win = LazyWindow(cls, np.asarray(S.origin(w), np.int64), 1024)
    tt["setup"] = time.perf_counter() - t0
    t = time.perf_counter(); roots, _ = T.stage1(geom, win); tt["stage1"] = time.perf_counter() - t
    t = time.perf_counter(); xc, xk = T.point_items(win, nbd, nin, seed); tt["points"] = time.perf_counter() - t
    tester = None; bf = None; rounds = []; attempts = 0
    rng = np.random.default_rng(seed + 1)
    res = None
    for r in range(R):
        t = time.perf_counter(); res = T.stage2(geom, win, xc, xk, roots, bf=bf, max_nodes_total=p1_budget); dt = time.perf_counter() - t
        rr = dict(round=r, items=len(xc), p1_s=dt, records=len(res["cell"]), B=res["U_total"], undet=res["stats"]["undet"])
        rounds.append(rr)
        cells = np.unique(res["cell"])
        if (len(res["cell"]) <= small_records or len(cells) == 1) and res["stats"]["undet"] == 0:
            out["path"] = "exact_small_B"; break
        if tester is None:
            t = time.perf_counter()
            tester = Tester(load_geometry(T.R / sc / "privileged" / "geometry.npz"), POLICY, S.crop(w, s), S.crop(w, -1), S.origin(w))
            tt["tester_init"] = time.perf_counter() - t
        # discovery: m fresh samples, every failure's violated receivers become constraints (not used to certify)
        t = time.perf_counter(); pts, why = sample_in_B(res, m_disc, rng); tt["sample_s"] = tt.get("sample_s", 0) + time.perf_counter() - t
        if pts is None:
            rr["sample"] = why; out["path"] = "fallback_exact"; break
        new = []
        for p in pts:
            ok, vox = tester.test(p)
            if not ok and vox is not None:
                new.append(vox)
        rr["disc_failures"] = len(new)
        if not new:
            # certification attempt c (at most C per label, each at level delta / C; Bonferroni): n_c fresh samples,
            # certify iff the one-sided Clopper-Pearson bound for p = vol(F)/vol(B) is >= 1 - eps
            attempts += 1
            n_c = cert_n(eps, delta / C)
            kmax = max_failures(n_c, eps, delta / C)
            pts, why = sample_in_B(res, n_c, rng)
            k = 0; cnew = []
            for i, p in enumerate(pts):
                ok, vox = tester.test(p)
                if not ok:
                    k += 1
                    if vox is not None:
                        cnew.append(vox)
                    if k > kmax:
                        break
            rr.update(cert_attempt=attempts, cert_n=n_c, cert_failures=k, cert_tested=i + 1, kmax=kmax)
            if k <= kmax:
                out["path"] = "stat_certified"; out["tv_cert"] = eps; out["delta"] = delta; out["k"] = k
                out["p_lower"] = cp_lower(n_c - k, n_c, delta / C)
                break
            if attempts >= C or not cnew:
                out["path"] = "fallback_exact"; break
            new = cnew                       # a failed attempt's violations become constraints

        nc, nk = items_from_vox(win, np.concatenate(new))
        xc = np.concatenate([xc, nc]); xk = np.concatenate([xk, nk])
        bf = T.build_filter(res)
    else:
        out["path"] = "fallback_exact"
    if out["path"] in ("exact_small_B", "fallback_exact"):
        t = time.perf_counter()
        codes, kls = win.top_items(); tt["items"] = time.perf_counter() - t
        t = time.perf_counter(); r2 = T.stage2(geom, win, codes, kls, roots, bf=T.build_filter(res)); tt["phase2"] = time.perf_counter() - t
        final = r2; out["tv_bound"] = r2["tv_bound"]
    else:
        final = res; out["tv_bound"] = None
    out["wall"] = time.perf_counter() - t0; out["cpu"] = time.process_time() - c0
    out["timing"] = tt; out["rounds"] = rounds; out["tester"] = tester.stats if tester else None
    rp = HERE / "refs" / f"{sc}_w{w:03d}_s{s}.npz"
    ref = np.load(rp if rp.exists() else T.R / sc / "labels" / "tasks" / f"w{w:03d}_s{s}.npz")
    lab = T.cell_masses(final["cell"], final["hi"], final["hi"]) if out["path"] == "stat_certified" else \
        T.cell_masses(final["cell"], final["lo"], final["hi"])          # q_B uses the piece volumes of B
    out["tv_vs_ref"] = T.tv(lab, T.cell_masses(ref["cell"], ref["lo"], ref["hi"]))
    out["vol_ratio"] = float(ref["lo"].sum()) / final["U_total"] if final["U_total"] > 0 else None
    return out


if __name__ == "__main__":
    a = sys.argv
    kw = json.loads(a[4]) if len(a) > 4 else {}
    print(json.dumps(run(a[1], int(a[2]), int(a[3]), **kw), default=float), flush=True)
