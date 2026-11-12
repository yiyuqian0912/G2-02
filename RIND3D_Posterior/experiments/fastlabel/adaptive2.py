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


def test_samples(tester, res, n, rng, stop_after=None, cull=True):
    """Draw n samples in B and test them (with certified tile culling per sample piece).
    Returns (n_tested, failures, violated-receiver arrays) or None if B cannot be sampled."""
    pts, why, picks = sample_in_B(res, n, rng, return_picks=True)
    if pts is None:
        return None
    k = 0; new = []; i = -1
    for i, (p, pc) in enumerate(zip(pts, picks)):
        need = None
        if cull:
            V = res["vpool"][res["voff"][pc]:res["voff"][pc] + res["vn"][pc]]
            need = tester.need_tiles(int(pc), V.min(0) - 1e-12, V.max(0) + 1e-12)
        ok, vox = tester.test(p, need)
        if not ok:
            k += 1
            if vox is not None:
                new.append(vox)
            if stop_after is not None and k > stop_after:
                break
    return i + 1, k, new


class Tester:
    """Exact membership test of candidate sources; on failure returns the mismatching receivers (window x, y, z)."""

    def __init__(self, bundle, policy, lit_zyx, solid_zyx, origin, N=1024, max_report=64):
        from rind3d.rendering.backends import BackendFactory
        self.fac = BackendFactory(cuda_batch=128); self.bundle = bundle; self.policy = policy
        self.ctx = self.fac.context(bundle, policy)
        self.o = np.asarray(origin, np.int64); self.N = int(N); self.L = lit_zyx.shape[0]; self.M = max_report
        self.lit_zyx = lit_zyx; self.solid_zyx = solid_zyx; self.tbox = None
        n = max(self.L // 64, 1)
        self.offs = [(a, b, c) for c in range(n) for b in range(n) for a in range(n)]
        self.origins = np.array([self.o + np.array(t) * 64 for t in self.offs], np.int32)
        self.exp = []
        for (a, b, c) in self.offs:
            sl = (slice(c * 64, (c + 1) * 64), slice(b * 64, (b + 1) * 64), slice(a * 64, (a + 1) * 64))
            self.exp.append((np.packbits(lit_zyx[sl].ravel(), bitorder="little"), np.packbits(solid_zyx[sl].ravel(), bitorder="little")))
        self.stats = dict(points=0, bind_s=0.0, render_s=0.0, legal_s=0.0, fail_legal=0, fail_box=0, fail_obs=0, tiles=0)
        import fastplanes
        self.fp = fastplanes.install(check=20)        # exact integer shadow planes; first 20 calls checked against the original
        self.fe = fastplanes.install_encode(check=20) # exact fast coefficient enclosures; first 20 calls checked
        self.rng = np.random.default_rng(12345)

    def set_cull(self, geom):
        from tilecull import tile_info
        self.geom = geom
        self.tbox, self.tcode = tile_info(self.lit_zyx, self.solid_zyx, self.o, self.N, self.offs)
        self._cull = {}

    def need_tiles(self, piece, qlo, qhi):
        from tilecull import tiles_to_render
        v = self._cull.get(piece)
        if v is None:
            P, K, clo, chi, _, _ = self.geom.tup()
            v = self._cull[piece] = tiles_to_render(qlo, qhi, self.tbox, self.tcode, P, K, clo, chi)
        return v

    def test(self, s, need=None):
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
            idx = np.arange(len(self.origins)) if need is None else np.flatnonzero(need)
            self.stats["tiles_skipped"] = self.stats.get("tiles_skipped", 0) + len(self.origins) - len(idx)
            for i0 in range(0, len(idx), 128):
                sel = idx[i0:i0 + 128]
                out = src.packed_batch(self.origins[sel], 64, self.N); self.stats["tiles"] += len(out)
                for jj, (pl, ps) in enumerate(out):
                    i, j = 0, int(sel[jj])
                    el, es = self.exp[i + j]
                    got = np.frombuffer(pl, np.uint8)
                    if not np.array_equal(got, el):
                        diff = np.flatnonzero(np.unpackbits(got ^ el, bitorder="little"))
                        z, y, x = np.unravel_index(diff, (64, 64, 64))
                        a, b, c = self.offs[j]
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


def run(sc, w, s, nbd=2048, nin=1024, n=None, C=3, R=8, m_disc=16, race_nodes=2000, race_records=20000, race_max_L=64, eps=0.05, delta=0.01, small_records=64, M=64, seed=0, p1_budget=5_000_000):
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
    tester = None; bf = None; rounds = []; attempts = 0; race_res = None; big_pool = False
    rng = np.random.default_rng(seed + 1)
    res = None
    for r in range(R):
        t = time.perf_counter(); res = T.stage2(geom, win, xc, xk, roots, bf=bf, max_nodes_total=p1_budget); dt = time.perf_counter() - t
        rr = dict(round=r, items=len(xc), p1_s=dt, records=len(res["cell"]), B=res["U_total"], undet=res["stats"]["undet"])
        rounds.append(rr)
        cells = np.unique(res["cell"])
        if len(cells) == 1 and res["stats"]["undet"] == 0 and len(res["cell"]) > 0:
            # support certificate: F <= B <= one 128^3 cell and F is nonempty (the true source generated the
            # observation), so q_F is exactly the one-hot cell (vol(F) > 0 almost surely); no phase 2 needed
            out["path"] = "single_cell_support"; out["cell"] = int(cells[0]); break
        if len(res["cell"]) <= small_records and res["stats"]["undet"] == 0:
            out["path"] = "exact_small_B"; break
        if tester is None:
            t = time.perf_counter()
            tester = Tester(load_geometry(T.R / sc / "privileged" / "geometry.npz"), POLICY, S.crop(w, s), S.crop(w, -1), S.origin(w))
            tester.set_cull(geom)
            tt["tester_init"] = time.perf_counter() - t
        # exact-first race: a budgeted exact phase 2 on B; kept if it completes without undetermined mass
        if race_nodes and r == 0 and win.L <= race_max_L and len(res["cell"]) <= race_records:
            t = time.perf_counter(); codes, kls = win.top_items()
            r2 = T.stage2(geom, win, codes, kls, roots, bf=T.build_filter(res), max_nodes_total=race_nodes)
            tt["race"] = time.perf_counter() - t
            if r2["stats"]["undet"] == 0:
                out["path"] = "exact_race"; race_res = r2; break
        # discovery: m fresh samples, every failure's violated receivers become constraints (not used to certify)
        t = time.perf_counter()
        got = test_samples(tester, res, m_disc, rng)
        tt["disc_s"] = tt.get("disc_s", 0) + time.perf_counter() - t
        if got is None and not big_pool:
            # vertex pool full (very broad B): rerun this round's phase 1 once with a larger vertex pool
            big_pool = True
            t = time.perf_counter(); res = T.stage2(geom, win, xc, xk, roots, bf=bf, max_nodes_total=p1_budget, max_vertices=1 << 25)
            rr["p1_s"] += time.perf_counter() - t; rr["big_pool"] = True
            got = test_samples(tester, res, m_disc, rng)
        if got is None:
            out["path"] = "fallback_exact"; break
        _, kd, new = got
        rr["disc_failures"] = kd
        if not new:
            attempts += 1
            n_c = cert_n(eps, delta / C); kmax = max_failures(n_c, eps, delta / C)
            t = time.perf_counter()
            nt, k, cnew = test_samples(tester, res, n_c, rng, stop_after=kmax)
            tt["cert_s"] = tt.get("cert_s", 0) + time.perf_counter() - t
            rr.update(cert_attempt=attempts, cert_n=n_c, cert_failures=k, cert_tested=nt, kmax=kmax)
            if k <= kmax and nt == n_c:
                out["path"] = "stat_certified"; out["tv_cert"] = eps; out["delta"] = delta; out["k"] = k
                out["p_lower"] = cp_lower(n_c - k, n_c, delta / C)
                break
            if attempts >= C or not cnew:
                out["path"] = "fallback_exact"; break
            new = cnew
        nc, nk = items_from_vox(win, np.concatenate(new))
        xc = np.concatenate([xc, nc]); xk = np.concatenate([xk, nk])
        bf = T.build_filter(res)
    else:
        out["path"] = "fallback_exact"
    if out["path"] == "exact_race":
        final = race_res; out["tv_bound"] = race_res["tv_bound"]
    elif out["path"] in ("exact_small_B", "fallback_exact"):
        t = time.perf_counter()
        codes, kls = win.top_items(); tt["items"] = time.perf_counter() - t
        t = time.perf_counter(); r2 = T.stage2(geom, win, codes, kls, roots, bf=T.build_filter(res)); tt["phase2"] = time.perf_counter() - t
        final = r2; out["tv_bound"] = r2["tv_bound"]
    elif out["path"] == "single_cell_support":
        final = dict(cell=np.array([out["cell"]]), lo=np.array([1.0]), hi=np.array([1.0]), U_total=res["U_total"]); out["tv_bound"] = 0.0
    else:
        final = res; out["tv_bound"] = None
    out["wall"] = time.perf_counter() - t0; out["cpu"] = time.process_time() - c0
    out["timing"] = tt; out["rounds"] = rounds; out["tester"] = tester.stats if tester else None
    rp = HERE / "refs" / f"{sc}_w{w:03d}_s{s}.npz"; rp2 = T.R / sc / "labels" / "tasks" / f"w{w:03d}_s{s}.npz"
    out["label_cells"] = None
    if not rp.exists() and not rp2.exists():
        lab = T.cell_masses(final["cell"], final["hi"], final["hi"]) if out["path"] == "stat_certified" else \
            T.cell_masses(final["cell"], final["lo"], final["hi"])
        np.savez_compressed(HERE / "labels_new" / f"{sc}_w{w:03d}_s{s}.npz", cells=np.array(list(lab)), mass=np.array(list(lab.values())))
        out["tv_vs_ref"] = None; out["vol_ratio"] = None
        return out
    ref = np.load(rp if rp.exists() else rp2)
    lab = T.cell_masses(final["cell"], final["hi"], final["hi"]) if out["path"] == "stat_certified" else \
        T.cell_masses(final["cell"], final["lo"], final["hi"])          # q_B uses the piece volumes of B
    out["tv_vs_ref"] = T.tv(lab, T.cell_masses(ref["cell"], ref["lo"], ref["hi"]))
    out["vol_ratio"] = float(ref["lo"].sum()) / final["U_total"] if final.get("U_total", 0) > 0 else None
    return out


if __name__ == "__main__":
    a = sys.argv
    kw = json.loads(a[4]) if len(a) > 4 else {}
    print(json.dumps(run(a[1], int(a[2]), int(a[3]), **kw), default=float), flush=True)
