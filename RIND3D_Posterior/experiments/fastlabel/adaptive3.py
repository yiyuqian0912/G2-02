"""FastLabel v3: exact outer bound + batched exact GPU point verification + round-wise k-tolerant certificate.

Target (contract rind3d-1k-contract-v1): q_F = cell masses (128^3) of Uniform(F), F = F_W^j = {s in D(G) minus the
closed window box : s reproduces R_j on every FREE receiver of W}.

Per label (window w, source channel s):
1. Phase 1, round r: B_r = F(X'_r) ∩ filter(B_{r-1}) by the TPE engine with a receiver subset X'_r (B_r ⊇ F; exact
   cell masses q_{B_r} and convex pieces).
2. Deterministic exits, checked every round:
   * single-cell support: B_r ⊆ one 128^3 cell and F ≠ ∅ (the true source generated R_j) -> q_F = δ_cell (TV = 0);
   * exact small B: <= small_records pieces -> exact phase 2 restricted to B_r (engine_p2x), deterministic bound;
   * round 1, L <= race_max_L: budgeted exact phase 2 (kept if it completes).
3. Statistical round r (fixed in advance: n_r = n_cert, delta_r = delta * 2^-r, k*_r = max{k : P(Bin(n_r, eps) <= k)
   <= delta_r}): n_r fresh i.i.d. points of Uniform(B_r), tested EXACTLY by gpuverify, in chunks; a chunk that makes
   K_r > k*_r stops the round (no certificate possible). Certify iff all n_r are tested and K_r <= k*_r.
   Otherwise the violated receivers of failing points are added to X'_{r+1}.
4. After R rounds: exact phase 2 on B_R (deterministic fallback).

Guarantee (CERTIFICATE_AUDIT.md): with p_r = vol(F)/vol(B_r), TV(q_{B_r}, q_F) <= 1 - p_r, and
P(a certificate is issued at a round with p_r < 1 - eps) <= sum_r delta_r < delta. The probability is over this label's
sampling randomness (keyed stream); n_r, k*_r and delta_r never depend on the round's own samples.
"""
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "twophase")); sys.path.insert(0, str(HERE))
import numpy as np
import twophase as T
from stat_tier import sample_in_B
from sampler3 import BSampler
from fastbf import build_filter2, items_from_vox2

VERSION = "fastlabel-v3.3"
N_DATASET = 159802                     # informative channels of the 1K (for the dataset-level delta policy)

DEFAULTS = dict(nbd=256, nin=128, R=12, n_cert=1024, n_stage1=512, chunks=(128, 384, 512), eps=0.05, delta=0.01 / N_DATASET,
                M=64, M_round=2048, small_records=0, race_nodes=0, race_records=20000, race_max_L=64,
                p1_budget=5_000_000, base_seed=20261112, refs=True, save_dir=None, validate_n=0,
                s1_d1=4, s1_nb=512, s1_nc=64, s1_learn=32, s1_tests=640, max_vpool=1 << 27)

_FAC = None
_SCENES = {}
GPU_LOCK = None


def process_factory():
    global _FAC
    if _FAC is None:
        from rind3d.rendering.backends import BackendFactory
        _FAC = BackendFactory(cuda_batch=128, geometry_cache=4)
        _FAC._cuda()
    return _FAC


def scene_cache(sc):
    v = _SCENES.get(sc)
    if v is None:
        from rind3d.geometry import load_geometry
        from rind3d_1k.physical import POLICY
        S = T.SceneObs(T.R / sc)
        bundle = load_geometry(T.R / sc / "privileged" / "geometry.npz")
        v = dict(S=S, geom=T.Geom(np.asarray(S.P, float), np.asarray(S.K)), bundle=bundle,
                 ctx=process_factory().context(bundle, POLICY))
        if len(_SCENES) >= 2:
            _SCENES.pop(next(iter(_SCENES)))
        _SCENES[sc] = v
    return v


from functools import lru_cache


@lru_cache(maxsize=4096)
def kstar(n, eps, d):
    """Largest k with P(Bin(n, eps) <= k) <= d (K = failures at the boundary p = 1 - eps); -1 if none."""
    from scipy.stats import binom
    k = -1
    while k + 1 < n and binom.cdf(k + 1, n, eps) <= d:
        k += 1
    return k


@lru_cache(maxsize=4096)
def design2(n1, n2, eps, d, frac1=0.5):
    """Two-stage sequential design at the boundary p = 1 - eps (failures K ~ Bin):
    stage 1: certify iff K_{n1} <= a1, with P(Bin(n1, eps) <= a1) <= frac1 * d;
    stage 2: otherwise continue to n2 points and certify iff K_{n2} <= a2, a2 the largest value with
    P(certify at stage 1 or 2) = P(K1 <= a1) + sum_{k1 = a1+1}^{a2} P(K1 = k1) P(Bin(n2 - n1, eps) <= a2 - k1) <= d.
    The certificate event is increasing in the per-point pass indicators, so for every p < 1 - eps its probability is
    below its value at p = 1 - eps. Returns (a1, a2, alpha)."""
    from scipy.stats import binom
    a1 = -1
    while a1 + 1 < n1 and binom.cdf(a1 + 1, n1, eps) <= frac1 * d:
        a1 += 1
    pm = binom.pmf(np.arange(n1 + 1), n1, eps); m = n2 - n1
    def tot(a2):
        p = binom.cdf(a1, n1, eps) if a1 >= 0 else 0.0
        for k1 in range(a1 + 1, min(a2, n1) + 1):
            p += pm[k1] * binom.cdf(a2 - k1, m, eps)
        return float(p)
    a2 = a1
    while tot(a2 + 1) <= d:
        a2 += 1
    return a1, a2, tot(a2)


def cp_lower(succ, n, alpha):
    from scipy.stats import beta
    return 0.0 if succ == 0 else float(beta.ppf(alpha, succ, n - succ + 1))


def label_rng(base_seed, sc, w, s):
    """Independent keyed streams per label: (phase-1 subset, sampling, constraint choice)."""
    ss = np.random.SeedSequence([int(base_seed), int(sc), int(w), int(s), 3])
    a, b, c = ss.spawn(3)
    return int(a.generate_state(1)[0]), np.random.default_rng(b), np.random.default_rng(c)


def items_from_vox(win, vox):
    c = win.cls
    x, y, z = vox[:, 0], vox[:, 1], vox[:, 2]
    k = c[x, y, z]; keep = k != 2
    codes = np.array([T.enc(0, int(a), int(b), int(d)) for a, b, d in vox[keep]], np.int64)
    return codes, k[keep].astype(np.int8)


def run(sc, w, s, **kw):
    cfg = dict(DEFAULTS); cfg.update(kw)
    import gpuverify as GV
    from fastwin import fast_cls3 as fast_cls, LazyWindow2 as LazyWindow
    eps, delta = cfg["eps"], cfg["delta"]
    out = dict(sc=sc, w=w, s=s, version=VERSION, eps=eps, delta=delta)
    tt = {}
    t0 = time.perf_counter(); c0 = time.process_time()
    scn = scene_cache(sc); S = scn["S"]; geom = scn["geom"]
    cls = fast_cls(S, w, s); win = LazyWindow(cls, np.asarray(S.origin(w), np.int64), 1024)
    out["L"] = win.L
    tt["setup"] = time.perf_counter() - t0
    seed_items, rng_s, rng_c = label_rng(cfg["base_seed"], sc, w, s)
    t = time.perf_counter()
    roots, _ = T.stage1(geom, win, d1=cfg["s1_d1"], n_boundary=cfg["s1_nb"], n_class=cfg["s1_nc"], n_learn=cfg["s1_learn"], max_tests=cfg["s1_tests"])
    tt["stage1"] = time.perf_counter() - t
    t = time.perf_counter(); xc, xk = T.point_items(win, cfg["nbd"], cfg["nin"], seed_items); tt["points"] = time.perf_counter() - t
    V = None; bf = None; rounds = []; race_res = None; vpool_n = 1 << 23; res = None; tri_cache = {}
    try:
        for r in range(1, cfg["R"] + 1):
            t = time.perf_counter(); res = T.stage2(geom, win, xc, xk, roots, bf=bf, max_nodes_total=cfg["p1_budget"], max_vertices=vpool_n)
            rr = dict(round=r, items=len(xc), p1_s=time.perf_counter() - t, records=len(res["cell"]), B=res["U_total"],
                      undet=res["stats"]["undet"])
            rounds.append(rr)
            cells = np.unique(res["cell"])
            if len(cells) == 1 and res["stats"]["undet"] == 0 and len(res["cell"]) > 0:
                out["path"] = "single_cell_support"; out["cell"] = int(cells[0]); break
            if len(res["cell"]) <= cfg["small_records"] and res["stats"]["undet"] == 0:
                out["path"] = "exact_small_B"; break
            if r == 1 and cfg["race_nodes"] and win.L <= cfg["race_max_L"] and len(res["cell"]) <= cfg["race_records"]:
                t = time.perf_counter(); codes, kls = win.top_items()
                r2 = T.stage2(geom, win, codes, kls, roots, bf=build_filter2(res), max_nodes_total=cfg["race_nodes"])
                tt["race"] = time.perf_counter() - t
                if r2["stats"]["undet"] == 0:
                    out["path"] = "exact_race"; race_res = r2; break
            if V is None:
                t = time.perf_counter()
                V = GV.WindowVerifier(process_factory(), scn["ctx"], S, w, s, cls, lock=GPU_LOCK)
                tt["verifier_init"] = time.perf_counter() - t
            # statistical round r
            n_r = int(cfg["n_cert"]); d_r = delta * 0.5 ** r; n1 = int(cfg["n_stage1"] or n_r)
            if n1 < n_r:
                a1, k_r, alpha_r = design2(n1, n_r, eps, d_r)
            else:
                a1, k_r = -1, kstar(n_r, eps, d_r); alpha_r = None
            rr.update(n=n_r, n1=n1, delta_r=d_r, a1=a1, kstar=k_r)
            t = time.perf_counter()
            smp = BSampler(res, rng_s, tri_cache)
            while smp.ok.startswith("piece without vertices") and vpool_n < cfg["max_vpool"]:
                # vertex pool full (very broad B): rerun this round's phase 1 with a 4x larger pool, kept for later rounds
                vpool_n = int(min(4 * vpool_n, cfg["max_vpool"]))
                t1 = time.perf_counter(); res = T.stage2(geom, win, xc, xk, roots, bf=bf, max_nodes_total=cfg["p1_budget"], max_vertices=vpool_n)
                rr["p1_s"] += time.perf_counter() - t1; rr["vpool"] = vpool_n
                smp = BSampler(res, rng_s, tri_cache)
            tt["sample_s"] = tt.get("sample_s", 0) + time.perf_counter() - t
            if smp.ok != "ok":
                rr["sample"] = smp.ok; out["path"] = "fallback_exact"; break
            K = 0; tested = 0; viol = []
            bounds = np.cumsum(cfg["chunks"]); bounds = [int(b) for b in bounds if b < n_r] + [n_r]
            lo = 0
            for hi in bounds:
                # lazy chunks: the round's points are i.i.d., so drawing them chunk by chunk is the same law
                t = time.perf_counter(); pts = smp.draw(hi - lo); tt["sample_s"] = tt.get("sample_s", 0) + time.perf_counter() - t
                t = time.perf_counter()
                for ok, vox, why in V.test_batch(pts):
                    if not ok:
                        K += 1
                        if vox is not None and len(vox):
                            viol.append(vox if len(vox) <= cfg["M"] else vox[rng_c.choice(len(vox), cfg["M"], replace=False)])
                tt["test_s"] = tt.get("test_s", 0) + time.perf_counter() - t
                tested = hi; lo = hi
                if K > k_r:
                    break
                if tested == n1 and n1 < n_r and K <= a1:
                    break                                   # stage-1 certificate
            rr.update(tested=tested, K=K)
            stage = 1 if (n1 < n_r and tested == n1 and K <= a1) else (2 if (tested == n_r and K <= k_r) else 0)
            if stage:
                out.update(path="stat_certified", round=r, n=tested, K=K, stage=stage, a1=a1, kstar=k_r, n1=n1,
                           n_max=n_r, delta_r=d_r, alpha_r=alpha_r, p_lower=cp_lower(tested - K, tested, d_r))
                if cfg["validate_n"]:
                    # independent empirical check (NOT part of the certificate): fresh points from a separate stream
                    from scipy.stats import beta
                    rv = np.random.default_rng(np.random.SeedSequence([int(cfg["base_seed"]), int(sc), w, s, 99]))
                    vp = BSampler(res, rv, tri_cache).draw(int(cfg["validate_n"]))
                    nv = len(vp); kv = sum(not g[0] for g in V.test_batch(vp))
                    out["validation"] = dict(n=nv, fail=int(kv), p_hat=1 - kv / nv,
                                             ci99=[float(beta.ppf(0.005, nv - kv, kv + 1)) if kv < nv else 0.0,
                                                   float(beta.ppf(0.995, nv - kv + 1, kv)) if kv > 0 else 1.0])
                break
            if not viol:
                out["path"] = "fallback_exact"; rr["note"] = "failures without violated receivers"; break
            v = np.unique(np.concatenate(viol), axis=0)
            if len(v) > cfg["M_round"]:
                v = v[rng_c.choice(len(v), cfg["M_round"], replace=False)]
            nc, nk = items_from_vox2(win, v)
            xc = np.concatenate([xc, nc]); xk = np.concatenate([xk, nk])
            bf = build_filter2(res)
        else:
            out["path"] = "fallback_exact"
        if out["path"] == "exact_race":
            final = race_res; out["tv_bound"] = race_res["tv_bound"]
        elif out["path"] in ("exact_small_B", "fallback_exact"):
            t = time.perf_counter(); codes, kls = win.top_items(); tt["items"] = time.perf_counter() - t
            t = time.perf_counter(); r2 = T.stage2(geom, win, codes, kls, roots, bf=build_filter2(res)); tt["phase2"] = time.perf_counter() - t
            final = r2; out["tv_bound"] = r2["tv_bound"]
        elif out["path"] == "single_cell_support":
            final = dict(cell=np.array([out["cell"]]), lo=np.array([1.0]), hi=np.array([1.0])); out["tv_bound"] = 0.0
        else:
            final = res; out["tv_bound"] = None; out["tv_cert"] = eps
    finally:
        if V is not None:
            out["verifier"] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in V.stats.items()}
            V.close()
    stat = out["path"] == "stat_certified"
    lab = T.cell_masses(final["cell"], final["hi"], final["hi"]) if stat else T.cell_masses(final["cell"], final["lo"], final["hi"])
    out["n_cells"] = len(lab)
    out["wall"] = time.perf_counter() - t0; out["cpu"] = time.process_time() - c0
    out["timing"] = tt; out["rounds"] = rounds
    if cfg["save_dir"]:
        d = Path(cfg["save_dir"]); d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"{sc}_w{w:03d}_s{s}.npz", cell=np.array(list(lab), np.int64), mass=np.array(list(lab.values())),
                            meta=json.dumps({k: v for k, v in out.items() if k not in ("rounds", "timing")}, default=float))
    if cfg["refs"]:
        rp = HERE / "refs" / f"{sc}_w{w:03d}_s{s}.npz"; rp2 = T.R / sc / "labels" / "tasks" / f"w{w:03d}_s{s}.npz"
        for p in (rp, rp2):
            if p.exists():
                ref = np.load(p)
                out["tv_vs_ref"] = T.tv(lab, T.cell_masses(ref["cell"], ref["lo"], ref["hi"]))
                out["ref_tv_bound"] = float(ref["tv_bound"]) if "tv_bound" in ref.files else None
                break
    return out


if __name__ == "__main__":
    a = sys.argv
    kw = json.loads(a[4]) if len(a) > 4 else {}
    print(json.dumps(run(a[1], int(a[2]), int(a[3]), **kw), default=float), flush=True)
