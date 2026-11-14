"""FastLabel v3.4 label pipeline (production copy of RIND3D_Posterior/experiments/fastlabel/adaptive3.py).

Target (contract rind3d-1k-contract-v1): q_F, the 128^3 cell masses of Uniform(F_W^j).
Entry point: label_window_fast(task, **cfg) with task an inputs.FastLabelWindow (from an adapter: window_from_1k,
window_from_100k). The computation depends only on the task record and cfg; the scene format enters only through the
adapter. The 1K entry point rind3d_1k.fastlabel.label_task(scene_dir, w, s) is label_window_fast(window_from_1k(...)).

Per task (scene, window w, source channel s):
1. Phase 1, round r: B_r = F(X'_r) ∩ filter(B_{r-1}) by the tau-pencil engine on a receiver subset (B_r ⊇ F).
2. Deterministic exits: single-cell support (TV 0); optional exact small B / exact race (deterministic tv_bound).
3. Statistical round r: two-stage test with fixed (n1, a1, n2, a2) at budget delta_r = delta * 2^-r; points i.i.d.
   Uniform(B_r) from the label's keyed stream, each tested EXACTLY by gpuverify (GPU, certified arithmetic).
   Certificate: TV(q_{B_r}, q_F) <= eps, with P(any false certificate at this label) <= sum_r delta_r < delta.
   Default delta = family_delta / n_tasks (0.01 / 159,802 for the 1K): by the union bound, P(any false statistical
   certificate among the n_tasks planned labels) <= family_delta. For another campaign, set n_tasks to its label count.
4. Otherwise: violated receivers of failing points join X'; after R rounds: exact phase 2 on B_R (deterministic).

Audit: RIND3D_Posterior/experiments/fastlabel/CERTIFICATE_AUDIT.md; kernel validation (Gate A): GPUVERIFY.md.
Status of the returned label:
* "certified": deterministic TV bound <= 0.05 (exact paths, single-cell support);
* "certified_stat": TV <= eps (= 0.05) with the stated statistical confidence (certificate in meta);
* "certified_loose" / "uncertified": exact fallback with a weaker deterministic bound (never relabelled).
"""
import json
import time
from functools import lru_cache

import numpy as np

from . import engine as T
from .filters import build_filter2, items_from_vox2
from .sampler import BSampler
from .window import window_classes, LazyWindow2

VERSION = "fastlabel-v3.4"
N_DATASET = 159802
STAT_KIND = 4            # record kind of a statistical label: per-cell volume of the certified outer bound B

DEFAULTS = dict(nbd=256, nin=128, R=12, n_cert=1024, n_stage1=512, chunks=(128, 384, 512), eps=0.05,
                delta=None, n_tasks=N_DATASET, family_delta=0.01, M=64, M_round=2048, small_records=0, race_nodes=0, race_records=20000,
                race_max_L=64, p1_budget=5_000_000, base_seed=20261112, validate_n=0,
                s1_d1=4, s1_nb=512, s1_nc=64, s1_learn=32, s1_tests=640, max_vpool=1 << 27)

_FAC = None
_SCENES = {}
GPU_LOCK = None


def set_gpu_lock(lock):
    """Worker initialiser (e.g. rind3d_1k.supervise): the cross-process semaphore around GPU submissions."""
    global GPU_LOCK
    GPU_LOCK = lock


def process_factory():
    global _FAC
    if _FAC is None:
        from rind3d.rendering.backends import BackendFactory
        _FAC = BackendFactory(cuda_batch=128, geometry_cache=4)
        if _FAC._cuda() is None:
            raise RuntimeError("FastLabel needs CUDA: " + str(_FAC.cuda_unavailable))
    return _FAC


def scene_context(scene_key, P, K, bundle, policy):
    """Per-scene solver objects, cached by scene_key (at most 2 scenes per process): tau-pencil geometry from the
    float64 rows P/K and the certified geometry context of the canonical bundle."""
    key = str(scene_key)
    v = _SCENES.get(key)
    if v is None:
        v = dict(geom=T.Geom(np.asarray(P, float), np.asarray(K)), bundle=bundle,
                 ctx=process_factory().context(bundle, policy))
        if len(_SCENES) >= 2:
            _SCENES.pop(next(iter(_SCENES)))
        _SCENES[key] = v
    return v


def task_context(task):
    return scene_context(task.scene_key, task.P, task.K, task.bundle, task.policy)


@lru_cache(maxsize=4096)
def kstar(n, eps, d):
    """Largest k with P(Bin(n, eps) <= k) <= d; -1 if none."""
    from scipy.stats import binom
    k = -1
    while k + 1 < n and binom.cdf(k + 1, n, eps) <= d:
        k += 1
    return k


@lru_cache(maxsize=4096)
def design2(n1, n2, eps, d, frac1=0.5):
    """Two-stage design at the boundary p = 1 - eps: stage 1 certifies iff K_{n1} <= a1 (P(Bin(n1, eps) <= a1) <=
    frac1 * d); stage 2 continues to n2 and certifies iff K_{n2} <= a2, the largest a2 with exact total <= d.
    The certificate event increases with the pass indicators, so its probability for p < 1 - eps is below the value
    at p = 1 - eps. Returns (a1, a2, alpha)."""
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


def label_rng(base_seed, scene_index, w, s):
    ss = np.random.SeedSequence([int(base_seed), int(scene_index), int(w), int(s), 3])
    a, b, c = ss.spawn(3)
    return int(a.generate_state(1)[0]), np.random.default_rng(b), np.random.default_rng(c)


def label_window_fast(task, **kw):
    """Label one FastLabelWindow (scene, window, source channel). Returns dict(cell, lo, hi, kind, tv_bound, status,
    meta). Deterministic given (task, cfg): every random stream is keyed by (base_seed, task.key)."""
    from . import gpuverify as GV
    cfg = dict(DEFAULTS); cfg.update(kw)
    # per-label delta: explicit, or family_delta / n_tasks for a simultaneous guarantee over the PLANNED workload
    # (union bound); n_tasks must be the number of labels the campaign will certify (1K: 159,802; 100K: set it)
    eps = float(cfg["eps"])
    delta = float(cfg["delta"]) if cfg["delta"] is not None else float(cfg["family_delta"]) / int(cfg["n_tasks"])
    sc, w, s = (int(v) for v in task.key)
    out = dict(version=VERSION, eps=eps, delta=delta)
    tt = {}
    t0 = time.perf_counter(); c0 = time.process_time()
    scn = task_context(task); geom = scn["geom"]
    cls = window_classes(task); win = LazyWindow2(cls, np.asarray(task.origin, np.int64), int(task.N))
    tt["setup"] = time.perf_counter() - t0
    seed_items, rng_s, rng_c = label_rng(cfg["base_seed"], sc, w, s)
    t = time.perf_counter()
    roots, _ = T.stage1(geom, win, d1=cfg["s1_d1"], n_boundary=cfg["s1_nb"], n_class=cfg["s1_nc"], n_learn=cfg["s1_learn"],
                        max_tests=cfg["s1_tests"])
    tt["stage1"] = time.perf_counter() - t
    xc, xk = T.point_items(win, cfg["nbd"], cfg["nin"], seed_items)
    V = None; bf = None; rounds = []; race_res = None; vpool_n = 1 << 23; res = None; tri_cache = {}
    try:
        for r in range(1, cfg["R"] + 1):
            t = time.perf_counter()
            res = T.stage2(geom, win, xc, xk, roots, bf=bf, max_nodes_total=cfg["p1_budget"], max_vertices=vpool_n)
            rr = dict(round=r, items=len(xc), p1_s=time.perf_counter() - t, records=len(res["cell"]),
                      undet=res["stats"]["undet"])
            rounds.append(rr)
            cells = np.unique(res["cell"])
            if len(cells) == 1 and res["stats"]["undet"] == 0 and len(res["cell"]) > 0:
                out["path"] = "single_cell_support"; out["cell"] = int(cells[0]); break
            if len(res["cell"]) <= cfg["small_records"] and res["stats"]["undet"] == 0:
                out["path"] = "exact_small_B"; break
            if r == 1 and cfg["race_nodes"] and win.L <= cfg["race_max_L"] and len(res["cell"]) <= cfg["race_records"]:
                codes, kls = win.top_items()
                r2 = T.stage2(geom, win, codes, kls, roots, bf=build_filter2(res), max_nodes_total=cfg["race_nodes"])
                if r2["stats"]["undet"] == 0:
                    out["path"] = "exact_race"; race_res = r2; break
            if V is None:
                V = GV.WindowVerifier(process_factory(), scn["ctx"], task, cls, lock=GPU_LOCK)
            n_r = int(cfg["n_cert"]); d_r = delta * 0.5 ** r; n1 = int(cfg["n_stage1"] or n_r)
            if n1 < n_r:
                a1, k_r, alpha_r = design2(n1, n_r, eps, d_r)
            else:
                a1, k_r, alpha_r = -1, kstar(n_r, eps, d_r), None
            rr.update(n=n_r, n1=n1, delta_r=d_r, a1=a1, kstar=k_r)
            smp = BSampler(res, rng_s, tri_cache)
            while smp.ok.startswith("piece without vertices") and vpool_n < cfg["max_vpool"]:
                vpool_n = int(min(4 * vpool_n, cfg["max_vpool"]))
                t1 = time.perf_counter()
                res = T.stage2(geom, win, xc, xk, roots, bf=bf, max_nodes_total=cfg["p1_budget"], max_vertices=vpool_n)
                rr["p1_s"] += time.perf_counter() - t1; rr["vpool"] = vpool_n
                smp = BSampler(res, rng_s, tri_cache)
            if smp.ok != "ok":
                rr["sample"] = smp.ok; out["path"] = "fallback_exact"; break
            K = 0; tested = 0; viol = []
            bounds = np.cumsum(cfg["chunks"]); bounds = [int(b) for b in bounds if b < n_r] + [n_r]
            lo = 0
            for hi in bounds:
                pts = smp.draw(hi - lo)
                for ok, vox, why in V.test_batch(pts):
                    if not ok:
                        K += 1
                        if vox is not None and len(vox):
                            viol.append(vox if len(vox) <= cfg["M"] else vox[rng_c.choice(len(vox), cfg["M"], replace=False)])
                tested = hi; lo = hi
                if K > k_r:
                    break
                if tested == n1 and n1 < n_r and K <= a1:
                    break
            rr.update(tested=tested, K=K)
            stage = 1 if (n1 < n_r and tested == n1 and K <= a1) else (2 if (tested == n_r and K <= k_r) else 0)
            if stage:
                out.update(path="stat_certified", round=r, n=tested, K=K, stage=stage, a1=a1, a2=k_r, n1=n1, n_max=n_r,
                           delta_r=d_r, alpha_r=alpha_r, p_lower=cp_lower(tested - K, tested, d_r))
                if cfg["validate_n"]:
                    from scipy.stats import beta
                    rv = np.random.default_rng(np.random.SeedSequence([int(cfg["base_seed"]), sc, w, s, 99]))
                    vp = BSampler(res, rv, tri_cache).draw(int(cfg["validate_n"]))
                    nv = len(vp); kv = sum(not g[0] for g in V.test_batch(vp))
                    out["validation"] = dict(n=nv, fail=int(kv), p_hat=1 - kv / nv, note="empirical, not part of the certificate",
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
            final = race_res
        elif out["path"] in ("exact_small_B", "fallback_exact"):
            codes, kls = win.top_items()
            final = T.stage2(geom, win, codes, kls, roots, bf=build_filter2(res))
    finally:
        if V is not None:
            out["verifier"] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in V.stats.items()}
            V.close()
    path = out["path"]
    if path == "stat_certified":
        cm = T.cell_masses(res["cell"], res["hi"], res["hi"])          # q_B: exact per-cell volume of B
        cell = np.array(list(cm), np.int64); vol = np.array(list(cm.values()))
        lab = dict(cell=cell.astype(np.int32), lo=vol, hi=vol.copy(), kind=np.full(len(cell), STAT_KIND, np.int8),
                   tv_bound=eps, status="certified_stat")
        out["certificate"] = dict(kind="statistical", target="Uniform(F_W^j) cell masses", label="q_B (outer bound B ⊇ F)",
                                  eps=eps, delta_label=delta, delta_round=out["delta_r"], round=out["round"], stage=out["stage"],
                                  n=out["n"], K=out["K"], a1=out["a1"], a2=out["a2"], n1=out["n1"], n_max=out["n_max"],
                                  alpha_round=out["alpha_r"], p_lower=out["p_lower"],
                                  policy=(f"simultaneous over {int(cfg['n_tasks'])} planned labels: delta = "
                                          f"{float(cfg['family_delta'])} / {int(cfg['n_tasks'])}") if cfg["delta"] is None
                                  else f"explicit per-label delta = {delta}",
                                  n_tasks=int(cfg["n_tasks"]) if cfg["delta"] is None else None,
                                  seed=[int(cfg["base_seed"]), sc, int(w), int(s)])
    elif path == "single_cell_support":
        lab = dict(cell=np.array([out["cell"]], np.int32), lo=np.array([1.0]), hi=np.array([1.0]),
                   kind=np.zeros(1, np.int8), tv_bound=0.0, status="certified")
        out["certificate"] = dict(kind="support", note="F ⊆ B ⊆ one 128^3 cell and F ≠ ∅ (true source): q_F = one-hot")
    else:
        tv = float(final["tv_bound"])
        lab = dict(cell=np.asarray(final["cell"], np.int32), lo=final["lo"], hi=final["hi"], kind=np.asarray(final["kind"], np.int8),
                   tv_bound=tv, status="certified" if tv <= 0.05 else ("certified_loose" if tv <= 0.10 else "uncertified"))
        out["certificate"] = dict(kind="deterministic", tv_bound=tv, engine="tau-pencil (engine_p2x, outer-bound filter)")
    out["rounds"] = rounds
    out["n_cells"] = int(len(np.unique(lab["cell"])))
    out["wall"] = time.perf_counter() - t0; out["cpu"] = time.process_time() - c0
    lab["meta"] = out
    return lab
