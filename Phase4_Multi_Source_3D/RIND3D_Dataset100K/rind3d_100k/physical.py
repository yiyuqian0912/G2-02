"""RIND3D-100K physical stage: one scene. Global geometry, 1-4 sources, exactly 4/4/4 windows (sides 64/128/256),
exact LOCAL responses only (no global response field), a rate stream, and one atomic scene file.

**Sampling law.** Identical in form to RIND3D 1K (docs/SELECTION_LAW_100K.md). The same keyed streams are used,
with the contract's seed and quotas:
1. **Geometry.** family = families[slot % 9], from the v0.2 generator and acceptance gate (`rind3d_1k.families`).
2. **Source count and thinning.** k and the thinning table c(m, L, k) come from the epoch-frozen controller plan.
   They are fixed before this scene's sources are drawn.
3. **Sources.** k i.i.d. legal uniform sources, with the 2^-20 anti-duplicate rule (`rind3d_1k.physical.sample_sources`).
4. **Proposals.** For each side, i.i.d. uniform origins from the stream keyed by (seed, slot, side, attempt), plus an
   independent U per proposal. A proposal is accepted iff A(W) and U < c(m, L, k), until 4 windows are accepted or the
   cap is reached. Repeated origins are skipped.
5. **Rate stream.** R proposals per side from separate keyed streams, with the same rule. The accept count gives
   Lambda_hat_L (training weight). The first `rate_exact` proposals are evaluated exactly in m (controller
   statistics).
6. **Scene acceptance.** All quotas must be filled within their caps.

**Contract v2 (declared law, `sampling.py`).** k ~ kappa from the slot's keyed draw (no controller); the natural
stream accepts every admissible proposal (no thinning); an optional coverage stream (L64, K = 4) accepts with c(m) from
separate keyed streams and falls back to the natural stream at its cap; the rate stream evaluates m exactly for every
proposal. Steps 1, 3 and 6 are unchanged; v1 (steps 2, 4, 5 above) stays runnable for regression.

**Exactness.** Every decision is the exact rule's decision (`local_eval`); screening changes cost only. Every stored
response is a certified exact local rendering at N = 1024."""
import hashlib
import io
import json
import time

import numpy as np
import zstandard

from rind3d.geometry import keyed_rng, pack_arrays
from rind3d_1k.families import generate, acceptance as geometry_acceptance
from rind3d_1k.physical import POLICY, STREAM, sample_sources, geometry_rows

from .local_eval import LocalScene, evaluate_proposal, render_accepted
from . import sampling as SM

SIDES = (64, 128, 256)


def _rng(seed, slot, stream, attempt):
    return keyed_rng(seed, slot, stream + 97 * attempt)


def _nprobe(cfg, Lp):
    n = cfg.get("n_probe", {"64": 64, "128": 256, "256": 512})
    return int(n[str(Lp)]) if isinstance(n, dict) else int(n)


def select_side(scene, sources, cfg, slot, attempt, Lp, c_table, probe_rng):
    seed, N = cfg["seed"], cfg["resolution"]
    L = Lp * N // 1024
    quota = int(cfg["quota"][str(Lp)]); cap = int(cfg["cap"][str(Lp)])
    c_row = c_table[str(Lp)]
    rq = _rng(seed, slot, STREAM["quota"] + Lp, attempt)
    ru = _rng(seed, slot, STREAM["thin"] + Lp, attempt)
    acc, seen, log = [], set(), []
    reasons = {}
    st0 = dict(scene.stats)
    nprop = 0; rendered_rejects = 0
    while len(acc) < quota and nprop < cap:
        o = tuple(int(v) for v in rq.integers(0, N - L + 1, 3))
        U = float(ru.random())
        nprop += 1
        if o in seen:
            reasons["repeat"] = reasons.get("repeat", 0) + 1
            continue
        seen.add(o)
        d = evaluate_proposal(scene, sources, o, L, U, c_row, probe_rng, n_probe=_nprobe(cfg, Lp))
        reasons[d.why] = reasons.get(d.why, 0) + 1
        ok = bool(d.accept)
        entry = dict(origin=o, U=U, A=d.A, m=(d.m if d.exact else -1), m_lo=d.m_lo, m_hi=d.m_hi, ok=int(ok))
        if ok:
            lits, sol, counts, m, mask = render_accepted(scene, o, L)
            if not (m >= 1 and U < c_row[m - 1]):
                raise AssertionError("lazy acceptance contradicted by exact rendering")
            if d.exact and m != d.m:
                raise AssertionError("exact m mismatch")
            entry.update(m=m)
            acc.append(dict(origin=o, L=L, Lp=Lp, m=m, mask=mask, counts=counts.tolist(), U=U, proposal=nprop,
                            lits=lits, solid=sol))
        elif d.exact and d.why not in ("contains_source",):
            rendered_rejects += 1
        log.append((o[0], o[1], o[2], entry["A"], entry["m"], entry["m_lo"], entry["m_hi"], U, int(ok)))
    st1 = dict(scene.stats)
    # rate stream: accept indicator for Lambda_hat; exact m on the first rate_exact proposals (controller statistics)
    R = int(cfg["rate_n"][str(Lp)]) if isinstance(cfg["rate_n"], dict) else int(cfg["rate_n"])
    R_exact = min(R, int(cfg.get("rate_exact", 8)))
    rr = _rng(seed, slot, STREAM["rate"] + Lp, attempt)
    rt = _rng(seed, slot, STREAM["rate_thin"] + Lp, attempt)
    lam = 0; mh = np.zeros(5, np.int64); nA_exact = 0
    for i in range(R):
        o = tuple(int(v) for v in rr.integers(0, N - L + 1, 3))
        U = float(rt.random())
        d = evaluate_proposal(scene, sources, o, L, U, c_row, probe_rng, n_probe=_nprobe(cfg, Lp),
                              need_exact=i < R_exact)
        if d.accept:
            lam += 1
        if i < R_exact and d.A:
            nA_exact += 1; mh[d.m] += 1
    st2 = dict(scene.stats)
    delta = lambda a, b: {kk: round(b[kk] - a[kk], 4) for kk in ("tiles", "render_s", "rendered_channels", "decided_by_probes", "decided_by_render", "probes")}
    rep = dict(Lp=Lp, L=L, quota=quota, proposals=nprop, reasons=reasons, filled=len(acc) >= quota,
               work_quota=delta(st0, st1), work_rate=delta(st1, st2),
               rate_n=R, rate_accepts=lam, Lambda_hat=lam / R, rate_exact=R_exact, rate_exact_A=nA_exact,
               rate_m=mh[1:].tolist(), rendered_rejects=rendered_rejects)
    return acc, rep, np.array(log, np.float64).reshape(-1, 9)


def select_side_v2(scene, sources, cfg, slot, attempt, Lp, k, probe_rng):
    """Contract v2/v3: natural stream (+ optional coverage stream with natural fallback) and an exact-m rate stream.
    The eligibility rule (v2 "mixed", v3 "direct_interface") comes from the contract; under v3 the rate stream also
    records the v2 rule of the same proposals (rate_m_mixed), for the sampling-impact comparison."""
    seed, N = cfg["seed"], cfg["resolution"]
    L = Lp * N // 1024
    rule = SM.eligibility(cfg); v3 = rule == "direct_interface"
    streams = SM.side_streams(cfg, Lp, k)
    n_probe = _nprobe(cfg, Lp)
    acc, log, taken = [], [], set()
    st0 = dict(scene.stats)

    def run(name, sid, n_want, c_row, rq, ru, cap, state):
        got = 0
        while got < n_want and state["proposals"] < cap:
            o = tuple(int(v) for v in rq.integers(0, N - L + 1, 3))
            U = float(ru.random())
            state["proposals"] += 1
            if o in state["seen"] or o in taken:
                state["reasons"]["repeat"] = state["reasons"].get("repeat", 0) + 1
                continue
            state["seen"].add(o)
            d = evaluate_proposal(scene, sources, o, L, U, c_row, probe_rng, n_probe=n_probe, rule=rule)
            state["reasons"][d.why] = state["reasons"].get(d.why, 0) + 1
            ok = bool(d.accept)
            m_log = d.m if d.exact else -1
            if ok:
                ra = render_accepted(scene, o, L, rule)
                lits, sol, counts, m, mask = ra[:5]
                extra = ra[5] if v3 else {}
                if not (m >= 1 and U < c_row[m - 1]):
                    raise AssertionError("lazy acceptance contradicted by exact rendering")
                if d.exact and m != d.m:
                    raise AssertionError("exact m mismatch")
                m_log = m; taken.add(o); got += 1
                acc.append(dict(origin=o, L=L, Lp=Lp, m=m, mask=mask, counts=counts.tolist(), U=U,
                                proposal=state["proposals"], stream=name, c_row=[float(x) for x in c_row],
                                lits=lits, solid=sol, **extra))
            log.append((o[0], o[1], o[2], d.A, m_log, d.m_lo, d.m_hi, U, int(ok), SM.STREAM_ID[name]))
        return got

    nat = dict(proposals=0, seen=set(), reasons={})
    nat_spec = streams[0]
    rq = _rng(seed, slot, STREAM["quota"] + Lp, attempt); ru = _rng(seed, slot, STREAM["thin"] + Lp, attempt)
    nat_acc = run("natural", 0, nat_spec["n"], nat_spec["c"], rq, ru, nat_spec["cap"], nat)
    rep_streams = [dict(name="natural", n_target=nat_spec["n"], c=nat_spec["c"], cap=nat_spec["cap"])]
    fallback = 0
    if len(streams) > 1:
        cov_spec = streams[1]
        cov = dict(proposals=0, seen=set(), reasons={})
        cq = _rng(seed, slot, SM.COV_STREAM["quota"] + Lp, attempt); cu = _rng(seed, slot, SM.COV_STREAM["thin"] + Lp, attempt)
        cov_acc = run("coverage", 1, cov_spec["n"], cov_spec["c"], cq, cu, cov_spec["cap"], cov)
        missing = cov_spec["n"] - cov_acc
        if missing > 0 and nat_acc == nat_spec["n"]:
            fallback = run("natural_fallback", 2, missing, nat_spec["c"], rq, ru, nat_spec["cap"], nat)
        rep_streams.append(dict(name="coverage", n_target=cov_spec["n"], c=cov_spec["c"], cap=cov_spec["cap"],
                                accepted=cov_acc, proposals=cov["proposals"], reasons=cov["reasons"]))
    rep_streams[0].update(accepted=nat_acc + fallback, proposals=nat["proposals"], reasons=nat["reasons"],
                          fallback=fallback)
    st1 = dict(scene.stats)
    # rate stream: natural rule, exact m for every proposal (P_A and the coverage acceptance probability)
    R = int(cfg["rate_n"][str(Lp)]) if isinstance(cfg["rate_n"], dict) else int(cfg["rate_n"])
    rr = _rng(seed, slot, STREAM["rate"] + Lp, attempt)
    mh = np.zeros(5, np.int64); nA = 0; why = {}; t_rate = time.perf_counter()
    mh2 = np.zeros(5, np.int64); nA2 = 0                     # v3: the v2 rule on the same proposals
    for i in range(R):
        o = tuple(int(v) for v in rr.integers(0, N - L + 1, 3))
        d = evaluate_proposal(scene, sources, o, L, 0.0, [1.0, 1.0, 1.0, 1.0], probe_rng, n_probe=n_probe, need_exact=True,
                              rule=rule)
        why[d.why] = why.get(d.why, 0) + 1
        if d.A:
            nA += 1; mh[d.m] += 1
        if v3 and d.m_mixed is not None and d.why not in ("contains_source", "no_free"):
            if d.m_mixed >= 1:
                nA2 += 1; mh2[d.m_mixed] += 1
    scene.stats["rate_s"] = scene.stats.get("rate_s", 0.0) + time.perf_counter() - t_rate
    st2 = dict(scene.stats)
    delta = lambda a, b: {kk: round(b[kk] - a[kk], 4) for kk in ("tiles", "render_s", "rendered_channels", "decided_by_probes", "decided_by_render", "probes")}
    rate_m = mh[1:].tolist()
    rep = dict(Lp=Lp, L=L, quota=int(cfg["quota"][str(Lp)]), streams=rep_streams, filled=len(acc) == int(cfg["quota"][str(Lp)]),
               proposals=int(sum(st.get("proposals", 0) for st in rep_streams)),
               reasons={kk: int(sum(st.get("reasons", {}).get(kk, 0) for st in rep_streams))
                        for kk in set().union(*[st.get("reasons", {}).keys() for st in rep_streams])},
               work_quota=delta(st0, st1), work_rate=delta(st1, st2),
               rate_n=R, rate_A=int(nA), rate_m=rate_m, rate_exact=R, rate_reasons=why,
               P_A_hat=nA / R, Lambda_hat=nA / R)
    if len(rep_streams) > 1:
        rep["Lambda_cov_hat"] = float(np.sum(np.asarray(rep_streams[1]["c"]) * np.asarray(rate_m, float)) / R)
    if v3:
        rep.update(eligibility=rule, rate_A_mixed=int(nA2), rate_m_mixed=mh2[1:].tolist())
    return acc, rep, np.array(log, np.float64).reshape(-1, 10)


def _packz(a, cctx):
    return cctx.compress(np.packbits(np.ascontiguousarray(a, dtype=bool).ravel()).tobytes())


def run_scene(cfg, slot, k, c_table, factory, attempt=0):
    """Returns (record, payload_bytes or None). The payload is the complete scene file (npz container)."""
    seed, N = cfg["seed"], cfg["resolution"]
    T = {}
    t0 = time.perf_counter()
    fams = cfg["families"]
    family = fams[slot % len(fams)]
    gcfg = cfg.get("geometry") or {}
    if gcfg:                                  # contract v3: observable thin slabs (geometry_v3, docs/THIN_SLABS_V3.md)
        from .geometry_v3 import generate_v3, thin_slab_report
        bundle, recipe = generate_v3(seed, slot, family, attempt, thin_min_vox=float(gcfg["thin_min_thickness_vox"]))
    else:
        bundle, recipe = generate(seed, slot, family, attempt)
    T["generate"] = time.perf_counter() - t0
    if bundle is None or recipe["proposal_failures"]:
        return dict(slot=slot, attempt=attempt, status="rejected", reason="geometry_proposal_failure", k=k), None
    t_ctx = time.perf_counter()
    context = factory.context(bundle, POLICY)
    T["context"] = time.perf_counter() - t_ctx
    thin = None
    if gcfg:
        t_th = time.perf_counter()
        thin = thin_slab_report(context, recipe, N, min_thickness_vox=float(gcfg["thin_min_thickness_vox"]))
        T["thin_check"] = time.perf_counter() - t_th
        if not all(r["ok"] for r in thin):
            return dict(slot=slot, attempt=attempt, status="rejected", reason="thin_slab_invalid", k=k, family=family,
                        thin_slabs=[r for r in thin if not r["ok"]], timing=dict(geometry=time.perf_counter() - t0)), None
    t_acc = time.perf_counter()
    acceptance = geometry_acceptance(context, family, seed, slot, attempt)
    T["acceptance"] = time.perf_counter() - t_acc
    T["geometry"] = time.perf_counter() - t0
    t = time.perf_counter()
    sources, srep = sample_sources(context, seed, slot, k, attempt)
    T["sources"] = time.perf_counter() - t
    sid = hashlib.sha256(cfg["contract"].encode() + bytes.fromhex(context.geometry_id) +
                         np.asarray(sources, "<f8").tobytes()).hexdigest()
    t = time.perf_counter()
    scene = LocalScene(factory, bundle, sources, POLICY, N, cfg.get("backend", "cuda_certified"))
    if cfg.get("hull_certificate", True):
        scene.enable_hull(context)
    T["bind"] = time.perf_counter() - t
    probe_rng = np.random.default_rng([seed, slot, attempt, 77001])    # probe locations: affect cost, never decisions
    try:
        t = time.perf_counter()
        windows, sides, logs = [], [], {}
        v2 = SM.declared(cfg)
        for Lp in SIDES:
            if v2:
                acc, rep, log = select_side_v2(scene, sources, cfg, slot, attempt, Lp, k, probe_rng)
            else:
                acc, rep, log = select_side(scene, sources, cfg, slot, attempt, Lp, c_table, probe_rng)
            windows += acc; sides.append(rep); logs[f"L{Lp}"] = log
        T["selection"] = time.perf_counter() - t
        work = dict(scene.stats)
        if scene.hull is not None:
            work.update(hull_calls=scene.hull.calls, hull_certified=scene.hull.certified)
    finally:
        scene.close()
    base = dict(slot=slot, attempt=attempt, k=k, family=family, scene_id=sid,
                sides=[{kk: v for kk, v in s.items()} for s in sides], timing=T, work=work)
    if not all(s["filled"] for s in sides):
        return dict(base, status="rejected", reason="quota_unfilled"), None
    t = time.perf_counter()
    cctx = zstandard.ZstdCompressor(level=3)
    blobs = io.BytesIO(); index = []; wstats = []
    for w, rec in enumerate(windows):
        b = _packz(rec["solid"], cctx); index.append((w, -1, blobs.tell(), len(b))); blobs.write(b)
        free = ~rec["solid"]
        for j in range(k):
            b = _packz(rec["lits"][j], cctx); index.append((w, j, blobs.tell(), len(b))); blobs.write(b)
        S = np.sum(np.stack(rec["lits"]), axis=0)[free]
        inf = [j for j in range(k) if (rec["mask"] >> j) & 1]
        ham = [float(np.mean(rec["lits"][a][free] != rec["lits"][b][free])) for ai, a in enumerate(inf) for b in inf[ai + 1:]]
        wstats.append(dict(overlap_hist=np.bincount(S, minlength=k + 1)[:k + 1].tolist(), free=int(free.sum()),
                           lit_frac=[float(rec["lits"][j][free].mean()) for j in range(k)],
                           min_pair_hamming=min(ham) if ham else None))
    fields = [("window", "<i4"), ("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("L", "<i4"), ("Lp", "<i4"), ("m", "<i1"),
              ("mask", "<u1"), ("k", "<i1")]
    v3 = v2 and SM.eligibility(cfg) == "direct_interface"
    W = np.zeros(len(windows), dtype=fields + ([("stream", "<i1")] if v2 else []) + ([("mixed_mask", "<u1")] if v3 else []))
    for w, r in enumerate(windows):
        W[w] = (w, *r["origin"], r["L"], r["Lp"], r["m"], r["mask"], k) + ((SM.STREAM_ID[r["stream"]],) if v2 else ()) \
            + ((int(r["mixed_mask"]),) if v3 else ())
    gbuf = io.BytesIO()
    np.savez_compressed(gbuf, **pack_arrays(bundle))     # same bytes layout as rind3d.geometry.save_geometry
    P, K = geometry_rows(context)
    lam = {str(s["Lp"]): s["Lambda_hat"] for s in sides}
    meta = dict(contract=cfg["contract"], scene_id=sid, slot=slot, attempt=attempt, seed=seed, resolution=N,
                family=family, n_bodies=len(context.bodies), faces=[len(b) for b in context.bodies],
                geometry_id=context.geometry_id, k=k, c_table=c_table, sides=sides, Lambda_hat=lam,
                quota=cfg["quota"], n_windows=len(windows), timing=T, work=work,
                windows_full=[dict({kk: v for kk, v in r.items() if kk not in ("lits", "solid")}, **wstats[w])
                              for w, r in enumerate(windows)],
                sources_report=srep, recipe_keys=sorted(recipe.keys()))
    if thin is not None:
        meta["geometry_rule"] = dict(gcfg, generator=recipe["version"])
        meta["thin_slabs"] = [dict(leaf_id=recipe["objects"][r["object"]]["leaf_id"], **r) for r in thin]
    if v2:
        ku = SM.kappa_draw(cfg, slot)
        meta["sampling"] = dict(law="declared", kappa=[float(x) for x in SM.kappa(cfg)], k=k, kappa_u=ku[1],
                                k_from_draw=ku[0], selection=cfg["selection"], rate_exact="all")
        if SM.eligibility(cfg) != "mixed":
            meta["sampling"]["eligibility"] = cfg["eligibility"]
    out = io.BytesIO()
    np.savez(out, meta=np.frombuffer(json.dumps(meta, default=str).encode(), np.uint8),
             windows=W, obs=np.frombuffer(blobs.getvalue(), np.uint8), obs_index=np.array(index, np.int64),
             geometry=np.frombuffer(gbuf.getvalue(), np.uint8), P=P, K=K,
             sources_hex=np.array([[float(v).hex() for v in s] for s in sources]),
             acceptance=np.frombuffer(json.dumps(acceptance, default=str)[:200000].encode(), np.uint8),
             **{f"log_{kk}": v for kk, v in logs.items()})
    T["serialize"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    rec = dict(base, status="accepted", sampling_law="declared" if v2 else "controller", m_hist=np.bincount([r["m"] for r in windows], minlength=5)[1:].tolist(),
               m_by_side={str(Lp): np.bincount([r["m"] for r in windows if r["Lp"] == Lp], minlength=5)[1:].tolist()
                          for Lp in SIDES},
               Lambda_hat=lam, bytes=len(out.getbuffer()),
               windows=[[int(r["L"]), int(r["mask"]), int(r["m"])] for r in windows])
    return rec, out.getvalue()
