"""Teacher interoperability audit (measured, real canary windows): v0.2 TeacherPhysics, Phase 3 Teacher, v0.3 fix.

  run.sh rind3d_training.interop_audit [--export canary-v03]
A1 CUDA window interface: v0.2 needs an external factory (AttributeError without) and returns SOLID=None;
   v0.3 owns the factory and returns the certified SOLID (compared with the stored mask)
A2 no-FREE windows: v0.2 receiver_bank, Phase 3 sample_probes / full_window_cost, v0.3 behaviour
A3 stratified sample size: v0.2 achieved k vs requested k on real mixed windows; v0.3 exact
A4 Phase 3 probe policy: how often the non-uniform "rare-free fallback" triggers (inclusion probabilities then
   unknown), as a function of the FREE fraction over all natural canary windows
A5 Phase 3 sample identity: observation dicts carry the real scene_id (a SHA256 commitment of the true sources
   in v0.2); the adapter substitutes the opaque sample id
A6 numerical policy: Phase 3 PhysicalOracle builds GeometryContext with the default policy; certified answers
   must agree with the production POLICY (compared on legality, points and a window)
A7 cost per Teacher call (route B budget): legality, point visibility, full window mismatch (cpu / cuda),
   sampled HT mismatch (k=256), Phase 3 full and sparse cost
Output: outputs/tests/interop_audit.json
"""
import json
import time

import numpy as np

from .env import OUT


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="canary-v03")
    a = ap.parse_args()
    from rind3d.rendering.backends import BackendFactory
    from rind3d_production.teacher import TeacherPhysics as T02
    from . import phase3_adapter as P3
    from .scenes import canary_scenes
    from .teacher import NoFreeReceivers, TeacherPhysics
    join = np.load(OUT/"privileged"/"export"/a.export/"join.npy")
    refs = canary_scenes()
    rng = np.random.default_rng(0)
    out = {}
    # ------------------------------------------------------------------ A1
    j = int(np.flatnonzero((join["modality"] == 0) & (join["category"] == 3) & (join["L"] == 128))[0])
    sc = refs[int(join["scene"][j])]
    o, L = (int(join["x"][j]), int(join["y"][j]), int(join["z"][j])), int(join["L"][j])
    s = join["sources"][j][:1]
    t02 = T02(sc.dir)
    try:
        t02.window(s, o, L, backend="cuda_certified")
        a1_nofactory = "no error"
    except Exception as e:                                 # noqa: BLE001 - recording the failure mode
        a1_nofactory = f"{type(e).__name__}: {e}"
    with BackendFactory(geometry_cache=1) as f:
        t02.factory = f
        y02, m02 = t02.window(s, o, L, backend="cuda_certified")
    tp = TeacherPhysics(sc)
    t = time.perf_counter()
    y03, m03 = tp.window(s, o, L, backend="cuda_certified")
    t_cuda = time.perf_counter()-t
    obs, solid = tp.observation(o, L)
    out["A1_cuda_window"] = {"v02_without_factory": a1_nofactory, "v02_with_factory_solid": None if m02 is None else "array",
                             "v02_response_equal_stored": bool(np.array_equal(y02[0], obs)),
                             "v03_response_equal_stored": bool(np.array_equal(y03[0], obs)),
                             "v03_solid_equal_stored": bool(np.array_equal(m03, solid)), "v03_L128_seconds_incl_init": t_cuda}
    # ------------------------------------------------------------------ A2
    jn = int(np.flatnonzero(join["no_free"])[0])
    scn = refs[int(join["scene"][jn])]
    on, Ln = (int(join["x"][jn]), int(join["y"][jn]), int(join["z"][jn])), int(join["L"][jn])
    a2 = {"no_free_windows_in_export": int(join["no_free"].sum()),
          "no_free_by_L": {str(L_): int((join["no_free"] & (join["L"] == L_)).sum()) for L_ in (64, 128, 256)}}
    try:
        T02(scn.dir).receiver_bank(on, Ln, "uniform", 64)
        a2["v02_receiver_bank"] = "no error"
    except Exception as e:                                 # noqa: BLE001
        a2["v02_receiver_bank"] = f"{type(e).__name__}: {e}"
    sid = int(join["sample_id"][jn])
    p3o = P3.observation_for_phase3(scn, on, Ln, sid)
    _, _, _, PH = P3._phase3()
    try:
        PH.sample_probes(p3o, count=64)
        a2["phase3_sample_probes"] = "no error"
    except Exception as e:                                 # noqa: BLE001
        a2["phase3_sample_probes"] = f"{type(e).__name__}: {e}"
    with P3.oracle_for_phase3(scn) as orc:
        src_n = join["sources"][jn][:int(join["K"][jn])]
        if scn.K == 1:
            try:
                PH.full_window_cost(orc, p3o, src_n)
                a2["phase3_full_window_cost"] = "no error"
            except Exception as e:                         # noqa: BLE001
                a2["phase3_full_window_cost"] = f"{type(e).__name__}: {e}"
            a2["adapter_full_cost"] = P3.phase3_full_cost(orc, p3o, src_n)["status"]
    tpn = TeacherPhysics(scn)
    a2["v03_bank_status"] = tpn.receiver_bank(on, Ln, "uniform", 64)["status"]
    try:
        tpn.receiver_bank(on, Ln, "uniform", 64, strict=True)
    except NoFreeReceivers as e:
        a2["v03_strict"] = f"NoFreeReceivers: {e}"
    obs_n, _ = tpn.observation(on, Ln)
    a2["v03_full_mismatch"] = tpn.full_mismatch(src_n, join["strengths"][jn][:int(join["K"][jn])], on, Ln, obs_n)
    tpn.close()
    out["A2_no_free"] = a2
    # ------------------------------------------------------------------ A3
    rows = []
    cand = np.flatnonzero((join["category"] == 3) & (join["L"] == 64))
    for jj in rng.choice(cand, 40, replace=False):
        scx = refs[int(join["scene"][jj])]
        ox = (int(join["x"][jj]), int(join["y"][jj]), int(join["z"][jj]))
        tpx = TeacherPhysics(scx)
        resp, _ = tpx.observation(ox, 64)
        for k in (7, 64, 256):
            b02 = T02(scx.dir).receiver_bank(ox, 64, "stratified", k, seed=1, response=resp)
            b03 = tpx.receiver_bank(ox, 64, "stratified", k, seed=1, response=resp)
            rows.append({"k": k, "v02": b02["k"], "v03": b03["k"], "strata": int(len(np.unique(resp[tpx.free_mask(ox, 64)])))})
        tpx.close()
    out["A3_stratified_k"] = {"cases": len(rows), "v02_mismatched": int(sum(r["v02"] != r["k"] for r in rows)),
                              "v02_max_abs_error": int(max(abs(r["v02"] - r["k"]) for r in rows)),
                              "v03_mismatched": int(sum(r["v03"] != r["k"] for r in rows))}
    # ------------------------------------------------------------------ A4
    nat = join[(join["subset"] == 0) & (join["modality"] == 0)]
    ff = nat["free"]/nat["L"].astype(float)**3
    a4 = {}
    for count in (64, 256):
        attempts = max(1024, count*64)
        # expected accepted distinct FREE positions after `attempts` uniform draws ~ attempts * f (f small)
        trig = (ff > 0) & (ff*attempts < 1.5*count)
        a4[f"count{count}"] = {"attempts": attempts, "fraction_of_natural_SB_windows_likely_fallback": float(trig.mean()),
                               "free_fraction_threshold": 1.5*count/attempts}
    # one measured case
    small = np.flatnonzero((join["modality"] == 0) & (join["free"] > 300) & (join["free"]/join["L"].astype(float)**3 < 0.005))
    if len(small):
        js = int(small[0])
        scs = refs[int(join["scene"][js])]
        os_ = (int(join["x"][js]), int(join["y"][js]), int(join["z"][js]))
        pr = PH.sample_probes(P3.observation_for_phase3(scs, os_, int(join["L"][js]), 1), count=256, seed=0)
        a4["measured_case"] = {"free": int(join["free"][js]), "L": int(join["L"][js]), "policy": pr.policy}
    out["A4_phase3_probe_policy"] = a4
    # ------------------------------------------------------------------ A5
    p3x = P3.observation_for_phase3(sc, o, L, int(join["sample_id"][j]))
    out["A5_identity"] = {"phase3_RIND3D_getitem_keys": ["scene_id", "window_id", "origin", "side", "observed_response", "local_solid_mask"],
                          "risk": "v0.2 scene_id = SHA256(profile, geometry_id, sources, strengths): a label commitment; never a feature",
                          "adapter_scene_id_field": p3x["scene_id"][:7] + "...", "adapter_uses_real_scene_id": sc.scene_id in p3x["scene_id"]}
    # ------------------------------------------------------------------ A6 + A7
    with P3.oracle_for_phase3(sc) as orc:
        a6 = {"phase3_policy_sha256": orc.context.policy_hash, "production_policy_sha256": tp.context.policy_hash}
        cands = []
        while len(cands) < 16:
            p = rng.random(3)
            if tp.legal([p])[0]:
                cands.append(p)
        cands = np.array(cands)
        test = np.vstack([cands, rng.random((16, 3))])
        a6["legality_agree"] = bool(np.array_equal(orc.source_valid(test).valid, tp.legal(test)))
        bank = tp.receiver_bank(o, L, "uniform", 2048, seed=3)
        v3 = tp.visibility(cands[:4], bank["world_xyz"])
        v_p3 = orc.evaluate_points(cands[:4], bank["world_xyz"]).response
        a6["points_agree"] = bool(np.array_equal(v3, np.where(v_p3 < 0, -1, v_p3)) or np.array_equal(v3 == 1, v_p3 == 1))
        w = P3.phase3_window(p3x)
        r = orc.rerender_window(cands[:1], w)
        y, m = tp.window(cands[:1], o, L)
        a6["window_agree"] = bool(np.array_equal(r.response, y[0]) and np.array_equal(r.solid, m))
        out["A6_policy"] = a6
        a7 = {}
        t = time.perf_counter()
        tp.legal(cands)
        a7["legality_ms_per_candidate"] = 1e3*(time.perf_counter()-t)/len(cands)
        t = time.perf_counter()
        tp.visibility(cands, bank["world_xyz"])
        a7["point_pairs_per_s_16_threads"] = len(cands)*len(bank["world_xyz"])/(time.perf_counter()-t)
        for LL in (64, 128, 256):
            jj = int(np.flatnonzero((join["scene"] == join["scene"][j]) & (join["L"] == LL) & (join["category"] == 3))[0])
            oo = (int(join["x"][jj]), int(join["y"][jj]), int(join["z"][jj]))
            ob, _ = tp.observation(oo, LL)
            for be in ("cpu_native_certified", "cuda_certified"):
                t = time.perf_counter()
                for c in cands[:3]:
                    tp.full_mismatch(c[None], [1.0], oo, LL, ob, backend=be)
                a7[f"full_mismatch_L{LL}_{be}_s_per_candidate"] = (time.perf_counter()-t)/3
            bk = tp.receiver_bank(oo, LL, "uniform", 256, seed=1)
            t = time.perf_counter()
            for c in cands:
                tp.sampled_mismatch(c[None], [1.0], bk, ob)
            a7[f"sampled_mismatch_k256_L{LL}_ms_per_candidate"] = 1e3*(time.perf_counter()-t)/len(cands)
        p3w = P3.observation_for_phase3(sc, o, L, 1)
        t = time.perf_counter()
        P3.phase3_full_cost(orc, p3w, cands[:1])
        a7["phase3_full_cost_L128_cpu_s"] = time.perf_counter()-t
        t = time.perf_counter()
        P3.phase3_sampled_cost(orc, p3w, cands, count=256)
        a7["phase3_sparse_cost_k256_ms_per_candidate"] = 1e3*(time.perf_counter()-t)/len(cands)
        out["A7_cost_per_call"] = a7
    tp.close()
    d = OUT/"tests"
    d.mkdir(parents=True, exist_ok=True)
    (d/"interop_audit.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
