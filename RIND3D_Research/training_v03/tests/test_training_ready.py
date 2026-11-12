"""v0.3 training-readiness integration tests on the ACTUAL exported files (student + privileged views).

  RIND3D_V03_EXPORT=canary-v03 ./run.sh pytest -q -p no:cacheprovider tests
Steps 1-10 follow the Teacher-interoperability checklist; the remaining tests audit the fixed Teacher interface,
selection weights, the sampler and representation exactness. Each test uses real canary scenes.
"""
import json
import os

import numpy as np
import pytest

from rind3d_training.env import OUT
from rind3d_training.scenes import canary_scenes
from rind3d_training.student import FEATURE_KEYS, SceneBalancedSampler, StudentShards
from rind3d_training.teacher import IllegalSource, MA_TOL, NoFreeReceivers, TeacherPhysics, in_closed_box

NAME = os.environ.get("RIND3D_V03_EXPORT", "canary-v03")
PRIV = OUT/"privileged"/"export"/NAME
pytestmark = pytest.mark.skipif(not PRIV.exists(), reason="export missing")


@pytest.fixture(scope="module")
def ctx():
    join = np.load(PRIV/"join.npy")
    refs = canary_scenes()
    sh = StudentShards(OUT/"student"/NAME/"train")
    by_sid = {int(s): i for i, s in enumerate(join["sample_id"])}
    return {"join": join, "refs": refs, "sh": sh, "by_sid": by_sid}


def pick(ctx, modality, L, pred=None, seed=0):
    sh, join, by = ctx["sh"], ctx["join"], ctx["by_sid"]
    rng = np.random.default_rng(seed)
    for i in rng.permutation(len(sh)):
        r = sh.index[i]
        if r["modality"] != modality or r["L"] != L:
            continue
        j = by[int(r["sample_id"])]
        if pred is None or pred(join[j]):
            return int(i), j
    pytest.skip("no matching sample")


def student_and_teacher(ctx, i, j):
    feats, sid = ctx["sh"].read(i)
    sc = ctx["refs"][int(ctx["join"]["scene"][j])]
    return feats, sid, sc, TeacherPhysics(sc, threads=8)


# ----------------------------------------------------------------------------------------------- steps 1-10
@pytest.mark.parametrize("mod", [0, 1])
def test_01_same_scene_both_views(ctx, mod):
    i, j = pick(ctx, mod, 64, lambda r: r["category"] == 3)
    feats, sid, sc, tp = student_and_teacher(ctx, i, j)
    assert int(sid) == int(ctx["join"]["sample_id"][j])
    o = tuple(int(v) for v in feats["window_origin"])
    assert o == (int(ctx["join"]["x"][j]), int(ctx["join"]["y"][j]), int(ctx["join"]["z"][j]))
    obs, solid = tp.observation(o, 64)
    assert np.array_equal(solid, feats["observed_solid"]) and np.array_equal(obs, feats["observed_response"])
    tp.close()


@pytest.mark.parametrize("mod", [0, 1])
def test_02_true_label(ctx, mod):
    i, j = pick(ctx, mod, 64)
    row = ctx["join"][j]
    sc = ctx["refs"][int(row["scene"])]
    s, a = sc.privileged_labels()
    K = int(row["K"])
    assert K == sc.K and np.all(np.isnan(row["sources"][K:]))
    got = {tuple(map(float, x)) for x in row["sources"][:K]}
    assert got == {tuple(map(float, x)) for x in s}                       # unordered set (MA) / single source
    tp = TeacherPhysics(sc, threads=8)
    assert tp.legal(row["sources"][:K]).all()
    tp.close()


@pytest.mark.parametrize("mod", [0, 1])
def test_03_source_visibility(ctx, mod):
    i, j = pick(ctx, mod, 128, lambda r: r["category"] == 3)
    feats, _, sc, tp = student_and_teacher(ctx, i, j)
    row = ctx["join"][j]
    K = int(row["K"])
    bank = tp.receiver_bank(tuple(feats["window_origin"]), 128, "uniform", 2048, seed=1)
    val, vis = tp.response(row["sources"][:K], row["strengths"][:K], bank["world_xyz"])
    obs = feats["observed_response"].ravel()[bank["local_indices"]].astype(float)
    assert np.all(vis >= 0) and np.max(np.abs(val - obs)) <= MA_TOL
    tp.close()


@pytest.mark.parametrize("mod,backend", [(0, "cpu_native_certified"), (1, "cpu_native_certified"), (0, "cuda_certified")])
def test_04_reconstruct_observation(ctx, mod, backend):
    i, j = pick(ctx, mod, 64, lambda r: r["category"] == 3, seed=2)
    feats, _, sc, tp = student_and_teacher(ctx, i, j)
    row = ctx["join"][j]
    K = int(row["K"])
    s, a = sc.privileged_labels()                                          # channel (generation) order
    lit, solid = tp.window(s, tuple(feats["window_origin"]), 64, backend)
    assert solid is not None and np.array_equal(solid, feats["observed_solid"])
    pred = lit[0] if K == 1 else tp.combined(lit, a)
    assert np.array_equal(pred, feats["observed_response"]) if K == 1 else np.max(np.abs(pred - feats["observed_response"])) == 0
    tp.close()


def test_05_uniform_and_stratified_banks(ctx):
    i, j = pick(ctx, 0, 128, lambda r: r["category"] == 3, seed=3)
    feats, _, sc, tp = student_and_teacher(ctx, i, j)
    o = tuple(feats["window_origin"])
    N = int((~feats["observed_solid"]).sum())
    u = tp.receiver_bank(o, 128, "uniform", 256, seed=0)
    assert u["k"] == 256 and np.allclose(u["inclusion_probability"], 256/N) and np.isclose(u["weight"].sum(), N)
    assert len(np.unique(u["local_indices"])) == 256 and not feats["observed_solid"].ravel()[u["local_indices"]].any()
    for k in (2, 7, 64, 255, 256, 1001):
        st = tp.receiver_bank(o, 128, "stratified", k, seed=k, response=feats["observed_response"])
        assert st["k"] == k                                                 # exact sample size (v0.2 bug fixed)
        assert np.isclose(st["weight"].sum(), N)                            # HT weights reproduce the FREE total
        assert set(np.unique(st["stratum"])) == {0, 1}                     # every stratum represented
    c = tp.receiver_bank(o, 128, "uniform", N + 5)
    assert c["status"] == "census" and c["k"] == N and np.all(c["inclusion_probability"] == 1)
    tp.close()


def test_06_inclusion_probabilities_and_weights(ctx):
    """HT unbiasedness (Monte Carlo) for both designs, and selection-law weights vs the natural population."""
    i, j = pick(ctx, 0, 64, lambda r: r["category"] == 3 and r["free"] > 2000, seed=4)
    feats, _, sc, tp = student_and_teacher(ctx, i, j)
    o = tuple(feats["window_origin"])
    obs = feats["observed_response"].ravel()
    free = ~feats["observed_solid"].ravel()
    c = (obs & free).astype(float)                                          # cost = lit indicator; total = lit count
    total = c.sum()
    for design in ("uniform", "stratified"):
        est = []
        for s in range(300):
            b = tp.receiver_bank(o, 64, design, 64, seed=1000+s, response=feats["observed_response"])
            est.append((b["weight"]*c[b["local_indices"]]).sum())
        est = np.array(est)
        assert abs(est.mean() - total) <= 4*est.std()/np.sqrt(len(est)) + 1e-9
    tp.close()
    from rind3d_training.selection import check_weights, load_index
    _, rec, reports, rates = load_index("canary-v0.2")
    chk = check_weights(rec, reports, rates, "global")
    for L, stats in chk.items():
        for k in ("mixed", "A_informative", "A_extreme", "contains_source"):
            v = stats[k]
            assert abs(v["weighted"] - v["population_rate_stream"]) <= max(0.25*v["population_rate_stream"], 0.01), (L, k, v)


def test_07_full_and_sampled_free_mismatch(ctx):
    i, j = pick(ctx, 0, 64, lambda r: r["category"] == 3 and not r["contains_source"], seed=5)
    feats, sid, sc, tp = student_and_teacher(ctx, i, j)
    row = ctx["join"][j]
    o = tuple(int(v) for v in feats["window_origin"])
    obs = feats["observed_response"]
    true = row["sources"][:1]
    f0 = tp.full_mismatch(true, [1.0], o, 64, obs)
    assert f0["status"] == "ok" and f0["mismatch_count"] == 0
    rng = np.random.default_rng(0)
    cand = None
    while cand is None:
        p = np.clip(true[0] + rng.normal(0, 0.05, 3), 0.01, 0.99)
        if tp.legal([p])[0]:
            cand = p[None]
    f1 = tp.full_mismatch(cand, [1.0], o, 64, obs)
    bank = tp.receiver_bank(o, 64, "uniform", 4096, seed=9)
    s1 = tp.sampled_mismatch(cand, [1.0], bank, obs)
    assert f1["mismatch_count"] > 0 or s1["estimate_count"] == 0
    se = np.sqrt(f1["free"]**2*f1["mismatch_fraction"]*(1-f1["mismatch_fraction"])/4096) + 1
    assert abs(s1["estimate_count"] - f1["mismatch_count"]) <= 5*se
    # Phase 3 Teacher agrees through the adapter (read-only import)
    from rind3d_training import phase3_adapter as P3
    p3obs = P3.observation_for_phase3(sc, o, 64, sid)
    with P3.oracle_for_phase3(sc) as orc:
        r = P3.phase3_full_cost(orc, p3obs, cand)
        assert r["status"] == "ok" and np.isclose(r["cost"], f1["mismatch_fraction"])
        r0 = P3.phase3_full_cost(orc, p3obs, true)
        assert r0["cost"] == 0
        sp = P3.phase3_sampled_cost(orc, p3obs, np.vstack([true, cand]), count=256, seed=1)
        assert sp["cost"][0] == 0
    assert not p3obs["scene_id"].endswith(sc.scene_id[:16])
    tp.close()


def test_08_candidate_legality(ctx):
    i, j = pick(ctx, 0, 256, lambda r: r["category"] == 3, seed=6)
    feats, _, sc, tp = student_and_teacher(ctx, i, j)
    o = tuple(int(v) for v in feats["window_origin"])
    rng = np.random.default_rng(1)
    cand = np.vstack([rng.random((60, 3)), [[0.0, 0.5, 0.5], [1.2, 0.5, 0.5]], (np.array(o) + 128.5)[None]/1024])
    leg = tp.legality(cand)
    legx = tp.legality(cand, window=(o, 256))
    assert leg[60] == (False, "outside_open_world") and leg[61] == (False, "outside_open_world")
    inside = in_closed_box(cand, o, 256)
    assert inside[-1] and all(legx[k] == (False, "inside_window_box") for k in np.flatnonzero(inside) if leg[k][1] != "outside_open_world")
    from rind3d_training import phase3_adapter as P3
    with P3.oracle_for_phase3(sc) as orc:
        v = orc.source_valid(cand)
        assert np.array_equal(v.valid, np.array([x for x, _ in leg]))
        C, _, _, _ = P3._phase3()
        vw = orc.source_valid(cand, window=C.Window(o, 256))
        assert np.array_equal(vw.valid, np.array([x for x, _ in legx]))       # agree off the box boundary
    tp.close()


def test_09_source_containing_windows(ctx):
    join = ctx["join"]
    sel = np.flatnonzero(join["contains_source"] & (join["subset"] == 0) & (join["split"] == 0) & (join["modality"] == 0))
    assert len(sel) > 0
    j = int(sel[0])
    row = join[j]
    assert row["w_srcexcl_mixture"] == 0 and row["w_global_mixture"] > 0     # excluded from the src-excluded task
    i = int(np.flatnonzero(ctx["sh"].index["sample_id"] == row["sample_id"])[0])
    feats, _, sc, tp = student_and_teacher(ctx, i, j)
    o, L = tuple(int(v) for v in feats["window_origin"]), int(feats["window_size"])
    true = row["sources"][:1]
    assert in_closed_box(true, o, L)[0]
    assert tp.legality(true)[0][0] and tp.legality(true, window=(o, L))[0] == (False, "inside_window_box")
    f = tp.full_mismatch(true, [1.0], o, L, feats["observed_response"])
    assert f["mismatch_count"] == 0
    assert not (join["contains_source"] & (join["subset"] > 0)).any()          # no subset window contains a source
    tp.close()


def test_10_student_inputs_contain_no_privileged_data(ctx):
    from rind3d_training.leakage import audit
    res = audit(NAME, sample=300)
    assert res["all_pass"], {k: v for k, v in res.items() if isinstance(v, dict) and not v["pass"]}
    feats, sid = ctx["sh"].read(0)
    assert set(feats) == set(FEATURE_KEYS)
    b = ctx["sh"].batch([0], packed=True)
    assert set(b) == {"features", "sample_id"} and "group" not in b["features"]
    assert not any(k in b["features"] for k in ("sample_id", "scene_id", "subset", "weight", "sources"))


# ----------------------------------------------------------------------------------------------- audits
def test_no_free_windows_flagged_never_zero(ctx):
    join = ctx["join"]
    sel = np.flatnonzero(join["no_free"])
    assert len(sel) > 0
    j = int(sel[0])
    sc = ctx["refs"][int(join["scene"][j])]
    tp = TeacherPhysics(sc, threads=4)
    o, L = (int(join["x"][j]), int(join["y"][j]), int(join["z"][j])), int(join["L"][j])
    b = tp.receiver_bank(o, L, "uniform", 64)
    assert b["status"] == "no_free" and b["k"] == 0
    with pytest.raises(NoFreeReceivers):
        tp.receiver_bank(o, L, "uniform", 64, strict=True)
    s, a = sc.privileged_labels()
    obs, solid = tp.observation(o, L)
    f = tp.full_mismatch(s, a, o, L, obs)
    assert f["status"] == "no_free" and np.isnan(f["mismatch_fraction"]) and f["mismatch_count"] is None
    assert np.isnan(tp.sampled_mismatch(s, a, b, obs)["estimate_fraction"])
    assert join["category"][j] == 0
    tp.close()


def test_ma_permutation_invariance(ctx):
    i, j = pick(ctx, 1, 64, lambda r: r["category"] == 3 and r["K"] >= 3, seed=7)
    feats, _, sc, tp = student_and_teacher(ctx, i, j)
    s, a = sc.privileged_labels()
    o = tuple(int(v) for v in feats["window_origin"])
    bank = tp.receiver_bank(o, 64, "uniform", 1024, seed=2)
    v1, _ = tp.response(s, a, bank["world_xyz"])
    perm = np.random.default_rng(0).permutation(len(s))
    v2, _ = tp.response(s[perm], a[perm], bank["world_xyz"])
    assert np.max(np.abs(v1 - v2)) <= MA_TOL
    m1 = tp.sampled_mismatch(s[perm], a[perm], bank, feats["observed_response"])
    assert m1["estimate_count"] == 0
    row = ctx["join"][j]
    K = int(row["K"])
    assert np.all(np.diff(row["sources"][:K, 0]) >= 0)                      # stored as a canonical (sorted) set
    tp.close()


def test_error_messages(ctx):
    i, j = pick(ctx, 0, 64)
    feats, _, sc, tp = student_and_teacher(ctx, i, j)
    with pytest.raises(IllegalSource, match=r"indices \[1\]"):
        tp.visibility([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]] if tp.legal([[0.5, 0.5, 0.5]])[0] else [[0.999, 0.999, 0.999], [1.5, 0.5, 0.5]], [[0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="strengths"):
        tp.response(ctx["join"]["sources"][j][:1], [1.0, 0.5], [[0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="unknown backend"):
        tp.window(ctx["join"]["sources"][j][:1], (0, 0, 0), 64, backend="gpu")
    with pytest.raises(ValueError, match="number of strata"):
        tp.receiver_bank(tuple(feats["window_origin"]), 64, "stratified", 1, response=np.tile([True, False], 64**3//2).reshape(64, 64, 64))
    tp.close()


def test_sampler_deterministic_balanced_bucketed(ctx):
    sh = ctx["sh"]
    s = SceneBalancedSampler(sh.index, {64: 32, 128: 8, 256: 2}, seed=7, per_scene={64: 10, 128: 10, 256: 4})
    e0, e0b, e1 = s.epoch(0), s.epoch(0), s.epoch(1)
    assert all(np.array_equal(a, b) for a, b in zip(e0, e0b)) and len(e0) == len(e0b)
    assert not all(len(a) == len(b) and np.array_equal(a, b) for a, b in zip(e0, e1))
    for b in e0:
        assert len(set(sh.index["L"][b])) == 1 and len(set(sh.index["modality"][b])) == 1
    chosen = np.concatenate(e0)
    assert len(np.unique(chosen)) == len(chosen)
    for L, m in ((64, 10), (128, 10), (256, 4)):
        g = sh.index["group"][chosen][sh.index["L"][chosen] == L]
        _, counts = np.unique(g, return_counts=True)
        assert counts.max() <= m and np.median(counts) == m


def test_v02_prefix_and_preserved_policy():
    s = json.loads((OUT/"privileged"/"index"/"canary-v0.2"/"P256"/"summary.json").read_text())
    assert s["v02_prefix_identical"] == f"{s['scenes']}/{s['scenes']}"


def test_space_to_depth_and_blocksparse_exact(ctx):
    from rind3d_training.blocksparse import decode, encode
    i, j = pick(ctx, 0, 256, lambda r: r["category"] == 3, seed=8)
    feats, _ = ctx["sh"].read(i)
    for B in (8, 16):
        code, pay = encode(feats["observed_response"], feats["observed_solid"], B)
        r, s = decode(code, pay, B)
        assert np.array_equal(r, feats["observed_response"]) and np.array_equal(s, feats["observed_solid"])
    x = feats["observed_response"]
    f, n = 4, 64
    y = x.reshape(n, f, n, f, n, f).transpose(1, 3, 5, 0, 2, 4).reshape(f**3, n, n, n)
    back = y.reshape(f, f, f, n, n, n).transpose(3, 0, 4, 1, 5, 2).reshape(256, 256, 256)
    assert np.array_equal(back, x)
