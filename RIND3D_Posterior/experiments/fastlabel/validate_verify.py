"""Gate A: the batched GPU verifier (gpuverify / verify.cu) against the reference certified renderer.

For each window: candidate sources of several kinds (uniform in the phase-1 outer bound B, near the true source at
scales 1e-9..1e-3, near B-piece vertices, uniform legal points in the world). For every legal candidate:
* reference: gpurender.render_tiles over ALL tiles (no culling; exact UNKNOWN fallback) -> lit bits of the window;
* new, per point (dump mode): the predicted inside/outside bit of EVERY FREE voxel, UNKNOWN voxels decided by the
  native exact predicate -> must equal the reference lit field bit for bit;
* new, batched: the pass/fail decision and the exact definite-mismatch count for the whole batch at once; every
  reported violated receiver must be a true mismatch of the reference.
  validate_verify.py <out.jsonl> sc:w:s [sc:w:s ...]"""
import json, sys, time
import numpy as np
sys.path[:0] = [".", "../twophase"]
import twophase as T, adaptive2 as A, gpurender, gpuverify as GV
from fastwin import fast_cls, LazyWindow
from stat_tier import sample_in_B
from rind3d_1k.physical import POLICY, legal_source


def ref_field(tester, p):
    lit, sol = gpurender.render_tiles(tester.dg, tester.ctx, p, tester.origins)
    L = tester.L; out = np.zeros((L, L, L), bool)
    for t, (a, b, c) in enumerate(tester.offs):
        out[c * 64:(c + 1) * 64, b * 64:(b + 1) * 64, a * 64:(a + 1) * 64] = np.unpackbits(lit[t], bitorder="little").reshape(64, 64, 64).astype(bool)
    return out                                                   # [z, y, x]


def plane_points(ctx, rng, n):
    """Points within ~1 ulp of the plane of a random body face (exercises the uncertifiable-sign host-row path)."""
    from fractions import Fraction
    out = []
    for _ in range(n):
        b = ctx.bodies[rng.integers(len(ctx.bodies))]; r = b[rng.integers(len(b))]
        k = max(range(3), key=lambda a: abs(r[a]))
        q = [Fraction(float(v)) for v in rng.random(3)]
        q[k] = Fraction(0)
        q[k] = -(r[3] + sum(r[a] * q[a] for a in range(3))) / r[k]
        p = np.array([float(v) for v in q])
        if np.all((p > 0) & (p < 1)):
            out.append(p)
    return out


def candidates(res, S, w, s, rng, n_b=40, n_near=16, n_vert=12, n_world=12, ctx=None, n_plane=24):
    pts, kinds = [], []
    if ctx is not None:
        P = plane_points(ctx, rng, n_plane); pts += P; kinds += ["plane"] * len(P)
    P, _ = sample_in_B(res, n_b, rng)
    if P is not None:
        pts += list(P); kinds += ["B"] * len(P)
    src = np.array([float.fromhex(v) for v in json.load(open(T.R / S.dir.name / "privileged" / "sources.json"))["sources_hex"][s]])
    for sc in np.logspace(-9, -3, n_near):
        pts.append(src + rng.normal(0, sc, 3)); kinds.append(f"near{sc:.0e}")
    V = res["vpool"]
    if len(V):
        for i in rng.choice(len(V), min(n_vert, len(V)), replace=False):
            pts.append(V[i] + rng.normal(0, 1e-12, 3)); kinds.append("vertex")
    for _ in range(n_world):
        pts.append(rng.random(3)); kinds.append("world")
    return np.array(pts), kinds


def main():
    out = open(sys.argv[1], "a")
    for spec in sys.argv[2:]:
        sc, w, s = spec.split(":"); w = int(w); s = int(s)
        t0 = time.perf_counter()
        scn = A.scene_cache(sc); S = scn["S"]; geom = scn["geom"]
        cls = fast_cls(S, w, s); win = LazyWindow(cls, np.asarray(S.origin(w), np.int64), 1024)
        roots, _ = T.stage1(geom, win); xc, xk = T.point_items(win, 1024, 512, 0)
        res = T.stage2(geom, win, xc, xk, roots)
        tester = A.Tester(scn["bundle"], POLICY, S.crop(w, s), S.crop(w, -1), S.origin(w))
        V = GV.WindowVerifier(A.process_factory(), tester.ctx, S, w, s, cls)
        rng = np.random.default_rng(hash(spec) % 2 ** 32)
        pts, kinds = candidates(res, S, w, s, rng, ctx=tester.ctx)
        sol = S.crop(w, -1); lit_obs = S.crop(w, s); free = ~sol
        rec = dict(spec=spec, L=win.L, bodies=V.G.nb, n=len(pts), legal=0, field_equal=0, field_diff=0, batch_agree=0,
                   batch_disagree=0, count_equal=0, count_diff=0, bad_violation=0, passes=0, unknown=0, kinds={})
        refs = {}
        for i, p in enumerate(pts):
            if legal_source(tester.ctx, p) is None or np.all((p >= tester.o / 1024) & (p <= (tester.o + tester.L) / 1024)):
                continue
            rec["legal"] += 1
            if tester.dg is None:
                tester.dg = gpurender.device_geometry(tester.fac, tester.bundle, POLICY, p)
            ref = ref_field(tester, p)
            refs[i] = ref
            V.test_batch(p[None], dump=True)
            d = V.dump.reshape(win.L, win.L, win.L)              # [z, y, x]: 0/1 inside, 2 unknown, 3 not evaluated
            unk = np.argwhere((d == 2) & free)
            pred_in = d.copy()
            if len(unk):
                rec["unknown"] += len(unk)
                cpu = tester.ctx.bind(p)
                cz, cy, cx = unk.T
                yy, _, _ = cpu.native.points(np.ascontiguousarray((np.stack([cx, cy, cz], 1) + tester.o + 0.5) / 1024, "<f8"), False)
                pred_in[cz, cy, cx] = 1 - yy
            ok_eval = np.all(pred_in[free] <= 1)
            same = ok_eval and np.array_equal((pred_in[free] == 0), ref[free])
            rec["field_equal" if same else "field_diff"] += 1
            rec["kinds"].setdefault(kinds[i], [0, 0]); rec["kinds"][kinds[i]][0 if same else 1] += 1
        # batched decisions for all legal points at once
        ids = sorted(refs)
        got = V.test_batch(pts[ids])
        for j, i in enumerate(ids):
            ref = refs[i]
            mis = free & (ref != lit_obs)
            ok_ref = not mis.any()
            ok, vox, why = got[j]
            rec["passes"] += ok_ref
            rec["batch_agree" if ok == ok_ref else "batch_disagree"] += 1
            if vox is not None:
                x, y, z = vox.T
                rec["bad_violation"] += int(np.sum(~mis[z, y, x]))
        # exact definite-mismatch counts (batched, raw kernel counters) vs reference
        rec["stats"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in V.stats.items()}
        rec["wall"] = time.perf_counter() - t0
        print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
        V.close()


if __name__ == "__main__":
    main()
