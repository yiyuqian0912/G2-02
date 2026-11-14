"""Gate A: the exact GPU verifier (gpuverify / verify.cu) against the certified CUDA renderer.

Production port of experiments/fastlabel/validate_verify.py (GPUVERIFY.md): same candidate kinds and checks, using
only this package and the rind3d backend (reference: BackendFactory.source(..., "cuda_certified").packed_batch over
ALL 64^3 tiles of the window, no culling, exact UNKNOWN fallback).

Per window (a FastLabelWindow from any adapter), candidates:
* uniform in the phase-1 outer bound B (BSampler), near the true source (scales 1e-9..1e-3), near engine vertices
  (+1e-12), uniform in the world, within ~1 ulp of a body face plane (forces uncertifiable signs and exact host rows).
For every legal candidate (exact legality, outside the closed window box):
* dump mode: the predicted inside/outside bit of EVERY FREE voxel (UNKNOWN voxels decided by the native exact
  predicate) must equal the reference lit field bit for bit;
* batch mode (all candidates at once): the pass/fail decision must agree with the reference, and every reported violated
  receiver must be a true mismatch of the reference.
Gate A passes iff field_diff = batch_disagree = bad_violation = 0 on every window.

  python -m rind3d_posterior.fastlabel.gate_a <out.jsonl> {1k|100k}:<scene_dir_or_container>:<w>:<s> ...
Validation builds: FASTLABEL_VERIFY_DEFINES="E32=1e-3f" (forces the FP64 interval path) or "UNK_BAND=2e-4f" (forces the
native GMP path); see GPUVERIFY.md."""
import json
import sys
import time
from fractions import Fraction

import numpy as np

from . import engine as T
from . import gpuverify as GV
from .inputs import legal_source, window_from_1k, window_from_100k
from .pipeline import process_factory, task_context
from .sampler import BSampler
from .window import LazyWindow2, window_classes


def unpack(bits, L):
    """bool [z, y, x] of a stored packed crop."""
    return np.unpackbits(np.asarray(bits, np.uint8), count=L ** 3).astype(bool).reshape(L, L, L)


def reference_lit(fac, bundle, policy, p, origin, L, N):
    """Certified CUDA renderer: lit field of source p over the whole window, bool [z, y, x]."""
    o = np.asarray(origin, np.int64); n = max(L // 64, 1)
    offs = [(a, b, c) for c in range(n) for b in range(n) for a in range(n)]
    src = fac.source(bundle, np.asarray(p, "<f8"), "cuda_certified", policy)
    try:
        out = src.packed_batch(np.array([o + np.array(t) * 64 for t in offs], np.int32), 64, int(N))
    finally:
        src.close()
    ref = np.zeros((L, L, L), bool)
    for (a, b, c), (pl, _) in zip(offs, out):
        ref[c * 64:(c + 1) * 64, b * 64:(b + 1) * 64, a * 64:(a + 1) * 64] = \
            np.unpackbits(np.frombuffer(pl, np.uint8), bitorder="little").reshape(64, 64, 64).astype(bool)
    return ref


def plane_points(ctx, rng, n):
    """Points within ~1 ulp of the plane of a random body face (exercises the uncertifiable-sign host-row path)."""
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


def candidates(task, res, ctx, rng, n_b=40, n_near=16, n_vert=12, n_world=12, n_plane=24):
    pts, kinds = [], []
    P = plane_points(ctx, rng, n_plane); pts += P; kinds += ["plane"] * len(P)
    smp = BSampler(res, rng, {})
    if smp.ok == "ok" and n_b:
        P = smp.draw(n_b); pts += list(P); kinds += ["B"] * len(P)
    src = np.asarray(task.bind_point, float)
    for sc in np.logspace(-9, -3, n_near):
        pts.append(src + rng.normal(0, sc, 3)); kinds.append(f"near{sc:.0e}")
    V = res["vpool"]
    if len(V) and n_vert:
        for i in rng.choice(len(V), min(n_vert, len(V)), replace=False):
            pts.append(V[i] + rng.normal(0, 1e-12, 3)); kinds.append("vertex")
    for _ in range(n_world):
        pts.append(rng.random(3)); kinds.append("world")
    return np.array(pts), kinds


def check_window(task, seed=0, **counts):
    t0 = time.perf_counter()
    fac = process_factory(); scn = task_context(task); ctx = scn["ctx"]
    L, N = int(task.L), int(task.N); o = np.asarray(task.origin, np.int64)
    cls = window_classes(task); win = LazyWindow2(cls, o, N)
    roots, _ = T.stage1(scn["geom"], win); xc, xk = T.point_items(win, 1024, 512, 0)
    res = T.stage2(scn["geom"], win, xc, xk, roots)
    V = GV.WindowVerifier(fac, ctx, task, cls)
    rng = np.random.default_rng(seed)
    pts, kinds = candidates(task, res, ctx, rng, **counts)
    sol = unpack(task.solid_bits, L); lit_obs = unpack(task.lit_bits, L); free = ~sol
    rec = dict(key=list(task.key), scene=str(task.scene_key), L=L, bodies=V.G.nb, n=len(pts), legal=0, field_equal=0,
               field_diff=0, batch_agree=0, batch_disagree=0, bad_violation=0, passes=0, unknown=0, kinds={})
    refs = {}
    for i, p in enumerate(pts):
        if legal_source(ctx, p) is None or np.all((p >= o / N) & (p <= (o + L) / N)):
            continue
        rec["legal"] += 1
        ref = refs[i] = reference_lit(fac, task.bundle, task.policy, p, o, L, N)
        V.test_batch(p[None], dump=True)
        d = V.dump.reshape(L, L, L).astype(np.int16)                # [z, y, x]: 0/1 inside, 2 unknown, 3 not evaluated
        unk = np.argwhere((d == 2) & free)
        if len(unk):
            rec["unknown"] += len(unk)
            cz, cy, cx = unk.T
            yy, _, _ = ctx.bind(p).native.points(np.ascontiguousarray((np.stack([cx, cy, cz], 1) + o + 0.5) / N, "<f8"), False)
            d[cz, cy, cx] = 1 - yy
        same = bool(np.all(d[free] <= 1) and np.array_equal(d[free] == 0, ref[free]))
        rec["field_equal" if same else "field_diff"] += 1
        rec["kinds"].setdefault(kinds[i], [0, 0]); rec["kinds"][kinds[i]][0 if same else 1] += 1
    ids = sorted(refs)
    got = V.test_batch(pts[ids]) if ids else []
    for j, i in enumerate(ids):
        mis = free & (refs[i] != lit_obs)
        ok_ref = not mis.any()
        ok, vox, _ = got[j]
        rec["passes"] += ok_ref
        rec["batch_agree" if ok == ok_ref else "batch_disagree"] += 1
        if vox is not None:
            x, y, z = vox.T
            rec["bad_violation"] += int(np.sum(~mis[z, y, x]))
    rec["stats"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in V.stats.items()}
    rec["ok"] = rec["field_diff"] == 0 and rec["batch_disagree"] == 0 and rec["bad_violation"] == 0
    rec["wall"] = time.perf_counter() - t0
    V.close()
    return rec


def task_from_spec(spec):
    fmt, rest = spec.split(":", 1)
    path, w, s = rest.rsplit(":", 2)
    return (window_from_1k if fmt == "1k" else window_from_100k)(path, int(w), int(s))


def main(argv=None):
    import zlib
    argv = sys.argv[1:] if argv is None else argv
    out = open(argv[0], "a"); ok = True
    for spec in argv[1:]:
        rec = check_window(task_from_spec(spec), seed=zlib.crc32(spec.encode()))
        rec["spec"] = spec; ok &= rec["ok"]
        print(json.dumps(rec), flush=True); out.write(json.dumps(rec) + "\n"); out.flush()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
