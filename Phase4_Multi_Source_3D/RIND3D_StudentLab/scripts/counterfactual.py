"""Counterfactual hidden worlds with an identical Student observation X (brief §8).

usage (labeller interpreter: rind3d built, numba, scipy, CUDA):
  counterfactual.py <out_dir> [--n-per-side 80] [--workers 4] [--seed 20261114]

Construction, exact for X:
* take a dev row (NAT-1K natural window W of a pilot-select scene, boundary-mixed channel j, true source s_j);
* H = conv({s_j} ∪ closed window box). Every segment from s_j to a receiver of W lies in H;
* G' = G minus every convex body that is disjoint from H, with a 2-voxel margin (LP feasibility test on the body's
  half-space rows). Hence:
  * the SOLID crop of W is unchanged (the removed bodies miss the box);
  * V_G'(s_j, x) = V_G(s_j, x) on every receiver x (the removed bodies miss every segment);
  * X = (SOLID ∩ W, r_j, L, o) is identical;
  * s_j stays legal in G';
* q_G' = Uniform(F_W(G')) is labelled with the production FastLabel v3.4 pipeline (same defaults and certificate as
  the 1K backfill), from the regenerated canonical leaves of the kept bodies (geometry hash of G checked).

q_G (stored label) and q_G' are full-world posteriors of two worlds that the Student cannot tell apart. For any X-only
predictor p, [KL(q_G || p) + KL(q_G' || p)] / 2 >= JSD(q_G, q_G'), attained at p = (q_G + q_G') / 2. So JSD is a
floor on the pair-averaged KL of every Student.
Writes <out>/cf_<scene>_w<w>_s<j>.npz (cell, lo, status, tv_bound, removed bodies, meta) and <out>/cf_index.jsonl."""
import argparse, glob, json, os, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB))
from slab.paths import FEATS, NAT, REL   # noqa: E402

MARGIN = 2.0 / 1024


def load_rows():
    shards = sorted(glob.glob(str(FEATS / "shard*.npy")), key=lambda p: int(Path(p).stem[5:]))
    return np.concatenate([np.load(p) for p in shards])


def leaves_of(meta):
    """Regenerate the scene's canonical leaves (recipe is deterministic); returns (leaves, geometry_id)."""
    import rind3d.lite_generators as LG
    from rind3d_1k import families
    cap = []; orig = LG.make_leaf
    def rec(*a, **k):
        it = orig(*a, **k); cap.append(it); return it
    LG.make_leaf = rec; families.make_leaf = rec
    try:
        _, recipe = families.generate(meta["seed"], meta["slot"], meta["family"], meta["attempt"])
    finally:
        LG.make_leaf = orig; families.make_leaf = orig
    return cap, recipe["geometry_id"]


def bundle_of(leaves):
    from rind.contracts import CSGNode, CSGOpcode, Geometry, RootKind
    from rind.storage import CanonicalBundle, pack_arrays, _decode_bundle
    b = CanonicalBundle(3); roots = []
    for it in leaves:
        b.add_leaf(it); roots.append(it.content_hash)
    if len(roots) > 1:
        node = CSGNode(3, CSGOpcode.UNION, tuple(roots)); b.add_node(node); root = node.content_hash
    else:
        root = roots[0]
    b.add_geometry(Geometry(3, RootKind.REF, root))
    return _decode_bundle(pack_arrays(b), 3)


def rows_of(ctx):
    B = ctx.bodies
    K = np.array([len(b) for b in B], np.int32); P = np.zeros((len(B), int(K.max()), 4)); P[..., 3] = -1.0
    for i, b in enumerate(B):
        P[i, :len(b)] = [[float(v) for v in r] for r in b]
    return P, K


def touches_hull(rows, V):
    """True if the body {x : a.x + d <= margin |a|} meets conv(V) (LP feasibility over the hull weights)."""
    from scipy.optimize import linprog
    A = rows[:, :3]; d = rows[:, 3]; na = np.linalg.norm(A, axis=1)
    # variables lambda (|V|): A (V^T lambda) <= -d + margin |a|, sum lambda = 1, lambda >= 0
    res = linprog(np.zeros(len(V)), A_ub=A @ V.T, b_ub=-d + MARGIN * na, A_eq=np.ones((1, len(V))), b_eq=[1.0],
                  bounds=[(0, None)] * len(V), method="highs")
    return res.status == 0


def job(args):
    rows_sub, out = args
    from rind3d_1k.fastlabel import DEFAULTS
    from rind3d_posterior.fastlabel.inputs import FastLabelWindow, legal_source
    from rind3d_posterior.fastlabel.pipeline import label_window_fast, process_factory
    import zstandard
    res = []
    for r in rows_sub:
        sc, w, j = int(r["scene"]), int(r["window"]), int(r["source"])
        p = Path(out) / f"cf_{sc:06d}_w{w:03d}_s{j}.npz"
        if p.exists():
            continue
        t0 = time.perf_counter()
        sd = NAT / "scenes" / f"{sc:06d}"
        meta = json.loads((sd / "scene.json").read_text())
        leaves, gid = leaves_of(meta)
        assert gid == meta["geometry_id"], "regenerated geometry differs"
        fac = process_factory()
        ctx = fac.context(bundle_of(leaves), _policy())
        P, K = rows_of(ctx)
        N = int(meta["resolution"])
        s = np.array([r["sx"], r["sy"], r["sz"]], float)
        o = np.array([r["ox"], r["oy"], r["oz"]], float) / N; L = float(r["Lp"]) / N
        V = np.array([s] + [o + L * np.array([a, b, c]) for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        # bodies and leaves are in the same order (one leaf per convex body); checked by the row count
        assert len(K) == len(leaves)
        keep = [i for i in range(len(K)) if touches_hull(P[i, :K[i]], V)]
        removed = [i for i in range(len(K)) if i not in keep]
        rec = dict(scene=sc, window=w, source=j, Lp=int(r["Lp"]), n_bodies=len(K), removed=removed, family=meta["family"])
        if not removed:
            rec["status"] = "nothing_removable"; res.append(rec); np.savez(p, meta=json.dumps(rec)); continue
        b2 = bundle_of([leaves[i] for i in keep]); ctx2 = fac.context(b2, _policy()); P2, K2 = rows_of(ctx2)
        assert legal_source(ctx2, s) is not None, "true source not legal in G'"
        W = np.load(sd / "public" / "windows.npy"); idx = np.load(sd / "public" / "observations_index.npy")
        pos = {(int(a), int(b)): (int(oo), int(n)) for a, b, oo, n in idx}; blob = (sd / "public" / "observations.bin").read_bytes()
        dz = zstandard.ZstdDecompressor()
        raw = lambda jj: np.frombuffer(dz.decompress(blob[pos[(w, jj)][0]:pos[(w, jj)][0] + pos[(w, jj)][1]]), np.uint8)
        wr = W[w]
        task = FastLabelWindow(P=P2, K=K2, bundle=b2, origin=(int(wr["x"]), int(wr["y"]), int(wr["z"])), L=int(wr["L"]), N=N,
                               lit_bits=raw(j), solid_bits=raw(-1), key=(sc, w, j), scene_key=f"cf:{sc}:{w}:{j}", bind_point=s)
        lab = label_window_fast(task, **DEFAULTS)
        rec.update(status=lab["status"], tv_bound=float(lab["tv_bound"]), wall=time.perf_counter() - t0, path=lab["meta"].get("path"),
                   n_cells=int(len(np.unique(lab["cell"]))))
        np.savez_compressed(p, cell=np.asarray(lab["cell"], np.int32), lo=np.asarray(lab["lo"], float), meta=json.dumps(rec, default=float))
        res.append(rec)
    return res


def _policy():
    from rind3d_posterior.fastlabel.inputs import POLICY
    return dict(POLICY)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n-per-side", type=int, default=80)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=20261114)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    rng = np.random.default_rng(a.seed); pick = []
    for L in (64, 128, 256):
        m = np.flatnonzero((rows["role"] == 1) & (rows["ds"] == 1) & (rows["type"] == 0) & (rows["Lp"] == L))   # dev only
        pick += list(rng.choice(m, min(a.n_per_side, len(m)), replace=False))
    pick = np.sort(np.array(pick))
    (out / "cf_rows.json").write_text(json.dumps([int(i) for i in pick]))
    chunks = [rows[pick[i::a.workers]] for i in range(a.workers)]
    with get_context("spawn").Pool(a.workers) as pool:            # one CUDA runtime per process
        for rs in pool.imap_unordered(job, [(c, str(out)) for c in chunks]):
            with open(out / "cf_index.jsonl", "a") as f:
                for r in rs:
                    f.write(json.dumps(r, default=float) + "\n")
    print(json.dumps(dict(out=str(out), rows=int(len(pick)))))


if __name__ == "__main__":
    main()
