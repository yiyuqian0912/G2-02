"""Depth counterfactuals: hidden casters scaled about the true source (brief §8, stronger construction).

usage (labeller interpreter): counterfactual_depth.py <out_dir> [--n-per-side 40] [--workers 4] [--lams 0.6,0.8,1.25,1.6]

For a dev row (natural window W, channel j, true source s):
* hidden casters are the convex bodies that meet H = conv({s} ∪ closed box) but not the box (2-voxel margin);
* G'_λ replaces every hidden caster O by its homothety about s, O' = s + λ (O - s) (leaf centre and matrix scaled;
  normals and offsets unchanged). The shadow cone of O from s is invariant, so the response of s in W should be unchanged;
* accepted only if all of these hold:
  * every O' stays outside the closed box (LP, margin), so the SOLID crop is identical;
  * O' stays inside the cube;
  * s is legal in G';
  * **the exact FastLabel verifier confirms that s reproduces the stored observation on every FREE receiver of W**.

  Then X is identical, by test and not by assumption;
* q'_λ = Uniform(F_W(G'_λ)) is labelled with FastLabel v3.4 (backfill defaults).

λ < 1 moves a caster toward the source and λ > 1 toward the window: the hidden depth of the occluders, which X cannot see.
Writes <out>/cfd_<scene>_w<w>_s<j>_l<λ>.npz and cfd_index.jsonl (one record per row and λ, accepted or with a reason)."""
import argparse, json, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from counterfactual import load_rows, bundle_of, rows_of, touches_hull, _policy, MARGIN   # noqa: E402
from slab.paths import NAT                                                                # noqa: E402


def leaf_calls(meta):
    """(leaf items, make_leaf call arguments) of the scene's recipe, in body order."""
    import rind3d.lite_generators as LG
    from rind3d_1k import families
    items, calls = [], []; orig = LG.make_leaf
    def rec(*a, **k):
        it = orig(*a, **k); items.append(it); calls.append((a, {kk: np.array(v) for kk, v in k.items()})); return it
    LG.make_leaf = rec; families.make_leaf = rec
    try:
        _, recipe = families.generate(meta["seed"], meta["slot"], meta["family"], meta["attempt"])
    finally:
        LG.make_leaf = orig; families.make_leaf = orig
    assert recipe["geometry_id"] == meta["geometry_id"]
    return items, calls


def inside_cube(rows, lo=0.002, hi=0.998):
    from scipy.optimize import linprog
    A = rows[:, :3]; d = rows[:, 3]
    for i in range(3):
        for sgn in (1, -1):
            c = np.zeros(3); c[i] = -sgn
            r = linprog(c, A_ub=A, b_ub=-d, bounds=[(None, None)] * 3, method="highs")
            if r.status != 0:
                return False
            v = r.x[i]
            if not (lo < v < hi):
                return False
    return True


def job(args):
    rows_sub, out, lams = args
    import zstandard
    import rind3d.lite_generators as LG
    from rind3d_1k.fastlabel import DEFAULTS
    from rind3d_posterior.fastlabel.inputs import FastLabelWindow, legal_source
    from rind3d_posterior.fastlabel.pipeline import label_window_fast, process_factory
    from rind3d_posterior.fastlabel.window import window_classes
    from rind3d_posterior.fastlabel import gpuverify as GV
    fac = process_factory(); res = []
    for r in rows_sub:
        sc, w, j = int(r["scene"]), int(r["window"]), int(r["source"])
        sd = NAT / "scenes" / f"{sc:06d}"; meta = json.loads((sd / "scene.json").read_text())
        items, calls = leaf_calls(meta)
        P, K = rows_of(fac.context(bundle_of(items), _policy())); assert len(K) == len(items)
        N = int(meta["resolution"]); s = np.array([r["sx"], r["sy"], r["sz"]], float)
        o = np.array([r["ox"], r["oy"], r["oz"]], float) / N; L = float(r["Lp"]) / N
        corners = np.array([o + L * np.array([a, b, c]) for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        H = np.vstack([s[None], corners])
        in_box = [touches_hull(P[i, :K[i]], corners) for i in range(len(K))]
        casters = [i for i in range(len(K)) if not in_box[i] and touches_hull(P[i, :K[i]], H)]
        W = np.load(sd / "public" / "windows.npy"); idx = np.load(sd / "public" / "observations_index.npy")
        pos = {(int(a), int(b)): (int(oo), int(n)) for a, b, oo, n in idx}; blob = (sd / "public" / "observations.bin").read_bytes()
        dz = zstandard.ZstdDecompressor()
        raw = lambda jj: np.frombuffer(dz.decompress(blob[pos[(w, jj)][0]:pos[(w, jj)][0] + pos[(w, jj)][1]]), np.uint8)
        wr = W[w]
        for lam in lams:
            p = Path(out) / f"cfd_{sc:06d}_w{w:03d}_s{j}_l{lam:g}.npz"
            rec = dict(scene=sc, window=w, source=j, Lp=int(r["Lp"]), lam=lam, family=meta["family"], casters=casters,
                       n_bodies=len(K))
            if p.exists():                         # resume: keep the stored record so the index stays complete
                with np.load(p) as z:
                    res.append(json.loads(str(z["meta"])))
                continue
            t0 = time.perf_counter()
            if not casters:
                rec["status"] = "no_hidden_caster"; res.append(rec); continue
            new = list(items); bad = None
            for i in casters:
                a, k = calls[i]; k2 = dict(k)
                k2["center"] = np.asarray(s + lam * (k["center"] - s), dtype="<f8"); k2["matrix"] = np.asarray(lam * k["matrix"], dtype="<f8")
                new[i] = LG.make_leaf(*a, **k2)
            try:
                b2 = bundle_of(new); ctx2 = fac.context(b2, _policy()); P2, K2 = rows_of(ctx2)
            except Exception as ex:                 # e.g. a scaled body outside the geometry kernel's world scope
                rec["status"] = f"geometry_unsupported:{type(ex).__name__}"; res.append(rec); np.savez(p, meta=json.dumps(rec)); continue
            for i in casters:
                if touches_hull(P2[i, :K2[i]], corners):
                    bad = "caster_enters_box"; break
                if not inside_cube(P2[i, :K2[i]]):
                    bad = "caster_leaves_cube"; break
            if bad is None and legal_source(ctx2, s) is None:
                bad = "source_illegal"
            if bad is None:
                task = FastLabelWindow(P=P2, K=K2, bundle=b2, origin=(int(wr["x"]), int(wr["y"]), int(wr["z"])), L=int(wr["L"]), N=N,
                                       lit_bits=raw(j), solid_bits=raw(-1), key=(sc, w, j), scene_key=f"cfd:{sc}:{w}:{j}:{lam}",
                                       bind_point=s)
                V = GV.WindowVerifier(fac, ctx2, task, window_classes(task))
                ok, vox, why = V.test_batch(s[None])[0]
                if not ok:
                    bad = f"observation_changed:{why}:{0 if vox is None else len(vox)}"
            if bad is not None:
                rec["status"] = bad; res.append(rec); np.savez(p, meta=json.dumps(rec)); continue
            try:
                lab = label_window_fast(task, **DEFAULTS)
            except Exception as ex:
                rec["status"] = f"label_error:{type(ex).__name__}"; res.append(rec); np.savez(p, meta=json.dumps(rec)); continue
            rec.update(status=lab["status"], tv_bound=float(lab["tv_bound"]), wall=time.perf_counter() - t0, path=lab["meta"].get("path"),
                       n_cells=int(len(np.unique(lab["cell"]))))
            np.savez_compressed(p, cell=np.asarray(lab["cell"], np.int32), lo=np.asarray(lab["lo"], float), meta=json.dumps(rec, default=float))
            res.append(rec)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n-per-side", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--lams", default="0.6,0.8,1.25,1.6")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); lams = [float(x) for x in a.lams.split(",")]
    rows = load_rows(); rng = np.random.default_rng(a.seed); pick = []
    for L in (64, 128, 256):
        m = np.flatnonzero((rows["role"] == 1) & (rows["ds"] == 1) & (rows["type"] == 0) & (rows["Lp"] == L))   # dev only
        pick += list(rng.choice(m, min(a.n_per_side, len(m)), replace=False))
    pick = np.sort(np.array(pick)); (out / "cfd_rows.json").write_text(json.dumps([int(i) for i in pick]))
    chunks = [rows[pick[i::a.workers]] for i in range(a.workers)]
    (out / "cfd_index.jsonl").write_text("")
    with get_context("spawn").Pool(a.workers) as pool:
        for rs in pool.imap_unordered(job, [(c, str(out), lams) for c in chunks]):
            with open(out / "cfd_index.jsonl", "a") as f:
                for r in rs:
                    f.write(json.dumps(r, default=float) + "\n")
    print(json.dumps(dict(out=str(out), rows=int(len(pick)))))


if __name__ == "__main__":
    main()
