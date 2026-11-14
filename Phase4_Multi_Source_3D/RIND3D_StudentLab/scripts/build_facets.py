"""Build the X-measurable facet table for every row of the pilot feature table (all roles except the sealed test split,
which the feature table never contains).

usage: build_facets.py [--workers 12] [--out PATH]
For each row (one window channel): the lit/dark FREE interface of the full-resolution crop (6-neighbour pairs of FREE
receivers with different response) is segmented into up to 8 planar facets by greedy local RANSAC (slab/facets.py).
Only the crop's SOLID mask and the channel's own response are read: the facets are a deterministic function of X.
Output npz: F (n_rows, 8, 12) float32 [mu_world_vox(3), normal(3), n_inliers, rms, sd1, sd2, remaining, 0], nf (n_rows,),
npts (n_rows,), plus provenance. Resumable per scene (part files in <out>.parts/)."""
import argparse, glob, json, os, sys, time, hashlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slab.paths import FEATS, REL, NAT, FACETS          # noqa: E402
from slab.facets import facets_of                        # noqa: E402

KMAX = 8
ROOT = {0: REL / "scenes", 1: NAT / "scenes"}


def load_rows():
    shards = sorted(glob.glob(str(FEATS / "shard*.npy")), key=lambda p: int(Path(p).stem[5:]))
    return np.concatenate([np.load(p) for p in shards])


def crops(d):
    W = np.load(d / "public" / "windows.npy"); idx = np.load(d / "public" / "observations_index.npy")
    pos = {(int(w), int(s)): (int(o), int(n)) for w, s, o, n in idx}
    blob = (d / "public" / "observations.bin").read_bytes(); dz = zstandard.ZstdDecompressor()
    Ls = {int(r["window"]): int(r["L"]) for r in W}

    def get(w, s):
        o, n = pos[(w, s)]; L = Ls[w]
        return np.unpackbits(np.frombuffer(dz.decompress(blob[o:o + n]), np.uint8), count=L ** 3).reshape(L, L, L).astype(bool)
    return get


def job(args):
    (ds, sc), idxs, rows_sub, part = args
    if os.path.exists(part):
        return part
    get = crops(ROOT[ds] / f"{sc:06d}")
    F = np.zeros((len(idxs), KMAX, 12), np.float32); nf = np.zeros(len(idxs), np.int16); npts = np.zeros(len(idxs), np.int64)
    for j, r in enumerate(rows_sub):
        if r["type"] != 0:                  # no FREE lit/dark adjacency: solid-separated, all-lit, all-dark
            continue
        f, n = facets_of(get(int(r["window"]), -1), get(int(r["window"]), int(r["source"])), max_facets=KMAX)
        o = np.array([r["ox"], r["oy"], r["oz"]], float)
        f = f.copy(); f[:, :3] += o
        F[j, :len(f), :11] = f[:, :11]; nf[j] = len(f); npts[j] = n
    np.savez(part + ".tmp.npz", idx=np.asarray(idxs), F=F, nf=nf, npts=npts); os.replace(part + ".tmp.npz", part)
    return part


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--out", default=str(FACETS))
    a = ap.parse_args()
    rows = load_rows(); out = Path(a.out); parts = Path(str(out) + ".parts"); parts.mkdir(parents=True, exist_ok=True)
    groups = {}
    for i in range(len(rows)):
        groups.setdefault((int(rows["ds"][i]), int(rows["scene"][i])), []).append(i)
    jobs = [(k, v, rows[v], str(parts / f"d{k[0]}_s{k[1]:06d}.npz")) for k, v in sorted(groups.items())]
    t0 = time.time()
    with Pool(a.workers) as p:
        for n, _ in enumerate(p.imap_unordered(job, jobs)):
            if n % 100 == 0:
                print(json.dumps(dict(done=n, of=len(jobs), wall=round(time.time() - t0))), flush=True)
    F = np.zeros((len(rows), KMAX, 12), np.float32); nf = np.zeros(len(rows), np.int16); npts = np.zeros(len(rows), np.int64)
    for _, _, _, part in jobs:
        z = np.load(part); F[z["idx"]] = z["F"]; nf[z["idx"]] = z["nf"]; npts[z["idx"]] = z["npts"]
    src = (Path(__file__).resolve().parent.parent / "slab" / "facets.py").read_bytes()
    np.savez(out, F=F, nf=nf, npts=npts, n_rows=len(rows), facets_py_sha256=hashlib.sha256(src).hexdigest(),
             params=json.dumps(dict(max_facets=KMAX, thr=0.9, min_frac=0.01, min_abs=24, iters=150, seed=0, cap=40000, nscore=3000)))
    print(json.dumps(dict(out=str(out), rows=len(rows), wall=round(time.time() - t0), mean_facets=float(nf.mean()))))


if __name__ == "__main__":
    main()
