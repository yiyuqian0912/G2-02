"""E16: real-data size of the balancing tilt for a tabular single-channel Student on a coarse X-partition.

For a partition of X into bins b (side, 4^3 cell of the window centre, lit-fraction bin, solid present), the population
optimum of any Student that only sees b is the weighted mean of the labels in b:
    p_pop(b) = sum_e w_e q_e / sum_e w_e         (16^3 coarse cells)
with w = omega (B1K) or omega / (n_L c) (T*, release only). TV(p_B1K(b), p_T*(b)) measures the tilt of the optimum
caused by the effective-layer balancing for this coarse Student (the real Student sees more of X; this is an
estimate for one sigma-algebra, not a bound). Noise reference: the same TV between two halves of the scenes, both
weighted for T* (split by scene hash), and the B1K-vs-T* TV computed separately on each half.
All non-test release scenes; labels must exist (run after the backfill). Read-only.
usage: e16_tabular_tilt.py <out.json> [workers]"""
import json, sys, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch/pilot")
REL = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
QUOTA = {64: 16, 128: 24, 256: 24}
LITB = np.array([0.01, 0.1, 0.5, 0.9, 0.99])


def scene_part(d):
    from data import omega_hat
    from rind3d_1k.loader import split_of
    meta = json.loads((d / "scene.json").read_text())
    if split_of(meta["scene_id"]) == "test":
        return None
    half = int(hashlib.sha256((meta["scene_id"] + ":e16").encode()).hexdigest(), 16) % 2
    W = np.load(d / "public" / "windows.npy")
    labs = json.loads((d / "labels" / "LABELS.json").read_text())["labels"]
    meta_l = {(x["window"], x["source"]): x for x in labs}
    out = []
    for r in W:
        Lp = int(r["Lp"]); m = int(r["m"]); c = float(meta["c_table"][str(Lp)][m - 1]); om = omega_hat(meta, Lp)
        ctr = (np.array([r["x"], r["y"], r["z"]]) + int(r["L"]) / 2) / 1024
        cb = tuple(np.minimum((ctr * 4).astype(int), 3))
        for j in range(int(r["k"])):
            if not (int(r["mask"]) >> j) & 1:
                continue
            p = d / "labels" / "tasks" / f"w{int(r['window']):03d}_s{j}.npz"
            if not p.exists():
                continue
            with np.load(p) as z:
                st = json.loads(str(z["meta"]))["status"]
                if st not in ("certified", "certified_stat"):
                    continue
                cell, lo = z["cell"].astype(np.int64), z["lo"]
            ix, iy, iz = cell // 16384, (cell // 128) % 128, cell % 128
            cc = (ix // 8) * 256 + (iy // 8) * 16 + iz // 8
            Q = np.bincount(cc, weights=lo, minlength=4096); Q /= Q.sum()
            lm = meta_l.get((int(r["window"]), j), {})
            lit = None
            out.append(dict(Lp=Lp, cb=cb, w_b=om, w_t=om / (QUOTA[Lp] * c), half=half, Q=Q.astype(np.float32),
                            m=m, wj=(int(r["window"]), j)))
    # lit fraction and solid flag need the crop counts: windows_full.json has per-channel lit_frac and counts
    wf = json.loads((d / "privileged" / "windows_full.json").read_text())
    for e in out:
        rec = wf[e["wj"][0]]
        e["lit"] = float(rec["lit_frac"][e["wj"][1]]); e["solid"] = int(rec["counts"][0][2] > 0)
        del e["wj"]
    return out


def main():
    outp = sys.argv[1]; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    dirs = sorted(p for p in (REL / "scenes").iterdir() if (p / "SEAL.json").exists())
    acc = {}
    n = 0
    with ProcessPoolExecutor(nw) as ex:
        for part in ex.map(scene_part, dirs, chunksize=4):
            if not part:
                continue
            for e in part:
                b = (e["Lp"], e["cb"], int(np.searchsorted(LITB, e["lit"])), e["solid"])
                a = acc.setdefault(b, dict(Sb=np.zeros(4096), St=np.zeros(4096), wb=0.0, wt=0.0,
                                           h=[dict(Sb=np.zeros(4096), St=np.zeros(4096), wb=0.0, wt=0.0) for _ in range(2)]))
                for tgt in (a, a["h"][e["half"]]):
                    tgt["Sb"] += e["w_b"] * e["Q"]; tgt["St"] += e["w_t"] * e["Q"]; tgt["wb"] += e["w_b"]; tgt["wt"] += e["w_t"]
                n += 1
    tv = lambda a, b: 0.5 * float(np.abs(a - b).sum())
    rows = []
    for b, a in acc.items():
        if min(a["h"][0]["wt"], a["h"][1]["wt"]) <= 0 or min(a["h"][0]["wb"], a["h"][1]["wb"]) <= 0:
            continue
        pb, pt = a["Sb"] / a["wb"], a["St"] / a["wt"]
        h0, h1 = a["h"]
        rows.append(dict(bin=[int(b[0]), list(map(int, b[1])), b[2], b[3]], mass_t=a["wt"],
                         tv_b1k_tstar=tv(pb, pt), tv_halves_tstar=tv(h0["St"] / h0["wt"], h1["St"] / h1["wt"]),
                         tv_b1k_tstar_half0=tv(h0["Sb"] / h0["wb"], h0["St"] / h0["wt"]),
                         tv_b1k_tstar_half1=tv(h1["Sb"] / h1["wb"], h1["St"] / h1["wt"])))
    res = dict(n_examples=n, n_bins=len(rows))
    for L in (64, 128, 256):
        rr = [r for r in rows if r["bin"][0] == L]
        w = np.array([r["mass_t"] for r in rr]); w = w / w.sum()
        res[f"L{L}"] = {k: float(np.sum(w * np.array([r[k] for r in rr]))) for k in
                        ("tv_b1k_tstar", "tv_halves_tstar", "tv_b1k_tstar_half0", "tv_b1k_tstar_half1")}
        res[f"L{L}"]["bins"] = len(rr)
    Path(outp).write_text(json.dumps(dict(summary=res, rows=rows), default=float))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
