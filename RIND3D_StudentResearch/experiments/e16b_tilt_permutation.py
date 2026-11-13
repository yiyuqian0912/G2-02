"""E16b: is the balancing tilt of the population optimum detectable on real labels? (permutation null on the same data)

For a coarse partition of X into bins, the optimum of a tabular Student is the weighted mean label of the bin. The
T* and B1K optima differ only through the per-example ratio r_e = w_T*/w_B1K = 1/(n_L c(m_e, L)), which depends on the
hidden effective-layer count. Null hypothesis "the hidden factor is unrelated to the label content within a bin": then
permuting r among the bin's WINDOWS (all channels of a window share one m, so windows are the exchangeable units) leaves the distribution of TV(p_B1K, p_T*) unchanged. Observed TV vs the
permutation distribution (8 permutations) measures the real tilt beyond finite-sample noise.
Two partitions:
  fine:   side x 4^3 window-centre cell x 6 lit-fraction bins x solid present  (as E16)
  coarse: side x 2^3 window-centre cell x 6 lit-fraction bins
Labels aggregated to 8^3 coarse cells (block size 1/8 of the world). Non-test release scenes; read-only.
usage: e16b_tilt_permutation.py <out.json> [workers]"""
import json, sys, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_StudentResearch/pilot")
sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K")
REL = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
QUOTA = {64: 16, 128: 24, 256: 24}
LITB = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
NB = 8


def scene_part(d):
    from data import omega_hat
    from rind3d_1k.loader import split_of
    meta = json.loads((d / "scene.json").read_text())
    if split_of(meta["scene_id"]) == "test":
        return None
    W = np.load(d / "public" / "windows.npy")
    wf = json.loads((d / "privileged" / "windows_full.json").read_text())
    out = []
    for r in W:
        Lp = int(r["Lp"]); m = int(r["m"]); c = float(meta["c_table"][str(Lp)][m - 1]); om = omega_hat(meta, Lp)
        ctr = (np.array([r["x"], r["y"], r["z"]]) + int(r["L"]) / 2) / 1024
        rec = wf[int(r["window"])]
        for j in range(int(r["k"])):
            if not (int(r["mask"]) >> j) & 1:
                continue
            p = d / "labels" / "tasks" / f"w{int(r['window']):03d}_s{j}.npz"
            with np.load(p) as z:
                cell, lo = z["cell"].astype(np.int64), z["lo"]
            ix, iy, iz = cell // 16384, (cell // 128) % 128, cell % 128
            cc = (ix // 16) * 64 + (iy // 16) * 8 + iz // 16
            u, inv = np.unique(cc, return_inverse=True)
            q = np.bincount(inv, weights=lo); q /= q.sum()
            lit = float(rec["lit_frac"][j]); sol = int(rec["counts"][0][2] > 0)
            lb = int(np.searchsorted(LITB, lit))
            fine = (Lp, tuple(np.minimum((ctr * 4).astype(int), 3)), lb, sol)
            coarse = (Lp, tuple(np.minimum((ctr * 2).astype(int), 1)), lb)
            out.append((fine, coarse, om, 1.0 / (QUOTA[Lp] * c), u.astype(np.int16), q.astype(np.float32),
                        (d.name, int(r["window"]))))
    return out


def tv_bin(ex, ratio):
    S_b = np.zeros(NB ** 3); S_t = np.zeros(NB ** 3); wb = wt = 0.0
    for (u, q, om), r in zip(ex, ratio):
        S_b[u] += om * q; S_t[u] += om * r * q; wb += om; wt += om * r
    return 0.5 * float(np.abs(S_b / wb - S_t / wt).sum())


def main():
    outp = sys.argv[1]; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dirs = sorted(p for p in (REL / "scenes").iterdir() if (p / "SEAL.json").exists())
    bins = {"fine": {}, "coarse": {}}
    n = 0
    with ProcessPoolExecutor(nw) as ex:
        for part in ex.map(scene_part, dirs, chunksize=4):
            for fine, coarse, om, r, u, q, win in part or []:
                bins["fine"].setdefault(fine, []).append(((u, q, om), r, win))
                bins["coarse"].setdefault(coarse, []).append(((u, q, om), r, win))
                n += 1
    rng = np.random.default_rng(16)
    res = dict(n_examples=n)
    for part, B in bins.items():
        for L in (64, 128, 256):
            obs, null, mass = [], [], []
            for key, items in B.items():
                if key[0] != L or len(items) < 20:
                    continue
                exs = [a for a, _, _ in items]; rr = np.array([b for _, b, _ in items])
                if rr.max() == rr.min():
                    continue                               # no hidden-factor variation in this bin
                # windows are the exchangeable units: all channels of one window share r (one m per window)
                wins = [w for _, _, w in items]; uw = sorted(set(wins)); wi = np.array([uw.index(w) for w in wins])
                rw = np.zeros(len(uw)); rw[wi] = rr
                obs.append(tv_bin(exs, rr))
                null.append(np.mean([tv_bin(exs, rng.permutation(rw)[wi]) for _ in range(8)]))
                mass.append(sum(o * r for (_, _, o), r in zip(exs, rr)))
            w = np.array(mass) / np.sum(mass)
            res[f"{part}:L{L}"] = dict(bins=len(obs), tv_observed=float(np.sum(w * np.array(obs))),
                                       tv_permutation_null=float(np.sum(w * np.array(null))),
                                       excess=float(np.sum(w * (np.array(obs) - np.array(null)))),
                                       share_bins_obs_gt_null=float(np.mean(np.array(obs) > np.array(null))))
    Path(outp).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
