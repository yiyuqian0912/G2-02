"""Pilot data: example table (from build_features shards), loss weights per declared population, label access.

Weights (per natural-proposal unit; A_TASK_DEFINITION.md section 5; omega_hat = combined_rb estimator):
  w_b1k   release rows: omega_hat                              (B1K: the release's balanced population)
  w_trel  release rows: omega_hat / (n_L c(m, L))             (T*, release only)
  w_mis   both streams: 1 / (n_L c(m, L) / omega_hat + n_p)    (T*, release + NAT-1K, balance heuristic)
  w_nat   natural rows: 1 / n_p                                (T*, natural supplement only)
Training samples examples with probability proportional to the arm's weight after equal-side normalisation (each side
1/3 of the mass; side is an input, so the optimum is unchanged), and uses an unweighted loss (same expected gradient)."""
import json
from pathlib import Path
import numpy as np

REL = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1")
NAT = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/natural_supplement_v1")
QUOTA = {64: 16, 128: 24, 256: 24}
ALPHA_QUOTA = {64: 0.20, 128: 0.09, 256: 0.04}
ROLES = ("pilot-train", "pilot-select", "pilot-eval")
TYPES = ("boundary_mixed", "solid_separated", "all_lit", "all_dark")
G = 128


def cell_of(xyz):
    i = np.minimum((np.asarray(xyz, float) * G).astype(np.int64), G - 1)
    return (i[..., 0] * G + i[..., 1]) * G + i[..., 2]


def omega_hat(meta, Lp):
    sd = {int(s["Lp"]): s for s in meta["sides"]}[Lp]
    c = np.array(meta["c_table"][str(Lp)], float)
    rb = float(np.sum(np.array(sd["rate_m"], float) * c) / int(sd["rate_n"]))
    n = int(sd["proposals"]) - int(sd["reasons"].get("repeat", 0)); r = QUOTA[Lp]
    quota = (r - 1) / (n - 1)
    a = ALPHA_QUOTA[Lp]
    return (1 - a) * rb + a * quota


class Table:
    def __init__(self, feat_dir):
        self.dir = Path(feat_dir)
        shards = sorted(self.dir.glob("shard*.npy"), key=lambda p: int(p.stem[5:]))
        rows, self.mm = [], []
        for si, p in enumerate(shards):
            r = np.load(p)
            f = self.dir / (p.stem + ".u8")
            self.mm.append(np.memmap(f, np.uint8, "r", shape=(len(r), 4, 32, 32, 32)))
            rr = np.zeros(len(r), dtype=r.dtype.descr + [("shard", "i2")])
            for k in r.dtype.names:
                rr[k] = r[k]
            rr["shard"] = si
            rows.append(rr)
        self.rows = np.concatenate(rows)
        self._weights()

    def _weights(self):
        r = self.rows
        n = len(r)
        self.w = {k: np.zeros(n) for k in ("b1k", "trel", "mis", "nat")}
        meta_cache, np_cache = {}, {}
        for i in range(n):
            sc = int(r["scene"][i])
            if sc not in meta_cache:
                meta_cache[sc] = json.loads((REL / "scenes" / f"{sc:06d}" / "scene.json").read_text())
                nj = NAT / "scenes" / f"{sc:06d}" / "natural.json"
                with open(nj) as f:
                    np_cache[sc] = int(json.load(f)["n_p"])
            meta = meta_cache[sc]; n_p = np_cache[sc]
            Lp = int(r["Lp"][i]); m = int(r["m"][i])
            if m < 1:                              # natural window with no mixed channel: OOD-only rows
                continue
            c = float(meta["c_table"][str(Lp)][m - 1]); om = omega_hat(meta, Lp); nL = QUOTA[Lp]
            self.w["mis"][i] = 1.0 / (nL * c / om + n_p)
            if r["ds"][i] == 0:
                self.w["b1k"][i] = om; self.w["trel"][i] = om / (nL * c)
            else:
                self.w["nat"][i] = 1.0 / n_p
        # rows that are not mixed carry no training weight under any population
        mixed = np.isin(r["type"], (0, 1))
        for k in self.w:
            self.w[k][~mixed] = 0.0

    def feats(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx), 4, 32, 32, 32), np.uint8)
        for j, i in enumerate(idx):
            out[j] = self.mm[self.rows["shard"][i]][self.rows["feat"][i]]
        return out

    def label_path(self, i):
        r = self.rows[i]
        root = REL if r["ds"] == 0 else NAT
        return root / "scenes" / f"{int(r['scene']):06d}" / "labels" / "tasks" / f"w{int(r['window']):03d}_s{int(r['source'])}.npz"

    def label(self, i, accept=("certified", "certified_stat"), max_tv=0.05):
        """(cells, q, status) of the certified label, or (None, None, status) if missing / not accepted."""
        p = self.label_path(i)
        if not p.exists():
            p = p.parent.parent / "ood" / p.name               # OOD reference labels of extreme channels (research)
        if not p.exists():
            return None, None, "unavailable"
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            if meta["status"] not in accept or float(z["tv_bound"]) > max_tv:
                return None, None, meta["status"]
            cell, lo = z["cell"].astype(np.int64), z["lo"]
        cells, inv = np.unique(cell, return_inverse=True)
        q = np.bincount(inv, weights=lo, minlength=len(cells))
        return cells, (q / q.sum()).astype(np.float32), meta["status"]

    def source_cell(self, i):
        r = self.rows[i]
        return int(cell_of([r["sx"], r["sy"], r["sz"]]))


def side_normalised(w, Lp):
    w = np.asarray(w, float).copy()
    for L in (64, 128, 256):
        s = (Lp == L) & (w > 0)
        if s.any():
            w[s] = w[s] / w[s].sum() / 3.0
    return w


def kish(w):
    w = np.asarray(w, float); w = w[w > 0]
    return float(w.sum() ** 2 / np.sum(w ** 2)) if len(w) else 0.0
