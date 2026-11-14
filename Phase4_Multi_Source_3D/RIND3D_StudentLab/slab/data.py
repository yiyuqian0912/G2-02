"""Lab data layer: the frozen pilot example table (features, weights, labels) plus splits, evaluation sets and the
X-measurable facet table.

Splits (scene-disjoint; declared by the pilot in natural/natural_supplement.py, reused unchanged):
  train    pilot-train   (684 scenes of the 1K train split)   training only
  dev      pilot-select  (102 scenes of the 1K train split)   all exploratory comparisons, OOD development
  confirm  pilot-eval    (121 scenes = the 1K val split)      confirmatory evaluation only (evaluate.py --confirm)
  test     the 1K test split: never present in the feature table; sealed.
Row types (X-measurable): 0 boundary-mixed, 1 solid-separated, 2 all-lit, 3 all-dark. Training uses type 0 only.
"""
import sys
import numpy as np
import torch

from .paths import PILOT, FEATS, FACETS

sys.path.insert(0, str(PILOT))
from data import Table, side_normalised, kish, ROLES   # noqa: E402  (frozen pilot code)

SPLIT_ROLE = {"train": 0, "dev": 1, "confirm": 2}
NC, NF, G = 16, 8, 128


class LabTable(Table):
    def __init__(self, feat_dir=FEATS, facets=FACETS, kmax=8):
        super().__init__(feat_dir)
        self.kmax = kmax
        if facets is not None and facets.exists():
            z = np.load(facets)
            assert int(z["n_rows"]) == len(self.rows), "facet table does not match the feature table"
            self.F = z["F"][:, :kmax].astype(np.float32); self.nf = z["nf"].astype(np.int64)
        else:
            self.F = None; self.nf = None

    def split_mask(self, split):
        return self.rows["role"] == SPLIT_ROLE[split]

    def eval_set(self, name, split):
        """Declared evaluation sets of one split (identical rows for every model)."""
        r = self.rows; base = self.split_mask(split)
        sets = {"tstar": base & (r["ds"] == 1) & (r["type"] == 0),        # T*: natural windows, boundary-mixed
                "b1k": base & (r["ds"] == 0) & (r["type"] == 0),          # release windows (B1K population)
                "sep": base & (r["type"] == 1),                           # solid-separated (natural + release)
                "alllit": base & (r["ds"] == 1) & (r["type"] == 2),       # OOD A
                "alldark": base & (r["ds"] == 1) & (r["type"] == 3)}      # OOD B
        idx = np.flatnonzero(sets[name])
        cap = {"b1k": 4000, "alllit": 2500}.get(name)
        if cap and len(idx) > cap:                                         # declared subsample, seed fixed
            # rows with a certified OOD reference label (confirm split only) are always kept; on dev there are none, and
            # the draw is then identical to a plain subsample of idx
            ref = np.array([self.label_path(i).exists() or (self.label_path(i).parent.parent / "ood" / self.label_path(i).name).exists()
                            for i in idx]) if name == "alllit" else np.zeros(len(idx), bool)
            rest = idx[~ref]
            pick = np.random.default_rng(20261114).choice(rest, cap - int(ref.sum()), replace=False)
            idx = np.sort(np.concatenate([idx[ref], pick]))
        return idx


def cf_of_cell(cell):
    """label cell index (ix*128+iy)*128+iz -> (coarse c, fine f) of the 16^3 x 8^3 layout."""
    cell = np.asarray(cell, np.int64)
    ix, iy, iz = cell // (G * G), (cell // G) % G, cell % G
    return (ix // NF) * 256 + (iy // NF) * 16 + iz // NF, (ix % NF) * 64 + (iy % NF) * 8 + iz % NF


def collate(T, pick, target, with_facets=True):
    r = T.rows[pick]
    out = dict(feats=torch.from_numpy(T.feats(pick)),
               origin=torch.from_numpy(np.stack([r["ox"], r["oy"], r["oz"]], 1).astype(np.int64)),
               side=torch.from_numpy(r["Lp"].astype(np.int64)), idx=torch.from_numpy(np.asarray(pick)))
    tb, tc, tf, tq, fb = [], [], [], [], 0
    src = np.array([T.source_cell(i) for i in pick])
    for b, i in enumerate(pick):
        cells = q = None
        if target == "posterior":
            cells, q, _ = T.label(i)
        if cells is None:
            cells = np.array([src[b]]); q = np.ones(1, np.float32); fb += target == "posterior"
        c, f = cf_of_cell(cells)
        tb.append(np.full(len(cells), b)); tc.append(c); tf.append(f); tq.append(q)
    sc, sf = cf_of_cell(src)
    out.update(tb=torch.from_numpy(np.concatenate(tb)), tc=torch.from_numpy(np.concatenate(tc)),
               tf=torch.from_numpy(np.concatenate(tf)), tq=torch.from_numpy(np.concatenate(tq).astype(np.float32)),
               sc=torch.from_numpy(sc), sf=torch.from_numpy(sf), fallback=fb)
    if with_facets and T.F is not None:
        out["facets"] = torch.from_numpy(T.F[pick]); out["nf"] = torch.from_numpy(T.nf[pick])
    return out


def to_dev(B, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in B.items()}
