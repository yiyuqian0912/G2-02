"""Loaders for RIND3D-100K (canonical structure scene -> window -> source; flattening only in `flatten`).

**Window record:**
* `solid` bool [L, L, L] (Z, Y, X), once;
* `responses` bool [k, L, L, L], one channel per physical source (stable source index);
* `informative[j]`;
* `posteriors[j]`: dict(cells, q, lo, hi, tv_bound, status) for a computed slot, otherwise dict(status=...), with
  status one of `not_computed`, `unavailable`, `engine_failure`, `timeout`, `memory_limit`;
* `weight`: the training weight omega, the contract default being the rate-stream Lambda_hat_L. It is a loss weight,
  never a model input;
* `weights`: the unbiased estimators rate / quota / combined (docs/SELECTION_LAW_100K.md §3).

**Splits:** sha256(scene_id) mod 100: < 80 train, < 90 val, else test.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .storage import SceneFile, scene_path, labels_path

ALPHA_QUOTA = {64: 0.12, 128: 0.05, 256: 0.035}      # fixed inverse-variance mixing weights for r = 4 (§3)


def split_of(scene_id):
    v = int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 100
    return "train" if v < 80 else ("val" if v < 90 else "test")


class Scene:
    def __init__(self, root, slot):
        self.root = Path(root); self.slot = int(slot)
        self.S = SceneFile(scene_path(root, slot))
        self.meta = self.S.meta
        self.side = {int(s["Lp"]): s for s in self.meta["sides"]}
        lp = labels_path(root, slot)
        self.lab = None
        if lp.exists():
            z = np.load(lp)
            self.lab = dict(cell=z["cell"], lo=z["lo"], hi=z["hi"], offsets=z["offsets"],
                            metas=json.loads(z["metas"].tobytes()), status=json.loads(z["status"].tobytes()))
            self.lab["by_key"] = {m["key"]: i for i, m in enumerate(self.lab["metas"])}

    def weights(self, Lp):
        Lp = int(Lp); sd = self.side[Lp]
        rate = float(sd["Lambda_hat"]); out = dict(rate=rate)
        r = int(sd["quota"]); n = int(sd["proposals"]) - int(sd["reasons"].get("repeat", 0))
        if n > 1 and r > 1:
            out["quota"] = (r - 1) / (n - 1)
            a = ALPHA_QUOTA[Lp]; out["combined"] = (1 - a) * rate + a * out["quota"]
        return out

    def posterior(self, w, j):
        key = f"{self.slot:06d}:{int(w):02d}:{int(j)}"
        r = self.S.W[w]; inf = bool((int(r["mask"]) >> j) & 1)
        if self.lab is None:
            return dict(status="unavailable" if inf else "not_computed", informative=inf)
        st = self.lab["status"].get(key, "unavailable")
        i = self.lab["by_key"].get(key)
        if i is None:
            return dict(status=st, informative=inf)
        a, b = self.lab["offsets"][i], self.lab["offsets"][i + 1]
        cells = self.lab["cell"][a:b].astype(np.int64); Lc = self.lab["lo"][a:b]; Uc = self.lab["hi"][a:b]   # per cell
        q = Lc / Lc.sum() if Lc.sum() > 0 else Lc
        m = self.lab["metas"][i]
        return dict(cells=cells, q=q, lo=Lc, hi=Uc, tv_bound=m["tv_bound"], status=m["status"], informative=inf)

    def window(self, w, with_labels=True):
        r = self.S.W[w]; k = int(r["k"])
        rec = dict(scene_id=self.meta["scene_id"], slot=self.slot, window=int(w), L=int(r["L"]), Lp=int(r["Lp"]),
                   origin=(int(r["x"]), int(r["y"]), int(r["z"])), k=k, m=int(r["m"]),
                   informative=[bool((int(r["mask"]) >> j) & 1) for j in range(k)],
                   solid=self.S.crop(w, -1), responses=np.stack([self.S.crop(w, j) for j in range(k)]))
        rec["weights"] = self.weights(r["Lp"]); rec["weight"] = rec["weights"]["rate"]
        if with_labels:
            rec["posteriors"] = [self.posterior(w, j) for j in range(k)]
        return rec


class RIND3D100K:
    def __init__(self, root, split=None, slots=None):
        self.root = Path(root)
        man = self.root / "manifest" / "physical.jsonl"
        acc = {}
        for line in man.read_text().splitlines():
            rr = json.loads(line)
            if rr.get("status") == "accepted":
                acc[int(rr["slot"])] = rr
        self.slots = sorted(s for s, rr in acc.items() if (slots is None or s in slots) and
                            (split is None or split_of(rr["scene_id"]) == split))
        self._cache = {}

    def scene(self, i):
        s = self.slots[i]
        if s not in self._cache:
            self._cache.clear(); self._cache[s] = Scene(self.root, s)
        return self._cache[s]

    def __len__(self):
        return 12 * len(self.slots)

    def __getitem__(self, idx):
        return self.scene(idx // 12).window(idx % 12)


def flatten(ds, informative_only=True, certified_only=True, max_tv=0.05, weight="rate"):
    """Downstream adapter: single-source examples (solid, response_j, L, origin, label_j, weight, ids)."""
    for i in range(len(ds.slots)):
        S = ds.scene(i)
        for w in range(len(S.S.W)):
            rec = S.window(w)
            for j in range(rec["k"]):
                if informative_only and not rec["informative"][j]:
                    continue
                lab = rec["posteriors"][j]
                if certified_only and not (lab.get("status") == "certified" and lab.get("tv_bound", 1) <= max_tv):
                    continue
                yield dict(scene_id=rec["scene_id"], window=w, source=j, L=rec["L"], origin=rec["origin"],
                           solid=rec["solid"], response=rec["responses"][j], label=lab,
                           weight=rec["weights"].get(weight, rec["weight"]))
