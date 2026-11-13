"""Loaders for RIND3D 1K.

Canonical access (`RIND3D1K`) keeps the source axis: one record per physical window, with
* the local SOLID mask once, bool [L, L, L] in ZYX order;
* responses[k, L, L, L], one binary channel per physical source (stable source index j);
* posteriors[j], the certified single-source label of channel j, or a slot with status 'unavailable' / 'not_computed'.

Flattening into single-source examples is a downstream adapter (`flatten`). It never changes the stored data.

Views:
* student: local inputs only (SOLID crop, one or all response channels, window side; the origin is optional);
* target: posterior labels, with TV bound and status;
* training weight (always present, a loss weight, never a model input): `weight` = omega, the scene-level acceptance
  rate of the window's side (docs/SELECTION_LAW_1K.md section 2). Training MUST weight each example by it, otherwise
  the per-channel target is not the certified posterior. `weights` holds the available unbiased estimators:
  - "rate": rate-stream estimate Lambda_hat (1024 proposals; contract default);
  - "quota": inverse-sampling estimate (r - 1) / (N - 1) from the quota stream (r = quota, N = proposals);
  - "combined": (1 - a_L) * rate + a_L * quota with fixed a_L = 0.20 / 0.09 / 0.04 (L = 64 / 128 / 256),
    strictly positive and of lower variance; unbiased because a_L is a constant and both streams are independent of
    the accepted windows' content given the scene.
* privileged (research_access=True only): source coordinates, global geometry, selection log.
Splits are by scene: sha256(scene_id) mod 100 < 80 train, < 90 val, else test."""

import hashlib
import json
from pathlib import Path

import numpy as np
import zstandard


QUOTA = {64: 16, 128: 24, 256: 24}
ALPHA_QUOTA = {64: 0.20, 128: 0.09, 256: 0.04}     # fixed inverse-variance mixing weights (release_1k_v1 medians)


def split_of(scene_id):
    v = int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 100
    return "train" if v < 80 else ("val" if v < 90 else "test")


class Scene:
    def __init__(self, d):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "scene.json").read_text())
        self.W = np.load(self.dir / "public" / "windows.npy")
        idx = np.load(self.dir / "public" / "observations_index.npy")
        self._pos = {(int(w), int(s)): (int(o), int(n)) for w, s, o, n in idx}
        self._blob = None
        self._dz = zstandard.ZstdDecompressor()
        lab = self.dir / "labels" / "LABELS.json"
        self.labels = {(x["window"], x["source"]): x for x in json.loads(lab.read_text())["labels"]} if lab.exists() else {}
        self.side = {int(x["Lp"]): x for x in self.meta.get("sides", [])}

    def weights(self, Lp):
        """Unbiased estimators of the training weight omega = Lambda_L(G, S) of side Lp (see the module docstring)."""
        Lp = int(Lp)
        rate = float(self.meta["Lambda_hat"][str(Lp)])
        out = dict(rate=rate)
        sd = self.side.get(Lp)
        if sd is not None:
            r = int(self.meta.get("quota", {}).get(str(Lp), QUOTA[Lp]))
            n = int(sd["proposals"]) - int(sd["reasons"].get("repeat", 0))
            if n > 1 and r > 1:
                out["quota"] = (r - 1) / (n - 1)
                a = ALPHA_QUOTA[Lp]
                out["combined"] = (1 - a) * rate + a * out["quota"]
        return out

    @property
    def blob(self):
        if self._blob is None:
            self._blob = (self.dir / "public" / "observations.bin").read_bytes()
        return self._blob

    def _crop(self, w, s):
        o, n = self._pos[(w, s)]
        L = int(self.W[w]["L"])
        return np.unpackbits(np.frombuffer(self._dz.decompress(self.blob[o:o + n]), np.uint8),
                             count=L ** 3).astype(bool).reshape(L, L, L)

    def posterior(self, w, j, dense=False):
        """Certified label of channel j: dict(cells, q, lo, hi, tv_bound, status), or a slot with status only."""
        p = self.dir / "labels" / "tasks" / f"w{w:03d}_s{j}.npz"
        meta = self.labels.get((w, j), {})
        if not p.exists():
            r = self.W[w]
            inf = bool((int(r["mask"]) >> j) & 1)
            return dict(status=meta.get("status", "unavailable" if inf else "not_computed"), informative=inf)
        with np.load(p) as z:
            cell, lo, hi = z["cell"], z["lo"], z["hi"]
            tv = float(z["tv_bound"])
            m = json.loads(str(z["meta"]))
        cells, inv = np.unique(cell, return_inverse=True)
        Lc = np.bincount(inv, weights=lo, minlength=len(cells)); Uc = np.bincount(inv, weights=hi, minlength=len(cells))
        q = Lc / Lc.sum() if Lc.sum() > 0 else Lc
        out = dict(cells=cells, q=q, lo=Lc, hi=Uc, tv_bound=tv, status=m["status"], informative=m["informative"],
                   certificate=m.get("certificate"), engine=m.get("engine"))
        if dense:
            g = np.zeros(128 ** 3); g[cells] = q; out["dense"] = g.reshape(128, 128, 128)
        return out

    def window(self, w, research_access=False, with_labels=True):
        r = self.W[w]
        k = int(r["k"])
        rec = dict(scene_id=self.meta["scene_id"], window=int(w), L=int(r["L"]), Lp=int(r["Lp"]),
                   origin=(int(r["x"]), int(r["y"]), int(r["z"])), k=k, m=int(r["m"]),
                   informative=[bool((int(r["mask"]) >> j) & 1) for j in range(k)],
                   solid=self._crop(w, -1), responses=np.stack([self._crop(w, j) for j in range(k)]))
        rec["weights"] = self.weights(r["Lp"])
        rec["weight"] = rec["weights"]["rate"]          # contract default omega (loss weight, not a model input)
        if with_labels:
            rec["posteriors"] = [self.posterior(w, j) for j in range(k)]
        if research_access:
            src = json.loads((self.dir / "privileged" / "sources.json").read_text())
            rec["sources"] = np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]])
        return rec


class RIND3D1K:
    def __init__(self, root, split=None):
        self.root = Path(root)
        ds = sorted(p for p in (self.root / "scenes").iterdir() if (p / "SEAL.json").exists())
        self.scenes = []
        for d in ds:
            sid = json.loads((d / "scene.json").read_text())["scene_id"]
            if split is None or split_of(sid) == split:
                self.scenes.append(d)
        self._cache = {}

    def scene(self, i):
        d = self.scenes[i]
        if d not in self._cache:
            self._cache.clear(); self._cache[d] = Scene(d)
        return self._cache[d]

    def __len__(self):
        return 64 * len(self.scenes)

    def __getitem__(self, idx):
        return self.scene(idx // 64).window(idx % 64)


def flatten(ds, informative_only=True, certified_only=True, max_tv=0.05, weight="rate", accept_statistical=False):
    """Downstream adapter: yields single-source examples (solid, response_j, L, origin, label_j, weight, ids).
    `weight` selects the omega estimator ("rate" = contract default, "quota", "combined"); every example of a window
    carries its window's omega, and training must use it as the loss weight.
    `accept_statistical`: also yield FastLabel labels with status "certified_stat" (TV <= 0.05 except with the
    probability stated in label["certificate"]; dataset-simultaneous policy). Off by default, so the behaviour on
    deterministic labels is unchanged; the status and certificate travel with every label."""
    for i in range(len(ds.scenes)):
        S = ds.scene(i)
        for w in range(len(S.W)):
            rec = S.window(w)
            for j in range(rec["k"]):
                if informative_only and not rec["informative"][j]:
                    continue
                lab = rec["posteriors"][j]
                ok_status = ("certified", "certified_stat") if accept_statistical else ("certified",)
                if certified_only and not (lab.get("status") in ok_status and lab.get("tv_bound", 1) <= max_tv):
                    continue
                yield dict(scene_id=rec["scene_id"], window=w, source=j, L=rec["L"], origin=rec["origin"],
                           solid=rec["solid"], response=rec["responses"][j], label=lab,
                           weight=rec["weights"].get(weight, rec["weight"]))
