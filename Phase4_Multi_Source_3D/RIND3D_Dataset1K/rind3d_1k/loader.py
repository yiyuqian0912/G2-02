"""Loaders for RIND3D 1K.

Canonical access (`RIND3D1K`) keeps the source axis: one record per physical window, with
* the local SOLID mask once, bool [L, L, L] in ZYX order;
* responses[k, L, L, L], one binary channel per physical source (stable source index j);
* posteriors[j], the certified single-source label of channel j, or a slot with status 'unavailable' / 'not_computed'.

Flattening into single-channel examples is a downstream adapter (`flatten`). It never changes the stored data.
`RIND3DNatural` reads the derived natural-window supplement (NAT-1K; RIND3D_StudentResearch/natural) the same way.

Views:
* student: local inputs only (SOLID crop, one or all response channels, window side; the origin is optional).
  NOT student inputs: m, mask, k, scene ids, weights, label status (they encode hidden channels or label paths);
* target: posterior labels, with TV bound, status and certificate;
* training weight (a loss weight, never a model input). The weight depends on the DECLARED target population
  (RIND3D_StudentResearch/A_TASK_DEFINITION.md); `flatten` requires it explicitly:
  - "B1K": the release's own balanced multi-source population. weight = omega_hat(G, S, L), an unbiased estimate of the
    scene-side acceptance rate Lambda_L(G, S). This removes the 1/Lambda quota normalisation (needed for the labels to be
    the posterior of this population) but keeps the effective-layer thinning c(m), which depends on hidden channels.
  - "Tstar": natural multi-source worlds, natural admissible windows, every mixed channel one example (the primary
    single-channel task). Release windows: weight = omega_hat / (n_L c(m, L)); natural-supplement windows: 1 / n_p.
  - "Tstar_mis": T* from the union of the release and the natural supplement (balance heuristic, bounded weights):
    weight = 1 / (n_L c(m, L) / omega_hat + n_p) for windows of either stream (needs the supplement root).
  All weights are in "per natural proposal" units, so they are comparable across scenes and streams. Any extra factor
  that depends only on the side L leaves the conditional optimum unchanged (`side_normalise`).
  omega_hat estimators (`Scene.weights`): "rate" (contract default stream count), "quota" (inverse sampling),
  "combined" (fixed mix), "rb" (Rao-Blackwellised rate: sum_m rate_m c(m) / rate_n, the stream's acceptance
  probability instead of its thinned count), "combined_rb" (fixed mix of rb and quota; default: unbiased, strictly
  positive, lowest variance).
* privileged (research_access=True / with_source=True only): source coordinates and the source's 128^3 cell.
Splits are by scene: sha256(scene_id) mod 100 < 80 train, < 90 val, else test."""

import hashlib
import json
from pathlib import Path

import numpy as np
import zstandard


QUOTA = {64: 16, 128: 24, 256: 24}
ALPHA_QUOTA = {64: 0.20, 128: 0.09, 256: 0.04}     # fixed inverse-variance mixing weights (release_1k_v1 medians)
GRID = 128                                          # label cells: cell = (ix * 128 + iy) * 128 + iz, i = floor(128 s)
ACCEPT_DEFAULT = ("certified", "certified_stat")    # every status whose TV bound <= 0.05 is certified
POPULATIONS = ("B1K", "Tstar", "Tstar_mis")
ESTIMATORS = ("rate", "quota", "combined", "rb", "combined_rb")


def split_of(scene_id):
    v = int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 100
    return "train" if v < 80 else ("val" if v < 90 else "test")


def cell_of(s):
    """128^3 label cell of a point of the unit cube (same convention as the label files)."""
    i = np.minimum((np.asarray(s, float) * GRID).astype(np.int64), GRID - 1)
    return int((i[0] * GRID + i[1]) * GRID + i[2])


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
        nat = self.dir / "natural.json"
        self.natural = json.loads(nat.read_text()) if nat.exists() else None   # set for NAT-1K supplement scenes
        self._sources = None

    def c(self, Lp, m):
        """Effective-layer thinning probability c(m, L) of this scene's release selection (m >= 1)."""
        return float(self.meta["c_table"][str(int(Lp))][int(m) - 1])

    def weights(self, Lp):
        """Unbiased estimators of omega = Lambda_L(G, S) of side Lp (see the module docstring)."""
        Lp = int(Lp)
        rate = float(self.meta["Lambda_hat"][str(Lp)])
        out = dict(rate=rate)
        sd = self.side.get(Lp)
        if sd is not None:
            c = np.array(self.meta["c_table"][str(Lp)], float)
            out["rb"] = float(np.sum(np.array(sd["rate_m"], float) * c) / int(sd["rate_n"]))
            r = int(self.meta.get("quota", {}).get(str(Lp), QUOTA[Lp]))
            n = int(sd["proposals"]) - int(sd["reasons"].get("repeat", 0))
            if n > 1 and r > 1:
                out["quota"] = (r - 1) / (n - 1)
                a = ALPHA_QUOTA[Lp]
                out["combined"] = (1 - a) * rate + a * out["quota"]
                out["combined_rb"] = (1 - a) * out["rb"] + a * out["quota"]
        return out

    def n_p(self, Lp=None):
        """Natural proposals per side of this scene in the supplement (0 if the scene has none)."""
        return int(self.natural["n_p"]) if self.natural else 0

    def population_weight(self, Lp, m, population, estimator="combined_rb", n_p_other=0):
        """Loss weight of a mixed example of a window with mixed count m (see the module docstring).
        For a release scene, n_p_other is the scene's natural proposal count in the supplement (Tstar_mis)."""
        Lp = int(Lp); nL = int(self.meta.get("quota", {}).get(str(Lp), QUOTA[Lp]))
        om = self.weights(Lp)
        om = om.get(estimator, om["rate"])
        if self.natural is None:                           # release window
            c = self.c(Lp, m)
            if population == "B1K":
                return om
            if population == "Tstar":
                return om / (nL * c)
            if population == "Tstar_mis":
                return 1.0 / (nL * c / om + n_p_other) if om > 0 else (1.0 / n_p_other if n_p_other else 0.0)
        else:                                              # natural-supplement window
            n_p = self.n_p()
            if population == "B1K":
                raise ValueError("natural-supplement windows are not part of the B1K population")
            if population == "Tstar":
                return 1.0 / n_p
            if population == "Tstar_mis":
                c = self.c(Lp, m)
                return 1.0 / (nL * c / om + n_p) if om > 0 else 1.0 / n_p
        raise ValueError(f"unknown population {population!r}; declare one of {POPULATIONS}")

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

    def sources(self):
        """Privileged: source coordinates [k, 3] (training targets / evaluation only, never a model input)."""
        if self._sources is None:
            src = json.loads((self.dir / "privileged" / "sources.json").read_text())
            self._sources = np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]])
        return self._sources

    def posterior(self, w, j, dense=False):
        """Certified label of channel j: dict(cells, q, lo, hi, tv_bound, status, certificate, engine, provenance), or a
        slot with status only ('unavailable', 'timeout', 'engine_failure', 'not_computed')."""
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
        prov = {kk: m.get(kk) for kk in ("engine", "engine_version", "path", "config_sha256")}
        prov["sha256"] = meta.get("sha256")
        out = dict(cells=cells, q=q, lo=Lc, hi=Uc, tv_bound=tv, status=m["status"], informative=m["informative"],
                   certificate=m.get("certificate"), engine=m.get("engine"), provenance=prov)
        if dense:
            g = np.zeros(GRID ** 3); g[cells] = q; out["dense"] = g.reshape(GRID, GRID, GRID)   # [x, y, z]
        return out

    def window(self, w, research_access=False, with_labels=True):
        r = self.W[w]
        k = int(r["k"])
        rec = dict(scene_id=self.meta["scene_id"], window=int(w), L=int(r["L"]), Lp=int(r["Lp"]),
                   origin=(int(r["x"]), int(r["y"]), int(r["z"])), k=k, m=int(r["m"]),
                   informative=[bool((int(r["mask"]) >> j) & 1) for j in range(k)],
                   solid=self._crop(w, -1), responses=np.stack([self._crop(w, j) for j in range(k)]))
        if self.natural is None:
            rec["weights"] = self.weights(r["Lp"])
            rec["weight"] = rec["weights"]["rate"]      # contract default omega (B1K loss weight, not a model input)
        if with_labels:
            rec["posteriors"] = [self.posterior(w, j) for j in range(k)]
        if research_access:
            rec["sources"] = self.sources()
        return rec


class RIND3D1K:
    kind = "release"

    def __init__(self, root, split=None, scenes=None):
        self.root = Path(root)
        ds = sorted(p for p in (self.root / "scenes").iterdir() if self._ok(p))
        self.scenes = []
        for d in ds:
            sid = json.loads((d / "scene.json").read_text())["scene_id"]
            if (split is None or split_of(sid) == split) and (scenes is None or d.name in scenes):
                self.scenes.append(d)
        self._cache = {}

    @staticmethod
    def _ok(p):
        return (p / "SEAL.json").exists()

    def scene(self, i):
        d = self.scenes[i]
        if d not in self._cache:
            self._cache.clear(); self._cache[d] = Scene(d)
        return self._cache[d]

    def __len__(self):
        return 64 * len(self.scenes)

    def __getitem__(self, idx):
        return self.scene(idx // 64).window(idx % 64)


class RIND3DNatural(RIND3D1K):
    """NAT-1K natural-window supplement (derived from the release's stored fields; see natural_supplement.py)."""
    kind = "natural"

    @staticmethod
    def _ok(p):
        return (p / "natural.json").exists()

    def __len__(self):
        return sum(len(np.load(d / "public" / "windows.npy")) for d in self.scenes)

    def __getitem__(self, idx):
        raise TypeError("index natural scenes with ds.scene(i).window(w)")


def _natural_np_table(natural_root):
    out = {}
    if natural_root is None:
        return out
    for d in (Path(natural_root) / "scenes").iterdir():
        p = d / "natural.json"
        if p.exists():
            with open(p) as f:
                head = json.load(f)
            out[d.name] = int(head["n_p"])
    return out


def side_means(ds, population, estimator="combined_rb", natural_root=None):
    """Mean loss weight per side over the mixed examples of ds (for side normalisation; reads metadata only)."""
    npt = _natural_np_table(natural_root) if population == "Tstar_mis" else {}
    tot = {}
    for i in range(len(ds.scenes)):
        S = ds.scene(i)
        for r in S.W:
            m = int(r["m"])
            if m == 0:
                continue
            Lp = int(r["Lp"])
            wv = S.population_weight(Lp, m, population, estimator, npt.get(S.dir.name, 0))
            a = tot.setdefault(Lp, [0.0, 0])
            a[0] += m * wv; a[1] += m
    return {L: v[0] / v[1] for L, v in tot.items() if v[1]}


def flatten(ds, population=None, *, estimator="combined_rb", accept=ACCEPT_DEFAULT, max_tv=0.05, fallback=None,
            missing="yield", supervision="posterior", informative_only=True, with_source=False, side_norm=None,
            natural_root=None, report=None, with_inputs=True, accept_statistical=None, certified_only=None,
            weight=None):
    """Downstream adapter: yields single-channel examples of the declared target population.

    population (required): "B1K", "Tstar" or "Tstar_mis" (module docstring). No default: the choice defines what the
        Student learns (A_TASK_DEFINITION.md) and must be explicit.
    accept: label statuses used as certified posterior targets (default: every certified status, deterministic and
        statistical, with tv_bound <= max_tv). Statistical labels carry their certificate in label["certificate"].
    supervision: "posterior" (certified label field) or "onehot" (true-source cell for every example: the one-hot
        baseline; needs privileged sources, which are used as the target only).
    fallback: None or "onehot". With "onehot", an example whose label is missing or not accepted gets the true-source
        one-hot target (supervision "onehot_fallback"); it is never presented as a certified posterior.
    missing: "yield" (default; example with supervision "missing" and target None) or "drop" (counted in `report`).
        Missing labels are never dropped silently: `report` (a dict) receives counts and weight mass per
        (side, supervision).
    side_norm: None or a dict {L: mean weight} (e.g. from side_means) to divide the weight by.

    Every example carries: ids (scene, window, source), L, origin, inputs (solid, response; omitted with
    with_inputs=False), supervision, target ({cells, q} or None), label (posterior dict or status slot), status,
    tv_bound, certificate, provenance, population, weight, and `privileged` (source, source_cell) when with_source or
    one-hot targets are requested. weight / privileged / status are never model inputs.

    Legacy arguments: accept_statistical=False restricts accept to ("certified",); certified_only=False accepts any
    stored label; weight=<name> is the old estimator argument."""
    if population not in POPULATIONS:
        raise ValueError(f"flatten: declare the target population explicitly: one of {POPULATIONS} "
                         "(see RIND3D_StudentResearch/A_TASK_DEFINITION.md)")
    if weight is not None:
        estimator = weight
    if accept_statistical is False:
        accept = ("certified",)
    if supervision not in ("posterior", "onehot"):
        raise ValueError("supervision must be 'posterior' or 'onehot'")
    if fallback not in (None, "onehot"):
        raise ValueError("fallback must be None or 'onehot'")
    if missing not in ("yield", "drop"):
        raise ValueError("missing must be 'yield' or 'drop'")
    npt = _natural_np_table(natural_root) if (population == "Tstar_mis" and getattr(ds, "kind", "release") == "release") else {}
    if population == "Tstar_mis" and getattr(ds, "kind", "release") == "release" and not npt:
        raise ValueError("Tstar_mis on release scenes needs natural_root (the supplement's n_p per scene)")
    need_src = with_source or supervision == "onehot" or fallback == "onehot"
    for i in range(len(ds.scenes)):
        S = ds.scene(i)
        for w in range(len(S.W)):
            r = S.W[w]
            k = int(r["k"]); Lp = int(r["Lp"]); m = int(r["m"])
            inf = [bool((int(r["mask"]) >> j) & 1) for j in range(k)]
            if informative_only and not any(inf):
                continue
            rec = S.window(w, with_labels=True) if with_inputs else None
            for j in range(k):
                if informative_only and not inf[j]:
                    continue
                lab = rec["posteriors"][j] if rec is not None else S.posterior(w, j)
                st = lab.get("status")
                ok = (certified_only is False and "cells" in lab) or (
                    st in accept and lab.get("tv_bound", 1.0) <= max_tv and "cells" in lab)
                wv = S.population_weight(Lp, max(m, 1), population, estimator, npt.get(S.dir.name, 0)) if inf[j] else None
                if wv is not None and side_norm:
                    wv = wv / side_norm[Lp]
                src = S.sources()[j] if need_src else None
                if supervision == "onehot":
                    sup = "onehot"; target = dict(cells=np.array([cell_of(src)]), q=np.array([1.0]))
                elif ok:
                    sup = "posterior"; target = dict(cells=lab["cells"], q=lab["q"])
                elif fallback == "onehot":
                    sup = "onehot_fallback"; target = dict(cells=np.array([cell_of(src)]), q=np.array([1.0]))
                else:
                    sup = "missing"; target = None
                if report is not None:
                    a = report.setdefault(f"L{Lp}:{sup}", [0, 0.0])
                    a[0] += 1; a[1] += float(wv or 0.0)
                    if sup != "posterior" and supervision == "posterior":
                        report.setdefault("status_not_accepted", {}).setdefault(str(st), 0)
                        report["status_not_accepted"][str(st)] += 1
                if sup == "missing" and missing == "drop":
                    continue
                ex = dict(scene_id=S.meta["scene_id"], scene=S.dir.name, dataset=getattr(ds, "kind", "release"),
                          window=w, source=j, L=int(r["L"]), Lp=Lp, origin=(int(r["x"]), int(r["y"]), int(r["z"])),
                          supervision=sup, target=target, label=lab, status=st, tv_bound=lab.get("tv_bound"),
                          certificate=lab.get("certificate"), provenance=lab.get("provenance"),
                          population=population, estimator=estimator, weight=wv)
                if rec is not None:
                    ex.update(solid=rec["solid"], response=rec["responses"][j])
                if need_src:
                    ex["privileged"] = dict(source=src, source_cell=cell_of(src))
                yield ex
