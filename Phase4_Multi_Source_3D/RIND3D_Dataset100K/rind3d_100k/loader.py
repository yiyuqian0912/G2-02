"""Loaders for RIND3D-100K (canonical structure scene -> window -> source; flattening only in `flatten`).

Port of the 1K M2 loader (`rind3d_1k.loader`) to the 100K containers and consolidated labels.

**Window record (`Scene.window`):**
* `solid` bool [L, L, L] (Z, Y, X), once;
* `responses` bool [k, L, L, L], one channel per physical source (stable source index);
* `informative[j]`, `stream` (natural / coverage / natural_fallback; v1: thinned);
* `posteriors[j]`: dict(cells, q, lo, hi, tv_bound, status, certificate, engine, provenance) for a computed slot,
  otherwise dict(status=...) with status `not_computed`, `unavailable`, `engine_failure`, `timeout`, `memory_limit`.

**Target populations** (`flatten(population=...)` is required; RIND3D_StudentResearch/A_TASK_DEFINITION.md):
* `"Tstar"`: T* from every stored window (natural and coverage streams combined by the balance heuristic,
  `sampling.tstar_weight`): weight = 1 / sum_s n_s c_s(m) / Lambda_s, per natural proposal. Where a side has only the
  natural stream this is P_A / n_nat. Contract v1 containers (one m-thinned stream): weight = omega / (n_L c(m, L)).
* `"Tstar_natural"`: T* from the natural stream only (coverage windows are skipped): weight = P_A / n_nat.
* `"stored"`: the stored windows as they are, weight 1 (diagnostics only; it is not T*).
Weights are loss weights, never model inputs. Estimators of P_A: "rate", "quota", "combined", "combined_rb"
(default; strictly positive when the natural stream accepted >= 2 windows). The coverage acceptance probability is
always the Rao-Blackwellised sum_m c(m) rate_m / rate_n of the exact-m rate stream.

**Supervision:** certified posterior labels (statuses in `accept`, deterministic `certified` and statistical
`certified_stat`, tv_bound <= max_tv). **Normalisation rule:** the label of a slot is the EXACT normalisation
lo / sum(lo) of its stored per-cell binary64 volumes `lo` (`certified_stat`: lo = hi = vol(B ∩ cell)), and tv_bound
refers to that vector. The returned `q` is its binary64 evaluation `lo / lo.sum()`; for n <= 128^3 cells it is within total
variation gamma_(n+1) / 2 < 1.2e-10 of the exact normalisation (any summation order); it need not sum to 1 exactly
(RIND3D_Posterior/docs/FASTLABEL_V34_CERTIFICATION.md §6.3). `fallback="onehot"` gives an example without an accepted label the true-source
cell, marked `onehot_fallback`; it is never presented as a certified posterior. `supervision="onehot"` is the one-hot
baseline. Missing labels are never dropped silently (`missing`, `report`).

**Student inputs** are `student_inputs(example)`: SOLID crop, one response channel, L and the origin. m, mask, k, the
stream, weights, statuses, other channels and source coordinates are never inputs.

**Splits:** sha256(scene_id) mod 100: < 80 train, < 90 val, else test."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import sampling as SM
from .storage import SceneFile, scene_path, labels_path, read_jsonl, task_key

ALPHA_QUOTA = {64: 0.12, 128: 0.05, 256: 0.035}      # fixed inverse-variance mixing weights of the quota estimator
GRID = 128
ACCEPT_DEFAULT = ("certified", "certified_stat")
POPULATIONS = ("Tstar", "Tstar_natural", "stored")
ESTIMATORS = ("rate", "quota", "combined", "combined_rb")
STREAM_NAME = {0: "natural", 1: "coverage", 2: "natural_fallback"}
INPUT_KEYS = ("solid", "response", "L", "origin")


def split_of(scene_id):
    v = int(hashlib.sha256(scene_id.encode()).hexdigest(), 16) % 100
    return "train" if v < 80 else ("val" if v < 90 else "test")


def cell_of(s):
    i = np.minimum((np.asarray(s, float) * GRID).astype(np.int64), GRID - 1)
    return int((i[0] * GRID + i[1]) * GRID + i[2])


def student_inputs(ex):
    """The only fields a Student may read."""
    return {kk: ex[kk] for kk in INPUT_KEYS}


class Scene:
    def __init__(self, root, slot):
        self.root = Path(root); self.slot = int(slot)
        self.S = SceneFile(scene_path(root, slot))
        self.meta = self.S.meta
        self.side = {int(s["Lp"]): s for s in self.meta["sides"]}
        self.v2 = "sampling" in self.meta
        lp = labels_path(root, slot)
        self.lab = None
        if lp.exists():
            z = np.load(lp)
            self.lab = dict(cell=z["cell"], lo=z["lo"], hi=z["hi"], offsets=z["offsets"],
                            metas=json.loads(z["metas"].tobytes()), status=json.loads(z["status"].tobytes()))
            self.lab["by_key"] = {m["key"]: i for i, m in enumerate(self.lab["metas"])}

    # -------------------------------------------------------------- selection metadata and weights
    def stream(self, w):
        if not self.v2:
            return "thinned"
        return STREAM_NAME[int(self.S.W[w]["stream"])]

    def side_record(self, Lp):
        """v2 side record with per-stream accepted / proposals / c rows and the exact-m rate histogram."""
        return self.side[int(Lp)]

    def weights_v1(self, Lp):
        """Contract v1 (one m-thinned stream): estimators of omega = Lambda_L (docs/SELECTION_LAW_100K.md §3)."""
        Lp = int(Lp); sd = self.side[Lp]
        rate = float(sd["Lambda_hat"]); out = dict(rate=rate)
        r = int(sd["quota"]); n = int(sd["proposals"]) - int(sd["reasons"].get("repeat", 0))
        if n > 1 and r > 1:
            out["quota"] = (r - 1) / (n - 1)
            a = ALPHA_QUOTA[Lp]; out["combined"] = out["combined_rb"] = (1 - a) * rate + a * out["quota"]
        return out

    def population_weight(self, w, population, estimator="combined_rb"):
        """Loss weight of a mixed example of window w (see the module docstring); None if the window is not part of
        the population."""
        r = self.S.W[w]; Lp = int(r["Lp"]); m = max(int(r["m"]), 1)
        if population == "stored":
            return 1.0
        if population not in POPULATIONS:
            raise ValueError(f"unknown population {population!r}; declare one of {POPULATIONS}")
        if not self.v2:
            om = self.weights_v1(Lp); om = om.get(estimator, om["rate"])
            c = float(self.meta["c_table"][str(Lp)][m - 1]); n = int(self.meta["quota"][str(Lp)])
            if population == "Tstar_natural":
                raise ValueError("contract v1 has no natural stream (m-thinned windows); use population='Tstar'")
            return om / (n * c)
        sd = self.side_record(Lp); a = ALPHA_QUOTA[Lp]
        if population == "Tstar":
            return SM.tstar_weight(m, sd, estimator, a)
        if self.stream(w) == "coverage":
            return None                                   # Tstar_natural: natural-stream windows only
        return SM.natural_weight(sd, estimator, a)

    # -------------------------------------------------------------- labels and windows
    def posterior(self, w, j):
        key = task_key(self.slot, w, j)
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
        prov = {kk: m.get(kk) for kk in ("engine", "engine_version", "path", "config_sha256")}
        return dict(cells=cells, q=q, lo=Lc, hi=Uc, tv_bound=m["tv_bound"], status=m["status"], informative=inf,
                    certificate=m.get("certificate"), engine=m.get("engine"), provenance=prov)

    def window(self, w, with_labels=True):
        r = self.S.W[w]; k = int(r["k"])
        rec = dict(scene_id=self.meta["scene_id"], slot=self.slot, window=int(w), L=int(r["L"]), Lp=int(r["Lp"]),
                   origin=(int(r["x"]), int(r["y"]), int(r["z"])), k=k, m=int(r["m"]), stream=self.stream(w),
                   informative=[bool((int(r["mask"]) >> j) & 1) for j in range(k)],
                   solid=self.S.crop(w, -1), responses=np.stack([self.S.crop(w, j) for j in range(k)]))
        if with_labels:
            rec["posteriors"] = [self.posterior(w, j) for j in range(k)]
        return rec

    def sources(self):
        """Privileged: source coordinates [k, 3] (targets / evaluation only, never a model input)."""
        return self.S.sources()


class RIND3D100K:
    def __init__(self, root, split=None, slots=None):
        self.root = Path(root)
        acc = {}
        for rr in read_jsonl(self.root / "manifest" / "physical.jsonl"):
            if rr.get("status") == "accepted":
                acc[int(rr["slot"])] = rr
        self.records = acc
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


def side_means(ds, population, estimator="combined_rb"):
    """Mean loss weight per side over the mixed examples of ds (for optional side normalisation; metadata only)."""
    tot = {}
    for i in range(len(ds.slots)):
        S = ds.scene(i)
        for w, r in enumerate(S.S.W):
            m = int(r["m"]); wv = S.population_weight(w, population, estimator)
            if m == 0 or wv is None:
                continue
            a = tot.setdefault(int(r["Lp"]), [0.0, 0]); a[0] += m * wv; a[1] += m
    return {L: v[0] / v[1] for L, v in tot.items() if v[1]}


def flatten(ds, population=None, *, estimator="combined_rb", accept=ACCEPT_DEFAULT, max_tv=0.05, fallback=None,
            missing="yield", supervision="posterior", informative_only=True, with_source=False, side_norm=None,
            report=None, with_inputs=True):
    """Single-channel examples of the declared target population (module docstring). Every example carries ids, L,
    origin, inputs (solid, response; omitted with with_inputs=False), supervision, target ({cells, q} or None), label,
    status, tv_bound, certificate, provenance, population, stream, weight, and `privileged` (source, source_cell) when
    with_source or one-hot targets are requested. Only `student_inputs(example)` may be given to a model."""
    if population not in POPULATIONS:
        raise ValueError(f"flatten: declare the target population explicitly: one of {POPULATIONS}")
    if supervision not in ("posterior", "onehot"):
        raise ValueError("supervision must be 'posterior' or 'onehot'")
    if fallback not in (None, "onehot"):
        raise ValueError("fallback must be None or 'onehot'")
    if missing not in ("yield", "drop"):
        raise ValueError("missing must be 'yield' or 'drop'")
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    need_src = with_source or supervision == "onehot" or fallback == "onehot"
    for i in range(len(ds.slots)):
        S = ds.scene(i)
        src_all = S.sources() if need_src else None
        for w in range(len(S.S.W)):
            r = S.S.W[w]; k = int(r["k"]); Lp = int(r["Lp"])
            inf = [bool((int(r["mask"]) >> j) & 1) for j in range(k)]
            if informative_only and not any(inf):
                continue
            wv_win = S.population_weight(w, population, estimator)
            if wv_win is None:
                continue                                   # window not part of the population (Tstar_natural)
            rec = S.window(w, with_labels=True) if with_inputs else None
            for j in range(k):
                if informative_only and not inf[j]:
                    continue
                lab = rec["posteriors"][j] if rec is not None else S.posterior(w, j)
                st = lab.get("status")
                ok = st in accept and "cells" in lab and float(lab.get("tv_bound", 1.0)) <= max_tv
                wv = wv_win / side_norm[Lp] if side_norm else wv_win
                src = src_all[j] if need_src else None
                if supervision == "onehot":
                    sup = "onehot"; target = dict(cells=np.array([cell_of(src)]), q=np.array([1.0]))
                elif ok:
                    sup = "posterior"; target = dict(cells=lab["cells"], q=lab["q"])
                elif fallback == "onehot":
                    sup = "onehot_fallback"; target = dict(cells=np.array([cell_of(src)]), q=np.array([1.0]))
                else:
                    sup = "missing"; target = None
                if report is not None:
                    a = report.setdefault(f"L{Lp}:{sup}", [0, 0.0]); a[0] += 1; a[1] += float(wv or 0.0)
                    if sup != "posterior" and supervision == "posterior":
                        report.setdefault("status_not_accepted", {}).setdefault(str(st), 0)
                        report["status_not_accepted"][str(st)] += 1
                if sup == "missing" and missing == "drop":
                    continue
                ex = dict(scene_id=S.meta["scene_id"], slot=S.slot, window=w, source=j, L=int(r["L"]), Lp=Lp,
                          origin=(int(r["x"]), int(r["y"]), int(r["z"])), stream=S.stream(w), supervision=sup,
                          target=target, label=lab, status=st, tv_bound=lab.get("tv_bound"),
                          certificate=lab.get("certificate"), provenance=lab.get("provenance"), population=population,
                          estimator=estimator, weight=wv)
                if rec is not None:
                    ex.update(solid=rec["solid"], response=rec["responses"][j])
                if need_src:
                    ex["privileged"] = dict(source=src, source_cell=cell_of(src))
                yield ex
