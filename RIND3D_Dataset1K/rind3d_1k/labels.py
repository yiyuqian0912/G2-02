"""RIND3D 1K, label stage: one certified single-source posterior per (scene, window, source).

Runs in the labeller environment (numba, optional CuPy) and does NOT need the producer natives.
Inputs per scene:
* public/observations.bin with its index: crops in ZYX order, packed bits, zstd;
* public/windows.npy;
* privileged/geometry_rows.npz.

Output: one file per task, labels/tasks/w{window:03d}_s{source}.npz with fields cell, lo, hi, kind and tv_bound,
written atomically. The scene's labels/LABELS.json is rebuilt from the task files.

Posterior target of a task (docs/SELECTION_LAW_1K.md): Uniform(F_W^j), with F_W^j = {s in D(G) minus the closed
box B(W) : s reproduces R_j on every FREE receiver of W}. The engine certifies per-cell volume intervals and a
deterministic TV bound for this target. The label is the posterior of the 1K sampling law under the documented
weights; the selection audit is in the doc."""
import json
import os
import time
from pathlib import Path

import numpy as np
import zstandard

STATUS_CERT = "certified"          # TV bound <= 0.05 for the declared target
STATUS_LOOSE = "certified_loose"   # 0.05 < TV bound <= 0.10
STATUS_UNCERT = "uncertified"      # TV bound > 0.10 (stored, but never presented as qualified)


class SceneObs:
    def __init__(self, scene_dir):
        self.dir = Path(scene_dir)
        self.W = np.load(self.dir / "public" / "windows.npy")
        self.index = np.load(self.dir / "public" / "observations_index.npy")
        self.blob = (self.dir / "public" / "observations.bin").read_bytes()
        self.dz = zstandard.ZstdDecompressor()
        g = np.load(self.dir / "privileged" / "geometry_rows.npz")
        self.P, self.K = g["P"], g["K"]
        self.scene = json.loads((self.dir / "scene.json").read_text())
        self._pos = {(int(w), int(s)): (int(o), int(n)) for w, s, o, n in self.index}

    def crop(self, w, s):
        """bool [Z, Y, X] crop: s = -1 for SOLID, else the response (lit) of source s."""
        o, n = self._pos[(w, s)]
        L = int(self.W[w]["L"])
        bits = np.frombuffer(self.dz.decompress(self.blob[o:o + n]), np.uint8)
        return np.unpackbits(bits, count=L ** 3).astype(bool).reshape(L, L, L)

    def cls(self, w, s):
        """Labeller classes cls[x, y, z]: 0 dark FREE, 1 lit FREE, 2 SOLID."""
        sol = self.crop(w, -1); lit = self.crop(w, s)
        c = np.where(sol, 2, np.where(lit, 1, 0)).astype(np.uint8)
        return np.ascontiguousarray(np.transpose(c, (2, 1, 0)))

    def origin(self, w):
        r = self.W[w]
        return np.array([r["x"], r["y"], r["z"]], np.int64)


def tasks_for_scene(scene_dir, include_uninformative=False):
    S = SceneObs(scene_dir)
    out = []
    for r in S.W:
        for j in range(int(r["k"])):
            inf = bool((int(r["mask"]) >> j) & 1)
            if inf or include_uninformative:
                out.append((str(scene_dir), int(r["window"]), j, int(r["L"]), inf))
    return out


def task_path(scene_dir, w, s):
    return Path(scene_dir) / "labels" / "tasks" / f"w{w:03d}_s{s}.npz"


_CACHE = {}


def run_task(t, engine, **kw):
    """Label one (scene, window, source). Returns a small metadata dict."""
    scene_dir, w, s, L, inf = t
    out = task_path(scene_dir, w, s)
    if out.exists():
        return dict(skip=True)
    S = _CACHE.get(scene_dir)
    if S is None:
        _CACHE.clear(); S = _CACHE[scene_dir] = SceneObs(scene_dir)
    from rind3d_posterior import label_window, ENGINE_VERSION
    cls = S.cls(w, s)
    c0 = time.process_time(); t0 = time.perf_counter()
    r = label_window(S.P, S.K, cls, S.origin(w), int(S.scene["resolution"]), engine=engine, **kw)
    cpu = time.process_time() - c0; wall = time.perf_counter() - t0
    tv = float(r["tv_bound"])
    status = STATUS_CERT if tv <= 0.05 else (STATUS_LOOSE if tv <= 0.10 else STATUS_UNCERT)
    meta = dict(window=w, source=s, L=L, informative=inf, tv_bound=tv, status=status, engine=engine,
                engine_version=ENGINE_VERSION, cpu=cpu, wall=wall, n_records=int(len(r["cell"])),
                vol_lo=float(np.sum(r["lo"])), vol_hi=float(np.sum(r["hi"])),
                entry_evals=int(r["stats"]["entry_evals"]), nodes=int(r["stats"]["nodes"]),
                engine_stats={kk: int(r["stats"][kk]) for kk in ("overflow", "budget_undet", "max_entries", "cells_out",
                                                                  "passes") if kk in r["stats"]},
                n_undet=int(np.sum(np.asarray(r["kind"]) == 2)),
                vol_undet=float(np.sum(np.asarray(r["hi"])[np.asarray(r["kind"]) == 2])),
                params={kk: v for kk, v in kw.items() if isinstance(v, (int, float, str))},
                gpu=(dict((k, v) for k, v in r["gpu"].items() if k != "tt") if r.get("gpu") else None))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, cell=np.asarray(r["cell"], np.int32), lo=r["lo"], hi=r["hi"],
                        kind=np.asarray(r["kind"], np.int8), tv_bound=tv, meta=json.dumps(meta))
    os.replace(tmp, out)
    return meta


def scene_label_summary(scene_dir, include_uninformative=False):
    """Collect task metadata into labels/LABELS.json; returns coverage counts."""
    S = SceneObs(scene_dir)
    rows, missing = [], []
    for r in S.W:
        for j in range(int(r["k"])):
            p = task_path(scene_dir, int(r["window"]), j)
            inf = bool((int(r["mask"]) >> j) & 1)
            if p.exists():
                with np.load(p) as z:
                    rows.append(json.loads(str(z["meta"])))
            else:
                rows.append(dict(window=int(r["window"]), source=j, L=int(r["L"]), informative=inf,
                                 status="unavailable"))
                if inf or include_uninformative:
                    missing.append((int(r["window"]), j))
    cov = {}
    for x in rows:
        key = ("inf" if x["informative"] else "uninf") + ":" + x["status"]
        cov[key] = cov.get(key, 0) + 1
    tmp = Path(scene_dir) / "labels" / "LABELS.json.tmp"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(dict(coverage=cov, labels=rows), indent=0))
    os.replace(tmp, Path(scene_dir) / "labels" / "LABELS.json")
    return cov, missing
