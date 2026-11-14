"""Validators for RIND3D 1K scenes.

validate_scene(dir) checks, with numpy and zstandard only:
* integrity: SEAL.json sha256 of every physical-stage file;
* contract: k in 1..4; exactly 64 windows with 16/24/24 of physical sides 64/128/256; valid, distinct origins;
* sources: in (0,1)^3; pairwise |.|_inf >= 2^-20; every source outside every CLOSED window box (continuous);
* windows: decoded SOLID / response crops complete; FREE > 0; no lit SOLID receiver; informative flags, mask and m
  recomputed from the crops; m >= 1;
* labels (if present): task files aligned with (window, source); metadata consistent; status consistent with the
  TV bound;
* views: no source coordinates in public/.

physics_spot_check(dir, n_points) re-renders random receivers of random windows with the canonical certified CPU
point predicate (rind3d native, interval clipping with rational fallback). This is independent of the CUDA field
renderer, and it also runs the rational shadow-projection predicate on a subsample. It needs the producer env."""
import hashlib
import json
from pathlib import Path

import numpy as np
import zstandard

SIDES = {64: 16, 128: 24, 256: 24}


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def validate_scene(d, check_seal=True, max_tv=0.05):
    d = Path(d)
    errs = []
    sc = json.loads((d / "scene.json").read_text())
    N = sc["resolution"]; k = sc["k"]
    if check_seal:
        seal = json.loads((d / "SEAL.json").read_text())
        for rel, h in seal.items():
            p = d / rel
            if not p.exists() or _sha(p) != h:
                errs.append(f"seal:{rel}")
    if not 1 <= k <= 4:
        errs.append("k")
    W = np.load(d / "public" / "windows.npy")
    if len(W) != 64:
        errs.append(f"n_windows={len(W)}")
    for Lp, n in SIDES.items():
        if int(np.sum(W["Lp"] == Lp)) != n:
            errs.append(f"quota L{Lp}")
    keys = set()
    for r in W:
        L = int(r["L"])
        if int(r["Lp"]) * N // 1024 != L:
            errs.append("L/Lp")
        for a in ("x", "y", "z"):
            if not 0 <= int(r[a]) <= N - L:
                errs.append(f"origin w{int(r['window'])}")
        key = (int(r["x"]), int(r["y"]), int(r["z"]), L)
        if key in keys:
            errs.append(f"duplicate w{int(r['window'])}")
        keys.add(key)
    src = json.loads((d / "privileged" / "sources.json").read_text())
    S = np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]])
    if S.shape != (k, 3) or not np.all((S > 0) & (S < 1)):
        errs.append("sources")
    for i in range(k):
        for j in range(i + 1, k):
            if np.max(np.abs(S[i] - S[j])) < 2.0 ** -20:
                errs.append("duplicate sources")
    for r in W:
        lo = np.array([r["x"], r["y"], r["z"]], float) / N; hi = (np.array([r["x"], r["y"], r["z"]], float) + r["L"]) / N
        if np.any(np.all((S >= lo) & (S <= hi), axis=1)):
            errs.append(f"source in closed box w{int(r['window'])}")
    idx = np.load(d / "public" / "observations_index.npy")
    pos = {(int(w), int(s)): (int(o), int(n)) for w, s, o, n in idx}
    blob = (d / "public" / "observations.bin").read_bytes()
    dz = zstandard.ZstdDecompressor()
    mh = np.zeros(5, int)
    ov = np.zeros(5, int)
    for r in W:
        w = int(r["window"]); L = int(r["L"])
        def crop(s):
            o, n = pos[(w, s)]
            return np.unpackbits(np.frombuffer(dz.decompress(blob[o:o + n]), np.uint8), count=L ** 3).astype(bool)
        if (w, -1) not in pos or any((w, j) not in pos for j in range(k)):
            errs.append(f"obs index w{w}"); continue
        sol = crop(-1); free = ~sol
        if not free.any():
            errs.append(f"no FREE w{w}")
        mask = 0; tot = np.zeros(int(free.sum()), np.int8)
        for j in range(k):
            lit = crop(j)
            if np.any(lit & sol):
                errs.append(f"lit SOLID w{w} s{j}")
            nl = int((lit & free).sum()); nd = int((free & ~lit).sum())
            if nl > 0 and nd > 0:
                mask |= 1 << j
            tot += lit[free]
        if mask != int(r["mask"]) or bin(mask).count("1") != int(r["m"]) or int(r["m"]) < 1:
            errs.append(f"layers w{w}")
        mh[int(r["m"])] += 1
        ov[:k + 1] += np.bincount(tot, minlength=k + 1)[:k + 1]
    # labels
    labs = {}
    tdir = d / "labels" / "tasks"
    if tdir.exists():
        for p in sorted(tdir.glob("w*_s*.npz")):
            w, j = int(p.stem[1:4]), int(p.stem.split("_s")[1])
            with np.load(p) as z:
                m = json.loads(str(z["meta"])); tv = float(z["tv_bound"])
                ok = len(z["cell"]) == len(z["lo"]) == len(z["hi"]) and np.all(z["hi"] >= z["lo"] - 1e-300)
            if m["window"] != w or m["source"] != j or not ok:
                errs.append(f"label align w{w} s{j}")
            if j >= k or w >= len(W):
                errs.append(f"label out of range w{w} s{j}")
            elif m["informative"] != bool((int(W[w]["mask"]) >> j) & 1):
                errs.append(f"label informative flag w{w} s{j}")
            if (m["status"] == "certified") != (tv <= max_tv):
                errs.append(f"label status w{w} s{j}")
            labs[m["status"]] = labs.get(m["status"], 0) + 1
    pub = b"".join((d / "public" / f).read_bytes()[:0] for f in ("windows.npy",))
    if "sources" in np.load(d / "public" / "windows.npy").dtype.names:
        errs.append("sources leaked to public")
    return dict(scene=d.name, ok=not errs, errors=errs[:20], n_errors=len(errs), k=k, m_hist=mh[1:].tolist(),
                overlap_hist=ov.tolist(), labels=labs)


def physics_spot_check(d, n_points=4096, n_windows=4, independent_points=64, seed=0):
    """Compare stored crops with the canonical certified point predicate at random receivers."""
    from rind3d.geometry import load_geometry
    from rind3d.rendering.backends import BackendFactory
    from .physical import POLICY
    d = Path(d)
    sc = json.loads((d / "scene.json").read_text()); N = sc["resolution"]
    bundle = load_geometry(d / "privileged" / "geometry.npz")
    fac = BackendFactory(cuda_allowed=False)
    ctx = fac.context(bundle, POLICY)
    src = json.loads((d / "privileged" / "sources.json").read_text())
    S = np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]])
    W = np.load(d / "public" / "windows.npy")
    idx = np.load(d / "public" / "observations_index.npy")
    pos = {(int(w), int(s)): (int(o), int(n)) for w, s, o, n in idx}
    blob = (d / "public" / "observations.bin").read_bytes(); dz = zstandard.ZstdDecompressor()
    rng = np.random.default_rng(seed)
    mism = 0; checked = 0; ind_mism = 0; ind_checked = 0
    for w in rng.choice(len(W), size=min(n_windows, len(W)), replace=False):
        r = W[w]; L = int(r["L"]); o = np.array([r["x"], r["y"], r["z"]])
        zyx = rng.integers(0, L, size=(n_points, 3))
        X = (o[None, :] + zyx[:, ::-1] + 0.5) / N
        for j in range(int(r["k"])):
            oo, n = pos[(int(w), j)]
            lit = np.unpackbits(np.frombuffer(dz.decompress(blob[oo:oo + n]), np.uint8), count=L ** 3).astype(bool).reshape(L, L, L)
            stored = lit[zyx[:, 0], zyx[:, 1], zyx[:, 2]]
            b = ctx.bind(S[j])
            y, m, _ = b.native.points(X)
            mism += int(np.sum((y == 1) != stored)); checked += len(X)
            bi = ctx.bind(S[j], independent=True)
            yi, mi, _ = bi.native.points(X[:independent_points])
            ind_mism += int(np.sum((yi == 1) != stored[:independent_points])); ind_checked += independent_points
    return dict(scene=d.name, checked=checked, mismatches=mism, independent_checked=ind_checked,
                independent_mismatches=ind_mism)
