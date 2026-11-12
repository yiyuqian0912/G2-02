"""obs-index-v0.3: expanded, extensible window index over stored complete fields (no physics regenerated).

The v0.2 observation policy is a *prefix* of v0.3: every subset of a scene and side is one deterministic
uniform proposal stream (v0.2 streams: keyed_rng(seed, slot, STREAM*8 + side)), and a larger profile simply
continues the same streams. For the canary (public integration seed) P102 reproduces the v0.2 records bit for
bit; for any release the seed is private (HMAC of a private 256-bit index key and the scene id), because
replaying a public proposal stream would reveal which mixed proposals were rejected for containing a source.

Profiles (per side L = 64/128/256):
  P102  natural 16/16/16, informative 8, extreme 4, rare 2, nonlocal 4            (= obs-policy-v0.2, 102)
  P256  natural 41/40/40, informative 20, extreme 11, rare 6, nonlocal 8          (256)
Every record carries exact counts and every rule indicator (A_informative, A_extreme, A_rare evaluated for
ALL records and ALL proposals, so inclusion probabilities can be reconstructed), the source-containment flag
(privileged), the no-FREE flag and a nonlocal estimate for mixed source-excluded records. A fixed-size
uniform rate stream (RATE_M proposals per side, stream 3051) gives unbiased per-scene P(A_s) estimates and
the natural category distribution. Outputs are privileged (they encode the source-containment of windows).
"""
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from rind3d.geometry import keyed_rng

from rind3d_production import windows as W2

from .env import INDEX_POLICY, OUT, keyed_int, private_secret

SIDES = (64, 128, 256)
SUBSETS = ("natural", "informative", "extreme", "rare", "nonlocal")
CODES = {s: i for i, s in enumerate(SUBSETS)}
STREAM = dict(W2.STREAM, rate=3051)
PROFILES = {
    "P102": {"natural": {64: 16, 128: 16, 256: 16}, "informative": 8, "extreme": 4, "rare": 2, "nonlocal": 4},
    "P256": {"natural": {64: 41, 128: 40, 256: 40}, "informative": 20, "extreme": 11, "rare": 6, "nonlocal": 8},
}
CAP_SCALE = 2.5                        # P256 proposal caps = v0.2 caps x 2.5 (same proposals per accepted window)
RATE_M = 1024
EXTREME, RARE, NONLOCAL, NL_SAMPLES = W2.EXTREME, W2.RARE, W2.NONLOCAL, W2.NL_SAMPLES

REC = np.dtype([("scene", "<u4"), ("window", "<u4"), ("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("L", "<u2"),
                ("subset", "u1"), ("rank", "<u4"), ("proposal_index", "<u4"),
                ("free", "<u4"), ("solid", "<u4"), ("lit", "<u4", (4,)),
                ("contains_source", "?"), ("no_free", "?"), ("category", "u1"),
                ("minority_fraction", "<f4"), ("minority_count", "<u4"),
                ("A_informative", "?"), ("A_extreme", "?"), ("A_rare", "?"),
                ("nonlocal_fraction", "<f4"), ("nonlocal_flag", "?")])
RATE = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("L", "<u2"), ("free", "<u4"), ("solid", "<u4"),
                 ("lit", "<u4", (4,)), ("contains_source", "?"), ("category", "u1"), ("minority_fraction", "<f4"),
                 ("minority_count", "<u4"), ("A_informative", "?"), ("A_extreme", "?"), ("A_rare", "?")])
CATEGORIES = {0: "no_free", 1: "all_lit", 2: "all_shadow", 3: "mixed", 4: "constant_lit"}


def category(free, lit, K):
    """SB: no_free / all_lit / all_shadow / mixed. MA: no_free / all_shadow (every channel dark) /
    constant_lit (every channel constant over FREE, some lit) / mixed (combined response non-constant)."""
    if free == 0:
        return 0
    lit = list(lit[:K])
    if K == 1:
        return 1 if lit[0] == free else 2 if lit[0] == 0 else 3
    if all(v == 0 for v in lit):
        return 2
    if all(v in (0, free) for v in lit):
        return 4
    return 3


def rules(free, lit, cs, K):
    mf, mc = W2.minority(free, list(lit[:K]))
    inf = mf > 0 and not cs
    return mf, mc, inf, inf and mf <= EXTREME, inf and mc <= RARE


def scene_seed(sc, key=None):
    """Canary: the public integration seed and slot (reproduces v0.2). Releases: private HMAC seed, slot 0."""
    if sc.release == "canary-v0.2":
        cfg = json.loads((sc.dir.parents[3]/"config.json").read_text())
        return int(cfg["seed"]), int(sc.v02_slot)
    return keyed_int(key, INDEX_POLICY, sc.release, sc.scene_id, bits=64) >> 1, 0


def build_scene(sc, ordinal, profile="P256", key=None, out_dir=None):
    from rind3d.rendering.context import GeometryContext
    from rind3d.geometry import load_geometry
    from rind3d_production.env import POLICY
    t0 = time.perf_counter()
    prof = PROFILES[profile]
    seed, slot = scene_seed(sc, key)
    sources, _ = sc.privileged_labels()
    K = sc.K
    packs, sols, cis = sc.open_counts()
    ctx = GeometryContext(load_geometry(sc.geometry), POLICY, source_cache_size=64)

    def lit_window_fn(o, L):
        return packs[0].read_window(*o, L), sols[0].read_window(*o, L)

    def counts(o, L):
        return W2.window_counts(cis, o, L)
    recs, report, pooled = [], {}, {}
    rate_rows = []
    for L in SIDES:
        si = SIDES.index(L)
        hits = {"proposals": 0, "A_informative": 0, "A_extreme": 0, "A_rare": 0, "contains_source": 0}
        for subset in SUBSETS:
            rng = keyed_rng(seed, slot, STREAM[subset]*8 + si)
            quota = prof["natural"][L] if subset == "natural" else prof[subset]
            cap = quota if subset == "natural" else int(W2.CAPS[subset]*(CAP_SCALE if profile != "P102" else 1))
            nl_rng = np.random.default_rng(np.random.SeedSequence([seed, slot, L, 77]))
            got = proposals = 0
            while got < quota and proposals < cap:
                proposals += 1
                o = tuple(int(v) for v in rng.integers(0, 1024-L+1, 3))
                free, solid, lit = counts(o, L)
                cs = W2.contains_source(o, L, sources)
                mf, mc, inf, ext, rare = rules(free, lit, cs, K)
                hits["proposals"] += 1
                hits["A_informative"] += inf
                hits["A_extreme"] += ext
                hits["A_rare"] += rare
                hits["contains_source"] += cs
                nl = np.nan
                ok = {"natural": True, "informative": inf, "extreme": ext, "rare": rare, "nonlocal": inf}[subset]
                if ok and subset == "nonlocal":
                    v = W2.nonlocal_fraction(ctx, sources, lit_window_fn, o, L, nl_rng)
                    nl = np.nan if v is None else v
                    ok = v is not None and v >= NONLOCAL
                if not ok:
                    continue
                if subset != "nonlocal" and inf:          # diagnostic estimate, independent recorded stream
                    r2 = np.random.default_rng(np.random.SeedSequence([seed, slot, L, 79, CODES[subset], proposals]))
                    v = W2.nonlocal_fraction(ctx, sources, lit_window_fn, o, L, r2)
                    nl = np.nan if v is None else v
                r = np.zeros((), REC)
                r["scene"], r["x"], r["y"], r["z"], r["L"] = ordinal, o[0], o[1], o[2], L
                r["subset"], r["rank"], r["proposal_index"] = CODES[subset], got, proposals
                r["free"], r["solid"] = free, solid
                lit4 = np.zeros(4, np.uint32)
                lit4[:K] = lit
                r["lit"] = lit4
                r["contains_source"], r["no_free"], r["category"] = cs, free == 0, category(free, lit4, K)
                r["minority_fraction"], r["minority_count"] = mf, mc
                r["A_informative"], r["A_extreme"], r["A_rare"] = inf, ext, rare
                r["nonlocal_fraction"] = nl
                r["nonlocal_flag"] = bool(inf and np.isfinite(nl) and nl >= NONLOCAL)
                recs.append(r)
                got += 1
            report[f"{subset}/{L}"] = {"quota": quota, "accepted": got, "proposals": proposals, "cap": cap,
                                       "shortfall": quota-got}
        rng = keyed_rng(seed, slot, STREAM["rate"]*8 + si)
        rh = {"proposals": RATE_M, "A_informative": 0, "A_extreme": 0, "A_rare": 0, "contains_source": 0}
        for _ in range(RATE_M):
            o = tuple(int(v) for v in rng.integers(0, 1024-L+1, 3))
            free, solid, lit = counts(o, L)
            cs = W2.contains_source(o, L, sources)
            mf, mc, inf, ext, rare = rules(free, lit, cs, K)
            rh["A_informative"] += inf
            rh["A_extreme"] += ext
            rh["A_rare"] += rare
            rh["contains_source"] += cs
            r = np.zeros((), RATE)
            r["x"], r["y"], r["z"], r["L"], r["free"], r["solid"] = o[0], o[1], o[2], L, free, solid
            lit4 = np.zeros(4, np.uint32)
            lit4[:K] = lit
            r["lit"], r["contains_source"], r["category"] = lit4, cs, category(free, lit4, K)
            r["minority_fraction"], r["minority_count"] = mf, mc
            r["A_informative"], r["A_extreme"], r["A_rare"] = inf, ext, rare
            rate_rows.append(r)
        pooled[str(L)] = {"rate_stream": rh, "all_streams": hits}
    arr = np.array(recs, REC)
    arr["window"] = np.arange(len(arr))
    rates = np.array(rate_rows, RATE)
    for p in packs + sols:
        p.close()
    meta = {"policy": INDEX_POLICY, "profile": profile, "release": sc.release, "scene_ordinal": ordinal,
            "scene_id": sc.scene_id, "profile_K": K, "data_profile": sc.profile, "split": sc.split, "family": sc.family,
            "seed_kind": "public-integration (canary)" if sc.release == "canary-v0.2" else "private HMAC",
            "report": report, "rule_hits": pooled, "seconds": time.perf_counter()-t0}
    if out_dir is not None:
        d = Path(out_dir)/f"{ordinal:05d}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d/"records.npy", arr)
        np.save(d/"rates.npy", rates)
        (d/"report.json").write_text(json.dumps(meta, indent=0, default=float))
    return arr, rates, meta


def v02_prefix_check(sc, arr):
    """Canary only: the P102 prefix of the expanded index equals the stored v0.2 records."""
    old = np.load(sc.v02_records)
    out = {"records_v02": len(old)}
    ok = True
    for L in SIDES:
        for subset in SUBSETS:
            q = W2.QUOTA[subset]
            cap = W2.CAPS.get(subset, q)                 # v0.2 proposal cap: later acceptances are P256-only
            a = arr[(arr["L"] == L) & (arr["subset"] == CODES[subset]) & (arr["rank"] < q) & (arr["proposal_index"] <= cap)]
            b = old[(old["L"] == L) & (old["subset"] == CODES[subset])]
            same = len(a) == len(b) and all(np.array_equal(a[k], b[k]) for k in
                                            ("x", "y", "z", "free", "solid", "lit", "contains_source", "proposal_index"))
            same = same and np.allclose(a["minority_fraction"], b["minority_fraction"])
            if same and subset == "nonlocal":
                same &= np.array_equal(np.nan_to_num(a["nonlocal_fraction"], nan=-1), np.nan_to_num(b["nonlocal_fraction"], nan=-1))
            ok &= bool(same)
    out["prefix_identical"] = ok
    return out


def _job(args):
    sc, ordinal, profile, key, out_dir = args
    arr, rates, meta = build_scene(sc, ordinal, profile, key, out_dir)
    chk = v02_prefix_check(sc, arr) if sc.release == "canary-v0.2" else None
    return ordinal, len(arr), meta["seconds"], chk


def main():
    import argparse
    from .scenes import canary_scenes, v01_scenes
    ap = argparse.ArgumentParser()
    ap.add_argument("release", choices=["canary", "v01"])
    ap.add_argument("--profile", default="P256")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    scenes = canary_scenes() if a.release == "canary" else v01_scenes(limit=a.limit)
    key = None if a.release == "canary" else private_secret("index_key", create=True)
    out_dir = OUT/"privileged"/"index"/{"canary": "canary-v0.2", "v01": "v0.1-1k"}[a.release]/a.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    (OUT/"privileged").chmod(0o700)
    t = time.perf_counter()
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, [(sc, k, a.profile, key, out_dir) for k, sc in enumerate(scenes)])):
            rows.append(r)
            if i % 50 == 0:
                print(i, r[:3], flush=True)
    scenes_json = [{"ordinal": k, "scene_id": sc.scene_id, "geometry_id": sc.geometry_id, "split": sc.split, "family": sc.family,
                    "profile": sc.profile, "K": sc.K, "dir": str(sc.dir)} for k, sc in enumerate(scenes)]
    checks = [r[3] for r in rows if r[3] is not None]
    summary = {"policy": INDEX_POLICY, "profile": a.profile, "release": a.release, "scenes": len(scenes),
               "records": int(sum(r[1] for r in rows)), "wall_seconds": time.perf_counter()-t, "workers": a.workers,
               "scene_seconds_median": float(np.median([r[2] for r in rows])),
               "scene_seconds_max": float(max(r[2] for r in rows)),
               "v02_prefix_identical": None if not checks else f"{sum(c['prefix_identical'] for c in checks)}/{len(checks)}"}
    (out_dir/"scenes.json").write_text(json.dumps(scenes_json, indent=0))
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
