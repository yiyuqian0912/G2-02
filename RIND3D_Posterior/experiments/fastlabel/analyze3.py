"""Summarise a runner3 jsonl (FastLabel v3): throughput, latency quantiles, paths, components, certificate stats.
  analyze3.py run.jsonl [summary_line_json_or_wall_seconds] [--weights]"""
import collections
import json
import sys

import numpy as np

N1K = 159802
SHARE_1K = {64: 0.2417, 128: 0.3769, 256: 0.3813}          # label shares of the 1K by side


def q(a, ps=(50, 90, 95, 99)):
    a = np.asarray(a, float)
    return {f"p{p}": round(float(np.percentile(a, p)), 2) for p in ps} if len(a) else {}


def main():
    rs = [json.loads(l) for l in open(sys.argv[1])]
    wall = None
    if len(sys.argv) > 2 and sys.argv[2] != "--weights":
        try:
            wall = float(json.loads(sys.argv[2])["wall"])
        except Exception:
            wall = float(sys.argv[2])
    err = [r for r in rs if "error" in r]; ok = [r for r in rs if "error" not in r]
    tw = np.array([r["task_wall"] for r in ok])
    print(f"labels {len(rs)}  errors {len(err)}  worker-s total {tw.sum():.0f}  mean {tw.mean():.3f}  {q(tw)}  max {tw.max():.1f}")
    for e in err[:5]:
        print("  ERROR", e["sc"], e["w"], e["s"], e["error"][:200])
    paths = collections.Counter((r["path"], r["L"]) for r in ok)
    print("paths:", dict(sorted(paths.items())))
    comp = collections.Counter()
    for r in ok:
        t = dict(r["timing"]); t["p1"] = sum(x["p1_s"] for x in r["rounds"])
        t["rest"] = r["task_wall"] - sum(t.values())
        for k, v in t.items():
            comp[k] += v
    tot = tw.sum()
    print("components (share of worker time):", {k: round(v / tot, 3) for k, v in comp.most_common()})
    byL = {}
    for L in (64, 128, 256):
        a = np.array([r["task_wall"] for r in ok if r["L"] == L])
        if len(a):
            byL[L] = a.mean()
            print(f"  L{L}: n {len(a)} mean {a.mean():.2f} {q(a)} max {a.max():.1f}")
    st = [r for r in ok if r["path"] == "stat_certified"]
    if st:
        K = np.array([r["K"] for r in st]); rd = np.array([r["round"] for r in st])
        tests = np.array([sum(x.get("tested", 0) for x in r["rounds"]) for r in st])
        print(f"stat: {len(st)}  rounds {collections.Counter(rd.tolist())}  K mean {K.mean():.2f} max {K.max()}  tests/label mean {tests.mean():.0f} max {tests.max()}"
              f"  p_lower min {min(r['p_lower'] for r in st):.4f}")
    ex = [r for r in ok if r.get("tv_bound") is not None]
    if ex:
        print(f"deterministic: {len(ex)}  max tv_bound {max(r['tv_bound'] for r in ex):.3g}  (> 0.05: {sum(r['tv_bound'] > 0.05 for r in ex)})")
    tv = [r["tv_vs_ref"] for r in ok if r.get("tv_vs_ref") is not None]
    if tv:
        print(f"vs reference labels: n {len(tv)} max TV {max(tv):.4f} mean {np.mean(tv):.4f}")
    mix = sum(SHARE_1K[L] * byL.get(L, 0) for L in SHARE_1K)
    print(f"1K-mix worker-s/label {mix:.3f}  -> {N1K * mix / 16 / 3600:.2f} h at 16 workers, {N1K * mix / 15 / 3600:.2f} h at 15 (no tail)")
    if wall:
        lps = len(rs) / wall
        print(f"machine: {len(rs)} labels in {wall:.0f} s = {lps:.2f} labels/s -> 1K {N1K / lps / 3600:.2f} h (this sample's mix, incl. tail)")


if __name__ == "__main__" and "--strata" not in sys.argv:
    main()


def strata(path):
    """Per-family and per-k (sources in the scene) breakdown of a runner3 jsonl."""
    import collections
    R = "/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes"
    rs = [json.loads(l) for l in open(path)]; meta = {}
    for r in rs:
        if r["sc"] not in meta:
            d = json.load(open(f"{R}/{r['sc']}/scene.json")); meta[r["sc"]] = (d["family"], d["k"])
    for name, key in (("family", 0), ("k", 1)):
        g = collections.defaultdict(list)
        for r in rs:
            g[meta[r["sc"]][key]].append(r)
        print(f"by {name}:")
        for kk in sorted(g):
            v = g[kk]; tw = np.array([r["task_wall"] for r in v]); err = sum("error" in r for r in v)
            st = sum(r.get("path") == "stat_certified" for r in v)
            print(f"  {str(kk):18s} n {len(v):4d} err {err} mean {tw.mean():.2f} p95 {np.percentile(tw, 95):.1f} max {tw.max():.1f} stat {st / len(v):.2f}")


if __name__ == "__main__" and "--strata" in sys.argv:
    strata(sys.argv[1])
