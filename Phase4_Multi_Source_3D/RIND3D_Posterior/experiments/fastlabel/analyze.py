import json, sys, collections, numpy as np
f = sys.argv[1]
refs = {(r["sc"], r["w"], r["s"]): r for r in map(json.loads, open("refs.jsonl")) if "cpu" in r}
side = {(a, int(b), int(c)): int(L) for L, a, b, c, fm in (l.split() for l in open("random120.txt"))}
sh = {64: 0.2417, 128: 0.3769, 256: 0.3813}
rs = [json.loads(l) for l in open(f)]
fail = [r for r in rs if r.get("failed")]
rs = [r for r in rs if not r.get("failed")]
print("results", len(rs), "failed", len(fail), [(r["sc"], r["w"], r["s"]) for r in fail])
agg = collections.defaultdict(lambda: dict(w=[], ref=[], tiles=[], paths=collections.Counter(), tv=[], refunc=0))
for r in rs:
    k = (r["sc"], r["w"], r["s"]); L = side[k]; a = agg[L]
    a["w"].append(r["wall"]); a["paths"][r["path"]] += 1; a["tiles"].append((r["tester"] or {}).get("tiles", 0))
    if k in refs:
        a["ref"].append(refs[k]["cpu"])
        if refs[k]["tv_bound"] > 0.05: a["refunc"] += 1
        elif r.get("tv_vs_ref") is not None: a["tv"].append(r["tv_vs_ref"])
mix_new = mix_ref = mix_tiles = 0
for L in (64, 128, 256):
    a = agg[L]; w = np.array(a["w"]); ref = np.array(a["ref"])
    mix_new += sh[L] * w.mean(); mix_ref += sh[L] * ref.mean(); mix_tiles += sh[L] * np.mean(a["tiles"])
    print(f"L{L}: n {len(w)} new mean {w.mean():.2f} p50 {np.median(w):.2f} p95 {np.percentile(w,95):.1f} max {w.max():.1f} | ref mean {ref.mean():.1f} (n {len(ref)}, uncertified refs {a['refunc']}) x{ref.mean()/w.mean():.1f} | max TV vs certified refs {max(a['tv']):.4f} | tiles/label {np.mean(a['tiles']):.0f} | {dict(a['paths'])}")
print(f"1K mix: new {mix_new:.2f} s/label, ref {mix_ref:.1f} s/label (x{mix_ref/mix_new:.1f}); 159,802 labels: {159802*mix_new/16/3600:.1f} h on 16 workers; GPU tiles/label {mix_tiles:.0f} -> {159802*mix_tiles*0.15e-3/3600:.1f}-{159802*mix_tiles*0.3e-3/3600:.1f} GPU-h at 0.15-0.30 ms/tile")
