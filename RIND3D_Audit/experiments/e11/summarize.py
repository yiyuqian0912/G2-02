"""Summarise out/e11_fastlabel_extreme.jsonl into a markdown table by (side, type)."""
import json, sys, collections
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "out/e11_fastlabel_extreme.jsonl")]
g = collections.defaultdict(list)
for r in rows:
    g[(r["L"], r["type"])].append(r)
print("| side | type | n | certified (det.) | certified_stat | uncertified / loose | timeout (150 s) or error | wall s median (max) | label volume range | cells median |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (L, t), rr in sorted(g.items()):
    st = collections.Counter(r.get("status") or ("timeout" if "timeout" in r.get("error", "") else "error") for r in rr)
    ok = [r for r in rr if r.get("status")]
    walls = [r["wall"] for r in ok]
    vols = [r.get("vol_label", 0) for r in ok]
    cells = [r.get("n_cells") for r in ok if r.get("n_cells")]
    print(f"| {L} | {t} | {len(rr)} | {st.get('certified', 0)} | {st.get('certified_stat', 0)} | {st.get('uncertified', 0) + st.get('certified_loose', 0)} | "
          f"{st.get('timeout', 0) + st.get('error', 0)} | {np.median(walls):.1f} ({max(walls):.1f}) | {min(vols):.3f}–{max(vols):.3f} | {int(np.median(cells)) if cells else '-'} |" if ok else
          f"| {L} | {t} | {len(rr)} | 0 | 0 | 0 | {st.get('timeout', 0) + st.get('error', 0)} | - | - | - |")
unc = [r for r in rows if r.get("status") in ("uncertified", "certified_loose")]
print()
print("Uncertified/loose labels: " + "; ".join(f"{r['scene']}/w{r['window']}/s{r['source']} L{r['L']} {r['type']} tv_bound {r['tv_bound']:.3f}" for r in unc))
