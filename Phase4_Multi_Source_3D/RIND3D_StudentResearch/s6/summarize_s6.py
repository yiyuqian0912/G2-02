"""Table: S6 prototype vs production v3.4 (audit E11) on the 18 audit all-dark tasks."""
import json
from pathlib import Path
E11 = (Path(__file__).resolve().parents[2] / "RIND3D_Audit/experiments/out/e11_fastlabel_extreme.jsonl")
prod = {}
for l in open(E11):
    r = json.loads(l)
    if r.get("type") == "alldark":
        prod[(r["scene"], int(r["window"]), int(r["source"]))] = r
rows = {}
for l in open(Path(__file__).with_name("s6_alldark.jsonl")):
    r = json.loads(l)
    rows[(r["scene"], int(r["window"]), int(r["source"]))] = r       # later lines (reruns) replace earlier ones
print("| scene/window/source | L | v3.4 (E11) | v3.4 wall s | S6 status | S6 wall s | rejected inside solid | solid volume removed / vol(B') | eta | rounds |")
print("|---|---|---|---|---|---|---|---|---|---|")
n_ok = 0
for k, p in prod.items():
    s = rows.get(k, {})
    sr = s.get("solid_removal") or {}
    st = s.get("status") or s.get("error", "not run")
    n_ok += st == "certified_stat"
    vol = s.get("vol") or 0
    eta = (s.get("tv_bound") or 0.05) - 0.05 if st == "certified_stat" else None
    print(f"| {k[0]}/w{k[1]}/s{k[2]} | {p.get('L')} | {p.get('status', p.get('error'))} | {round(p.get('wall', 0)) or '—'} | {st} | "
          f"{round(s.get('wall', 0))} | {s.get('rejected')} | {round(sr.get('removed_volume', 0) / vol, 4) if vol else '—'} | "
          f"{'%.1e' % eta if eta is not None else '—'} | {len(s.get('rounds') or [])} |")
print(f"\nS6 certified {n_ok} / {len(prod)}; v3.4 certified {sum(1 for p in prod.values() if p.get('status') == 'certified_stat')} / {len(prod)}")
