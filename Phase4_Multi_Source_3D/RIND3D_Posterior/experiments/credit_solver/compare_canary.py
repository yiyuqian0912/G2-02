"""Compare CBC outputs on release-canary channels with the production FastLabel v3.4 labels.

Reads OUT/runs.jsonl and OUT/results/KEY.json (cbc_run envelopes) and the canary's
consolidated label files. FastLabel labels carry either a deterministic TV bound
("certified") or the statistical guarantee TV <= 0.05 except with probability <= delta
("certified_stat"). Consistency: TV(p_cbc, q_fl) <= tv_cbc + tv_fl.
Also reports FastLabel worker seconds per label when recorded in the label metadata.
"""
import argparse
import base64
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "xlf_assurance"))
from ledger import decode  # noqa: E402

CANARY = Path("/home/yushu/Projects/RIND/RIND3D_100K_v3_work/release/canary")


def fl_label(root, slot, w, s):
    z = np.load(root / "labels" / "scenes" / f"{slot // 1000:04d}" / f"{slot:06d}.labels.npz")
    metas = json.loads(z["metas"].tobytes())
    key = f"{slot:06d}:{w:02d}:{s}"
    for i, m in enumerate(metas):
        if m["key"] == key:
            a, b = z["offsets"][i], z["offsets"][i + 1]
            lo = z["lo"][a:b]
            q = lo / lo.sum()
            return dict(zip(z["cell"][a:b].astype(np.int64).tolist(), q.tolist())), m
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--root", type=Path, default=CANARY)
    args = ap.parse_args()
    rows = []
    for line in (args.out / "runs.jsonl").read_text().splitlines():
        rec = json.loads(line)
        t = rec["task"]
        s = rec["summary"] or {}
        row = dict(key=t["key"], L=t["L"], family=t["family"], status=s.get("status"), wall=rec["process_wall"],
                   tv=None if s.get("tv") is None else float(Fraction(s["tv"])))
        q, meta = fl_label(args.root, t["slot"], t["w"], t["s"])
        row["fl_status"] = meta and meta["status"]
        row["fl_bound"] = meta and float(meta["tv_bound"])
        row["fl_seconds"] = meta and (meta.get("seconds") or meta.get("label_seconds") or meta.get("wall"))
        env = args.out / "results" / f"{t['key']}.json"
        if row["status"] == "CERTIFIED" and env.exists() and q is not None:
            p = {c: float(v) for c, v in decode(base64.b64decode(json.loads(env.read_bytes())["wire"])).items()}
            d = 0.5 * sum(abs(p.get(c, 0.0) - q.get(c, 0.0)) for c in set(p) | set(q))
            row.update(tv_to_fl=d, cells=len(p), fl_cells=len(q), consistent=d <= row["tv"] + row["fl_bound"] + 1e-9)
        rows.append(row)
    (args.out / "compare_canary.json").write_text(json.dumps(rows, indent=1))
    cert = [r for r in rows if r["status"] == "CERTIFIED"]
    print(f"canary tasks {len(rows)}  certified {len(cert)}  inconsistent {sum(r.get('consistent') is False for r in cert)}")
    for L in (64, 128, 256):
        sub = [r for r in rows if r["L"] == L]
        w = sorted(r["wall"] for r in sub)
        c = [r for r in sub if r["status"] == "CERTIFIED"]
        if sub:
            print(f"  L={L}: {len(c)}/{len(sub)}  wall median {w[len(w)//2]:.2f} s  p90 {w[int(.9*len(w))]:.2f} s")
    fams = sorted({r["family"] for r in rows})
    print("  by family:", {f: f"{sum(r['status']=='CERTIFIED' for r in rows if r['family']==f)}/{sum(r['family']==f for r in rows)}" for f in fams})
    if cert:
        d = sorted(r["tv_to_fl"] for r in cert if "tv_to_fl" in r)
        print(f"  TV(cbc, FastLabel): median {d[len(d)//2]:.2e}  max {d[-1]:.3g}")
        ex = [r for r in cert if r["fl_status"] == "certified"]
        if ex:
            print(f"  vs FastLabel deterministic labels: {len(ex)}, max TV {max(r['tv_to_fl'] for r in ex):.3g}")


if __name__ == "__main__":
    main()
