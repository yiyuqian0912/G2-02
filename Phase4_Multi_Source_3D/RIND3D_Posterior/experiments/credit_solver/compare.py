"""Post-run comparison of credit-solver outputs with independent reference labels.

Only this script reads reference labels; the solver never does. For each task with a
CERTIFIED envelope, decode the released wire bytes and compute the TV distance to the
reference distribution. Consistency requires TV(p, q_ref) <= tv_mine + tv_ref.

References (deterministic, certified by other engines):
  archive tasks: RIND3D_100K_v3_work/cmp_v3/labels/KEY_tp.npz (Tau-pencil lo/hi; bound in report_rows.json)
  holdout/fresh: global_inverse/records/{reference,fresh_reference}/KEY_cpu.npz (random-subset + Tau-pencil, q)
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

GI = HERE.parent / "global_inverse" / "records"
CMP = Path("/home/yushu/Projects/RIND/RIND3D_100K_v3_work/cmp_v3")


def reference_bounds():
    bounds = {}
    for f in (GI / "reference" / "holdout_cpu.jsonl", GI / "reference" / "holdout_large_cpu.jsonl",
              GI / "fresh_reference" / "fresh_cpu.jsonl"):
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                bounds[r["task"]["key"]] = (r["status"], r["tv_bound"])
    rows = CMP / "report_rows.json"
    if rows.exists():
        for r in json.loads(rows.read_text()):
            bounds.setdefault(r["key"], (r["tp"]["status"], r["tp"]["tv_bound"]))
    return bounds


def reference(key):
    for d in (GI / "reference", GI / "fresh_reference"):
        f = d / f"{key}_cpu.npz"
        if f.exists():
            z = np.load(f)
            return dict(zip(z["cell"].tolist(), z["q"].tolist())), f.name
    f = CMP / "labels" / f"{key}_tp.npz"
    if f.exists():
        z = np.load(f)
        lo = z["lo"] if z["lo"].sum() > 0 else z["hi"]
        q = lo / lo.sum()
        return dict(zip(z["cell"].tolist(), q.tolist())), f.name
    return None, None


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(float(p.get(k, 0)) - q.get(k, 0.0)) for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path, help="batch output directory")
    args = ap.parse_args()
    bounds = reference_bounds()
    rows = []
    for line in (args.out / "runs.jsonl").read_text().splitlines():
        rec = json.loads(line)
        key = rec["task"]["key"]
        env_path = args.out / "results" / f"{key}.json"
        row = dict(key=key, L=rec["task"]["L"], family=rec["task"]["family"], split=rec["task"]["split"],
                   exit=rec["exit"], wall=rec["process_wall"])
        s = rec["summary"] or {}
        row.update(status=s.get("status"), reason=s.get("reason"),
                   tv=None if s.get("tv") is None else float(Fraction(s["tv"])),
                   cpu=s.get("cpu"), rss_mib=None if s.get("peak_rss_kib") is None else s["peak_rss_kib"] / 1024)
        if row["status"] == "CERTIFIED" and env_path.exists():
            envelope = json.loads(env_path.read_bytes())
            p = decode(base64.b64decode(envelope["wire"]))
            row["cells"] = len(p)
            q, name = reference(key)
            if q is not None:
                st, b = bounds.get(key, (None, None))
                row.update(ref=name, ref_status=st, ref_bound=b, ref_cells=len(q), tv_to_ref=tv(p, q))
                if b is not None:
                    row["consistent"] = row["tv_to_ref"] <= row["tv"] + b + 1e-9
        rows.append(row)
    (args.out / "compare.json").write_text(json.dumps(rows, indent=1))
    n = len(rows)
    cert = [r for r in rows if r["status"] == "CERTIFIED"]
    bad = [r for r in cert if r.get("consistent") is False]
    print(f"tasks {n}  certified {len(cert)}  inconsistent {len(bad)}  errors {sum(r['exit'] != 0 for r in rows)}")
    for L in (64, 128, 256):
        sub = [r for r in rows if r["L"] == L]
        c = [r for r in sub if r["status"] == "CERTIFIED"]
        walls = sorted(r["wall"] for r in sub)
        if sub:
            print(f"  L={L}: {len(c)}/{len(sub)} certified; wall median {walls[len(walls)//2]:.2f} s max {walls[-1]:.2f} s")
    for split in sorted({r["split"] for r in rows}):
        sub = [r for r in rows if r["split"] == split]
        print(f"  {split}: {sum(r['status'] == 'CERTIFIED' for r in sub)}/{len(sub)}")
    if cert:
        worst = max(cert, key=lambda r: r.get("tv_to_ref") or 0)
        print("  max tv_to_ref", worst.get("tv_to_ref"), worst["key"], "bound", worst["tv"], "ref", worst.get("ref_bound"))
    for r in bad:
        print("  INCONSISTENT", r)


if __name__ == "__main__":
    main()
