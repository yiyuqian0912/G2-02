"""Join XLF batch results with the certified v3 reference labels (tau-pencil labeller + independent verifier).

usage: summarize.py BATCH.jsonl [...]   (reference: RIND3D_V3_WORK/xlf/ref/bench_tp.jsonl, else the label meta of
RIND3D_V3_WORK/perf/*_tp_label.npz)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

W = Path(os.environ.get("RIND3D_V3_WORK", "/home/yushu/Projects/RIND/RIND3D_100K_v3_work"))


def refs():
    out = {}
    f = W / "xlf/ref/bench_tp.jsonl"
    if f.exists():
        for l in f.read_text().splitlines():
            r = json.loads(l)
            out[r["key"]] = dict(wall=r.get("wall"), label_s=r.get("label_s"), verify_s=r.get("verify_s"), tv=r.get("tv_bound"),
                                 cells=r.get("cells"))
    for p in (W / "perf").glob("*_tp_label.npz"):
        k = p.name.replace("_tp_label.npz", "")
        if k in out:
            continue
        with np.load(p) as z:
            m = json.loads(z["meta"].tobytes().decode())
            out[k] = dict(wall=None, label_s=m.get("label_seconds"), verify_s=m.get("verify_seconds"),
                          tv=float(z["tv_bound"]), cells=int(np.sum(z["lo"] > 0)))
    return out


def main():
    R = refs(); rows = []
    for fn in sys.argv[1:]:
        for l in Path(fn).read_text().splitlines():
            if l.strip():
                rows.append(json.loads(l))
    print(f"{'task':14s} {'L':>3s} | {'XLF s':>7s} {'check s':>7s} {'TV':>9s} {'cells':>6s} {'conf':>4s} {'chk':>4s} | "
          f"{'v3 s':>7s} {'(lab+ver)':>13s} {'TV':>9s} {'cells':>6s} | {'speedup':>7s}")
    tot_x = tot_r = 0.0
    for d in sorted(rows, key=lambda d: d.get("key", "")):
        k = d["key"]; r = R.get(k, {})
        if d.get("error"):
            print(f"{k:14s} ERROR"); continue
        c = d.get("check", {})
        rs = (r.get("label_s") or 0) + (r.get("verify_s") or 0)
        xs = d["solve_s"]
        sp = rs / xs if xs and rs else float("nan")
        tot_x += xs; tot_r += rs
        print(f"{k:14s} {d['L']:3d} | {xs:7.1f} {c.get('seconds', float('nan')):7.1f} {d['tv'] if d['tv'] is not None else float('nan'):9.2e} "
              f"{d['cells']:6d} {str(d.get('enclosure_conflicts')):>4s} {('ok' if c.get('ok') else 'FAIL') if c else '-':>4s} | "
              f"{rs:7.1f} {'%.1f+%.1f' % (r.get('label_s') or 0, r.get('verify_s') or 0):>13s} "
              f"{r.get('tv', float('nan')) if r.get('tv') is not None else float('nan'):9.2e} {str(r.get('cells')):>6s} | {sp:7.2f}"
              + ("  TIMEOUT" if d.get("timed_out") else ""))
    print(f"total XLF {tot_x:.1f} s, v3 {tot_r:.1f} s, ratio {tot_r / tot_x if tot_x else float('nan'):.2f}")


if __name__ == "__main__":
    main()
