#!/usr/bin/env python3
"""Complexity coverage and training ESS of a RIND3D-100K root (metadata only; safe while production runs).

  coverage_report.py <root> [--estimator combined_rb] [--out COVERAGE.json]

Reports, from the manifest records and the scene containers' window tables (no crops are read):
* the realised source-count histogram against the contract's kappa;
* per side L and mixed count m: windows, mixed channels, independent scenes (per family as well), stream shares;
* T* weights (loader.Scene.population_weight): per side Kish ESS fraction of the mixed-channel examples, the
  max / median weight ratio, and the coverage stream's share of the T* weight mass;
* labels: statuses and certified fraction (if consolidated).
Compare with the coverage targets of RIND3D_StudentResearch/J_COMPLEXITY_COVERAGE.md §2 (scaled to the canary)."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "RIND3D_Dataset1K"), str(REPO / "RIND3D_Posterior" / "src")]
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--estimator", default="combined_rb")
    ap.add_argument("--out"); a = ap.parse_args()
    from rind3d_100k.loader import RIND3D100K, Scene
    from rind3d_100k.storage import read_jsonl
    root = Path(a.root); cfg = json.loads((root / "CONTRACT.json").read_text())
    ds = RIND3D100K(root)
    kh = Counter(); win = Counter(); chan = Counter(); scenes = defaultdict(set); fam_scenes = defaultdict(set)
    streams = Counter(); wts = defaultdict(list); mass = defaultdict(lambda: [0.0, 0.0])
    for slot in ds.slots:
        S = Scene(root, slot); fam = S.meta["family"]; kh[S.S.k] += 1
        for w, r in enumerate(S.S.W):
            L = int(r["Lp"]); m = int(r["m"]); st = S.stream(w)
            win[(L, m)] += 1; chan[(L, m)] += m; scenes[(L, m)].add(slot); fam_scenes[(fam, L, m)].add(slot); streams[(L, st)] += 1
            wv = S.population_weight(w, "Tstar", a.estimator)
            wts[L] += [wv] * m
            mass[L][1 if st == "coverage" else 0] += wv * m
    n = len(ds.slots)
    ess = {}
    for L, v in wts.items():
        v = np.asarray(v, float)
        ess[str(L)] = dict(examples=len(v), kish_ess_fraction=float(v.sum() ** 2 / (len(v) * (v ** 2).sum())) if len(v) else None,
                           max_over_median=float(v.max() / np.median(v)) if len(v) else None,
                           coverage_weight_share=float(mass[L][1] / max(mass[L][0] + mass[L][1], 1e-300)))
    labels = Counter()
    last = {}                                  # consolidated.jsonl: one record per consolidation, the last one wins
    for r in read_jsonl(root / "labels" / "consolidated.jsonl"):
        last[int(r["slot"])] = r
    for r in last.values():
        labels.update(r.get("statuses", {}))
    kap = (cfg.get("source_count") or {}).get("kappa")
    out = dict(root=str(root), contract=cfg["contract"], scenes=n,
               k_hist={str(k): kh[k] for k in range(1, 5)}, kappa=kap,
               by_side_m={f"L{L}:m{m}": dict(windows=win[(L, m)], channels=chan[(L, m)], scenes=len(scenes[(L, m)]))
                          for L in (64, 128, 256) for m in (1, 2, 3, 4)},
               min_family_scenes_m4={f"L{L}": min((len(fam_scenes[(f, L, 4)]) for f in cfg["families"]), default=0)
                                      for L in (64, 128, 256)},
               streams={f"L{L}:{st}": c for (L, st), c in sorted(streams.items())},
               tstar=ess, estimator=a.estimator, label_statuses=dict(labels))
    s = json.dumps(out, indent=1)
    Path(a.out or root / "COVERAGE.json").write_text(s); print(s)


if __name__ == "__main__":
    main()
