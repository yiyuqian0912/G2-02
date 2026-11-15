"""Q6 generalisation readout on dev T* (B_RESULTS §8). usage: generalisation.py side|fam [--sets tstar,sep,...]

side: G0side / F1side (trained on L64 + L128) against G0 / F1, on the trained sides and on L256. Uses the evaluation-time
      diagnostic eval_dev_sb1.npz (side_bucket_max = 1) when present.
fam:  G0fam / F1fam (corridors and thin_slabs held out) against G0 / F1, by held-out family; for T*, also split by
      facet-law violation (scripts/failure_analysis.facet_miss: the true source is used only to stratify).
Paired differences use identical rows; brackets are 95% scene-bootstrap intervals; means are side-balanced."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slab.analyze import load, boot, hpd_mae     # noqa: E402
from slab.paths import OUT                       # noqa: E402

RUNS = OUT / "runs"
FAMILIES = {2: "corridors", 7: "thin_slabs"}


def fmt(b):
    return f"{b['mean']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}]"


def cov50(h):
    """Coverage of the nominal 50% HPD region minus 0.5 (negative = over-confident)."""
    h = h[~np.isnan(h)]
    return float(np.mean(h < 0.5) - 0.5) if len(h) else np.nan


def compare(A, B, m, label):
    ba = boot(A["ig"][m], A["scene"][m], A["Lp"][m])
    d = boot(A["ig"][m] - B["ig"][m], A["scene"][m], A["Lp"][m])
    print(f"| {label} | {int(m.sum())} | {fmt(ba)} | {fmt(d)} | {hpd_mae(A['hpd'][m]):.3f} / {hpd_mae(B['hpd'][m]):.3f} | "
          f"{cov50(A['hpd'][m]):+.3f} / {cov50(B['hpd'][m]):+.3f} |")


def main():
    mode = sys.argv[1]
    sets = sys.argv[sys.argv.index("--sets") + 1].split(",") if "--sets" in sys.argv else ["tstar"]
    if mode == "side":
        pairs = [("G0side_s0", "eval_dev", "G0_s0"), ("G0side_s0", "eval_dev_sb1", "G0_s0"),
                 ("F1side_s0", "eval_dev", "F1_s0"), ("F1side_s0", "eval_dev_sb1", "F1_s0")]
    else:
        pairs = [("G0fam_s0", "eval_dev", "G0_s0"), ("F1fam_s0", "eval_dev", "F1_s0")]
    for s in sets:
        print(f"\n### {s}\n| arm vs full | n | IG | difference | HPD-MAE arm / full | cov50 − 0.5 arm / full |\n|---|---|---|---|---|---|")
        for a, f, b in pairs:
            if not (RUNS / a / f"{f}.npz").exists():
                continue
            A, B = load(RUNS / a / f"{f}.npz")[s], load(RUNS / b / "eval_dev.npz")[s]
            assert np.array_equal(A["idx"], B["idx"])
            tag = f"{a}{' (L256 bucket -> L128)' if f.endswith('sb1') else ''} vs {b}"
            if mode == "side":
                compare(A, B, np.isin(A["Lp"], [64, 128]), f"{tag}, L64 + L128 (trained)")
                compare(A, B, A["Lp"] == 256, f"{tag}, L256 (held out)")
            else:
                compare(A, B, ~np.isin(A["family"], list(FAMILIES)), f"{tag}, other families (trained)")
                for k, nm in FAMILIES.items():
                    compare(A, B, A["family"] == k, f"{tag}, {nm} (held out)")
    if mode == "fam" and "tstar" in sets:
        from failure_analysis import facet_miss
        E = {k: load(RUNS / f"{k}_s0" / "eval_dev.npz")["tstar"] for k in ("G0", "G0fam", "F1", "F1fam")}
        r = E["F1"]; _, nbig, nviol = facet_miss(r["idx"]); viol = nviol > 0
        print("\n### tstar, held-out families by facet-law violation (large facet >= 100 inliers misses s by > 4 vox)")
        print("| stratum | n | G0fam − G0 | F1fam − F1 | F1 IG |\n|---|---|---|---|---|")
        for k, nm in FAMILIES.items():
            for sn, sm in (("no large facet", nbig == 0), ("every large facet obeys", (nbig > 0) & ~viol), ("violating", viol)):
                m = (r["family"] == k) & sm
                g = boot(E["G0fam"]["ig"][m] - E["G0"]["ig"][m], r["scene"][m], r["Lp"][m])
                f = boot(E["F1fam"]["ig"][m] - E["F1"]["ig"][m], r["scene"][m], r["Lp"][m])
                print(f"| {nm}, {sn} | {int(m.sum())} | {fmt(g)} | {fmt(f)} | {boot(r['ig'][m], r['scene'][m], r['Lp'][m])['mean']:.2f} |")


if __name__ == "__main__":
    main()
