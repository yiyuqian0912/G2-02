"""E4b: cluster-aware reading of E4. The 333 labels come from 4 scenes and only 13 distinct true sources, so their
HPD ranks are strongly dependent; a KS test over all 333 is not valid. Here: per-source summaries and the
distribution of KS p-values when ONE label per source is drawn at random (independent across sources).
Run: python e4b_label_coverage_clusters.py out/e4_label_coverage.json > out/e4b_label_coverage_clusters.json"""
import json, sys
import numpy as np
from scipy.stats import kstest
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "out/e4_label_coverage.json"))
rows = d["rows"]; rng = np.random.default_rng(11)
groups = {}
for r in rows:
    groups.setdefault((r["scene"], r["source"]), []).append(r["true"]["u"])
per = {f"{a}/s{b}": dict(n=len(v), mean_u=float(np.mean(v)), sd_u=float(np.std(v))) for (a, b), v in sorted(groups.items())}
keys = list(groups)
ps = []; means = []
for _ in range(2000):
    u = np.array([groups[k][rng.integers(len(groups[k]))] for k in keys])
    ps.append(kstest(u, "uniform").pvalue); means.append(u.mean())
ps = np.array(ps)
# under H0 the mean of 13 independent U(0,1) has sd sqrt(1/12/13)
out = dict(n_sources=len(keys), per_source=per, one_label_per_source=dict(
    share_ks_p_below_0p05=float(np.mean(ps < 0.05)), median_ks_p=float(np.median(ps)), mean_of_means=float(np.mean(means)),
    h0_sd_of_mean=float(np.sqrt(1 / 12 / len(keys)))),
    intra_source_correlation_note="labels of one source share s_j and G; across windows its HPD ranks are dependent")
json.dump(out, sys.stdout, indent=1)
