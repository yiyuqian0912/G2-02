"""Markdown tables from analyze.py output (pilot_analysis.json). usage: tables.py <pilot_analysis.json>"""
import json, sys

R = json.load(open(sys.argv[1]))
A = R["arms"]
SETS = [("tstar", "T* (natural, in-dist.)"), ("b1k", "B1K (release, in-dist.)"), ("ood_nat_sep", "OOD solid-separated (natural)"),
        ("ood_rel_sep", "OOD solid-separated (release)"), ("ood_nat_alllit", "OOD all-lit (natural)"),
        ("ood_nat_alldark", "OOD all-dark (natural)")]


def f(x, d=3):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def rci(d, k, kc):
    if d.get(k) is None:
        return "—"
    c = d.get(kc)
    return f"{d[k]:.3f}" + (f" [{c[0]:.3f}, {c[1]:.3f}]" if c else "")


def ci(m):
    return f"{m['mean']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"


print("### Information gain over the uniform-legal prior (nats; higher is better), scene-bootstrap 95% CI\n")
print("| arm | " + " | ".join(n for _, n in SETS) + " |")
print("|---|" + "---|" * len(SETS))
for a, D in A.items():
    print(f"| {a} | " + " | ".join(ci(D[s]["info_gain_vs_prior"]) if s in D else "—" for s, _ in SETS) + " |")
print("\n### KL(q_G || p) to the certified full-world label (nats; lower is better; only where a label exists)\n")
print("| arm | " + " | ".join(n for _, n in SETS) + " |")
print("|---|" + "---|" * len(SETS))
for a, D in A.items():
    print(f"| {a} | " + " | ".join(ci(D[s]["kl"]) if s in D and "kl" in D[s] else "—" for s, _ in SETS) + " |")
print("\n### Calibration MAE of HPD coverage (10..90%) | mass on the label support / prior mass | adaptivity rho(ig, ig_ref)\n")
print("| arm | set | calib MAE [CI] | support-mass ratio | rho(KL(p||u), KL(q||u)) | rho(gain, KL(q||u)) |")
print("|---|---|---|---|---|---|")
for a, D in A.items():
    for s, n in SETS:
        if s not in D:
            continue
        d = D[s]
        sm = d.get("supp_mass_over_prior")
        print(f"| {a} | {n} | {d['calib_mae']:.3f} [{d['calib_mae_ci95'][0]:.3f}, {d['calib_mae_ci95'][1]:.3f}] | "
              f"{ci(sm) if sm else '—'} | {rci(d, 'adaptivity_spearman_ig_vs_ig_ref', 'adaptivity_ig_ci95')} | {rci(d, 'adaptivity_spearman_gain_vs_ig_ref', 'adaptivity_gain_ci95')} |")
print("\n### By side (T* set): information gain\n")
print("| arm | L64 | L128 | L256 |")
print("|---|---|---|---|")
for a, D in A.items():
    bs = D["tstar"]["by_side"]
    print(f"| {a} | " + " | ".join(f(bs[L]["gain"]) for L in ("64", "128", "256")) + " |")
print("\n### Paired differences (row arm minus column arm; nats; NLL, lower is better)\n")
for k, v in R["paired"].items():
    for s, m in v.items():
        if "nll" in m:
            print(f"* {k} on {s}: NLL {ci(m['nll'])}" + (f"; KL {ci(m['kl'])}" if "kl" in m else ""))
