"""E14: the only hidden-source factor of T* for a single-channel example: g_kappa(beta) = E_kex[(1 - beta)^(K-1)],
beta = vol(D(G) & box(W)) / vol(D(G)), K ~ example-level (size-biased) source count. Uses each scene's SOLID volume
fraction (sum of brick populations of the stored 1024^3 SOLID field; metadata only) and the realised k histogram.
Reports per side the worst-case bound TV(p*_T*, p*_N1) <= 1 - g(beta_max)/g(beta_min), with beta ranging over the solid
fractions of the window's family (an upper bound on the variation across worlds compatible with the same X)."""
import os
import json, sys
from pathlib import Path
import numpy as np
from rind3d.pack import RINDPack
ROOT = (Path(os.environ.get("RIND3D_1K_DATA", "/home/yushu/Projects/RIND/RIND3D_1K_data")) / "release_1k_v1")
phi, fam, ks = [], [], []
for d in sorted((ROOT / "scenes").iterdir()):
    meta = json.loads((d / "scene.json").read_text())
    with RINDPack(d / "privileged" / "fields" / "src0" / "solid.rindb") as m:
        pop = sum(int(r[7]) for r in m.index)
    phi.append(pop / 1024 ** 3); fam.append(meta["family"]); ks.append(int(meta["k"]))
phi = np.array(phi); fam = np.array(fam); ks = np.array(ks)
kh = np.bincount(ks, minlength=5)[1:] / len(ks)
kex = np.arange(1, 5) * kh / np.sum(np.arange(1, 5) * kh)
g = lambda b: float(np.sum(kex * (1 - b) ** np.arange(0, 4)))
out = dict(k_hist=kh.tolist(), k_example_level=kex.tolist(),
           solid_fraction=dict(min=float(phi.min()), median=float(np.median(phi)), max=float(phi.max())),
           by_family={f: [float(phi[fam == f].min()), float(phi[fam == f].max())] for f in sorted(set(fam))})
for L in (64, 128, 256):
    vbox = (L / 1024) ** 3                      # FREE volume in the box <= box volume
    worst = 0.0
    for f in sorted(set(fam)):
        lo, hi = phi[fam == f].min(), phi[fam == f].max()
        bmin, bmax = vbox / (1 - lo), vbox / (1 - hi)
        worst = max(worst, 1 - g(bmax) / g(bmin))
    out[f"L{L}"] = dict(box_volume=vbox, g_at_full_box_phi0=g(vbox), tv_bound_worst_family=worst,
                        tv_bound_any_world=1 - g(vbox / (1 - phi.max())) / g(vbox))
json.dump(out, sys.stdout, indent=1)
