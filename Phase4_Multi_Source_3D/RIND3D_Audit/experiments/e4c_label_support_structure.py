"""E4c: structure of the 333 stored labels: support size, number of 26-connected components of the 128^3 support,
mass outside the largest component, entropy; by side. (Multimodality / disconnected support in real labels.)
Run: nice -n 19 python e4c_label_support_structure.py > out/e4c_label_support_structure.json"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import label

ROOT = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes"); G = 128
rows = []
for sc in ("000000", "000001", "000002", "000003"):
    d = ROOT / sc
    for x in json.loads((d / "labels" / "LABELS.json").read_text())["labels"]:
        if x.get("status") != "certified":
            continue
        with np.load(d / "labels" / "tasks" / f"w{x['window']:03d}_s{x['source']}.npz") as z:
            cell = z["cell"].astype(np.int64); lo = z["lo"]
        cells, inv = np.unique(cell, return_inverse=True); m = np.bincount(inv, weights=lo); keep = m > 0
        cells, m = cells[keep], m[keep] / m[keep].sum()
        occ = np.zeros(G ** 3, bool); occ[cells] = True
        lab, nc = label(occ.reshape(G, G, G), structure=np.ones((3, 3, 3)))
        comp = lab.reshape(-1)[cells]
        cm = np.bincount(comp, weights=m)[1:]
        rows.append(dict(scene=sc, window=x["window"], source=x["source"], L=x["L"], n_cells=int(len(cells)), n_comp=int(nc),
                         mass_outside_largest=float(1 - cm.max()), n_comp_mass_ge_1pct=int((cm >= 0.01).sum()),
                         entropy_bits=float(-(m * np.log2(m)).sum())))
out = {}
for Lsel in (None, 64, 128, 256):
    rr = [r for r in rows if Lsel is None or r["L"] == Lsel]
    if not rr:
        continue
    nc = np.array([r["n_comp"] for r in rr]); mo = np.array([r["mass_outside_largest"] for r in rr]); n1 = np.array([r["n_comp_mass_ge_1pct"] for r in rr])
    out["all" if Lsel is None else f"L{Lsel}"] = dict(n=len(rr), share_single_cell=float(np.mean([r["n_cells"] == 1 for r in rr])),
        share_multi_component=float(np.mean(nc > 1)), share_multi_component_mass_ge_1pct=float(np.mean(n1 > 1)),
        share_mass_outside_largest_ge_5pct=float(np.mean(mo >= 0.05)), max_components=int(nc.max()),
        n_cells_quantiles={str(q): float(np.quantile([r["n_cells"] for r in rr], q)) for q in (.1, .5, .9)})
out["rows"] = rows
json.dump(out, sys.stdout, indent=1)
