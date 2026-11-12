"""Geometry-diversity study: per-family complexity, labelling cost and tails, certification, observational and
posterior diversity. Usage: geometry_study.py <root-with-one-campaign-per-family> <out.json>"""
import sys, json
from pathlib import Path
import numpy as np
from scipy import ndimage

root = Path(sys.argv[1]); out = {}
for fam_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    log = fam_dir / "labels_log.jsonl"
    if not log.exists():
        continue
    R = [json.loads(l) for l in log.read_text().splitlines()]
    R = [r for r in R if "error" not in r]
    scenes = sorted((fam_dir / "scenes").iterdir())
    faces, bodies, phys = [], [], []
    litf, ovl4 = [], []
    for s in scenes:
        sc = json.loads((s / "scene.json").read_text())
        faces += sc["faces"]; bodies.append(sc["n_bodies"]); phys.append(sc["timing"].get("fields", 0) + sc["timing"].get("selection", 0))
        for w in json.loads((s / "privileged" / "windows_full.json").read_text()):
            litf += [f for j, f in enumerate(w["lit_frac"]) if (w["mask"] >> j) & 1]
    supp, modes, vol = [], [], []
    for r in R:
        p = fam_dir / "scenes" / r["scene"] / "labels" / "tasks" / f"w{r['window']:03d}_s{r['source']}.npz"
        with np.load(p) as z:
            cells = np.unique(z["cell"][z["lo"] > 0])
        supp.append(len(cells)); vol.append(r["vol_lo"])
        if len(cells):
            g = np.zeros(128 ** 3, bool); g[cells] = True
            modes.append(int(ndimage.label(g.reshape(128, 128, 128))[1]))
    c = np.array([r["cpu"] for r in R]); tv = np.array([r["tv_bound"] for r in R])
    out[fam_dir.name] = dict(
        scenes=len(scenes), bodies_mean=float(np.mean(bodies)), planes_mean=float(np.mean(faces)), planes_max=int(np.max(faces)),
        physical_s_mean=float(np.mean(phys)), labels=len(R),
        label_cpu=dict(mean=float(c.mean()), p50=float(np.median(c)), p95=float(np.percentile(c, 95)), max=float(c.max())),
        certified_frac=float(np.mean(tv <= 0.05)), tv_max=float(tv.max()),
        support_cells=dict(p50=float(np.median(supp)), p95=float(np.percentile(supp, 95))),
        modes=dict(mean=float(np.mean(modes)), frac_multimodal=float(np.mean(np.array(modes) > 1)), max=int(np.max(modes))),
        posterior_volume_log10_p50=float(np.median(np.log10(np.maximum(vol, 1e-300)))),
        lit_frac=dict(p10=float(np.percentile(litf, 10)), p50=float(np.median(litf)), p90=float(np.percentile(litf, 90))))
    print(fam_dir.name, json.dumps(out[fam_dir.name]), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
