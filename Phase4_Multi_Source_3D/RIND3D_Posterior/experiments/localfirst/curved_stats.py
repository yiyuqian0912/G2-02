"""Local interface statistics of curved-occluder research scenes (no labels exist for balls).

usage: curved_stats.py --out DIR [--n 4]
Generates n scenes per curved family (contract rind3d-100k-curved-research-v0, run_scene_curved), keeps them under
DIR/scenes, and for every direct-interface channel reports: interface edges, the share of edges explained by the
planar components (RANSAC peeling, at most 8 planes), the components' rms off-plane residual (curvature indicator),
the conditioning of their normals, and whether the HEURISTIC local band contains the true driver source."""
import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
D100K = HERE.parents[3] / "RIND3D_Dataset100K"
sys.path.insert(0, str(D100K))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    from rind3d.rendering.backends import BackendFactory
    from rind3d_100k import sampling as SM
    from rind3d_100k.physical_curved import run_scene_curved
    from rind3d_100k.storage import scene_path, atomic_write, SceneFile
    from rind3d_posterior.localfirst.local import interface_edges, plane_components, conditioning, local_band_cells
    cfg = json.loads((D100K / "contract" / "RIND3D_100K_CURVED_RESEARCH_v0.json").read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); fac = BackendFactory()
    rows = []
    for f, fam in enumerate(cfg["families"]):
        for i in range(a.n):
            slot = f + len(cfg["families"]) * i
            k, _ = SM.kappa_draw(cfg, slot)
            for att in range(3):
                rec, payload = run_scene_curved(cfg, slot, k, fac, att)
                if payload is not None:
                    break
            if payload is None:
                continue
            atomic_write(scene_path(out, slot), payload)
            S = SceneFile(scene_path(out, slot)); src = S.sources()
            for w, r in enumerate(S.W):
                o = np.array([r["x"], r["y"], r["z"]]); sol = S.crop(w, -1)
                for j in range(S.k):
                    if not (int(r["mask"]) >> j) & 1:
                        continue
                    cls = np.where(sol, 2, np.where(S.crop(w, j), 1, 0)).transpose(2, 1, 0)
                    li, da = interface_edges(cls)
                    comps = plane_components(li, da, seed=0)
                    cond = conditioning(comps)
                    band = local_band_cells(comps, o, 1024)
                    ii = np.minimum((src[j] * 128).astype(int), 127)
                    big = [c for c in comps if c["n"] >= 24]
                    rows.append(dict(family=fam, slot=slot, L=int(r["Lp"]), edges=int(len(li)),
                                     explained=float(sum(c["n"] for c in comps) / max(len(li), 1)) if len(li) <= 20000 else None,
                                     rms=[round(c["rms"], 3) for c in big[:4]], rank=cond["rank"], sigma=cond["sigma"],
                                     source_in_band=bool(band[(ii[0] * 128 + ii[1]) * 128 + ii[2]]), band_cells=int(band.sum())))
            print(fam, slot, "windows done", flush=True)
    (out / "curved_stats.json").write_text(json.dumps(rows, indent=1))
    for fam in cfg["families"]:
        g = [x for x in rows if x["family"] == fam]
        if not g:
            continue
        rms = [v for x in g for v in x["rms"]]
        print(json.dumps(dict(family=fam, channels=len(g), source_in_band=sum(x["source_in_band"] for x in g),
                              median_component_rms_vox=float(np.median(rms)) if rms else None,
                              rank_counts={kk: sum(x["rank"] == kk for x in g) for kk in range(4)},
                              median_explained=float(np.median([x["explained"] for x in g if x["explained"] is not None])))))


if __name__ == "__main__":
    main()
