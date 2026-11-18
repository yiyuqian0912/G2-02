"""How often is hypothesis H1 (every direct interface edge is free) violated in stored 100K scenes?

usage: measure_h1.py --root ROOT [--root ROOT2 ...] --out out.json [--max-scenes 200]
For every stored window and every channel with a direct interface: the interface edges (rind3d_posterior.localfirst
.local.interface_edges) and the exact free-edge test against the scene's canonical bodies (edge_free_mask). Reports
per root and family: share of interface edges that are non-free, share of channels with any non-free edge, and the
share with a non-free planar component of >= 24 edges (the size the local planar decomposition acts on)."""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "RIND3D_Dataset100K"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-scenes", type=int, default=200)
    a = ap.parse_args()
    from rind3d.rendering.backends import BackendFactory
    from rind3d_1k.physical import POLICY
    from rind3d_100k.storage import SceneFile
    from rind3d_posterior.localfirst.local import interface_edges, edge_free_mask, plane_components
    fac = BackendFactory(cuda_allowed=False)
    out = {}
    for root in a.root:
        agg = defaultdict(lambda: dict(channels=0, edges=0, nonfree=0, ch_nonfree=0, ch_nonfree_comp=0, by_L=defaultdict(lambda: [0, 0])))
        for p in sorted(Path(root).glob("scenes/*/*.npz"))[:a.max_scenes]:
            S = SceneFile(p); fam = S.meta["family"]
            ctx = fac.context(S.geometry(), POLICY); rows = [c.rows for c in ctx.components]
            for w, r in enumerate(S.W):
                o = np.array([r["x"], r["y"], r["z"]]); sol = S.crop(w, -1)
                for j in range(S.k):
                    if not (int(r["mask"]) >> j) & 1:
                        continue
                    lit = S.crop(w, j)                                  # [Z, Y, X]
                    cls = np.where(sol, 2, np.where(lit, 1, 0)).transpose(2, 1, 0)   # -> [x, y, z]
                    li, da = interface_edges(cls)
                    free = edge_free_mask(rows, o, int(S.meta["resolution"]), li, da)
                    g = agg[fam]; g["channels"] += 1; g["edges"] += len(li); g["nonfree"] += int((~free).sum())
                    g["by_L"][int(r["Lp"])][0] += 1
                    if (~free).any():
                        g["ch_nonfree"] += 1; g["by_L"][int(r["Lp"])][1] += 1
                        nf = np.flatnonzero(~free)
                        if len(nf) >= 24 and any(c["n"] >= 24 for c in plane_components(li[nf], da[nf], max_planes=2)):
                            g["ch_nonfree_comp"] += 1
        res = {}
        for fam, g in sorted(agg.items()):
            res[fam] = dict(channels=g["channels"], edge_nonfree_share=g["nonfree"] / max(g["edges"], 1),
                            channel_nonfree_share=g["ch_nonfree"] / max(g["channels"], 1),
                            channel_nonfree_component_share=g["ch_nonfree_comp"] / max(g["channels"], 1),
                            by_L={str(L): dict(channels=v[0], nonfree_share=v[1] / max(v[0], 1)) for L, v in sorted(g["by_L"].items())})
        tot = {k: sum(g[k] for g in agg.values()) for k in ("channels", "edges", "nonfree", "ch_nonfree", "ch_nonfree_comp")}
        res["ALL"] = dict(channels=tot["channels"], edge_nonfree_share=tot["nonfree"] / max(tot["edges"], 1),
                          channel_nonfree_share=tot["ch_nonfree"] / max(tot["channels"], 1),
                          channel_nonfree_component_share=tot["ch_nonfree_comp"] / max(tot["channels"], 1))
        out[root] = res
        print(json.dumps({root: res["ALL"]}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
