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


def scene_stats(p):
    """Per-channel H1 statistics of one stored scene (one task of the process pool)."""
    from rind3d.rendering.backends import BackendFactory
    from rind3d_1k.physical import POLICY
    from rind3d_100k.storage import SceneFile
    from rind3d_posterior.localfirst.local import interface_edges, edge_free_mask, plane_components
    S = SceneFile(p); fam = S.meta["family"]
    ctx = BackendFactory(cuda_allowed=False).context(S.geometry(), POLICY); rows = [c.rows for c in ctx.components]
    out = []
    for w, r in enumerate(S.W):
        o = np.array([r["x"], r["y"], r["z"]]); sol = S.crop(w, -1)
        for j in range(S.k):
            if not (int(r["mask"]) >> j) & 1:
                continue
            lit = S.crop(w, j)
            cls = np.where(sol, 2, np.where(lit, 1, 0)).transpose(2, 1, 0)
            li, da = interface_edges(cls)
            free = edge_free_mask(rows, o, int(S.meta["resolution"]), li, da)
            nf = np.flatnonzero(~free)
            comp = bool(len(nf) >= 24 and any(c["n"] >= 24 for c in plane_components(li[nf], da[nf], max_planes=2, iters=64)))
            out.append(dict(family=fam, L=int(r["Lp"]), edges=int(len(li)), nonfree=int(len(nf)), comp=comp))
    return str(p), out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-scenes", type=int, default=200); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    out = {}
    for root in a.root:
        paths = sorted(Path(root).glob("scenes/*/*.npz"))[:a.max_scenes]
        per = []
        with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
            for _, rows in ex.map(scene_stats, paths):
                per += rows
        res = {}
        fams = sorted({x["family"] for x in per})
        for fam in fams + ["ALL"]:
            g = [x for x in per if fam == "ALL" or x["family"] == fam]
            E = sum(x["edges"] for x in g); NF = sum(x["nonfree"] for x in g)
            res[fam] = dict(channels=len(g), edge_nonfree_share=NF / max(E, 1),
                            channel_nonfree_share=sum(x["nonfree"] > 0 for x in g) / max(len(g), 1),
                            channel_nonfree_component_share=sum(x["comp"] for x in g) / max(len(g), 1),
                            by_L={str(L): sum(x["nonfree"] > 0 for x in g if x["L"] == L) / max(sum(x["L"] == L for x in g), 1) for L in (64, 128, 256)})
        out[root] = res
        print(json.dumps({root: res["ALL"]}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
