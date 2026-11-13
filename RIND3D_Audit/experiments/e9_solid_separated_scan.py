"""E9: how often are informative channels 'solid-separated' (lit and dark FREE receivers never 6-adjacent, i.e. every
6-connected FREE component of the window is monochromatic)? Also 26-adjacency (checked only when 6-adjacency fails).
Reads public/observations.bin crops (ZYX packed bits) read-only.
Lemma (A): no 6-adjacent lit/dark FREE pair <=> every 6-connected component of FREE(W) is single-coloured (a path inside
FREE from a lit to a dark voxel must contain a 6-adjacent lit/dark step).
Run: nice -n 19 python e9_solid_separated_scan.py <Lp> <scene_from> <scene_to> > out/e9_L<Lp>.json"""
import json, sys
from pathlib import Path
import numpy as np, zstandard

ROOT = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/release_1k_v1/scenes")
Lp, a, b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dz = zstandard.ZstdDecompressor()


def adj(c, dirs):
    L = c.shape[0]
    for d in dirs:
        sl_a = tuple(slice(max(x, 0), L + min(x, 0)) for x in d); sl_b = tuple(slice(max(-x, 0), L + min(-x, 0)) for x in d)
        A, B = c[sl_a], c[sl_b]
        if np.any(((A == 0) & (B == 1)) | ((A == 1) & (B == 0))):
            return True
    return False


D6 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
D26 = [d for d in [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)] if d > (0, 0, 0)]
n = sep6 = sep26 = 0; examples = []; solid_any = 0
for d in sorted(ROOT.iterdir())[a:b]:
    W = np.load(d / "public" / "windows.npy"); idx = np.load(d / "public" / "observations_index.npy")
    pos = {(int(w), int(s)): (int(o), int(m)) for w, s, o, m in idx}
    blob = (d / "public" / "observations.bin").read_bytes()
    for r in W:
        if int(r["Lp"]) != Lp:
            continue
        L = int(r["L"]); w = int(r["window"])
        o, m = pos[(w, -1)]
        sol = np.unpackbits(np.frombuffer(dz.decompress(blob[o:o + m]), np.uint8), count=L ** 3).astype(bool).reshape(L, L, L)
        for j in range(int(r["k"])):
            if not (int(r["mask"]) >> j) & 1:
                continue
            o, m = pos[(w, j)]
            lit = np.unpackbits(np.frombuffer(dz.decompress(blob[o:o + m]), np.uint8), count=L ** 3).astype(bool).reshape(L, L, L)
            c = np.where(sol, 2, np.where(lit, 1, 0)).astype(np.int8)
            n += 1; solid_any += bool(sol.any())
            if not adj(c, D6):
                sep6 += 1
                s26 = not adj(c, D26); sep26 += s26
                if len(examples) < 20:
                    examples.append(dict(scene=d.name, window=w, source=j, sep26=bool(s26), n_solid=int(sol.sum()),
                                         n_lit=int((c == 1).sum()), n_dark=int((c == 0).sum())))
json.dump(dict(Lp=Lp, scenes=[a, b], informative_channels=n, with_any_solid=solid_any, solid_separated_6=sep6,
               solid_separated_26=sep26, share_6=sep6 / max(n, 1), examples=examples), sys.stdout, indent=1)
