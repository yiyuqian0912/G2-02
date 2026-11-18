"""One-core read-only mixed-interface pilot. Run from this directory (see README)."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import zstandard

from rind3d.geometry import load_geometry
from rind3d.rendering.polyhedra import decode_convex
from certificates import compile_band, integer_row, mixed_chart, segment_hits, value


def cropz(blob, entries, w, s, side):
    offset, length = entries[(w, s)]
    packed = zstandard.ZstdDecompressor().decompress(blob[offset:offset + length])
    return np.unpackbits(np.frombuffer(packed, np.uint8), count=side**3).reshape((side,)*3).T


def patch_origins(cls, side=8):
    """All aligned patches with an internal active/inactive FREE edge, then rank sample."""
    n = len(cls) // side
    blocks = cls.reshape(n, side, n, side, n, side).transpose(0, 2, 4, 1, 3, 5)
    mixed = np.zeros((n, n, n), dtype=bool)
    for axis in (3, 4, 5):
        l, r = [slice(None)] * 6, [slice(None)] * 6
        l[axis], r[axis] = slice(1, None), slice(None, -1)
        a, b = blocks[tuple(l)], blocks[tuple(r)]
        mixed |= (((a == 1) & (b == 0)) | ((a == 0) & (b == 1))).any(axis=(3, 4, 5))
    origins = np.argwhere(mixed) * side
    k = min(4, len(origins))
    ranks = [(2*i+1)*len(origins)//(2*k) for i in range(k)]
    return len(origins), [tuple(map(int, origins[i])) for i in ranks]


def top_count(p):
    if (p == p.flat[0]).all():
        return int(p.flat[0] != 2)
    h = len(p) // 2
    return sum(top_count(p[x:x+h,y:y+h,z:z+h]) for x in (0,h) for y in (0,h) for z in (0,h))


def source_box(cell, width, denom):
    xyz = (int(cell) // 16384, int(cell) // 128 % 128, int(cell) % 128)
    lo = tuple((k * denom // 128) // width * width for k in xyz)
    return lo, tuple(k + width for k in lo)


def audit_task(task, base):
    started = perf_counter()
    with np.load(task["container"], allow_pickle=False) as z:
        w, s = task["w"], task["s"]
        win = z["windows"][w]
        L = int(win["L"])
        denom = 2 * int(json.loads(z["meta"].tobytes())["resolution"])
        o = tuple(int(win[k]) for k in ("x","y","z"))
        entries = {(int(i),int(j)): (int(off),int(n)) for i,j,off,n in z["obs_index"]}
        blob = z["obs"].tobytes()
        cls = cropz(blob, entries, w, s, L)
        cls[cropz(blob, entries, w, -1, L).astype(bool)] = 2
        geometry = z["geometry"].tobytes()
        bodies = [tuple(integer_row(row) for row in b.rows) for b in decode_convex(load_geometry(io.BytesIO(geometry)))]
    key = task["key"]
    with np.load(base/"labels"/(key+"_tp.npz")) as lab:
        indices, inv = np.unique(lab["cell"], return_inverse=True)
        mass = np.bincount(inv, weights=lab["lo"])
        if not len(mass) or not max(mass) > 0:
            raise ValueError(f"{key} has no reference lower mass")
        ref = int(indices[np.argmax(mass)])
    blind = int.from_bytes(hashlib.sha256(("astra-mixed-patch-v1/"+key).encode()).digest()[:8], "big") % 128**3
    n_patches, origins = patch_origins(cls)
    runs = []
    checked = False
    for loc in origins:
        patch = cls[tuple(slice(a,a+8) for a in loc)]
        lo = tuple(2*(a+b)+1 for a,b in zip(o,loc))
        obs = [(tuple(a+2*int(i) for a,i in zip(lo,p)), bool(patch[tuple(p)]))
               for p in np.argwhere(patch != 2)]
        rbox = lo, tuple(c+14 for c in lo)
        items = top_count(patch)
        for mode, cell, div in (("blind",blind,128), ("support",ref,128), ("parent",ref,16)):
            box = source_box(cell,denom//div,denom)
            start = perf_counter()
            state, chart = mixed_chart(bodies,box,rbox,denom)
            rec = dict(loc=loc, box=mode, state=state, free=len(obs), pure_items=items, screen_seconds=perf_counter()-start,
                       source_cell_128=cell)
            # Do not charge pure-block diagnostic count to chart screening.
            if state == "pencil":
                c,i,j = chart
                f,b = bodies[c][i], bodies[c][j]
                band = compile_band(f,b,obs,denom)
                vol = band.fraction_of_box(box,denom)
                rec.update(chart=chart, lower=str(band.lower), upper=str(band.upper),
                           witnesses=[band.active_witness, band.inactive_witness], fraction_exact=str(vol),
                           fraction=float(vol))
                if mode == "support" and not checked:
                    # Two distinct points in the closed cell; independent primal segment clipping
                    # checks the complete labelled response pattern at each, including mismatches.
                    mid = tuple((a+h)//2 for a,h in zip(*box))
                    for pos in (box[0], mid):
                        predicates = []
                        for point, active in obs:
                            hit = any(segment_hits(body,pos,point,denom) for body in bodies)
                            test_hit = value(f,pos,denom)*value(b,point,denom) <= value(b,pos,denom)*value(f,point,denom)
                            if hit != test_hit:
                                raise AssertionError(f"ray mismatch on {key}")
                            predicates.append(active != hit)
                        if all(predicates) != band.accepts(pos,denom):
                            raise AssertionError(f"compressed patch mismatch on {key}")
                    rec["reference_rays"] = 2*len(obs)
                    checked = True
                if mode == "support":
                    # Nine repeated evaluations in Q: threshold form vs direct per-receiver conjunction.
                    from certificates import corners
                    points = (*corners(box), tuple((a+h)//2 for a,h in zip(*box)))
                    t = perf_counter()
                    out1 = []
                    for pos in points:
                        fs,bs=value(f,pos,denom),value(b,pos,denom)
                        out1.append(all((fs*value(b,r,denom) > bs*value(f,r,denom)) == active for r,active in obs))
                    rec["direct9_seconds"] = perf_counter()-t
                    t = perf_counter()
                    fresh = compile_band(f,b,obs,denom)
                    out2 = [fresh.accepts(pos,denom) for pos in points]
                    rec["compiled9_including_build_seconds"] = perf_counter()-t
                    if out1 != out2:
                        raise AssertionError("benchmark predicates disagree")
            elif state.startswith("constant"):
                rec.update(fraction_exact="0",fraction=0.0)
            runs.append(rec)
    return dict(key=key, family=task["family"],L=L, geometry_sha256=hashlib.sha256(geometry).hexdigest(),
                eligible_patches=n_patches, records=runs,wall_s=perf_counter()-started)


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("comparison", type=Path, help="directory containing tasks.json and labels/")
    arg.add_argument("output", type=Path)
    a = arg.parse_args()
    raw = (a.comparison/"tasks.json").read_bytes()
    tasks = json.loads(raw)
    rows = [audit_task(t,a.comparison) for t in tasks]
    print(len(rows), "tasks", Counter(r["state"] for t in rows for r in t["records"]))
    a.output.write_text(json.dumps(dict(base_commit="18465250870cbdb7a540be4537bb96efb08b122c",
                                       task_manifest_sha256=hashlib.sha256(raw).hexdigest(),tasks=rows),
                                  separators=(",",":"))+"\n")


if __name__ == "__main__":
    main()
