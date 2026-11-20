"""Audit archived accepted observations and archived capped solver attempts. Never reads source positions."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import zstandard
from scipy.ndimage import label
import evidence


def features(active, inactive, edges):
    minority = min(active, inactive)
    n = sum(edges)
    probs = np.asarray([i/n for i in edges if i])
    return dict(minority=minority, fraction=minority/(active+inactive), edges=n,
                orientation=float(-np.sum(probs*np.log(probs))))


def spatial_features(cls):
    """Occupied 8-by-8-by-8 bins of negative endpoints and their 26-neighbor connectivity."""
    side = cls.shape[0]
    occupied = np.zeros((8,8,8),bool)
    for axis in range(3):
        a = [slice(None)]*3
        b = a.copy()
        a[axis], b[axis] = slice(None,-1), slice(1,None)
        left, right = cls[tuple(a)], cls[tuple(b)]
        for sign in (0,1):
            ijk = np.nonzero((left == sign) & (right == 1-sign))
            if sign:
                ijk[axis][:] += 1
            occupied[tuple(i//(side//8) for i in ijk)] = True
    return int(occupied.sum()), int(label(occupied, structure=np.ones((3,3,3)))[1])


def unpack(blob, offsets, window, channel, side):
    start,size = offsets[window,channel]
    bits = zstandard.ZstdDecompressor().decompress(blob[start:start+size],max_output_size=side**3//8)
    if len(bits) != side**3//8:
        raise ValueError("invalid packed crop")
    return np.unpackbits(np.frombuffer(bits,np.uint8)).reshape((side,)*3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    archive=args.work/"xlf_cbc/v301_canary_10s"
    runs={r["task"]["key"]:r for r in map(json.loads,(archive/"runs.jsonl").read_text().splitlines())}
    matched=set()
    with args.output.open("x") as out:
        for file in sorted((args.work/"sampling_impact/v3/scenes").glob("*/*.npz")):
            with np.load(file,allow_pickle=False) as z:
                meta=json.loads(z["meta"].tobytes())
                blob=z["obs"].tobytes()
                offsets={(int(w),int(j)):(int(start),int(size)) for w,j,start,size in z["obs_index"]}
                geom=hashlib.sha256(z["geometry"].tobytes()).hexdigest()
                for w,win in enumerate(meta["windows_full"]):
                    side=win["L"]
                    solid=unpack(blob,offsets,w,-1,side).astype(bool)
                    for j,cnt in enumerate(win["counts"]):
                        if not win["mask"] & (1<<j):
                            continue
                        key=f"{meta['slot']:06d}_w{w:02d}_s{j}"
                        row=dict(key=key,slot=meta["slot"],w=w,j=j,L=side,family=meta["family"],
                                 scene=meta["scene_id"],stream=win["stream"],proposal=win["proposal"],
                                 mask=win["mask"],K=meta["k"],bodies=meta["n_bodies"],
                                 **features(*cnt[:2], win["interface_edges"][j]))
                        cls=unpack(blob,offsets,w,j,side)
                        if np.any(cls[solid]) or int(cls.sum())!=cnt[0] or int(solid.sum())!=cnt[2]:
                            raise ValueError("metadata/crop mismatch")
                        cls[solid]=2
                        t=perf_counter()
                        row["bins"],row["clusters"]=spatial_features(cls)
                        row["spatial_s"]=perf_counter()-t
                        if key in runs:
                            run=runs[key]
                            rec=json.loads((archive/"results"/(key+".json")).read_bytes())
                            problem=rec["problem"]
                            if (problem["geometry_sha256"],problem["origin"],problem["side"],problem["channel"],
                                problem["observation_xyz_sha256"]) != (geom,win["origin"],side,j,evidence.crop_digest(cls.T)):
                                raise ValueError("archive identity differs: "+key)
                            matched.add(key)
                            s=run["summary"]
                            row.update(wall=s["wall"],process_wall=run["process_wall"],status=rec["status"],
                                       cells=rec.get("cells"),lower=rec.get("lower_volume"),live=rec.get("unknown_volume"),
                                       convergence=rec.get("trace"),crop=problem["observation_xyz_sha256"])
                            if "wire" in rec:
                                v=np.frombuffer(base64.b64decode(rec["wire"]),dtype=[("c","<u4"),("p","<f8")])["p"]
                                row["effective_cells"]=float(np.exp(-np.sum(v*np.log(v))))
                        print(json.dumps(row,separators=(",",":")),file=out)
            out.flush()
    if set(runs)!=matched:
        raise ValueError("some archive keys lacked an identity match")


if __name__=="__main__":
    main()

