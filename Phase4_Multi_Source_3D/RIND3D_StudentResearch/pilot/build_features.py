"""Student pilot: example tables and pooled input features (CPU; labeller venv: numpy, numba, zstandard).

usage: build_features.py <out_dir> <worker_i> <n_workers>
Examples (one per window channel):
* release_1k_v1, scenes of roles pilot-train / pilot-select / pilot-eval: every MIXED channel (label tasks of the backfill);
* NAT-1K supplement: pilot-train scenes: every mixed channel; pilot-select / pilot-eval scenes: EVERY channel of every
  admissible window (mixed, all-lit, all-dark), for the T* evaluation and the natural OOD strata.
The 1K test split is never read.
Per example (structured row): ids, side, origin, m, k, family, role, response type (0 boundary-mixed, 1 solid-separated,
2 all-lit, 3 all-dark; from the crop: X-measurable), counts, the true source (privileged; target / evaluation only),
and the row of its uint8 feature block [4, 32, 32, 32] (XYZ; lit-FREE, dark-FREE, SOLID, boundary fractions x 255).
Writes shard files <out>/shard{i}.npy (rows) and <out>/shard{i}.u8 (features), resumable per scene."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "natural"))
from featlib import pooled  # noqa: E402

REL = (Path(os.environ.get("RIND3D_1K_DATA", "/home/yushu/Projects/RIND/RIND3D_1K_data")) / "release_1k_v1")
NAT = (Path(os.environ.get("RIND3D_1K_DATA", "/home/yushu/Projects/RIND/RIND3D_1K_data")) / "natural_supplement_v1")
P = 32
ROLES = ("pilot-train", "pilot-select", "pilot-eval")
FAMILIES = ("barriers", "convex_polyhedra", "corridors", "multiscale_convex", "occluder_clusters", "prisms",
            "rotated_boxes", "thin_slabs", "wedges")
ROW = np.dtype([("ds", "i1"), ("scene", "i4"), ("window", "i4"), ("source", "i1"), ("Lp", "i2"), ("ox", "i4"),
                ("oy", "i4"), ("oz", "i4"), ("m", "i1"), ("k", "i1"), ("family", "i1"), ("role", "i1"), ("type", "i1"),
                ("nlit", "i8"), ("ndark", "i8"), ("nsol", "i8"), ("npairs", "i8"), ("sx", "f8"), ("sy", "f8"),
                ("sz", "f8"), ("feat", "i8")])


def role_of(sid):
    from natural_supplement import role_of as r
    return r(sid)


def scene_jobs():
    jobs = []
    for d in sorted((REL / "scenes").iterdir()):
        if not (d / "SEAL.json").exists():
            continue
        meta = json.loads((d / "scene.json").read_text())
        role = role_of(meta["scene_id"])
        if role == "test":
            continue
        jobs.append((d.name, role))
    return jobs


class Crops:
    def __init__(self, d):
        self.W = np.load(d / "public" / "windows.npy")
        idx = np.load(d / "public" / "observations_index.npy")
        self.pos = {(int(w), int(s)): (int(o), int(n)) for w, s, o, n in idx}
        self.blob = (d / "public" / "observations.bin").read_bytes()
        self.dz = zstandard.ZstdDecompressor()

    def crop(self, w, s):
        o, n = self.pos[(w, s)]
        L = int(self.W[w]["L"])
        return np.unpackbits(np.frombuffer(self.dz.decompress(self.blob[o:o + n]), np.uint8), count=L ** 3).reshape(L, L, L)


def do_scene(name, role, rows, feats):
    meta = json.loads((REL / "scenes" / name / "scene.json").read_text())
    src = json.loads((REL / "scenes" / name / "privileged" / "sources.json").read_text())
    S = np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]])
    fam = FAMILIES.index(meta["family"]); ri = ROLES.index(role)
    for ds, root in ((0, REL), (1, NAT)):
        d = root / "scenes" / name
        if ds == 1 and not (d / "natural.json").exists():
            raise RuntimeError(f"natural supplement missing for {name}")
        C = Crops(d)
        for r in C.W:
            w = int(r["window"]); k = int(r["k"]); mask = int(r["mask"])
            chans = [j for j in range(k) if (mask >> j) & 1]
            if ds == 1 and role != "pilot-train":
                chans = list(range(k))                      # every channel: natural OOD strata
            if not chans:
                continue
            sol = C.crop(w, -1)
            for j in chans:
                f, nl, nd, ns, npair = pooled(sol, C.crop(w, j), P)
                mixed = nl > 0 and nd > 0
                assert mixed == bool((mask >> j) & 1), (name, ds, w, j)
                typ = (0 if npair > 0 else 1) if mixed else (2 if nd == 0 else 3)
                rows.append((ds, int(name), w, j, int(r["Lp"]), int(r["x"]), int(r["y"]), int(r["z"]), int(r["m"]), k, fam,
                             ri, typ, nl, nd, ns, npair, S[j, 0], S[j, 1], S[j, 2], -1))
                feats.append(f)


def main():
    out = Path(sys.argv[1]); i, n = int(sys.argv[2]), int(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    jobs = scene_jobs()[i::n]
    done_p = out / f"shard{i}.done.json"
    done = set(json.loads(done_p.read_text())) if done_p.exists() else set()
    rows_p, feat_p = out / f"shard{i}.npy", out / f"shard{i}.u8"
    old = list(np.load(rows_p)) if rows_p.exists() else []
    nfeat = (feat_p.stat().st_size // (4 * P ** 3)) if feat_p.exists() else 0
    assert len(old) == nfeat, "shard inconsistent"
    t0 = time.perf_counter()
    with open(feat_p, "ab") as ff:
        for name, role in jobs:
            if name in done:
                continue
            rows, feats = [], []
            do_scene(name, role, rows, feats)
            for r_, f in zip(rows, feats):
                r_ = list(r_); r_[-1] = nfeat; nfeat += 1
                old.append(tuple(r_)); ff.write(f.tobytes())
            ff.flush(); os.fsync(ff.fileno())
            arr = np.array([tuple(x) for x in old], dtype=ROW)
            np.save(str(rows_p) + ".tmp.npy", arr); os.replace(str(rows_p) + ".tmp.npy", rows_p)
            done.add(name); done_p.write_text(json.dumps(sorted(done)))
            print(json.dumps(dict(scene=name, role=role, examples=len(rows), total=len(old), wall=time.perf_counter() - t0)),
                  flush=True)


if __name__ == "__main__":
    main()
