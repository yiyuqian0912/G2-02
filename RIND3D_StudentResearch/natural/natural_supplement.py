"""NAT-1K v1: natural-window supplement of release_1k_v1 (derived product; the release is only read).

Why: the release's windows are thinned by the effective-layer count m (c(m) in [0.01, 1]); for the declared Student
target T* (natural multi-source worlds, single-channel mixed examples) the release needs weights omega / c(m), whose Kish
ESS is 11.8% of the examples (E13). Windows drawn from p_nat on the SAME stored 1024^3 fields, with a FIXED number of
proposals per scene and side and no thinning, are a self-weighting sample of T* (equal weights within a side), and can be
combined with the release by multiple importance sampling with bounded weights.

Design (declared before any label is computed):
* stream: keyed_rng(seed, slot, 57000 + Lp) (attempt 0; unused by the physical stage: 51001, 52000+L .. 55000+L, 56000);
* n_p proposals per side: 8 for pilot-train scenes, 16 for pilot-select and pilot-eval scenes, none for test scenes;
* roles: split_of(scene_id) == "test" -> skipped (the 1K test split stays untouched); "val" -> pilot-eval;
  "train" and sha256(scene_id + ":pilot-select") mod 8 == 0 -> pilot-select; other train -> pilot-train;
* every proposal is kept in natural_log (origin, admissible, per-channel counts); a proposal is admissible iff no source
  lies in the closed box and FREE > 0 (the physical rule without the m >= 1 condition and without thinning);
* every admissible window is materialised (SOLID crop + every source's response crop, same packed format as the
  release); `mask` marks MIXED channels (>= 1 lit and >= 1 dark FREE receiver) = the label tasks; all-lit / all-dark
  channels are stored (types in natural.json) but not labelled here.
Output tree: <out>/scenes/<id>/{scene.json -> release, privileged -> release (read-only symlinks), public/windows.npy,
public/observations.bin, public/observations_index.npy, natural.json, natural_log.npz}, written to a stage dir and
renamed atomically. Resumable: finished scenes are skipped.

usage: natural_supplement.py <release_root> <out_root> <a>:<b>"""
import hashlib, io, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
import zstandard

sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Dataset1K")
from rind3d.geometry import keyed_rng
from rind3d_1k.physical import Fields, contains_any
from rind3d_1k.loader import split_of

SIDES = (64, 128, 256)
STREAM_NAT = 57000
NP = {"pilot-train": 8, "pilot-select": 16, "pilot-eval": 16}
VERSION = "nat-1k-v1"


def role_of(scene_id):
    sp = split_of(scene_id)
    if sp == "test":
        return "test"
    if sp == "val":
        return "pilot-eval"
    v = int(hashlib.sha256((scene_id + ":pilot-select").encode()).hexdigest(), 16) % 8
    return "pilot-select" if v == 0 else "pilot-train"


def packz(a, cctx):
    return cctx.compress(np.packbits(np.ascontiguousarray(a, dtype=bool).ravel()).tobytes())


def do_scene(rel, out_root):
    meta = json.loads((rel / "scene.json").read_text())
    role = role_of(meta["scene_id"])
    dst = out_root / "scenes" / rel.name
    if role == "test" or (dst / "natural.json").exists():
        return dict(scene=rel.name, role=role, skipped=True)
    t0 = time.perf_counter()
    N = int(meta["resolution"]); k = int(meta["k"]); slot = int(meta["slot"]); seed = int(meta["seed"])
    assert int(meta["attempt"]) == 0
    src = json.loads((rel / "privileged" / "sources.json").read_text())
    sources = np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]])
    dirs = [rel / "privileged" / "fields" / f"src{j}" for j in range(k)]
    F = Fields(dirs, N)
    stage = out_root / "stage" / rel.name
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "public").mkdir(parents=True)
    cctx = zstandard.ZstdCompressor(level=3)
    blobs = io.BytesIO(); index = []; wins = []; logs = {}; wmeta = []
    for Lp in SIDES:
        L = Lp * N // 1024
        rq = keyed_rng(seed, slot, STREAM_NAT + Lp)
        log = []
        for p in range(NP[role]):
            o = tuple(int(v) for v in rq.integers(0, N - L + 1, 3))
            if contains_any(sources, o, L, N):
                log.append((*o, 0, -1, 0, 0, 0)); continue
            cnt = F.channel_counts(o, L)                    # per channel (lit, shadow, solid)
            free = L ** 3 - int(cnt[0, 2])
            if free <= 0:
                log.append((*o, 0, -2, 0, 0, 0)); continue
            mixed = (cnt[:, 0] > 0) & (cnt[:, 1] > 0)
            alllit = (cnt[:, 1] == 0); alldark = (cnt[:, 0] == 0)
            m = int(mixed.sum()); mask = int(sum(1 << j for j in range(k) if mixed[j]))
            log.append((*o, 1, m, mask, int(alllit.sum()), int(alldark.sum())))
            w = len(wins)
            sol = F.solid.read_window(*o, L)
            b = packz(sol, cctx); index.append((w, -1, blobs.tell(), len(b))); blobs.write(b)
            fr = ~sol
            for j in range(k):
                lit = F.resp[j].read_window(*o, L)
                assert not np.any(lit & sol)
                assert [int(lit.sum()), int((fr & ~lit).sum()), int(sol.sum())] == cnt[j].tolist(), (rel.name, o, j)
                b = packz(lit, cctx); index.append((w, j, blobs.tell(), len(b))); blobs.write(b)
            wins.append((w, *o, L, Lp, m, mask, k))
            types = ["mixed" if mixed[j] else ("all_lit" if alllit[j] else "all_dark") for j in range(k)]
            wmeta.append(dict(window=w, Lp=Lp, origin=list(o), proposal=p, counts=cnt.tolist(), free=free, types=types,
                              lit_frac=[float(cnt[j, 0] / free) for j in range(k)]))
        logs[f"L{Lp}"] = np.array(log, dtype=np.int64).reshape(-1, 8)
    W = np.zeros(len(wins), dtype=[("window", "<i4"), ("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("L", "<i4"),
                                   ("Lp", "<i4"), ("m", "<i1"), ("mask", "<u1"), ("k", "<i1")])
    for i, r in enumerate(wins):
        W[i] = r
    (stage / "public" / "observations.bin").write_bytes(blobs.getvalue())
    np.save(stage / "public" / "observations_index.npy", np.array(index, dtype=np.int64).reshape(-1, 4))
    np.save(stage / "public" / "windows.npy", W)
    np.savez_compressed(stage / "natural_log.npz", **logs)
    os.symlink(rel / "scene.json", stage / "scene.json")
    os.symlink(rel / "privileged", stage / "privileged")
    nat = dict(version=VERSION, release_scene=str(rel), scene_id=meta["scene_id"], role=role, n_p=NP[role],
               stream=dict(key="keyed_rng(seed, slot, 57000 + Lp)", seed=seed, slot=slot), k=k, family=meta["family"],
               sides={str(Lp): dict(proposals=int(len(logs[f"L{Lp}"])), admissible=int(np.sum(logs[f"L{Lp}"][:, 3] == 1)),
                                    contains_source=int(np.sum(logs[f"L{Lp}"][:, 4] == -1)),
                                    no_free=int(np.sum(logs[f"L{Lp}"][:, 4] == -2)),
                                    mixed_channels=int(np.sum(np.clip(logs[f"L{Lp}"][:, 4], 0, None))))
                      for Lp in SIDES},
               windows=wmeta, wall=time.perf_counter() - t0)
    (stage / "natural.json").write_text(json.dumps(nat, indent=0))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(stage, dst)
    return dict(scene=rel.name, role=role, windows=len(wins), mixed=int(sum(r[6] for r in wins)), wall=nat["wall"])


def main():
    rel_root = Path(sys.argv[1]); out_root = Path(sys.argv[2]); a, b = (int(x) for x in sys.argv[3].split(":"))
    out_root.mkdir(parents=True, exist_ok=True)
    for d in sorted((rel_root / "scenes").iterdir()):
        if a <= int(d.name) < b and (d / "SEAL.json").exists():
            print(json.dumps(do_scene(d, out_root)), flush=True)


if __name__ == "__main__":
    main()
