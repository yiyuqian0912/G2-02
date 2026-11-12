"""Field-side benchmarks (producer venv): the v0.2 per-scene reader, dynamic windows, and the earlier numbers.

  run.sh rind3d_training.bench_fields
  F1 v0.2 ProductionDataset random cross-scene access per profile x L: cold (page cache dropped, LRU empty),
     warm (same windows again), scene-local (consecutive windows of one scene); with the time split into scene
     open (RINDPack headers + windows.npy) and window read/decode, and the bytes handed to the caller.
  F2 reproduction of the v0.2 C8 numbers: 156 SB-L64 windows/s and the 2-worker DataLoader pass of 256
     natural windows (10.64 s), decomposed by size/profile and by IPC payload bytes.
  F3 dynamic natural windows from stored fields (trusted process): random scene + uniform origin, exact counts
     (CountIndex) for the privileged flags, SB response + SOLID read; MA combined from private channels.
Output: outputs/bench/field_bench.json
"""
import json
import os
import time

import numpy as np

from .env import CANARY, OUT


def drop(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def canary_public_files():
    return [p for p in CANARY.glob("shards/*/scenes/*/public/*") if p.is_file()] + \
           [p for p in CANARY.glob("shards/*/scenes/*/private/src*/*.rindb")]


def f1_v02(n=120, seed=3):
    from rind3d_production.loader import ProductionDataset
    out = {}
    rng = np.random.default_rng(seed)
    for prof, name in (("rind3d-csg-single-binary-v0.2", "SB"), ("rind3d-csg-multi-additive-v0.2", "MA")):
        ds = ProductionDataset(CANARY, profiles=[prof], subsets=("natural", "informative", "extreme", "rare", "nonlocal"))
        Ls = np.array([int(ds._scene(int(si))["public_windows"][int(w)]["L"]) for si, w in ds.index])
        ds.cache.clear()
        for L in (64, 128, 256):
            idx = rng.choice(np.flatnonzero(Ls == L), n, replace=False)
            row = {}
            for state in ("cold", "warm"):
                if state == "cold":
                    drop(canary_public_files())
                ds.cache.clear()
                t_open = t_read = 0.0
                nbytes = 0
                for i in idx:
                    si, w = map(int, ds.index[i])
                    t = time.perf_counter()
                    ds._scene(si)
                    t2 = time.perf_counter()
                    x = ds[int(i)]
                    t3 = time.perf_counter()
                    t_open += t2-t
                    t_read += t3-t2
                    nbytes += x["observed_response"].nbytes + x["observed_solid"].nbytes
                row[state] = {"windows_per_s": n/(t_open+t_read), "scene_open_ms": 1e3*t_open/n, "read_decode_ms": 1e3*t_read/n,
                              "returned_bytes_per_window": nbytes/n}
            # scene-local: n consecutive windows of the same L from few scenes (LRU hits)
            ds.cache.clear()
            loc = np.flatnonzero(Ls == L)
            by = {}
            for i in loc:
                by.setdefault(int(ds.index[i][0]), []).append(int(i))
            seq = [i for v in list(by.values())[:12] for i in v][:n]
            t = time.perf_counter()
            for i in seq:
                ds[i]
            row["scene_local_windows_per_s"] = len(seq)/(time.perf_counter()-t)
            out[f"{name}/L{L}"] = row
    return out


def f2_reproduce(seed=5):
    """Same construction as rind3d_production.usability (rng seed 5): the 2-worker pass over 256 natural windows."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from rind3d_production.loader import ProductionDataset, collate_numpy
    rng = np.random.default_rng(5)
    for prof in ("rind3d-csg-single-binary-v0.2", "rind3d-csg-multi-additive-v0.2"):
        ds0 = ProductionDataset(CANARY, profiles=[prof], subsets=("natural", "informative", "extreme", "rare", "nonlocal"))
        for L in (64, 128, 256):                               # consume the same RNG draws as usability.py
            _ = rng.permutation(len(ds0))[:4000]
    ds = ProductionDataset(CANARY, subsets=("natural",))
    sel = rng.permutation(len(ds))[:256].tolist()
    meta = []
    for i in sel:
        si, w = map(int, ds.index[i])
        meta.append((ds.scenes[si]["profile"].split("-")[2], int(ds._scene(si)["public_windows"][w]["L"])))
    comp = {}
    for p, L in meta:
        comp[f"{p}/L{L}"] = comp.get(f"{p}/L{L}", 0) + 1
    out = {"composition": comp}
    for w in (0, 2):
        drop(canary_public_files())
        ds.cache.clear()
        t = time.perf_counter()
        n = nb = 0
        for batch in DataLoader(Subset(ds, sel), batch_size=8, num_workers=w, collate_fn=collate_numpy):
            n += len(batch["window_id"])
            nb += sum(a.nbytes for a in batch["observed_response"]) + sum(a.nbytes for a in batch["observed_solid"])
        out[f"workers{w}_cold"] = {"windows": n, "seconds": time.perf_counter()-t, "payload_bytes_through_IPC": nb if w else 0,
                                   "returned_bytes": nb}
    # per-item single-process cost by class (warm)
    per = {}
    ds.cache.clear()
    for i, (p, L) in zip(sel, meta):
        t = time.perf_counter()
        x = ds[i]
        per.setdefault(f"{p}/L{L}", []).append((time.perf_counter()-t, x["observed_response"].nbytes + x["observed_solid"].nbytes))
    out["per_class_single_process"] = {k: {"n": len(v), "ms_mean": 1e3*float(np.mean([a for a, _ in v])),
                                           "bytes_mean": float(np.mean([b for _, b in v])),
                                           "seconds_total": float(np.sum([a for a, _ in v]))} for k, v in per.items()}
    out["torch"] = torch.__version__
    return out


def f3_dynamic(n=100, seed=11):
    from rind3d_production import windows as W2
    from rind3d_production.production import combined_window
    from .scenes import canary_scenes
    refs = canary_scenes()
    rng = np.random.default_rng(seed)
    out = {}
    for prof in ("SB", "MA"):
        pool = [r for r in refs if r.profile == prof]
        opened = {}
        for L in (64, 128, 256):
            for state in ("cold", "warm"):
                rng2 = np.random.default_rng([seed, L])
                if state == "cold":
                    drop(canary_public_files())
                    for v in opened.values():
                        for p in v[0] + v[1]:
                            p.close()
                    opened = {}
                t_open = t_cnt = t_read = 0.0
                for _ in range(n):
                    sc = pool[int(rng2.integers(len(pool)))]
                    o = tuple(int(v) for v in rng2.integers(0, 1024-L+1, 3))
                    t = time.perf_counter()
                    if sc.scene_id not in opened:
                        opened[sc.scene_id] = sc.open_counts() + (sc.privileged_labels(),)
                    packs, sols, cis, (src, a) = opened[sc.scene_id]
                    t2 = time.perf_counter()
                    free, solid, lit = W2.window_counts(cis, o, L)
                    W2.contains_source(o, L, src)
                    t3 = time.perf_counter()
                    if prof == "SB":
                        packs[0].read_window(*o, L, packed=True)
                    else:
                        combined_window(packs, a, sols[0], o, L)
                    sols[0].read_window(*o, L, packed=True)
                    t4 = time.perf_counter()
                    t_open += t2-t
                    t_cnt += t3-t2
                    t_read += t4-t3
                out[f"{prof}/L{L}/{state}"] = {"windows_per_s": n/(t_open+t_cnt+t_read), "scene_open_ms": 1e3*t_open/n,
                                               "counts_flags_ms": 1e3*t_cnt/n, "read_ms": 1e3*t_read/n,
                                               "scenes_open": len(opened)}
        for v in opened.values():
            for p in v[0] + v[1]:
                p.close()
    return out


def main():
    res = {}
    t = time.perf_counter()
    res["F1_v02_reader"] = f1_v02()
    res["F2_reproduce_v02_C8"] = f2_reproduce()
    res["F3_dynamic"] = f3_dynamic()
    res["seconds"] = time.perf_counter()-t
    d = OUT/"bench"
    d.mkdir(parents=True, exist_ok=True)
    (d/"field_bench.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:5000])


if __name__ == "__main__":
    main()
