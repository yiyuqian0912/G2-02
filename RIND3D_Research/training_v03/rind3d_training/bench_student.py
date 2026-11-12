"""Student-side loader and training benchmarks (benchmark-only CUDA venv: run_gpu.sh).

  run_gpu.sh rind3d_training.bench_student NAME [--quick]
Measures on the actual student export outputs/student/NAME/train:
  B1 per-sample read path decomposition (pread, zstd, CPU unpack) per modality x L, cold vs warm page cache,
     random cross-shard order vs shard-sequential order
  B2 DataLoader throughput: workers 0/1/2/4/8, size-bucketed packed batches vs mixed-size per-sample lists,
     persistent workers, repeated epochs (epoch 1 after cache drop vs epoch 2)
  B3 CPU->GPU: packed uint8 vs unpacked fp16, pinned vs pageable, on-GPU bit unpack and lossless space-to-depth
  B4 training throughput: a small 3D CNN on lossless space-to-depth inputs (64^3 token grid for every L),
     GPU-resident synthetic batches (compute bound) vs real loader-fed steps (end to end), AMP fp16
Targets for B4 come from the privileged join table (trusted training process) by opaque sample id.
Output: outputs/bench/student_bench_NAME.json
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from .env import OUT
from .student import SceneBalancedSampler, StudentShards, unpack_bits

BATCH = {64: 32, 128: 8, 256: 2}


def drop_cache(paths):
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.fsync(fd)
        except OSError:
            pass
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def b1_read_paths(sh, n=150, seed=0):
    import zstandard as zstd
    rng = np.random.default_rng(seed)
    out = {}
    for mod in (0, 1):
        for L in (64, 128, 256):
            cand = np.flatnonzero((sh.index["modality"] == mod) & (sh.index["L"] == L))
            if not len(cand):
                continue
            take = rng.choice(cand, min(n, len(cand)), replace=False)
            row = {}
            for state in ("cold", "warm"):
                if state == "cold":
                    drop_cache(sh.files)
                sh._pid = None
                fds, dz = sh._io()
                t_io = t_z = t_u = 0.0
                nbytes = dbytes = 0
                for i in take:
                    r = sh.index[i]
                    fd = fds[sh.file_of[i]]
                    t = time.perf_counter()
                    a = os.pread(fd, int(r["resp_bytes"]), int(r["resp_offset"]))
                    b = os.pread(fd, int(r["solid_bytes"]), int(r["solid_offset"]))
                    t2 = time.perf_counter()
                    pa, pb = dz.decompress(a), dz.decompress(b)
                    t3 = time.perf_counter()
                    sh.decode_response(r, pa)
                    unpack_bits(pb, L)
                    t4 = time.perf_counter()
                    t_io += t2-t
                    t_z += t3-t2
                    t_u += t4-t3
                    nbytes += len(a)+len(b)
                    dbytes += len(pa)+len(pb)
                k = len(take)
                row[state] = {"windows": k, "read_ms": 1e3*t_io/k, "zstd_ms": 1e3*t_z/k, "cpu_unpack_ms": 1e3*t_u/k,
                              "packed_path_windows_per_s": k/(t_io+t_z), "unpacked_path_windows_per_s": k/(t_io+t_z+t_u),
                              "stored_bytes_per_window": nbytes/k, "decoded_packed_bytes_per_window": dbytes/k}
            # shard-sequential order (same count, contiguous rows of one shard)
            f0 = sh.file_of[cand[0]]
            seq = cand[sh.file_of[cand] == f0][:k]
            drop_cache(sh.files)
            sh._pid = None
            t = time.perf_counter()
            for i in seq:
                sh.raw(int(i))
            row["cold_sequential_packed_windows_per_s"] = len(seq)/(time.perf_counter()-t)
            out[f"{'SB' if mod == 0 else 'MA'}/L{L}"] = row
    return out


class TorchDS:
    """Map-style dataset over StudentShards returning packed payloads (decoded on the GPU).
    tensors=True converts arrays to torch tensors inside the worker, so the DataLoader moves them through shared
    memory (file descriptors) instead of pickling the payload bytes through a pipe."""

    def __init__(self, sh, packed=True, tensors=False):
        self.sh, self.packed, self.tensors = sh, packed, tensors

    def __len__(self):
        return len(self.sh)

    def __getitem__(self, batch_indices):
        b = self.sh.batch(batch_indices, packed=self.packed)
        if self.tensors:
            import torch
            b["features"] = {k: torch.from_numpy(v) for k, v in b["features"].items()}
        return b


def identity(x):
    return x


def mixed_collate(sh):
    def f(idx):
        return [sh.read(int(i)) for i in idx]
    return f


def b2_dataloader(sh, workers=(0, 1, 2, 4, 8), max_batches=None, seed=0):
    import torch
    from torch.utils.data import DataLoader
    out = {}
    sampler = SceneBalancedSampler(sh.index, BATCH, seed=seed)
    plan = sampler.epoch(0)
    if max_batches:
        plan = plan[:max_batches]
    nwin = sum(len(b) for b in plan)
    for w in workers:
        for state in ("cold", "warm"):
            if state == "cold":
                drop_cache(sh.files)
            ds = TorchDS(sh)
            dl = DataLoader(ds, batch_size=None, sampler=[list(b) for b in plan], num_workers=w,
                            collate_fn=identity, pin_memory=False, prefetch_factor=4 if w else None, persistent_workers=False)
            t = time.perf_counter()
            n = nb = 0
            for batch in dl:
                n += len(batch["sample_id"])
                nb += sum(v.nbytes for v in batch["features"].values())
            dt = time.perf_counter()-t
            out[f"bucketed_packed/workers{w}/{state}"] = {"windows": n, "seconds": dt, "windows_per_s": n/dt, "MB_per_s": nb/dt/1e6}
    for w in (2, 4, 8):
        for state in ("cold", "warm"):
            if state == "cold":
                drop_cache(sh.files)
            dl = DataLoader(TorchDS(sh, tensors=True), batch_size=None, sampler=[list(b) for b in plan], num_workers=w,
                            collate_fn=identity, prefetch_factor=4, persistent_workers=False)
            t = time.perf_counter()
            n = nb = 0
            for batch in dl:
                n += len(batch["sample_id"])
                nb += sum(v.numel()*v.element_size() for v in batch["features"].values())
            dt = time.perf_counter()-t
            out[f"bucketed_packed_shm_tensors/workers{w}/{state}"] = {"windows": n, "seconds": dt, "windows_per_s": n/dt, "MB_per_s": nb/dt/1e6}
    # mixed-size per-sample lists of unpacked arrays (the v0.2-style path), 4 workers, same windows
    mix = SceneBalancedSampler(sh.index, BATCH, seed=seed, mixed=True).epoch(0)
    mix_plan = []
    budget = min(nwin, 2000)                       # the unpacked mixed path is ~20-70 windows/s: bounded sample
    for b in mix:
        mix_plan.append(b)
        budget -= len(b)
        if budget <= 0:
            break
    for w in (0, 4):
        drop_cache(sh.files)
        dl = DataLoader(list(range(len(mix_plan))), batch_size=None, num_workers=w,
                        collate_fn=lambda j, p=mix_plan, f=mixed_collate(sh): f(p[j]), prefetch_factor=4 if w else None)
        t = time.perf_counter()
        n = nb = 0
        for batch in dl:
            n += len(batch)
            nb += sum(f["observed_response"].nbytes + f["observed_solid"].nbytes for f, _ in batch)
        dt = time.perf_counter()-t
        out[f"mixed_unpacked/workers{w}/cold"] = {"windows": n, "seconds": dt, "windows_per_s": n/dt, "MB_per_s": nb/dt/1e6}
    out["plan_windows"] = nwin
    return out


def s2d(x, f, torch):
    """Lossless space-to-depth: [B,C,L,L,L] -> [B, C f^3, L/f, L/f, L/f]."""
    B, C, L = x.shape[0], x.shape[1], x.shape[2]
    n = L//f
    x = x.reshape(B, C, n, f, n, f, n, f).permute(0, 1, 3, 5, 7, 2, 4, 6)
    return x.reshape(B, C*f**3, n, n, n)


def to_input(batch, dev, torch, pinned=False):
    """Packed numpy batch -> fp16 [B, 2 (f^3), 64,64,64] on GPU (SB: lit, SOLID; MA: value, SOLID)."""
    from .student import gpu_unpack
    f = batch["features"]
    L = int(f["window_size"][0])
    sol = torch.as_tensor(f["observed_solid_packed"])
    if pinned:
        sol = sol.pin_memory()
    sol = gpu_unpack(sol.to(dev, non_blocking=True), torch)
    if "observed_response_packed" in f:
        r = torch.as_tensor(f["observed_response_packed"])
        if pinned:
            r = r.pin_memory()
        resp = gpu_unpack(r.to(dev, non_blocking=True), torch).half()
    else:
        pl = torch.as_tensor(f["observed_code_planes"]).to(dev, non_blocking=True)
        bits = gpu_unpack(pl, torch).to(torch.int64)
        code = (bits << torch.arange(bits.shape[1], device=dev).view(1, -1, 1, 1, 1)).sum(1)
        pal = torch.as_tensor(f["observed_palette"]).to(dev)
        resp = torch.gather(pal, 1, code.view(code.shape[0], -1)).view_as(code).half()
    x = torch.stack([resp, sol.half()], 1)
    return s2d(x, L//64, torch) if L > 64 else x


def make_net(torch):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.ModuleDict({str(L): nn.Conv3d(2*(L//64)**3, 32, 3, padding=1) for L in (64, 128, 256)})
            ch = [32, 64, 128, 128]
            layers = []
            for a, b in zip(ch[:-1], ch[1:]):
                layers += [nn.Conv3d(a, b, 3, stride=2, padding=1), nn.GroupNorm(8, b), nn.GELU()]
            self.trunk = nn.Sequential(nn.GELU(), *layers)
            self.head = nn.Sequential(nn.Linear(128 + 5, 128), nn.GELU(), nn.Linear(128, 3))

        def forward(self, x, L, cond):
            h = self.trunk(self.stem[str(L)](x)).mean((2, 3, 4))
            return self.head(torch.cat([h, cond], 1))
    return Net()


def b3_transfer(sh, torch, dev, n=6):
    out = {}
    rng = np.random.default_rng(1)
    for mod in (0, 1):
        for L in (64, 128, 256):
            cand = np.flatnonzero((sh.index["modality"] == mod) & (sh.index["L"] == L))
            if not len(cand):
                continue
            idx = rng.choice(cand, min(BATCH[L], len(cand)), replace=False)
            bp = sh.batch(idx, packed=True)
            bu = sh.batch(idx, packed=False)
            row = {}
            for pinned in (False, True):
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(n):
                    x = to_input(bp, dev, torch, pinned)
                torch.cuda.synchronize()
                row[f"packed_to_gpu_input_ms{'_pinned' if pinned else ''}"] = 1e3*(time.perf_counter()-t)/n
            resp = bu["features"]["observed_response"]
            sol = bu["features"]["observed_solid"]
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                x2 = torch.stack([torch.from_numpy(resp.astype(np.float16)), torch.from_numpy(sol.astype(np.float16))], 1).to(dev)
            torch.cuda.synchronize()
            row["unpacked_fp16_cpu_convert_and_copy_ms"] = 1e3*(time.perf_counter()-t)/n
            row["packed_host_bytes"] = int(sum(v.nbytes for k, v in bp["features"].items() if k.startswith("observed")))
            row["unpacked_fp16_host_bytes"] = int(resp.size*2*2)
            x_ref = x2[:, 0].float()
            x_pk = to_input(bp, dev, torch)
            if L > 64:
                f = L//64
                B = x_pk.shape[0]
                inv = x_pk.reshape(B, 2, f, f, f, 64, 64, 64).permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(B, 2, L, L, L)
            else:
                inv = x_pk
            row["lossless_roundtrip"] = bool(torch.equal(inv[:, 0].float(), x_ref) and torch.equal(inv[:, 1], x2[:, 1]))
            row["batch"] = int(len(idx))
            out[f"{'SB' if mod == 0 else 'MA'}/L{L}"] = row
    return out


def b4_training(sh, torch, dev, join, steps=20, workers=4, seed=0):
    from torch.utils.data import DataLoader
    by_sid = {int(s): i for i, s in enumerate(join["sample_id"])}
    net = make_net(torch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3)
    scaler = torch.amp.GradScaler("cuda")
    out = {}

    def step(x, L, cond, y):
        with torch.autocast("cuda", dtype=torch.float16):
            p = net(x, L, cond)
            loss = torch.nn.functional.mse_loss(p.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    def targets(batch):
        rows = [by_sid[int(s)] for s in batch["sample_id"]]
        y = torch.from_numpy(join["sources"][rows, 0, :].astype(np.float32)).to(dev)
        f = batch["features"]
        cond = np.column_stack([np.asarray(f["window_origin"])/1024, np.asarray(f["window_size"])/1024, np.asarray(f["modality"])]).astype(np.float32)
        return y, torch.from_numpy(cond).to(dev)
    sb = sh.index["modality"] == 0
    for L in (64, 128, 256):
        cand = np.flatnonzero(sb & (sh.index["L"] == L))
        rng = np.random.default_rng(seed)
        bt = sh.batch(rng.choice(cand, BATCH[L], replace=False), packed=True)
        x = to_input(bt, dev, torch)
        y, cond = targets(bt)
        for _ in range(3):
            step(x, L, cond, y)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(steps):
            step(x, L, cond, y)
        torch.cuda.synchronize()
        dt = (time.perf_counter()-t)/steps
        out[f"SB/L{L}/gpu_resident"] = {"batch": BATCH[L], "step_ms": 1e3*dt, "windows_per_s": BATCH[L]/dt,
                                         "peak_GPU_MiB": torch.cuda.max_memory_allocated()/2**20}
        # end-to-end: real loader-fed steps (size-bucketed packed batches, `workers` workers, cold cache)
        sel = sh.index.copy()
        sampler = SceneBalancedSampler(sh.index[cand], BATCH[L], seed=seed, drop_last=True)
        plan = [cand[b] for b in sampler.epoch(1)][:max(steps*3, 30)]
        drop_cache(sh.files)
        dl = DataLoader(TorchDS(sh, tensors=True), batch_size=None, sampler=[list(b) for b in plan], num_workers=workers,
                        prefetch_factor=4, collate_fn=identity, persistent_workers=False)
        it = iter(dl)
        first = next(it)
        x = to_input(first, dev, torch)
        y, cond = targets(first)
        step(x, L, cond, y)
        torch.cuda.synchronize()
        t = time.perf_counter()
        n = 0
        for b in it:
            x = to_input(b, dev, torch)
            y, cond = targets(b)
            step(x, L, cond, y)
            n += len(b["sample_id"])
        torch.cuda.synchronize()
        dt = time.perf_counter()-t
        out[f"SB/L{L}/loader_fed_workers{workers}_cold"] = {"windows": n, "seconds": dt, "windows_per_s": n/dt}
        del sel
    out["model_parameters"] = int(sum(p.numel() for p in net.parameters()))
    out["model"] = "stem conv3d(2 f^3 -> 32) on lossless space-to-depth 64^3 grid, 3 strided conv3d blocks (64,128,128) + GN + GELU, global mean, MLP(133->128->3); AdamW, AMP fp16, MSE on the true source (SB)"
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--b2-batches", type=int, default=None, help="bound the DataLoader passes to the first N planned batches")
    ap.add_argument("--workers", type=int, nargs="*", default=None)
    a = ap.parse_args()
    import torch
    root = OUT/"student"/a.name/"train"
    sh = StudentShards(root)
    join = np.load(OUT/"privileged"/"export"/a.name/"join.npy")
    dev = torch.device("cuda")
    res = {"export": a.name, "split": "train", "samples": len(sh), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "cpu_count": os.cpu_count(),
           "student_bytes": int(sum(p.stat().st_size for p in sh.files))}
    t = time.perf_counter()
    res["B1_read_paths"] = b1_read_paths(sh, n=60 if a.quick else 150)
    wl = tuple(a.workers) if a.workers else ((0, 4) if a.quick else (0, 1, 2, 4, 8))
    res["B2_dataloader"] = b2_dataloader(sh, workers=wl, max_batches=60 if a.quick else a.b2_batches)
    res["B3_transfer"] = b3_transfer(sh, torch, dev)
    res["B4_training"] = b4_training(sh, torch, dev, join, steps=10 if a.quick else 30)
    res["seconds"] = time.perf_counter()-t
    d = OUT/"bench"
    d.mkdir(parents=True, exist_ok=True)
    (d/f"student_bench_{a.name}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
