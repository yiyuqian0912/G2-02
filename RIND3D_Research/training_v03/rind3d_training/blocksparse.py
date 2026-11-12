"""Memory-aware, lossless L256 (and L128) representations, measured on the actual student export.

  run.sh rind3d_training.blocksparse NAME [--per-class 120]
Representations of one window (SB: lit + SOLID; voxel state in {SOLID, FREE-lit, FREE-shadow}):
  dense-packed     2 bits/voxel (response + SOLID bit planes), zstd on disk (the export format)
  block-sparse B   per B^3 block a 2-bit code {uniform SOLID, uniform lit, uniform shadow, mixed}; mixed blocks
                   keep their full-resolution packed bits. Exact (verified bit-for-bit). GPU memory scales
                   with the mixed-block fraction.
  space-to-depth   exact rearrangement to a 64^3 grid with 2 (L/64)^3 channels (used by bench_student B4)
Lossy alternatives are only REPORTED, never used: for a k^3 average/any-pool we count windows in which the
pooled representation cannot represent the minority receivers (rare windows) or thin SOLID (1-2 voxel) parts.
Output: outputs/bench/blocksparse_NAME.json
"""
import json
import time

import numpy as np
import zstandard as zstd

from .env import OUT
from .student import StudentShards

CODES = {"solid": 0, "lit": 1, "shadow": 2, "mixed": 3}


def encode(resp, solid, B):
    L = resp.shape[0]
    n = L//B
    r = resp.reshape(n, B, n, B, n, B).transpose(0, 2, 4, 1, 3, 5).reshape(n, n, n, B**3)
    s = solid.reshape(n, B, n, B, n, B).transpose(0, 2, 4, 1, 3, 5).reshape(n, n, n, B**3)
    all_s, any_s = s.all(-1), s.any(-1)
    lit = r & ~s
    all_lit = lit.all(-1)
    no_lit = ~lit.any(-1)
    code = np.full((n, n, n), 3, np.uint8)
    code[all_s] = 0
    code[~any_s & all_lit] = 1
    code[~any_s & no_lit] = 2
    mixed = code == 3
    payload = np.packbits(np.stack([r[mixed], s[mixed]], 1), axis=-1, bitorder="little")
    return code, payload


def decode(code, payload, B):
    n = code.shape[0]
    L = n*B
    r = np.zeros((n, n, n, B**3), bool)
    s = np.zeros_like(r)
    s[code == 0] = True
    r[code == 1] = True
    m = code == 3
    bits = np.unpackbits(payload, axis=-1, bitorder="little", count=B**3).astype(bool)
    r[m], s[m] = bits[:, 0], bits[:, 1]
    f = lambda a: a.reshape(n, n, n, B, B, B).transpose(0, 3, 1, 4, 2, 5).reshape(L, L, L)
    return f(r), f(s)


def lossy_report(resp, solid, k):
    """Would a k^3 pooling lose information? (any non-uniform block = information not representable)."""
    L = resp.shape[0]
    n = L//k
    st = np.where(solid, 0, np.where(resp, 1, 2)).astype(np.uint8)
    b = st.reshape(n, k, n, k, n, k).transpose(0, 2, 4, 1, 3, 5).reshape(n, n, n, k**3)
    nonuni = (b.min(-1) != b.max(-1))
    free = ~solid
    lit = resp & free
    nl, nf = int(lit.sum()), int(free.sum())
    minority = min(nl, nf-nl)
    # minority receivers living in blocks where they are a strict minority vanish under majority pooling
    if minority and nf:
        minority_is_lit = nl <= nf-nl
        mb = (lit if minority_is_lit else (free & ~resp)).reshape(n, k, n, k, n, k).transpose(0, 2, 4, 1, 3, 5).reshape(n, n, n, k**3).sum(-1)
        fb = free.reshape(n, k, n, k, n, k).transpose(0, 2, 4, 1, 3, 5).reshape(n, n, n, k**3).sum(-1)
        lost = int(mb[(mb > 0) & (2*mb < fb)].sum())
    else:
        lost = 0
    sb = solid.reshape(n, k, n, k, n, k).transpose(0, 2, 4, 1, 3, 5).reshape(n, n, n, k**3).sum(-1)
    thin_lost = int(sb[(sb > 0) & (2*sb < k**3)].sum())           # SOLID voxels that majority pooling erases
    return {"nonuniform_block_fraction": float(nonuni.mean()), "minority_receivers": minority,
            "minority_lost_by_majority_pool": lost, "solid_voxels_lost_by_majority_pool": thin_lost,
            "solid_voxels": int(solid.sum())}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--per-class", type=int, default=120)
    a = ap.parse_args()
    sh = StudentShards(OUT/"student"/a.name/"train", modalities=[0])
    rng = np.random.default_rng(0)
    comp = zstd.ZstdCompressor(level=3)
    dz = zstd.ZstdDecompressor()
    out = {}
    for L in (128, 256):
        cand = np.flatnonzero(sh.index["L"] == L)
        take = rng.choice(cand, min(a.per_class, len(cand)), replace=False)
        rows = []
        for i in take:
            r, pa, pb = sh.raw(int(i))
            stored = int(r["resp_bytes"] + r["solid_bytes"])
            resp = np.unpackbits(np.frombuffer(pa, np.uint8), bitorder="little").reshape(L, L, L).astype(bool)
            solid = np.unpackbits(np.frombuffer(pb, np.uint8), bitorder="little").reshape(L, L, L).astype(bool)
            row = {"dense_packed_zstd_bytes": stored, "dense_packed_raw_bytes": 2*L**3//8}
            for B in (8, 16):
                t = time.perf_counter()
                code, pay = encode(resp, solid, B)
                te = time.perf_counter()-t
                blob = comp.compress(code.tobytes()) + comp.compress(pay.tobytes())
                t = time.perf_counter()
                c2 = np.frombuffer(dz.decompress(comp.compress(code.tobytes())), np.uint8).reshape(code.shape)
                r2, s2 = decode(c2, pay, B)
                td = time.perf_counter()-t
                assert np.array_equal(r2, resp) and np.array_equal(s2, solid), "block-sparse roundtrip failed"
                mixed = float((code == 3).mean())
                row[f"B{B}"] = {"mixed_block_fraction": mixed, "zstd_bytes": len(blob),
                                "raw_bytes": int(code.size//4 + pay.nbytes), "encode_ms": 1e3*te, "decode_ms": 1e3*td,
                                "gpu_dense_fp16_MiB": 2*2*L**3/2**20, "gpu_sparse_fp16_MiB": 2*2*mixed*L**3/2**20 + code.size*1/2**20}
            for k in (2, 4):
                row[f"pool{k}"] = lossy_report(resp, solid, k)
            rows.append(row)
        agg = {"windows": len(rows), "roundtrip_exact": True,
               "dense_packed_zstd_bytes_median": float(np.median([x["dense_packed_zstd_bytes"] for x in rows]))}
        for B in (8, 16):
            for key in ("mixed_block_fraction", "zstd_bytes", "encode_ms", "decode_ms", "gpu_sparse_fp16_MiB"):
                v = [x[f"B{B}"][key] for x in rows]
                agg[f"B{B}_{key}_median"] = float(np.median(v))
                agg[f"B{B}_{key}_p95"] = float(np.percentile(v, 95))
            agg[f"B{B}_gpu_dense_fp16_MiB"] = rows[0][f"B{B}"]["gpu_dense_fp16_MiB"]
        for k in (2, 4):
            agg[f"pool{k}_windows_losing_minority_receivers"] = int(sum(x[f"pool{k}"]["minority_lost_by_majority_pool"] > 0 for x in rows))
            agg[f"pool{k}_windows_losing_solid_voxels"] = int(sum(x[f"pool{k}"]["solid_voxels_lost_by_majority_pool"] > 0 for x in rows))
            agg[f"pool{k}_windows_all_minority_lost"] = int(sum(0 < x[f"pool{k}"]["minority_receivers"] == x[f"pool{k}"]["minority_lost_by_majority_pool"] for x in rows))
            agg[f"pool{k}_nonuniform_block_fraction_median"] = float(np.median([x[f"pool{k}"]["nonuniform_block_fraction"] for x in rows]))
        out[f"SB/L{L}"] = agg
    d = OUT/"bench"
    d.mkdir(parents=True, exist_ok=True)
    (d/f"blocksparse_{a.name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
