"""Stage B2: response materialization strategies, measured on the nine stored benchmark fields.

A  full 1024^3 response + solid fields (RINDPack, certified CUDA stream; rendered in B1), random window reads
B  selected dense windows rendered once and stored as packed+Zstandard payloads, then read
C  lazy: geometry + source metadata only; every access renders the window (warm or cold renderer)
D  hybrid: reusable geometry context / renderer per scene + LRU cache of rendered windows
Windows: sides 64, 128, 256 at random integer origins (overlap allowed). Every window obtained by B/C is
compared bit for bit with the A read (same certified physics, same voxel convention).
Teacher workload (independent of the true-source field): candidate-source batches against window receivers.
"""
import json
import time
import zlib

import numpy as np
import zstandard as zstd

from rind3d.counts import CountIndex
from rind3d.pack import RINDPack
from rind3d.rendering.backends import BackendFactory

from . import scenes as S
from .bench_backends import SEED, window_points
from .env import POLICY, ROOT

OUT = ROOT/"outputs"/"benchmarks"


def tiles(origin, L):
    return [(origin[0]+64*i, origin[1]+64*j, origin[2]+64*k) for k in range(L//64) for j in range(L//64) for i in range(L//64)]


def render_window(renderer, origin, L):
    """Certified CUDA render of an L^3 window at an arbitrary origin, tiled by 64^3 blocks (batches of 32)."""
    t = tiles(origin, L)
    out = np.zeros((L, L, L), bool)
    for pos in range(0, len(t), 32):
        chunk = t[pos:pos+32]
        packed = renderer.packed_batch([list(map(int, c)) for c in chunk], 64, 1024)
        for (x, y, z), (resp, _) in zip(chunk, packed):
            a = np.unpackbits(np.frombuffer(resp, np.uint8), bitorder="little").reshape(64, 64, 64).astype(bool)
            out[z-origin[2]:z-origin[2]+64, y-origin[1]:y-origin[1]+64, x-origin[0]:x-origin[0]+64] = a
    return out


def main():
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 9]))
    rows = []
    comp = zstd.ZstdCompressor(level=3)
    with BackendFactory(geometry_cache=2, cuda_batch=32) as factory:
        for i, family in enumerate(S.FAMILIES):
            root = OUT/"fields"/f"{family}-cuda_certified"
            bundle, _ = S.geometry(SEED, 100+i, family)
            t0 = time.perf_counter()
            ctx = factory.context(bundle, POLICY)
            t_ctx = time.perf_counter()-t0
            truth = S.sample_sources(ctx, SEED, 100+i, 1)[0][0]
            row = {"family": family, "bytes": {p.name: p.stat().st_size for p in root.iterdir() if p.suffix in (".rindb", ".rindi")}}
            with RINDPack(root/"response.rindb", cache_bricks=0) as y, RINDPack(root/"solid.rindb", cache_bricks=0) as m:
                t0 = time.perf_counter()
                ci = CountIndex.build(y, m)
                row["count_index_build_seconds"] = time.perf_counter()-t0
                row["count_index_bytes"] = ci.save(OUT/"fields"/f"{family}-counts.rindc")
                t0 = time.perf_counter()
                for _ in range(1000):
                    o = rng.integers(0, 1024-64, 3)
                    ci.query(tuple(int(v) for v in o), 64)
                row["count_query_seconds"] = (time.perf_counter()-t0)/1000
            t0 = time.perf_counter()
            renderer = factory.source(bundle, truth, "cuda_certified", POLICY)
            t_bind = time.perf_counter()-t0
            row["setup"] = {"context_seconds": t_ctx, "cuda_source_bind_seconds": t_bind}
            for L in (64, 128, 256):
                origins = [rng.integers(0, 1024-L, 3) for _ in range(8 if L < 256 else 4)]
                cold, warm, rend, stored, ok = [], [], [], [], True
                for o in origins:
                    o = tuple(int(v) for v in o)
                    t0 = time.perf_counter()
                    with RINDPack(root/"response.rindb", cache_bricks=0) as y:
                        a = y.read_window(*o, L)
                    cold.append(time.perf_counter()-t0)
                    with RINDPack(root/"response.rindb", cache_bricks=256) as y:
                        y.read_window(*o, L)
                        t0 = time.perf_counter()
                        y.read_window(*o, L)
                        warm.append(time.perf_counter()-t0)
                    t0 = time.perf_counter()
                    b = render_window(renderer, o, L)
                    rend.append(time.perf_counter()-t0)
                    ok &= bool(np.array_equal(a, b))
                    packed = np.packbits(b, axis=-1, bitorder="little")
                    blob = comp.compress(packed.tobytes())
                    stored.append(len(blob))
                    t0 = time.perf_counter()
                    np.unpackbits(np.frombuffer(zstd.ZstdDecompressor().decompress(blob), np.uint8).reshape(L, L, L//8),
                                  axis=-1, bitorder="little").astype(bool)
                    row.setdefault(f"B_read_seconds_{L}", []).append(time.perf_counter()-t0)
                row[f"L{L}"] = {"A_read_cold_seconds": float(np.mean(cold)), "A_read_warm_seconds": float(np.mean(warm)),
                                "C_render_seconds": float(np.mean(rend)), "B_stored_bytes": float(np.mean(stored)),
                                "B_read_seconds": float(np.mean(row.pop(f"B_read_seconds_{L}"))),
                                "raw_packed_bytes": L**3//8, "render_equals_stored_field": ok}
            # Teacher: 32 candidate sources, full 64^3 window cost by CUDA raster (bind + render) vs native points.
            hyp, _ = S.hypothetical_sources(ctx, rng, 32)
            o = tuple(int(v) for v in rng.integers(0, 960, 3))
            t0 = time.perf_counter()
            for h in hyp:
                r = factory.source(bundle, h, "cuda_certified", POLICY)
                r.packed_batch([list(o)], 64, 1024)
                r.close()
            t_cuda = time.perf_counter()-t0
            pts = window_points(o, 64)
            free = pts[~ctx.contains(pts)]
            t0 = time.perf_counter()
            for h in hyp[:4]:
                ctx.bind(h).native.points(np.ascontiguousarray(free), False)
            t_pts = (time.perf_counter()-t0)/4
            row["teacher_full_window_64"] = {"candidates": int(len(hyp)), "cuda_seconds_per_candidate": t_cuda/len(hyp),
                                             "native_points_seconds_per_candidate_single_thread": t_pts, "free_receivers": int(len(free))}
            renderer.close()
            rows.append(row)
            print(family, {L: {k: round(v, 4) if isinstance(v, float) else v for k, v in row[f"L{L}"].items()} for L in (64, 128, 256)},
                  row["teacher_full_window_64"], flush=True)
    (OUT/"materialization-raw.json").write_text(json.dumps({"seed": SEED, "scenes": rows}, indent=1, default=float))


if __name__ == "__main__":
    main()
