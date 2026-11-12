"""The certified tile renderer's packed_batch, split so that only true GPU work runs under a cross-process lock.

Same arithmetic as rind3d.rendering.cuda_backend.CUDASource.packed_batch:
* the same kernels ('regions' per level, then 'packed') on the same DeviceGeometry;
* the same shadow-plane rows, encoded with the same directed enclosures (fastplanes: verified identical);
* the same exact CPU recomputation of every UNKNOWN voxel (native.points of the certified CPU binding).

Not computed: the per-tile certificate-coverage labels download (statistics only)."""
import contextlib

import numpy as np

import fastplanes


def device_geometry(fac, bundle, policy, probe_source):
    """The factory's DeviceGeometry for this bundle (created through one ordinary CUDASource)."""
    src = fac.source(bundle, probe_source, "cuda_certified", policy)
    dg = src.device; src.close()
    return dg


def render_tiles(dg, ctx, s, origins, side=64, N=1024, lock=None):
    """Exact (lit, solid) packed bits for 64^3 tiles at `origins`, source s. Returns (lit (n, 32768) uint8, solid)."""
    cpu = ctx.bind(s)
    planes = [fastplanes.shadow_planes_fast(b, cpu.source) for b in ctx.bodies]
    rows, offs = fastplanes.encode_fast(planes)
    origins = np.ascontiguousarray(origins, dtype="<i4").reshape(-1, 3)
    r = dg.runtime; g = dg; body_count = len(ctx.bodies); last = int(np.log2(side // 8))
    lit_all, sol_all, unk = [], [], []
    with (lock if lock is not None else contextlib.nullcontext()):
        prow = r.upload(rows); poff = r.upload(offs)
        try:
            for i0 in range(0, len(origins), g.batch):
                o = origins[i0:i0 + g.batch]; count = len(o)
                r.upload(o, g.origins)
                for level in range(last + 1):
                    r.launch('regions', ((8 ** level + 127) // 128, count, 1), (128, 1, 1),
                             [g.rows, g.offsets, prow, poff, body_count, g.origins, count, side, N, level, g.labels, 0])
                r.launch('packed', (128, count, 1), (256, 1, 1),
                         [g.rows, g.offsets, prow, poff, body_count, g.origins, count, side, N, last, g.labels, *g.output, 0])
                vals = [r.download(p, (count, 32768), np.uint8) for p in g.output]
                lit_all.append(vals[0]); sol_all.append(vals[1]); unk.append((vals[2], vals[3]))
        finally:
            r.free(prow); r.free(poff)
    lit = np.concatenate(lit_all); sol = np.concatenate(sol_all)
    # exact CPU recomputation of UNKNOWN voxels (outside the lock), exactly as packed_batch does
    k = 0
    for (u0, u1) in unk:
        for i in range(len(u0)):
            for which, uarr in ((0, u0), (1, u1)):
                if not np.any(uarr[i]):
                    continue
                bad = np.flatnonzero(np.unpackbits(uarr[i], bitorder='little'))
                z, y, x = np.unravel_index(bad, (64,) * 3)
                pts = (np.stack([x, y, z], axis=1) + origins[k + i] + .5) / N
                yy, mm, _ = cpu.native.points(np.ascontiguousarray(pts, dtype='<f8'), False)
                actual = (yy, mm)[which]
                if np.any(actual < 0):
                    raise RuntimeError('Native exact UNKNOWN fallback did not certify')
                tgt = lit if which == 0 else sol
                bits = np.unpackbits(tgt[k + i], bitorder='little'); bits[bad] = actual
                tgt[k + i] = np.packbits(bits, bitorder='little')
        k += len(u0)
    if np.any(lit & sol):
        raise AssertionError('CUDA solid receiver lit')
    return lit, sol
