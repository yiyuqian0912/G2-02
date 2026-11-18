"""GPU checks of the ball tile layer (trivial GPU work; docs/CURVED_ASSESSMENT.md §1.3 checks C2-C3 and §2.2).

* `full`:    whole-tile comparison GPU (rind3d_curved.cuda_balls) vs CPU (BallSet: outward-rounded intervals plus exact
             rational fallback), every voxel of 64^3 tiles centred on near-tangent receivers of each ball.
             (The unit test samples 300 voxels per tile.)
* `tangent`: the exact-tangency configuration of test_exact_tangency_is_a_hit_on_every_path rendered as a GPU tile.
* `micro`:   per-tile time split: kernel / device-to-host copy / host decode, and kernel time vs ball count.

Usage: python gpu_checks.py {full,tangent,micro}"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
from test_curved_spheres import _near_tangent_rays  # noqa: E402
from rind3d_curved.spheres import BallSet  # noqa: E402
from rind3d_curved.cuda_balls import BallTiles  # noqa: E402
from rind3d_curved.families import generate_curved  # noqa: E402

N = 1024; T64 = 64


def _grid(o):
    zz, yy, xx = np.meshgrid(*(np.arange(T64),) * 3, indexing="ij")
    return (np.c_[xx.ravel(), yy.ravel(), zz.ravel()] + o + 0.5) / N


def full():
    rng = np.random.default_rng(11)
    tot = dict(tiles=0, voxels=0, blk_mismatch=0, sol_mismatch=0, blocked=0, gpu_unknown=0, cpu_exact=0)
    for slot in (2, 4):
        _, balls, _ = generate_curved(20261119, slot, "spheres", 0)
        B = BallSet(balls[:, :3], balls[:, 3]); T = BallTiles(balls)
        srcs = []
        while len(srcs) < 3:
            s = rng.uniform(0.02, 0.98, 3)
            if B.legal_source(s):
                srcs.append(s)
        for s in srcs:
            origins = [np.clip(np.floor(_near_tangent_rays(rng, b[:3], b[3], s, 1)[0] * N).astype(int) - 32, 0, N - T64)
                       for b in balls[:4]]
            for o, (blk, sol) in zip(origins, T.tiles(s, origins)):
                X = _grid(o)
                cb = B.blocked(s, X).reshape((T64,) * 3); cs = B.contains(X).reshape((T64,) * 3)
                tot["tiles"] += 1; tot["voxels"] += len(X); tot["blocked"] += int(cb.sum())
                tot["blk_mismatch"] += int((cb != blk).sum()); tot["sol_mismatch"] += int((cs != sol).sum())
        tot["gpu_unknown"] += T.stats["unknown"]; tot["cpu_exact"] += B.stats["exact"]
    print(tot)


def tangent():
    yl = 1279 / 2048; zl = 1025 / 2048; c = np.array([0.5, 0.5, zl]); r = yl - 0.5
    s = np.array([0.0625, yl, zl]); T = BallTiles(np.array([[*c, r]])); o = np.array([900 - 32, 639 - 32, 512 - 32])
    blk, _ = T.tile(s, o)
    B = BallSet(c[None], [r]); cb = B.blocked(s, _grid(o)).reshape((T64,) * 3)
    print(dict(gpu_unknown=T.stats["unknown"], tangent_receiver_blocked=bool(blk[32, 32, 32]), next_row_blocked=bool(blk[32, 33, 32]),
               tile_mismatch_vs_cpu=int((cb != blk).sum()), cpu_exact_fallbacks=B.stats["exact"]))


def micro():
    import cupy as cp
    _, balls, _ = generate_curved(20261119, 2, "spheres", 0)
    s = np.array([0.03, 0.05, 0.07]); origins = np.random.default_rng(1).integers(0, N - T64, (32, 3)).astype(np.int32)
    for nb in (1, 3, len(balls)):
        T = BallTiles(balls[:nb]); T.tiles(s, origins)                       # warm-up (NVRTC)
        t = time.perf_counter()
        for _ in range(5):
            T.tiles(s, origins)
        total = (time.perf_counter() - t) / 5 / len(origins)
        prev = T.drv.ctxGetCurrent(); T.drv.ctxSetCurrent(T.ctx)
        o = cp.asarray(origins); out = cp.empty(T64 ** 3 * len(origins), cp.uint8)
        launch = lambda: T.k(((T64 ** 3 + 255) // 256, len(origins)), (256,), (T.db, np.int32(nb), np.float64(s[0]), np.float64(s[1]),
                                                                            np.float64(s[2]), o, np.int32(T64), np.int32(N), out))
        launch(); cp.cuda.Device().synchronize(); t = time.perf_counter()
        for _ in range(5):
            launch()
        cp.cuda.Device().synchronize(); kern = (time.perf_counter() - t) / 5 / len(origins)
        t = time.perf_counter()
        for _ in range(5):
            cp.asnumpy(out)
        copy = (time.perf_counter() - t) / 5 / len(origins)
        T.drv.ctxSetCurrent(prev)
        print("balls %d, per 64^3 tile: total %.2f ms | kernel %.2f ms (%.2f ns per voxel-ball test) | D2H copy %.3f ms | host %.2f ms"
              % (nb, 1e3 * total, 1e3 * kern, 1e9 * kern / (T64 ** 3 * nb), 1e3 * copy, 1e3 * (total - kern - copy)))


if __name__ == "__main__":
    dict(full=full, tangent=tangent, micro=micro)[sys.argv[1]]()
