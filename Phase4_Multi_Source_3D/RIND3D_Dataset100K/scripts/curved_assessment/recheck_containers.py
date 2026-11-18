"""Independent re-check of stored curved scene containers against the canonical exact oracle (rind.ellipsoid3d).

docs/CURVED_ASSESSMENT.md §1.3, check C1. CPU only, one process.

* Sphere-only containers: every stored bit is determined by the balls. SOLID(x) iff x lies in some closed ball; a FREE
  receiver is response-active iff the closed segment [s, x] misses every ball. Per channel the sample is up to
  `cap_if` FREE receivers 6-adjacent to a stored direct response interface (the near-tangent, hardest receivers) plus
  `n_uni` uniform FREE receivers; SOLID is checked on a uniform sample plus the SOLID boundary.
* Mixed (spheres_polyhedra) containers: only the necessary ball conditions are checked (a receiver stored active must
  miss every ball; a receiver inside a ball must be stored SOLID).

Usage: python recheck_containers.py WORK_DIR_WITH_scenes/ [cap_if=60] [n_uni=30]"""
import glob
import json
import sys
import time

import numpy as np
import zstandard
from rind.ellipsoid3d import ExactEllipsoid3D

N = 1024


def main(root, cap_if=60, n_uni=30):
    dctx = zstandard.ZstdDecompressor(); rng = np.random.default_rng(0)
    tot = dict(checked=0, mismatch=0, interface=0, solid_checked=0, solid_mismatch=0, mixed_nec=0, mixed_nec_fail=0)
    t0 = time.perf_counter()
    for p in sorted(glob.glob(f"{root}/scenes/*/*.npz")):
        z = np.load(p, allow_pickle=False); m = json.loads(bytes(z["meta"]).decode())
        balls = z["balls"]; orcs = [ExactEllipsoid3D(b[:3], np.eye(3) * b[3]) for b in balls]
        srcs = np.array([[float.fromhex(v) for v in s] for s in z["sources_hex"]])
        obs = bytes(z["obs"]); idx = z["obs_index"]; W = z["windows"]
        sphere_only = m["n_bodies"] == 0

        def arr(w, j, L):
            r = idx[(idx[:, 0] == w) & (idx[:, 1] == j)][0]
            bits = np.unpackbits(np.frombuffer(dctx.decompress(obs[r[2]:r[2] + r[3]]), np.uint8))[:L ** 3]
            return bits.reshape(L, L, L).astype(bool)                       # [Z, Y, X]

        sc = dict(checked=0, mismatch=0)
        for w in range(len(W)):
            L = int(W["L"][w]); o = np.array([W["x"][w], W["y"][w], W["z"][w]])
            sol = arr(w, -1, L)
            X = lambda zyx: (o + zyx[:, ::-1] + 0.5) / N
            zyx = rng.integers(0, L, (n_uni, 3))
            bd = np.argwhere(sol & ~np.all([np.roll(sol, s, a) for a in range(3) for s in (1, -1)], axis=0))
            if len(bd):
                zyx = np.r_[zyx, bd[rng.choice(len(bd), min(len(bd), n_uni), replace=False)]]
            for q, x in zip(zyx, X(zyx)):
                ins = any(orc.value(tuple(x)) <= 1 for orc in orcs)
                if sphere_only:
                    tot["solid_checked"] += 1; tot["solid_mismatch"] += int(ins != sol[tuple(q)])
                elif ins:
                    tot["mixed_nec"] += 1; tot["mixed_nec_fail"] += int(not sol[tuple(q)])
            free = ~sol
            for j in range(int(m["k"])):
                lit = arr(w, j, L)
                e = np.zeros_like(free)                                     # receivers on a direct FREE-FREE interface
                for a in range(3):
                    s0 = [slice(None)] * 3; s1 = [slice(None)] * 3; s0[a] = slice(0, -1); s1[a] = slice(1, None)
                    d = free[tuple(s0)] & free[tuple(s1)] & (lit[tuple(s0)] != lit[tuple(s1)])
                    e[tuple(s0)] |= d; e[tuple(s1)] |= d
                ie = np.argwhere(e); fr = np.argwhere(free)
                pick = ie[rng.choice(len(ie), min(len(ie), cap_if), replace=False)] if len(ie) else np.zeros((0, 3), int)
                uni = fr[rng.choice(len(fr), min(len(fr), n_uni), replace=False)] if len(fr) else np.zeros((0, 3), int)
                tot["interface"] += len(pick)
                zyx = np.r_[pick, uni]
                for q, x in zip(zyx, X(zyx)):
                    blocked = any(orc.segment_hit(srcs[j], x) for orc in orcs)
                    active = bool(lit[tuple(q)])
                    if sphere_only:
                        tot["checked"] += 1; sc["checked"] += 1
                        if blocked == active:
                            tot["mismatch"] += 1; sc["mismatch"] += 1
                    elif active:
                        tot["mixed_nec"] += 1; tot["mixed_nec_fail"] += int(blocked)
        print(p.split("/")[-1], m["family"], "k", m["k"], "balls", len(balls), sc, flush=True)
    print(json.dumps(tot), "%.1f s" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main(sys.argv[1], *(int(v) for v in sys.argv[2:4]))
