"""Local-first root restriction plans for the certified pipeline (labeller side, UNTRUSTED).

For every stage-1 root box and a deterministic set of H1-free direct response-interface edges (l, k), the octree
explanation summary of certregion (Proposition A') proposes, per root,
* a DROP (some interface edge has no explaining occluder edge on the box), or
* CLIP recipes: for each occluder edge e that alone explains some interface edges on the box, the two extreme wedge
  planes of its pencil, each named (body, facet i, facet j, l, k, endpoint) -- the plane through line(e) and the
  receiver l (endpoint 0) or k (endpoint 1).
The engine clips with float rows; the verifier (certify.propa) re-proves every DROP and CLIP from exact geometry and
only applies what it proves. Selection of interface edges: all H1-free ones if at most n_edges, else a deterministic
spread (farthest-point order over the edge midpoints, seeded by the task key)."""
import zlib

import numpy as np

from . import certregion as CR
from .local import interface_edges, edge_free_mask


def _fps(P, n, rng):
    first = int(rng.integers(len(P))); idx = [first]; d = np.linalg.norm(P - P[first], axis=1)
    for _ in range(n - 1):
        j = int(np.argmax(d)); idx.append(j); d = np.minimum(d, np.linalg.norm(P - P[j], axis=1))
    return np.array(idx, np.int64)


def plan(scene, cls, o, N, roots, key="", n_edges=256, max_edges=12, m=1e-10):
    """Returns dict(keep (bool per root), drop {root index: (l_flat, k_flat)}, clips {root index: [recipes]},
    rows (float clip rows per kept root, list), seconds)."""
    import time
    t0 = time.perf_counter()
    L = cls.shape[0]
    lit, dark = interface_edges(cls)
    if len(lit) == 0:
        return dict(keep=np.ones(len(roots), bool), drop={}, clips={}, rows={}, seconds=0.0, n_pairs=0)
    rows_exact = [list(r) for r in scene.rows]
    free = edge_free_mask(rows_exact, o, N, lit, dark)
    elig = np.flatnonzero(free)
    rng = np.random.default_rng([20261118, zlib.crc32(key.encode())])
    if len(elig) > n_edges:
        mid = 0.5 * (lit[elig] + dark[elig]).astype(float)
        elig = np.sort(elig[_fps(mid, n_edges, rng)])
    lit, dark = lit[elig], dark[elig]
    Lp = (o + lit + 0.5) / N; Kp = (o + dark + 0.5) / N
    flat = lambda v: int((v[0] * L + v[1]) * L + v[2])
    # occluder edges in the verifier's indexing (exact edge list of certify.scene)
    E = scene.edges
    EA = np.array([[float(v) for v in A] for (_, _, _, A, _) in E]).reshape(-1, 3)
    EB = np.array([[float(v) for v in B] for (_, _, _, _, B) in E]).reshape(-1, 3)
    Fr = np.zeros((len(E), 2, 4))
    for t, (c, i, j, _, _) in enumerate(E):
        for s_, f in enumerate((i, j)):
            r = scene.RM[c, f]; Fr[t, s_] = r / np.linalg.norm(r[:3])
    off, idx = CR.admissible_edges(Fr, Lp, Kp)
    wlo = o / N; whi = (o + L) / N
    expl, _ = CR.explanations_for_roots(EA, EB, Lp, Kp, off, idx, roots, wlo, whi, m=m)
    # the plan only proposes what the verifier's own single-pair whole-box test can prove (certify.propa); the octree
    # summary of certregion only orders the candidates
    from ..certify import propa
    out = np.zeros(4, np.int64)
    pair_ok = np.array([propa.seg_free(Lp[i], Kp[i], scene.C, scene.K, scene.RM, scene.RL, scene.RR2, scene.aabb_lo,
                                       scene.aabb_hi) for i in range(len(Lp))], bool)

    def n_expl(q, i):
        d_, ix, iy, iz = (int(v) for v in roots[q]); h = 1.0 / (1 << d_)
        qlo = np.array([ix * h, iy * h, iz * h]); qhi = qlo + h
        n = propa.explaining_edges(qlo, qhi, Lp[i], Kp[i], scene.E_c, scene.E_i, scene.E_j, scene.E_A, scene.E_B,
                                   scene.E_r, int(scene.nE), scene.RM, scene.RL, scene.RR2, out)
        return n, out[:min(n, len(out))].tolist()

    keep = np.ones(len(roots), bool); drops = {}; clips = {}; rows = {}
    for q in range(len(roots)):
        u = expl[q]
        if np.any(u == -1):
            dropped = False
            for i0 in np.flatnonzero((u == -1) & pair_ok)[:8]:
                if n_expl(q, int(i0))[0] == 0:
                    keep[q] = False; drops[q] = (flat(lit[i0]), flat(dark[i0])); dropped = True
                    break
            if dropped:
                continue
        es, cnt = np.unique(u[u >= 0], return_counts=True)
        rec = []; rr = []
        for e in es[np.argsort(-cnt, kind="stable")][:max_edges]:
            c, fi, fj, _, _ = E[int(e)]
            a, b = EA[e], EB[e]; d = (b - a) / np.linalg.norm(b - a)
            w1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0]); w1 /= np.linalg.norm(w1); w2 = np.cross(d, w1)
            cand = []
            for i in np.flatnonzero((u == e) & pair_ok):
                nn, lst = n_expl(q, int(i))
                if not (nn == 0 or (nn == 1 and lst[0] == int(e))):
                    continue
                wr = CR.wedge_rows(a, b, Lp[i], Kp[i], m)
                if wr is None:
                    continue
                for which, row in enumerate(wr):
                    cand.append((which, int(i), row))
            if not cand:
                continue
            Nn = np.array([cr[2][:3] for cr in cand]); ang = np.arctan2(Nn @ w2, Nn @ w1)
            ref = np.angle(np.mean(np.exp(1j * ang))); rel = np.angle(np.exp(1j * (ang - ref)))
            for jx in sorted({int(np.argmin(rel)), int(np.argmax(rel))}):
                which, i, row = cand[jx]
                rec.append((int(c), int(fi), int(fj), flat(lit[i]), flat(dark[i]), int(which)))
                rr.append(row)
        clips[q] = rec; rows[q] = rr
    return dict(keep=keep, drop=drops, clips=clips, rows=rows, seconds=time.perf_counter() - t0, n_pairs=int(len(lit)))
