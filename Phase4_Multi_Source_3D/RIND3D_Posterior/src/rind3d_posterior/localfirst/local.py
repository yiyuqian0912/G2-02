"""X-only geometry of a window's direct response interface (no world geometry is read here).

Conventions: cls[x, y, z] with 0 response-inactive FREE (legacy "dark"), 1 response-active FREE ("lit"), 2 SOLID;
window coordinates in voxels, receiver v at v + 1/2; world coordinates (o + v + 1/2) / N.

Everything in this module is a heuristic *description* of the observation, used to choose receivers and to report
diagnostics. The band of `local_band_cells` is not an outer bound of the feasible set: it relies on hypotheses H1-H2
of LOCAL_INVERSE_GEOMETRY.md that the observation cannot certify."""
import numpy as np

DARK, LIT, SOLID = 0, 1, 2


def interface_edges(cls):
    """Direct FREE-FREE interface edges: (lit voxel (n,3), dark voxel (n,3)) with |lit - dark|_1 = 1."""
    cls = np.asarray(cls)
    L_, D_ = [], []
    for ax in range(3):
        a = [slice(None)] * 3; b = [slice(None)] * 3
        a[ax] = slice(0, -1); b[ax] = slice(1, None)
        ca, cb = cls[tuple(a)], cls[tuple(b)]
        m = (ca != SOLID) & (cb != SOLID) & (ca != cb)
        u = np.argwhere(m); v = u.copy(); v[:, ax] += 1
        lit_u = ca[m] == LIT
        L_.append(np.where(lit_u[:, None], u, v)); D_.append(np.where(lit_u[:, None], v, u))
    return np.concatenate(L_).astype(np.int64), np.concatenate(D_).astype(np.int64)


def _fit(P):
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T) if len(P) > 3 else np.eye(3))
    return c, V[:, 0], w


def plane_components(lit, dark, max_planes=8, min_pts=24, tol=0.75, iters=256, max_pts=20000, seed=0):
    """Greedy RANSAC peeling of the oriented interface edges into planar components.
    A plane (n, d) explains edge (l, k) iff it separates it up to tol voxels: n.l - d >= -tol and n.k - d <= tol
    (n oriented from the inactive to the active side). Returns a list of dicts (centroid, normal, n, extent, rms,
    idx = indices of the edges explained), largest first. Heuristic: the decomposition affects cost, never safety."""
    rng = np.random.default_rng(seed)
    l = lit.astype(float) + 0.5; k = dark.astype(float) + 0.5
    mid = 0.5 * (l + k); g = l - k
    alive = np.arange(len(mid))
    if len(alive) > max_pts:
        alive = np.sort(rng.choice(alive, max_pts, replace=False))
    comps = []
    while len(alive) >= min_pts and len(comps) < max_planes:
        M = mid[alive]; best = None
        for _ in range(iters):
            t = rng.choice(len(alive), 3, replace=False)
            n = np.cross(M[t[1]] - M[t[0]], M[t[2]] - M[t[0]]); nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            if np.sum(g[alive] @ n) < 0:
                n = -n
            d = n @ M[t[0]]
            inl = ((l[alive] @ n - d) >= -tol) & ((k[alive] @ n - d) <= tol)
            c = int(inl.sum())
            if best is None or c > best[0]:
                best = (c, n, d, inl)
        if best is None or best[0] < min_pts:
            break
        inl = best[3]
        for _ in range(3):                                         # least-squares refinement on the inliers
            c0, n, w = _fit(M[inl])
            if np.sum(g[alive][inl] @ n) < 0:
                n = -n
            d = n @ c0
            inl2 = ((l[alive] @ n - d) >= -tol) & ((k[alive] @ n - d) <= tol)
            if inl2.sum() < min_pts:
                break
            inl = inl2
        c0, n, w = _fit(M[inl])
        if np.sum(g[alive][inl] @ n) < 0:
            n = -n
        P = M[inl] - c0
        u = np.linalg.svd(P, full_matrices=False)[2] if len(P) >= 3 else np.eye(3)
        ext = np.sqrt(max(np.var(P @ u[1]), 1e-12)) * 2.0                  # in-plane minor spread (voxels)
        comps.append(dict(centroid=c0, normal=n, n=int(inl.sum()), extent=float(ext),
                          major=float(np.sqrt(max(np.var(P @ u[0]), 1e-12)) * 2.0),
                          rms=float(np.sqrt(np.mean((P @ n) ** 2))), idx=alive[inl]))
        alive = alive[~inl]
    return comps


def conditioning(comps, min_pts=24):
    """Singular values of the stacked unit normals of the components with >= min_pts edges (zeros if fewer)."""
    N = np.array([c["normal"] for c in comps if c["n"] >= min_pts]).reshape(-1, 3)
    s = np.linalg.svd(N, compute_uv=False) if len(N) else np.zeros(0)
    s = np.concatenate([s, np.zeros(3 - len(s))])[:3]
    return dict(sigma=s.tolist(), rank=int(np.sum(s > 0.05)), n_planes=int(len(N)))


def local_band_cells(comps, origin, N, grid=128, min_pts=24, kappa=1.0):
    """HEURISTIC local candidate region on the global grid (bool [grid^3], cell = (ix*G+iy)*G+iz): cells meeting
    every component's band |n.(s - m)| <= tau(s), tau = kappa h sqrt(3)/2 (1 + 2 |s - m| / extent), assuming each
    component is one exact shadow plane through the source (H1-H2). Not an outer bound in general."""
    G = grid; h = 1.0 / N; cs = 1.0 / G
    ax = (np.arange(G) + 0.5) * cs
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    S = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    keep = np.ones(len(S), bool)
    half = cs * np.sqrt(3) / 2
    o = np.asarray(origin, float)
    for c in comps:
        if c["n"] < min_pts:
            continue
        m = (o + c["centroid"]) * h; n = c["normal"]
        dist = np.linalg.norm(S - m, axis=1)
        tau = kappa * h * np.sqrt(3) / 2 * (1 + 2 * dist / max(c["extent"] * h, h))
        keep &= np.abs((S - m) @ n) <= tau + half
    return keep


def select_receivers(cls, n_boundary=256, n_interior=128, seed=0, comps=None, per_plane=None):
    """Receiver subset X' (voxel coords (n,3) [x,y,z]) chosen from the local interface geometry: for each planar
    component, edges spread by farthest-point sampling over the component (both endpoints), then edges outside every
    component, then random interior FREE receivers. Same budget as FastLabel's random boundary choice."""
    rng = np.random.default_rng(seed)
    lit, dark = interface_edges(cls)
    if comps is None:
        comps = plane_components(lit, dark, seed=seed)
    mid = 0.5 * (lit + dark).astype(float)
    picks = []
    used = np.zeros(len(lit), bool)
    budget_edges = n_boundary // 2
    big = [c for c in comps if c["n"] >= 24]
    if per_plane is None:
        per_plane = max(4, budget_edges // max(1, len(big) + 1))
    for c in big:
        idx = c["idx"]; P = mid[idx]
        first = int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))
        chosen = [first]; dmin = np.linalg.norm(P - P[first], axis=1)
        for _ in range(min(per_plane, len(idx)) - 1):
            j = int(np.argmax(dmin)); chosen.append(j); dmin = np.minimum(dmin, np.linalg.norm(P - P[j], axis=1))
        sel = idx[chosen]; used[sel] = True; picks.append(sel)
    rest = np.flatnonzero(~used)
    n_left = budget_edges - sum(len(p) for p in picks)
    if n_left > 0 and len(rest):
        picks.append(rng.choice(rest, min(n_left, len(rest)), replace=False))
    e = np.concatenate(picks) if picks else np.zeros(0, np.int64)
    vox = np.concatenate([lit[e], dark[e]]) if len(e) else np.zeros((0, 3), np.int64)
    vox = np.unique(vox, axis=0)
    free = np.argwhere(np.asarray(cls) != SOLID)
    if n_interior and len(free):
        vox = np.concatenate([vox, free[rng.choice(len(free), min(n_interior, len(free)), replace=False)]])
    return np.unique(vox, axis=0), comps
