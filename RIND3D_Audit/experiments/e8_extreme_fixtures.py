"""E8: exact 3D physics fixtures for mixed, all-lit, all-dark, solid-separated, thin-obstacle and narrow-opening
responses; the tau-pencil CPU engine (rind3d_posterior.label_window) vs an independent brute-force Monte Carlo
feasibility check.

World (0,1)^3 with axis-aligned closed boxes (rows a.x + d <= 0), receiver grid N = 32, window side L = 4 (world side
1/8). The response is rendered by an independent float64 Liang-Barsky segment/box test (closed solid, closed segment).
Box coordinates are non-dyadic so no voxel centre lies on a shadow boundary of the chosen sources (checked: the
minimum slab-test margin is reported).
Checks per fixture:
  * true source cell in the engine's support;
  * vol(F): engine [sum lo, sum hi] vs MC estimate from n uniform candidates (binomial s.e.);
  * every MC-feasible candidate lies in an engine cell with hi > 0 (no false exclusion);
  * TV between the engine label and the MC histogram at 8^3 blocks (MC noise reported by a split-half TV);
  * connected components (26-connectivity) of the engine support at 128^3 and of the MC-feasible 8^3 blocks.
Run one fixture: nice -n 19 python e8_extreme_fixtures.py <fixture> > out/e8_<fixture>.json
"""
import json, sys, time
import numpy as np

sys.path.insert(0, "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Posterior/src")
N, L = 32, 4
ORIGIN = np.array([4, 14, 14])          # window box [0.125, 0.25] x [0.4375, 0.5625]^2


def box(lo, hi):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    return [[-1, 0, 0, lo[0]], [1, 0, 0, -hi[0]], [0, -1, 0, lo[1]], [0, 1, 0, -hi[1]], [0, 0, -1, lo[2]], [0, 0, 1, -hi[2]]]


WALL_X = (0.6013, 0.6237)
HOLES = (((0.2013, 0.3137), (0.4413, 0.5587)), ((0.6863, 0.7987), (0.4413, 0.5587)))


def wall_with_holes():
    """Wall x in WALL_X, y,z in [0.0313, 0.9687] minus two holes (built from 7 boxes)."""
    x0, x1 = WALL_X; a, b = 0.0313, 0.9687
    (h1y, hz), (h2y, _) = HOLES
    bx = [box((x0, a, a), (x1, b, hz[0])), box((x0, a, hz[1]), (x1, b, b)),                      # below / above holes
          box((x0, a, hz[0]), (x1, h1y[0], hz[1])), box((x0, h1y[1], hz[0]), (x1, h2y[0], hz[1])),
          box((x0, h2y[1], hz[0]), (x1, b, hz[1]))]
    return bx


FIXTURES = {
    # name: (bodies, source, expected response type)
    "box_mixed": ([box((0.4113, 0.4137, 0.4171), (0.5913, 0.5137, 0.5871))], (0.8713, 0.5913, 0.5013), "mixed"),
    "box_alllit": ([box((0.4113, 0.4137, 0.4171), (0.5913, 0.5137, 0.5871))], (0.1013, 0.8913, 0.1187), "alllit"),
    "box_alldark": ([box((0.4113, 0.4137, 0.4171), (0.5913, 0.6137, 0.5871))], (0.8713, 0.5113, 0.5013), "alldark"),
    # slab through the window: SOLID voxel layers y = 15.5/32, 16.5/32; lit above, dark below (no lit-dark adjacency)
    "slab_separated": ([box((0.0313, 0.4713, 0.0313), (0.9687, 0.5287, 0.9687))], (0.2013, 0.8113, 0.5013), "separated"),
    # thin plate (half-thickness 2^-14) between window and source
    "thin_plate_mixed": ([box((0.3913, 0.3013, 0.3013), (0.3913 + 2 ** -13, 0.5013, 0.7013))], (0.6813, 0.4513, 0.5113), "mixed"),
    # wall with two holes: mixed through one hole (narrow opening) and all-lit (disconnected support)
    "holes_mixed": (wall_with_holes(), (0.9413, 0.1673, 0.4987), "mixed"),
    "holes_alllit": (wall_with_holes(), (0.3013, 0.5113, 0.8013), "alllit"),
}


def rows_of(bodies):
    K = np.array([len(b) for b in bodies], np.int32); P = np.zeros((len(bodies), int(K.max()), 4)); P[..., 3] = -1
    for i, b in enumerate(bodies):
        P[i, :len(b)] = b
    return P, K


def aabbs(bodies):
    out = []
    for b in bodies:
        r = np.array(b); out.append((np.array([r[0, 3], r[2, 3], r[4, 3]]), np.array([-r[1, 3], -r[3, 3], -r[5, 3]])))
    return out


def seg_box(P, Q, lo, hi):
    """Closed segment P->Q meets closed box [lo, hi]? Returns (hit, margin) with margin = t1 - t0 (signed slack)."""
    d = Q - P; t0 = np.zeros(P.shape[:-1]); t1 = np.ones(P.shape[:-1]); ok = np.ones(P.shape[:-1], bool)
    for a in range(3):
        da = d[..., a]; pa = P[..., a]; par = np.abs(da) < 1e-300
        ok &= ~(par & ((pa < lo[a]) | (pa > hi[a])))
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo[a] - pa) / da; tb = (hi[a] - pa) / da
        t0 = np.maximum(t0, np.where(par, -np.inf, np.minimum(ta, tb))); t1 = np.minimum(t1, np.where(par, np.inf, np.maximum(ta, tb)))
    return ok & (t0 <= t1), t1 - t0


def receivers():
    g = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1).reshape(-1, 3)
    return g, (ORIGIN + g + 0.5) / N


def inside(bodies, X):
    s = np.zeros(len(X), bool)
    for lo, hi in aabbs(bodies):
        s |= np.all((X >= lo) & (X <= hi), 1)
    return s


def render(bodies, s, X):
    lit = np.ones(len(X), bool); margin = np.inf
    for lo, hi in aabbs(bodies):
        hit, m = seg_box(np.repeat(s[None], len(X), 0), X, lo, hi)
        lit &= ~hit; margin = min(margin, float(np.min(np.abs(m))))
    return lit, margin


def main(name, n_mc=400_000, seed=1):
    t0 = time.time()
    bodies, s_true, kind = FIXTURES[name]
    s_true = np.array(s_true)
    g, X = receivers()
    solid = inside(bodies, X)
    litv, margin = render(bodies, s_true, X[~solid])
    lit = np.zeros(len(X), bool); lit[np.flatnonzero(~solid)] = litv
    cls = np.where(solid, 2, np.where(lit, 1, 0)).astype(np.uint8).reshape(L, L, L)     # [x, y, z]
    free = ~solid
    nl, nd = int((lit & free).sum()), int((~lit & free).sum())
    # 6- and 26-adjacency of lit and dark FREE receivers
    c3 = cls
    adj6 = adj26 = False
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                a = c3[max(dx, 0):L + min(dx, 0), max(dy, 0):L + min(dy, 0), max(dz, 0):L + min(dz, 0)]
                b = c3[max(-dx, 0):L + min(-dx, 0), max(-dy, 0):L + min(-dy, 0), max(-dz, 0):L + min(-dz, 0)]
                x = bool(np.any(((a == 0) & (b == 1)) | ((a == 1) & (b == 0))))
                adj26 |= x
                if abs(dx) + abs(dy) + abs(dz) == 1:
                    adj6 |= x
    rtype = "alllit" if nd == 0 else ("alldark" if nl == 0 else ("separated" if not adj6 else "mixed"))
    out = dict(fixture=name, expected=kind, response_type=rtype, n_lit=nl, n_dark=nd, n_solid=int(solid.sum()),
               lit_dark_adjacent_6=adj6, lit_dark_adjacent_26=adj26, min_slab_margin=margin, source=s_true.tolist())
    # ---- engine
    from rind3d_posterior import label_window
    P, K = rows_of(bodies)
    te = time.time()
    r = label_window(P, K, cls, ORIGIN, N, engine="cpu")
    out["engine_seconds"] = time.time() - te
    cells, inv = np.unique(np.asarray(r["cell"], np.int64), return_inverse=True)
    lo = np.bincount(inv, weights=r["lo"], minlength=len(cells)); hi = np.bincount(inv, weights=r["hi"], minlength=len(cells))
    out.update(engine_tv_bound=float(r["tv_bound"]), engine_vol_lo=float(lo.sum()), engine_vol_hi=float(hi.sum()),
               engine_cells=int(len(cells)))
    q = lo / lo.sum() if lo.sum() > 0 else hi / hi.sum()
    G = 128
    ci = np.minimum((s_true * G).astype(np.int64), G - 1); ct = (ci[0] * G + ci[1]) * G + ci[2]
    out["true_cell_in_support"] = bool(np.isin(ct, cells[hi > 0]))
    # ---- brute-force MC feasibility
    rng = np.random.default_rng(seed)
    wlo, whi = ORIGIN / N, (ORIGIN + L) / N
    Xf, rf = X[free], lit[free]
    feas = []
    tm = time.time()
    for _ in range(n_mc // 20000):
        S = rng.random((20000, 3))
        ok = ~inside(bodies, S) & ~np.all((S >= wlo) & (S <= whi), 1)
        idx = np.flatnonzero(ok); Sx = S[idx]
        litm = np.ones((len(Sx), len(Xf)), bool)
        for blo, bhi in aabbs(bodies):
            hit, _ = seg_box(np.repeat(Sx[:, None, :], len(Xf), 1), np.repeat(Xf[None], len(Sx), 0), blo, bhi)
            litm &= ~hit
        match = np.all(litm == rf[None], 1)
        feas.append(Sx[match])
    feas = np.concatenate(feas)
    out["mc_seconds"] = time.time() - tm
    p = len(feas) / n_mc
    out.update(mc_n=n_mc, mc_vol=p, mc_vol_se=float(np.sqrt(p * (1 - p) / n_mc)))
    fi = np.minimum((feas * G).astype(np.int64), G - 1); fc = (fi[:, 0] * G + fi[:, 1]) * G + fi[:, 2]
    supp = set(cells[hi > 0].tolist())
    out["mc_feasible_outside_engine_support"] = int(sum(c not in supp for c in fc.tolist()))
    # 8^3 block comparison
    B = 8; f = G // B

    def blocks(cidx, w):
        cx, cy, cz = cidx // (G * G), (cidx // G) % G, cidx % G
        b = (cx // f * B + cy // f) * B + cz // f
        return np.bincount(b, weights=w, minlength=B ** 3)

    qe = blocks(cells, q); qe /= qe.sum()
    if len(fc):
        qm = blocks(fc, np.ones(len(fc))); qm /= qm.sum()
        h = len(fc) // 2
        q1 = blocks(fc[:h], np.ones(h)); q2 = blocks(fc[h:], np.ones(len(fc) - h))
        out["tv_engine_vs_mc_8cubed"] = float(0.5 * np.abs(qe - qm).sum())
        out["tv_mc_splithalf_8cubed"] = float(0.5 * np.abs(q1 / q1.sum() - q2 / q2.sum()).sum())
    # connected components
    from scipy.ndimage import label
    occ = np.zeros(G ** 3, bool); occ[cells[hi > 0]] = True
    _, ncomp = label(occ.reshape(G, G, G), structure=np.ones((3, 3, 3)))
    out["engine_support_components_26"] = int(ncomp)
    if len(fc):
        ob = np.zeros(B ** 3, bool); ob[np.unique(blocks(fc, np.ones(len(fc))).nonzero()[0])] = True
        _, nb = label(ob.reshape(B, B, B), structure=np.ones((3, 3, 3)))
        out["mc_feasible_block_components_26"] = int(nb)
    # where is the posterior mass relative to the box/wall: share of label mass with x > wall/box max x
    xs = (cells // (G * G) + 0.5) / G
    xmax = max(hi_[0] for _, hi_ in aabbs(bodies))
    out["label_mass_beyond_obstacle_x"] = float(q[xs > xmax].sum())
    out["seconds"] = time.time() - t0
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], n_mc=int(sys.argv[2]) if len(sys.argv) > 2 else 400_000)
