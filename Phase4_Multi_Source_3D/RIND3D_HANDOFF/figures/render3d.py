"""Minimal orthographic ray caster for the figures: convex bodies (half-spaces), boxes and voxel cutaways, with a
depth buffer so that overlays (rays, edges, points) are drawn only where visible. Display only (float64)."""
import numpy as np
from matplotlib import colors as mcolors


class Camera:
    def __init__(self, elev=22.0, azim=-58.0, size=(900, 900), center=(0.5, 0.5, 0.5), half_width=0.92):
        e, a = np.radians(elev), np.radians(azim)
        self.to_cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        self.f = -self.to_cam
        r = np.cross(self.f, [0, 0, 1.0]); self.r = r / np.linalg.norm(r); self.u = np.cross(self.r, self.f)
        self.W, self.H = size; self.c = np.asarray(center, float)
        self.s = 2 * half_width / self.W                                         # world units per pixel

    def rays(self):
        j, i = np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)
        P = self.c + ((j - self.W / 2) * self.s)[..., None] * self.r - ((i - self.H / 2) * self.s)[..., None] * self.u
        return (P - 4.0 * self.f).reshape(-1, 3), self.f

    def project(self, X):
        X = np.asarray(X, float).reshape(-1, 3) - self.c
        return np.stack([X @ self.r / self.s + self.W / 2, self.H / 2 - X @ self.u / self.s], 1), (X @ self.f + 4.0)


def halfspaces_from_facets(facets):
    """[(polygon pts, outward normal)] of one body -> (n, d) rows with n.x <= d."""
    return np.array([np.r_[n, np.dot(n, p[0])] for p, n in facets])


def hit_convex(O, v, bodies):
    """Nearest entry of rays O + t v into convex bodies (list of (m, 4) rows n.x <= d). Returns t, normal, body id."""
    n_r = len(O); tbest = np.full(n_r, np.inf); nbest = np.zeros((n_r, 3)); bid = np.full(n_r, -1)
    for b, H in enumerate(bodies):
        N, d = H[:, :3], H[:, 3]
        den = N @ v                                                               # (m,)
        num = d[None, :] - O @ N.T                                                # (n_r, m)
        t0 = np.zeros(n_r); t1 = np.full(n_r, np.inf); ent = np.full(n_r, -1); ok = np.ones(n_r, bool)
        for k in range(len(d)):
            if abs(den[k]) < 1e-15:
                ok &= num[:, k] >= 0; continue
            t = num[:, k] / den[k]
            if den[k] < 0:
                upd = t > t0; t0 = np.where(upd, t, t0); ent = np.where(upd, k, ent)
            else:
                t1 = np.minimum(t1, t)
        hit = ok & (t0 <= t1) & (ent >= 0) & (t0 < tbest)
        tbest[hit] = t0[hit]; nbest[hit] = N[ent[hit]]; bid[hit] = b
    return tbest, nbest, bid


def hit_boxes(O, v, boxes):
    """Nearest entry into axis-aligned boxes [(lo, hi)]: t, normal, box id."""
    n_r = len(O); tbest = np.full(n_r, np.inf); nbest = np.zeros((n_r, 3)); bid = np.full(n_r, -1)
    for b, (lo, hi) in enumerate(boxes):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo - O) / v; tb = (hi - O) / v
        tmin = np.minimum(ta, tb); tmax = np.maximum(ta, tb)
        t0 = tmin.max(1); ax = tmin.argmax(1); t1 = tmax.min(1)
        hit = (t0 <= t1) & (t1 > 0) & (t0 < tbest)
        tbest[hit] = t0[hit]; bid[hit] = b
        nn = np.zeros((hit.sum(), 3)); nn[np.arange(hit.sum()), ax[hit]] = -np.sign(v[ax[hit]])
        nbest[hit] = nn
    return tbest, nbest, bid


def shade(rgb, N, light=(0.35, -0.55, 0.75), amb=0.58):
    l = np.asarray(light, float); l /= np.linalg.norm(l)
    k = amb + (1 - amb) * np.clip(N @ l, 0, 1)
    return np.clip(rgb * k[:, None] + 0.05 * (1 - k)[:, None], 0, 1)


def edges_from_ids(ids, normals, H, W):
    """Outline pixels: neighbours with a different body id or a different face normal."""
    I = ids.reshape(H, W); Nn = normals.reshape(H, W, 3)
    e = np.zeros((H, W), bool)
    for di, dj in ((0, 1), (1, 0)):
        a = I[: H - di, : W - dj]; b = I[di:, dj:]
        na = Nn[: H - di, : W - dj]; nb = Nn[di:, dj:]
        diff = (a != b) | ((a >= 0) & (np.abs(na - nb).sum(-1) > 1e-6))
        e[: H - di, : W - dj] |= diff; e[di:, dj:] |= diff
    return e


def render_bodies(cam, bodies, base, bg=(1, 1, 1), edge=(0.22, 0.27, 0.33), alpha_edge=0.75):
    O, v = cam.rays()
    t, N, bid = hit_convex(O, v, bodies)
    rgb = np.tile(np.asarray(bg, float), (len(O), 1)); hit = bid >= 0
    rgb[hit] = shade(np.asarray(mcolors.to_rgb(base)), N[hit])
    E = edges_from_ids(bid, N, cam.H, cam.W).ravel()
    rgb[E] = (1 - alpha_edge) * rgb[E] + alpha_edge * np.asarray(edge)
    depth = np.where(hit, t, np.inf)
    alpha = hit.astype(float); alpha[E] = np.maximum(alpha[E], alpha_edge)
    return np.dstack([rgb.reshape(cam.H, cam.W, 3), alpha.reshape(cam.H, cam.W)]), depth.reshape(cam.H, cam.W)


def visible_polyline(cam, P0, P1, depth, n=400, eps=2e-3):
    """Pixel polylines of the visible parts of segment P0-P1 under the depth buffer."""
    T = np.linspace(0, 1, n)[:, None]
    X = P0 + T * (P1 - P0)
    uv, z = cam.project(X)
    i = np.clip(uv[:, 1].astype(int), 0, cam.H - 1); j = np.clip(uv[:, 0].astype(int), 0, cam.W - 1)
    vis = z <= depth[i, j] + eps
    out, cur = [], []
    for k in range(n):
        if vis[k]:
            cur.append(uv[k])
        elif cur:
            out.append(np.array(cur)); cur = []
    if cur:
        out.append(np.array(cur))
    return out, uv


def render_cutaway(cam, cls, cut=True, base=None):
    """Window class tensor cls[x, y, z] (0 dark, 1 lit, 2 SOLID) in the unit cube; optional octant cut at the corner
    facing the camera. Returns RGBA image and the hit mask."""
    L = cls.shape[0]
    corner = (cam.to_cam > 0).astype(float)                                       # the cube corner facing the camera
    lo_c = np.where(corner > 0, 0.5, 0.0); hi_c = np.where(corner > 0, 1.0, 0.5)
    boxes = []
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                lo = np.array([bx, by, bz]) * 0.5; hi = lo + 0.5
                if cut and np.allclose(lo, lo_c) and np.allclose(hi, hi_c):
                    continue
                boxes.append((lo, hi))
    O, v = cam.rays()
    t, N, bid = hit_boxes(O, v, boxes)
    hit = bid >= 0
    X = O[hit] + (t[hit] - 1e-9)[:, None] * v + 1e-7 * (-N[hit])                # nudge inside the box
    ijk = np.clip((X * L).astype(int), 0, L - 1)
    lut = np.array([mcolors.to_rgb(c) for c in (base or ("#22344F", "#E8B04B", "#9AA3AE"))])
    col = lut[cls[ijk[:, 0], ijk[:, 1], ijk[:, 2]]]
    rgb = np.ones((len(O), 3)); rgb[hit] = shade(col, N[hit], amb=0.72)
    alpha = hit.astype(float)
    return np.dstack([rgb.reshape(cam.H, cam.W, 3), alpha.reshape(cam.H, cam.W)])


def cube_segments(o, s):
    o = np.asarray(o, float); e = []
    for a in range(3):
        for b in (0, 1):
            for c in (0, 1):
                p = o.copy(); ax2 = [i for i in range(3) if i != a]
                p[ax2[0]] += b * s; p[ax2[1]] += c * s; q = p.copy(); q[a] += s
                e.append((p, q))
    return e


def render_voxels(cam, vals, cmap, norm, n=128, step=0.5, chunk=40000, amb=0.62, mip=True, mode=None):
    """First-hit rendering of the nonzero cells of vals (dict cell -> value on an n^3 grid in [0,1]^3), coloured by
    cmap(norm(value)) and shaded by the entered face. Returns RGBA image and depth."""
    occ = np.zeros((n, n, n), np.float64)
    for c, q in vals.items():
        occ[c // (n * n), (c // n) % n, c % n] = q
    O, v = cam.rays()
    # smooth normals: gradient of a box-blurred occupancy (display only)
    B = (occ > 0).astype(np.float64)
    for ax_ in range(3):
        B = (np.roll(B, 1, ax_) + B + np.roll(B, -1, ax_)) / 3
        B = (np.roll(B, 1, ax_) + B + np.roll(B, -1, ax_)) / 3
    G = np.stack(np.gradient(B), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (0 - O) / v; tb = (1 - O) / v
    t0 = np.maximum(np.minimum(ta, tb).max(1), 0); t1 = np.maximum(ta, tb).min(1)
    rgb = np.ones((len(O), 3)); alpha = np.zeros(len(O)); depth = np.full(len(O), np.inf)
    dt = step / n; ns = int(np.ceil(np.sqrt(3) / dt)) + 2
    for a in range(0, len(O), chunk):
        sl = slice(a, a + chunk); hitp = np.where(t1[sl] > t0[sl])[0]
        if not len(hitp):
            continue
        Os = O[sl][hitp]; ts = t0[sl][hitp]
        T = ts[:, None] + dt * np.arange(ns)[None, :]
        X = Os[:, None, :] + T[..., None] * v
        inside = np.all((X >= 0) & (X < 1), axis=2)
        I = np.clip((X * n).astype(int), 0, n - 1)
        val = np.where(inside, occ[I[..., 0], I[..., 1], I[..., 2]], 0)
        has = val.max(1) > 0
        k = np.argmax(val > 0, axis=1)
        idx = np.where(has)[0]
        if not len(idx):
            continue
        kk = k[idx]
        Ih = I[idx, kk]
        N = -G[Ih[:, 0], Ih[:, 1], Ih[:, 2]]
        nn = np.linalg.norm(N, axis=1); bad = nn < 1e-9
        N[bad] = -v; N /= np.maximum(nn, 1e-9)[:, None]; N[bad] = -v
        if mode == "sum":                                            # column sum along the ray, flat colour
            col = np.asarray(cmap(norm(val[idx].sum(1) * step)))[:, :3]
            N = np.tile(-v, (len(idx), 1))
        else:
            col = np.asarray(cmap(norm(val[idx].max(1) if mip else val[idx, kk])))[:, :3]
        g = a + hitp[idx]
        rgb[g] = shade(col, N, amb=amb); alpha[g] = 1.0; depth[g] = T[idx, kk]
    return np.dstack([rgb.reshape(cam.H, cam.W, 3), alpha.reshape(cam.H, cam.W)]), depth.reshape(cam.H, cam.W)
