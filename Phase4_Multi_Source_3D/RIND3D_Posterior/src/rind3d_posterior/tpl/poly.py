"""Compact convex polytopes for region-based (BSP) source-space search (numba).

Same algorithm as the validated rpl/polytope.py clipper (tested against qhull to 4e-14), but array capacities are
taken from the array shapes, so many small polytopes can live in preallocated stacks.

A polytope:  V (MV,3) vertices (some unreferenced),  Fc (MF,MPV) int32 face vertex cycles (CCW from outside),
Fn (MF,) face sizes, scalars nv, nf.  clip() keeps {x : a.x + b <= 0}; vertices within EPS (metric) of the
plane are snapped onto it.
"""
import numpy as np
from numba import njit

EPS = 1e-13
OK, EMPTY, OVERFLOW = 0, 1, 2


@njit(cache=True)
def set_box(V, Fc, Fn, lo, hi):
    k = 0
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                V[k, 0] = hi[0] if ix else lo[0]
                V[k, 1] = hi[1] if iy else lo[1]
                V[k, 2] = hi[2] if iz else lo[2]
                k += 1
    F = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    for f in range(6):
        for j in range(4):
            Fc[f, j] = F[f][j]
        Fn[f] = 4
    return 8, 6


@njit(cache=True)
def volume(V, Fc, Fn, nf):
    if nf == 0:
        return 0.0
    x0 = V[Fc[0, 0]]
    vol = 0.0
    for f in range(nf):
        n = Fn[f]
        p0 = V[Fc[f, 0]]
        for j in range(1, n - 1):
            p1 = V[Fc[f, j]]
            p2 = V[Fc[f, j + 1]]
            ax, ay, az = p0[0] - x0[0], p0[1] - x0[1], p0[2] - x0[2]
            bx, by, bz = p1[0] - x0[0], p1[1] - x0[1], p1[2] - x0[2]
            cx, cy, cz = p2[0] - x0[0], p2[1] - x0[1], p2[2] - x0[2]
            vol += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    return vol / 6.0


@njit(cache=True)
def referenced(V, Fc, Fn, nv, nf, out):
    """Copy the referenced vertices into out; returns their number."""
    used = np.zeros(nv, np.bool_)
    for f in range(nf):
        for j in range(Fn[f]):
            used[Fc[f, j]] = True
    k = 0
    for v in range(nv):
        if used[v]:
            out[k, 0] = V[v, 0]; out[k, 1] = V[v, 1]; out[k, 2] = V[v, 2]
            k += 1
    return k


@njit(cache=True)
def copy_poly(V, Fc, Fn, nv, nf, V2, Fc2, Fn2):
    for v in range(nv):
        V2[v, 0] = V[v, 0]; V2[v, 1] = V[v, 1]; V2[v, 2] = V[v, 2]
    for f in range(nf):
        Fn2[f] = Fn[f]
        for j in range(Fn[f]):
            Fc2[f, j] = Fc[f, j]


@njit(cache=True)
def compact(V, Fc, Fn, nv, nf):
    remap = -np.ones(nv, np.int32)
    k = 0
    V2 = np.empty((nv, 3))
    for f in range(nf):
        for j in range(Fn[f]):
            v = Fc[f, j]
            if remap[v] < 0:
                remap[v] = k
                V2[k] = V[v]
                k += 1
            Fc[f, j] = remap[v]
    for v in range(k):
        V[v] = V2[v]
    return k


@njit(cache=True)
def clip(V, Fc, Fn, nv, nf, a0, a1, a2, b):
    """Keep a.x + b <= 0. Returns (status, nv, nf); arrays modified in place."""
    MV = V.shape[0]; MF = Fc.shape[0]; MPV = Fc.shape[1]
    if nf == 0:
        return EMPTY, 0, 0
    nrm = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    if not (nrm > 0.0) or not np.isfinite(nrm):
        if nrm == 0.0 and b <= 0.0:
            return OK, nv, nf
        if nrm == 0.0:
            return EMPTY, 0, 0
        return OVERFLOW, nv, nf
    a0 /= nrm; a1 /= nrm; a2 /= nrm; b /= nrm
    used = np.zeros(nv, np.bool_)
    for f in range(nf):
        for j in range(Fn[f]):
            used[Fc[f, j]] = True
    d = np.zeros(nv)
    n_out = 0
    n_in = 0
    for v in range(nv):
        if not used[v]:
            continue
        dv = a0 * V[v, 0] + a1 * V[v, 1] + a2 * V[v, 2] + b
        if dv > EPS:
            n_out += 1
        elif dv < -EPS:
            n_in += 1
        else:
            V[v, 0] -= dv * a0; V[v, 1] -= dv * a1; V[v, 2] -= dv * a2
            dv = 0.0
        d[v] = dv
    if n_out == 0:
        return OK, nv, nf
    if n_in == 0:
        return EMPTY, 0, 0
    ncapmax = 4 * MPV
    ek0 = np.empty(ncapmax, np.int32); ek1 = np.empty(ncapmax, np.int32); ekv = np.empty(ncapmax, np.int32)
    ne = 0
    newF = np.empty((MF, MPV), np.int32)
    newN = np.zeros(MF, np.int32)
    nnf = 0
    cap = np.empty(ncapmax, np.int32)
    ncap = 0
    for f in range(nf):
        n = Fn[f]
        m = 0
        for j in range(n):
            u = Fc[f, j]
            w = Fc[f, (j + 1) % n]
            du = d[u]; dw = d[w]
            if du <= 0.0:
                if m >= MPV:
                    return OVERFLOW, nv, nf
                newF[nnf, m] = u
                m += 1
                if du == 0.0:
                    dup = False
                    for q in range(ncap):
                        if cap[q] == u:
                            dup = True
                            break
                    if not dup:
                        if ncap >= ncapmax:
                            return OVERFLOW, nv, nf
                        cap[ncap] = u
                        ncap += 1
            if (du < 0.0 and dw > 0.0) or (du > 0.0 and dw < 0.0):
                k0 = min(u, w); k1 = max(u, w)
                vid = -1
                for q in range(ne):
                    if ek0[q] == k0 and ek1[q] == k1:
                        vid = ekv[q]
                        break
                if vid < 0:
                    if nv >= MV or ne >= ncapmax:
                        return OVERFLOW, nv, nf
                    t = du / (du - dw)
                    V[nv, 0] = V[u, 0] + t * (V[w, 0] - V[u, 0])
                    V[nv, 1] = V[u, 1] + t * (V[w, 1] - V[u, 1])
                    V[nv, 2] = V[u, 2] + t * (V[w, 2] - V[u, 2])
                    dd = a0 * V[nv, 0] + a1 * V[nv, 1] + a2 * V[nv, 2] + b
                    V[nv, 0] -= dd * a0; V[nv, 1] -= dd * a1; V[nv, 2] -= dd * a2
                    vid = nv
                    nv += 1
                    ek0[ne] = k0; ek1[ne] = k1; ekv[ne] = vid
                    ne += 1
                    if ncap >= ncapmax:
                        return OVERFLOW, nv, nf
                    cap[ncap] = vid
                    ncap += 1
                if m >= MPV:
                    return OVERFLOW, nv, nf
                newF[nnf, m] = vid
                m += 1
        if m >= 3:
            newN[nnf] = m
            nnf += 1
            if nnf >= MF - 1:
                return OVERFLOW, nv, nf
    if ncap >= 3:
        cx = 0.0; cy = 0.0; cz = 0.0
        for q in range(ncap):
            cx += V[cap[q], 0]; cy += V[cap[q], 1]; cz += V[cap[q], 2]
        cx /= ncap; cy /= ncap; cz /= ncap
        if abs(a0) < 0.9:
            tx, ty, tz = 1.0, 0.0, 0.0
        else:
            tx, ty, tz = 0.0, 1.0, 0.0
        ux = a1 * tz - a2 * ty; uy = a2 * tx - a0 * tz; uz = a0 * ty - a1 * tx
        un = np.sqrt(ux * ux + uy * uy + uz * uz)
        ux /= un; uy /= un; uz /= un
        wx = a1 * uz - a2 * uy; wy = a2 * ux - a0 * uz; wz = a0 * uy - a1 * ux
        ang = np.empty(ncap)
        for q in range(ncap):
            px = V[cap[q], 0] - cx; py = V[cap[q], 1] - cy; pz = V[cap[q], 2] - cz
            ang[q] = np.arctan2(px * wx + py * wy + pz * wz, px * ux + py * uy + pz * uz)
        order = np.argsort(ang)
        if ncap > MPV:
            return OVERFLOW, nv, nf
        for q in range(ncap):
            newF[nnf, q] = cap[order[q]]
        newN[nnf] = ncap
        nnf += 1
    for f in range(nnf):
        Fn[f] = newN[f]
        for j in range(newN[f]):
            Fc[f, j] = newF[f, j]
    nf = nnf
    if nf < 4:
        return EMPTY, 0, 0
    if nv > MV // 2:
        nv = compact(V, Fc, Fn, nv, nf)
    return OK, nv, nf


@njit(cache=True)
def sample_uniform(V, Fc, Fn, nf, n, rng_u, out):
    """n exactly uniform points in the polytope (fan tetrahedra from vertex 0 of face 0).
    rng_u: (n, 4) uniforms in [0,1). Returns the number written (n)."""
    x0 = V[Fc[0, 0]]
    # tetrahedra (x0, p0, p1, p2) over all face fans
    nt = 0
    for f in range(nf):
        nt += max(Fn[f] - 2, 0)
    T = np.empty((nt, 3), np.int32)
    W = np.empty(nt)
    k = 0
    tot = 0.0
    for f in range(nf):
        for j in range(1, Fn[f] - 1):
            a = Fc[f, 0]; b = Fc[f, j]; c = Fc[f, j + 1]
            ax, ay, az = V[a, 0] - x0[0], V[a, 1] - x0[1], V[a, 2] - x0[2]
            bx, by, bz = V[b, 0] - x0[0], V[b, 1] - x0[1], V[b, 2] - x0[2]
            cx, cy, cz = V[c, 0] - x0[0], V[c, 1] - x0[1], V[c, 2] - x0[2]
            w = (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6.0
            T[k, 0] = a; T[k, 1] = b; T[k, 2] = c
            W[k] = max(w, 0.0)
            tot += W[k]
            k += 1
    cdf = np.cumsum(W) / tot
    for i in range(n):
        u = rng_u[i, 0]
        t = np.searchsorted(cdf, u)
        if t >= nt:
            t = nt - 1
        # uniform in tetra via sorted-sum trick on 3 uniforms
        s = rng_u[i, 1]; tt = rng_u[i, 2]; uu = rng_u[i, 3]
        if s + tt > 1.0:
            s = 1.0 - s; tt = 1.0 - tt
        if tt + uu > 1.0:
            tmp = uu; uu = 1.0 - s - tt; tt = 1.0 - tmp
        elif s + tt + uu > 1.0:
            tmp = uu; uu = s + tt + uu - 1.0; s = 1.0 - tt - tmp
        a0 = 1.0 - s - tt - uu
        for q in range(3):
            out[i, q] = (a0 * x0[q] + s * V[T[t, 0], q] + tt * V[T[t, 1], q] + uu * V[T[t, 2], q])
    return n
