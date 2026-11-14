"""Convex polytope clipping and exact (floating-point) volume, numba.

A polytope is stored as
    V  (MV, 3) float64   vertex coordinates (some may be unreferenced)
    Fc (MF, MPV) int32   vertex indices of each face polygon, outward (counter-clockwise seen from outside)
    Fn (MF,) int32       number of vertices of each face
and the scalars nv, nf.

`clip(V, Fc, Fn, nv, nf, a, b)` keeps {x : a.x + b <= 0}.  Planes are normalised inside so that the
on-plane tolerance EPS is a metric distance.  Vertices within EPS of the plane are snapped onto it.
The volume is computed by the divergence theorem (fan triangulation of each face).

This is plain float64 geometry.  It is *not* a certificate by itself: callers that need a rigorous
statement must account for its rounding error separately (see docs/METHOD.md, section "numerics").
"""
import numpy as np
from numba import njit

MV = 4096
MF = 512
MPV = 128
EPS = 1e-13
OK, EMPTY, OVERFLOW = 0, 1, 2


@njit(cache=True)
def box_polytope(lo, hi):
    V = np.zeros((MV, 3))
    Fc = np.zeros((MF, MPV), np.int32)
    Fn = np.zeros(MF, np.int32)
    k = 0
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                V[k, 0] = hi[0] if ix else lo[0]
                V[k, 1] = hi[1] if iy else lo[1]
                V[k, 2] = hi[2] if iz else lo[2]
                k += 1
    # vertex index = 4*ix + 2*iy + iz ; faces listed counter-clockwise seen from outside
    faces = ((0, 1, 3, 2),   # x = lo  (normal -x)
             (4, 6, 7, 5),   # x = hi  (normal +x)
             (0, 4, 5, 1),   # y = lo  (normal -y)
             (2, 3, 7, 6),   # y = hi  (normal +y)
             (0, 2, 6, 4),   # z = lo  (normal -z)
             (1, 5, 7, 3))   # z = hi  (normal +z)
    for f in range(6):
        for j in range(4):
            Fc[f, j] = faces[f][j]
        Fn[f] = 4
    return V, Fc, Fn, 8, 6


@njit(cache=True)
def volume(V, Fc, Fn, nv, nf):
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
def clip(V, Fc, Fn, nv, nf, a0, a1, a2, b):
    """Keep a.x + b <= 0.  Returns (status, nv, nf); arrays are modified in place."""
    if nf == 0:
        return EMPTY, nv, nf
    nrm = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
    if nrm == 0.0:
        if b <= 0.0:
            return OK, nv, nf
        return EMPTY, 0, 0
    a0 /= nrm
    a1 /= nrm
    a2 /= nrm
    b /= nrm
    # classify referenced vertices only (mark referenced)
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
            # snap onto the plane
            V[v, 0] -= dv * a0
            V[v, 1] -= dv * a1
            V[v, 2] -= dv * a2
            dv = 0.0
        d[v] = dv
    if n_out == 0:
        return OK, nv, nf
    if n_in == 0:
        return EMPTY, 0, 0
    # edge cache for intersection vertices
    ek0 = np.empty(MPV * 4, np.int32)
    ek1 = np.empty(MPV * 4, np.int32)
    ekv = np.empty(MPV * 4, np.int32)
    ne = 0
    newF = np.empty((MF, MPV), np.int32)
    newN = np.zeros(MF, np.int32)
    nnf = 0
    cap = np.empty(MPV * 4, np.int32)
    ncap = 0
    for f in range(nf):
        n = Fn[f]
        m = 0
        for j in range(n):
            u = Fc[f, j]
            w = Fc[f, (j + 1) % n]
            du = d[u]
            dw = d[w]
            if du <= 0.0:
                if m >= MPV:
                    return OVERFLOW, nv, nf
                newF[nnf, m] = u
                m += 1
                if du == 0.0:
                    # on-plane vertex belongs to the cap
                    dup = False
                    for q in range(ncap):
                        if cap[q] == u:
                            dup = True
                            break
                    if not dup:
                        if ncap >= cap.shape[0]:
                            return OVERFLOW, nv, nf
                        cap[ncap] = u
                        ncap += 1
            if (du < 0.0 and dw > 0.0) or (du > 0.0 and dw < 0.0):
                k0 = min(u, w)
                k1 = max(u, w)
                vid = -1
                for q in range(ne):
                    if ek0[q] == k0 and ek1[q] == k1:
                        vid = ekv[q]
                        break
                if vid < 0:
                    if nv >= MV or ne >= ek0.shape[0]:
                        return OVERFLOW, nv, nf
                    t = du / (du - dw)
                    V[nv, 0] = V[u, 0] + t * (V[w, 0] - V[u, 0])
                    V[nv, 1] = V[u, 1] + t * (V[w, 1] - V[u, 1])
                    V[nv, 2] = V[u, 2] + t * (V[w, 2] - V[u, 2])
                    # snap exactly onto the plane
                    dd = a0 * V[nv, 0] + a1 * V[nv, 1] + a2 * V[nv, 2] + b
                    V[nv, 0] -= dd * a0
                    V[nv, 1] -= dd * a1
                    V[nv, 2] -= dd * a2
                    vid = nv
                    nv += 1
                    ek0[ne] = k0
                    ek1[ne] = k1
                    ekv[ne] = vid
                    ne += 1
                    if ncap >= cap.shape[0]:
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
    # cap polygon ordered counter-clockwise around +a (outward normal of kept side)
    if ncap >= 3:
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for q in range(ncap):
            cx += V[cap[q], 0]
            cy += V[cap[q], 1]
            cz += V[cap[q], 2]
        cx /= ncap
        cy /= ncap
        cz /= ncap
        # basis u, w with u x w = a
        if abs(a0) < 0.9:
            tx, ty, tz = 1.0, 0.0, 0.0
        else:
            tx, ty, tz = 0.0, 1.0, 0.0
        ux = a1 * tz - a2 * ty
        uy = a2 * tx - a0 * tz
        uz = a0 * ty - a1 * tx
        un = np.sqrt(ux * ux + uy * uy + uz * uz)
        ux /= un
        uy /= un
        uz /= un
        wx = a1 * uz - a2 * uy
        wy = a2 * ux - a0 * uz
        wz = a0 * uy - a1 * ux
        ang = np.empty(ncap)
        for q in range(ncap):
            px = V[cap[q], 0] - cx
            py = V[cap[q], 1] - cy
            pz = V[cap[q], 2] - cz
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
    # compact vertices when the array is getting full
    if nv > MV // 2:
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
        nv = k
    return OK, nv, nf


@njit(cache=True)
def clipped_box_volume(lo, hi, planes, nplanes):
    """Volume of box[lo,hi] intersected with {x : planes[k,:3].x + planes[k,3] <= 0, k < nplanes}.
    Returns (volume, status)."""
    V, Fc, Fn, nv, nf = box_polytope(lo, hi)
    for k in range(nplanes):
        st, nv, nf = clip(V, Fc, Fn, nv, nf, planes[k, 0], planes[k, 1], planes[k, 2], planes[k, 3])
        if st == EMPTY:
            return 0.0, EMPTY
        if st == OVERFLOW:
            return -1.0, OVERFLOW
    return volume(V, Fc, Fn, nv, nf), OK
