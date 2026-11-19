"""Exact plane-based convex polytopes for the certificate verifier (numba).

Representation (plane-based geometry: Sugihara and Iri 1989; Bernstein and Fussell 2009; Campen and Kobbelt 2010).
* A global plane table PL[id] = (a0, a1, a2, b): binary64 numbers read as EXACT rationals. A signed reference
  ref = 2 id + neg denotes the closed half-space  sigma (a.x + b) <= 0,  sigma = -1 if neg else +1.
* A polytope is the intersection of its faces' half-spaces. Each face has a plane reference and a cycle of vertices
  (counter-clockwise seen from outside); Fe[f, j] is the plane reference of the neighbouring face across the edge
  from Fv[f, j] to Fv[f, j+1]. A vertex is the common point of three face planes (Vp, unsigned ids); its coordinates
  are never used for combinatorial decisions, only an enclosure (Vm +- Vr, componentwise) of the exact rational point.
* Every sign "plane h at vertex (p, q, r)" is EXACT: sign(det4(p, q, r, h)) * sign(det3(p, q, r)) with the filters
  and expansions of ar.py, under Simulation of Simplicity (Edelsbrunner and Muecke 1990): plane offsets are perturbed
  b_id -> b_id + eps_id with eps_0 >> eps_1 >> ... (ids in creation order), so no vertex ever lies on a plane that
  does not define it. The perturbed polytopes are simple; their volumes and their vertex sets tend to the unperturbed
  ones as eps -> 0, so every volume and every max/min of an affine function computed here is that of the exact
  polytope. Two children {h <= 0}, {h >= 0} of a split tile their parent exactly (same perturbation of h).
No snapping, no tolerance: the clipping combinatorics is exact; only vertex coordinates are enclosed."""
import numpy as np
from numba import njit

from .ar import (U, TINY, INFL, G3, G4, G5, dn, up, det3_float, det3_exp, det4_rows_sign, det3_sign, exp_approx)

MV = 256        # vertices per polytope
MF = 96         # faces per polytope
MPV = 64        # vertices per face
OK, EMPTY, OVERFLOW, NOOP = 0, 1, 2, 3


def new_slots(S):
    return dict(Fp=np.zeros((S, MF), np.int64), Fn=np.zeros((S, MF), np.int32), Fv=np.zeros((S, MF, MPV), np.int32),
                Fe=np.zeros((S, MF, MPV), np.int64), Vp=np.zeros((S, MV, 3), np.int64), Vm=np.zeros((S, MV, 3)),
                Vr=np.zeros((S, MV, 3)), nV=np.zeros(S, np.int64), nF=np.zeros(S, np.int64))


# ------------------------------------------------------------------------------------------- vertex enclosure
@njit(cache=True)
def vertex_enclosure(PL, p, q, r, out_m, out_r):
    """Enclosure of the common point of planes p, q, r (Cramer's rule, rigorous). Determinants are evaluated in
    binary64 with an a priori bound and, whenever that bound is not small relative to the result (ill-conditioned
    triples, e.g. nearly parallel pencil planes), EXACTLY by expansions: the enclosure radius is then ~1 ulp of the
    coordinates independently of the conditioning. Returns False if the triple is singular."""
    a = PL[p]; b = PL[q]; c = PL[r]
    d0, e0 = det3_float(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2])
    if not (e0 <= 2.0 ** -45 * abs(d0)):
        ex = np.empty(32)
        n = det3_exp(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], ex)
        if ex[n - 1] == 0.0:
            return False
        d0, e0 = exp_approx(ex, n)
        if not (abs(d0) > e0):
            return False
    m = np.empty((3, 3))
    for k in range(3):
        # numerator: column k replaced by -b
        for i in range(3):
            R = a if i == 0 else (b if i == 1 else c)
            for j in range(3):
                m[i, j] = R[j]
            m[i, k] = -R[3]
        dk, ek = det3_float(m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1], m[2, 2])
        if not (ek <= 2.0 ** -52 * abs(d0)):          # coordinate error ek/|d0| above ~2^-52: evaluate exactly
            ex = np.empty(32)
            n = det3_exp(m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1], m[2, 2], ex)
            dk, ek = exp_approx(ex, n)
        mid = dk / d0
        den = abs(d0) - e0
        rad = (ek + abs(mid) * e0) / den
        rad = up(up(rad * INFL) + up(abs(mid) * U)) + TINY
        out_m[k] = mid
        out_r[k] = rad
    return True


@njit(cache=True)
def _affine_at(PL, h, vm, vr):
    """(mid, rad) of plane h's affine function at the enclosed point vm +- vr (rigorous)."""
    a = PL[h]
    s = a[0] * vm[0] + a[1] * vm[1] + a[2] * vm[2] + a[3]
    ab = abs(a[0] * vm[0]) + abs(a[1] * vm[1]) + abs(a[2] * vm[2]) + abs(a[3])
    # three products and three additions: gamma_4 (the former gamma_3 is violated by ~1e-3 of random 4-term sums
    # as an enclosure; it was nevertheless sufficient for sos_sign, whose decisions only need the bound near zero)
    rad = (abs(a[0]) * vr[0] + abs(a[1]) * vr[1] + abs(a[2]) * vr[2]) * INFL + G4 * ab * INFL + TINY
    return s, rad


@njit(cache=True)
def sos_sign(PL, h, p, q, r, vm, vr):
    """Exact sign (+1/-1) of plane h (unsigned id, not in {p, q, r}) at vertex (p, q, r), under SoS."""
    s, rad = _affine_at(PL, h, vm, vr)
    if s > rad:
        return 1
    if s < -rad:
        return -1
    A = PL[p]; B = PL[q]; C = PL[r]; H = PL[h]
    s3 = det3_sign(A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2])
    s4 = det4_rows_sign(A, B, C, H)
    if s4 != 0:
        return s4 * s3
    # SoS: smallest id among rows with a nonzero cofactor of its offset entry decides
    ids = np.array([p, q, r, h])
    order = np.argsort(ids)
    for t in range(4):
        i = order[t]
        if i == 3:
            cs = s3
        elif i == 0:
            cs = -det3_sign(B[0], B[1], B[2], C[0], C[1], C[2], H[0], H[1], H[2])
        elif i == 1:
            cs = det3_sign(A[0], A[1], A[2], C[0], C[1], C[2], H[0], H[1], H[2])
        else:
            cs = -det3_sign(A[0], A[1], A[2], B[0], B[1], B[2], H[0], H[1], H[2])
        if cs != 0:
            return cs * s3
    return 1        # unreachable: the cofactor of h's row is det3(p, q, r) != 0


@njit(cache=True, inline="always")
def ref_id(ref):
    return ref >> 1


@njit(cache=True, inline="always")
def ref_sigma(ref):
    return -1 if (ref & 1) else 1


# ------------------------------------------------------------------------------------------- construction
@njit(cache=True)
def set_box(PL, s, axis_ids, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF):
    """Slot s := box given by six axis planes: axis_ids[k, 0] = plane id of x_k = lo_k, axis_ids[k, 1] of x_k = hi_k
    (both stored as (e_k, -value)); faces {x_k >= lo_k} = ref(id_lo, neg), {x_k <= hi_k} = ref(id_hi, pos)."""
    # vertices: corner (ix, iy, iz) -> index ix + 2 iy + 4 iz
    for iz in range(2):
        for iy in range(2):
            for ix in range(2):
                v = ix + 2 * iy + 4 * iz
                Vp[s, v, 0] = axis_ids[0, ix]; Vp[s, v, 1] = axis_ids[1, iy]; Vp[s, v, 2] = axis_ids[2, iz]
                for k, ii in ((0, ix), (1, iy), (2, iz)):
                    Vm[s, v, k] = -PL[axis_ids[k, ii], 3]
                    Vr[s, v, k] = 0.0
    # faces with outward CCW cycles; refs: lo face of axis k: {-(x_k - lo) <= 0} -> neg; hi face: pos
    L0 = 2 * axis_ids[0, 0] + 1; H0 = 2 * axis_ids[0, 1]
    L1 = 2 * axis_ids[1, 0] + 1; H1 = 2 * axis_ids[1, 1]
    L2 = 2 * axis_ids[2, 0] + 1; H2 = 2 * axis_ids[2, 1]
    # face order: x-lo, x-hi, y-lo, y-hi, z-lo, z-hi
    cyc = ((0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4), (2, 6, 7, 3), (0, 2, 3, 1), (4, 5, 7, 6))
    refs = (L0, H0, L1, H1, L2, H2)
    for f in range(6):
        Fp[s, f] = refs[f]
        Fn[s, f] = 4
        for j in range(4):
            Fv[s, f, j] = cyc[f][j]
    # neighbours: edge (u -> w) of face f is shared with the face containing both u, w other than f
    for f in range(6):
        for j in range(4):
            u = cyc[f][j]; w = cyc[f][(j + 1) % 4]
            for g in range(6):
                if g == f:
                    continue
                cu = False; cw = False
                for t in range(4):
                    if cyc[g][t] == u:
                        cu = True
                    if cyc[g][t] == w:
                        cw = True
                if cu and cw:
                    Fe[s, f, j] = refs[g]
    nV[s] = 8
    nF[s] = 6


@njit(cache=True)
def copy_slot(a, b, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF):
    nf = nF[a]; nv = nV[a]
    for f in range(nf):
        Fp[b, f] = Fp[a, f]; n = Fn[a, f]; Fn[b, f] = n
        for j in range(n):
            Fv[b, f, j] = Fv[a, f, j]; Fe[b, f, j] = Fe[a, f, j]
    for v in range(nv):
        for k in range(3):
            Vp[b, v, k] = Vp[a, v, k]; Vm[b, v, k] = Vm[a, v, k]; Vr[b, v, k] = Vr[a, v, k]
    nV[b] = nv; nF[b] = nf


@njit(cache=True)
def clip(PL, s, h, Fp, Fn, Fv, Fe, Vp, Vm, Vr, nV, nF):
    """Slot s := slot s n {half-space h} (signed ref). Returns OK, NOOP (h already a face or redundant), EMPTY or
    OVERFLOW (slot unchanged)."""
    nf = nF[s]; nv = nV[s]
    if nf == 0:
        nV[s] = 0
        return EMPTY
    hid = h >> 1; hs = -1 if (h & 1) else 1
    for f in range(nf):
        if (Fp[s, f] >> 1) == hid:
            if Fp[s, f] == h:
                return NOOP
            nF[s] = 0; nV[s] = 0          # {h <= 0} n {h >= 0} is the face plane: no volume
            return EMPTY
    sg = np.empty(nv, np.int64)
    npos = 0; nneg = 0
    for v in range(nv):
        x = sos_sign(PL, hid, Vp[s, v, 0], Vp[s, v, 1], Vp[s, v, 2], Vm[s, v], Vr[s, v]) * hs
        sg[v] = x
        if x > 0:
            npos += 1
        else:
            nneg += 1
    if npos == 0:
        return NOOP
    if nneg == 0:
        nF[s] = 0; nV[s] = 0
        return EMPTY
    # new vertex per cut edge, keyed by the unordered pair of face plane refs
    MAXE = 2 * MF
    ek0 = np.empty(MAXE, np.int64); ek1 = np.empty(MAXE, np.int64); ekv = np.empty(MAXE, np.int64)
    ne = 0
    # remap kept vertices
    remap = -np.ones(nv, np.int64)
    nVm = np.empty((MV, 3)); nVr = np.empty((MV, 3)); nVp = np.empty((MV, 3), np.int64)
    k = 0
    for v in range(nv):
        if sg[v] < 0:
            remap[v] = k
            for t in range(3):
                nVm[k, t] = Vm[s, v, t]; nVr[k, t] = Vr[s, v, t]; nVp[k, t] = Vp[s, v, t]
            k += 1
    nnv = k
    newFp = np.empty(MF, np.int64); newFn = np.zeros(MF, np.int32)
    newFv = np.empty((MF, MPV), np.int32); newFe = np.empty((MF, MPV), np.int64)
    capS = np.empty(MF, np.int64); capE = np.empty(MF, np.int64); capN = np.empty(MF, np.int64)
    ncap = 0
    nnf = 0
    tmpm = np.empty(3); tmpr = np.empty(3)
    for f in range(nf):
        n = Fn[s, f]
        F = Fp[s, f]
        m = 0
        xin = -1; xout = -1
        for j in range(n):
            u = Fv[s, f, j]; w = Fv[s, f, (j + 1) % n]; G = Fe[s, f, j]
            su = sg[u]; sw = sg[w]
            x = -1
            if su != sw:
                a0 = min(F, G); a1 = max(F, G)
                for t in range(ne):
                    if ek0[t] == a0 and ek1[t] == a1:
                        x = ekv[t]
                        break
                if x < 0:
                    if nnv >= MV or ne >= MAXE:
                        return OVERFLOW
                    if not vertex_enclosure(PL, F >> 1, G >> 1, hid, tmpm, tmpr):
                        return OVERFLOW
                    for t in range(3):
                        nVm[nnv, t] = tmpm[t]; nVr[nnv, t] = tmpr[t]
                    nVp[nnv, 0] = F >> 1; nVp[nnv, 1] = G >> 1; nVp[nnv, 2] = hid
                    x = nnv; nnv += 1
                    ek0[ne] = a0; ek1[ne] = a1; ekv[ne] = x; ne += 1
            if su < 0:
                if m >= MPV:
                    return OVERFLOW
                newFv[nnf, m] = remap[u]; newFe[nnf, m] = G; m += 1
                if sw > 0:
                    if m >= MPV:
                        return OVERFLOW
                    newFv[nnf, m] = x; newFe[nnf, m] = h; m += 1
                    xout = x
            else:
                if sw < 0:
                    if m >= MPV:
                        return OVERFLOW
                    newFv[nnf, m] = x; newFe[nnf, m] = G; m += 1
                    xin = x
        if m == 0:
            continue
        if nnf >= MF - 1:
            return OVERFLOW
        newFp[nnf] = F; newFn[nnf] = m
        nnf += 1
        if xout >= 0:
            # the cap traverses this face's cap edge (xout -> xin on face F) in reverse: xin -> xout, neighbour F
            if ncap >= MF:
                return OVERFLOW
            capS[ncap] = xin; capE[ncap] = xout; capN[ncap] = F; ncap += 1
    # cap face cycle
    if ncap < 3 or ncap > MPV:
        return OVERFLOW
    used = np.zeros(ncap, np.bool_)
    cur = 0
    used[0] = True
    newFp[nnf] = h
    newFv[nnf, 0] = capS[0]; newFe[nnf, 0] = capN[0]
    m = 1
    end = capE[0]
    while m < ncap:
        nxt = -1
        for t in range(ncap):
            if not used[t] and capS[t] == end:
                nxt = t
                break
        if nxt < 0:
            return OVERFLOW         # inconsistent cap (cannot happen with exact signs)
        used[nxt] = True
        newFv[nnf, m] = capS[nxt]; newFe[nnf, m] = capN[nxt]; m += 1
        end = capE[nxt]
    if end != capS[0]:
        return OVERFLOW
    newFn[nnf] = m
    nnf += 1
    # commit
    for f in range(nnf):
        Fp[s, f] = newFp[f]; Fn[s, f] = newFn[f]
        for j in range(newFn[f]):
            Fv[s, f, j] = newFv[f, j]; Fe[s, f, j] = newFe[f, j]
    for v in range(nnv):
        for t in range(3):
            Vm[s, v, t] = nVm[v, t]; Vr[s, v, t] = nVr[v, t]; Vp[s, v, t] = nVp[v, t]
    nF[s] = nnf; nV[s] = nnv
    return OK


# ------------------------------------------------------------------------------------------- volume
@njit(cache=True)
def _det3_mr(am, ar, bm, br, cm, cr):
    """(mid, rad) enclosure of det[a; b; c] for mid-radius vectors (rigorous)."""
    mid = (am[0] * (bm[1] * cm[2] - bm[2] * cm[1]) - am[1] * (bm[0] * cm[2] - bm[2] * cm[0])
           + am[2] * (bm[0] * cm[1] - bm[1] * cm[0]))
    rad = 0.0; ab = 0.0
    # six terms sigma(i, j, k): a_i b_j c_k
    for (i, j, kk) in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        A = abs(am[i]); B = abs(bm[j]); C = abs(cm[kk])
        rad += ar[i] * (B + br[j]) * (C + cr[kk]) + A * br[j] * (C + cr[kk]) + A * B * cr[kk]
        ab += A * B * C
    rad = (rad + G5 * ab) * INFL + TINY
    return mid, rad


@njit(cache=True)
def volume(s, Fn, Fv, Vm, Vr, nF):
    """Rigorous enclosure [lo, hi] of the exact polytope volume (fan tetrahedra from vertex 0, divergence formula)."""
    nf = nF[s]
    if nf == 0:
        return 0.0, 0.0
    x0m = Vm[s, Fv[s, 0, 0]]; x0r = Vr[s, Fv[s, 0, 0]]
    am = np.empty(3); ar = np.empty(3); bm = np.empty(3); br = np.empty(3); cm = np.empty(3); cr = np.empty(3)
    tlo = 0.0; thi = 0.0
    for f in range(nf):
        n = Fn[s, f]
        p0 = Fv[s, f, 0]
        for t in range(3):
            d = Vm[s, p0, t] - x0m[t]
            am[t] = d; ar[t] = (Vr[s, p0, t] + x0r[t]) * INFL + U * abs(d) * INFL + TINY
        for j in range(1, n - 1):
            p1 = Fv[s, f, j]; p2 = Fv[s, f, j + 1]
            for t in range(3):
                d = Vm[s, p1, t] - x0m[t]
                bm[t] = d; br[t] = (Vr[s, p1, t] + x0r[t]) * INFL + U * abs(d) * INFL + TINY
                d = Vm[s, p2, t] - x0m[t]
                cm[t] = d; cr[t] = (Vr[s, p2, t] + x0r[t]) * INFL + U * abs(d) * INFL + TINY
            dm, dr = _det3_mr(am, ar, bm, br, cm, cr)
            # interval accumulation with outward rounding: each fan determinant lies in [dm - dr, dm + dr]
            tlo = dn(tlo + dn(dm - dr)); thi = up(thi + up(dm + dr))
    # (the former mid-radius sum covered the rounding of the radius sum only by INFL = 1 + 64u, i.e. up to ~64 fan
    # triangles; directed accumulation needs no a priori count)
    lo = dn(tlo / 6.0); hi = up(thi / 6.0)
    return max(lo, 0.0), hi


@njit(cache=True)
def affine_range(s, a0, a1, a2, b, r_coef, Vm, Vr, nV):
    """Rigorous [lo, hi] of the affine function a.x + b over the exact polytope in slot s (max/min at vertices).
    r_coef bounds |coefficients - (a0, a1, a2, b)| componentwise as (r_a (scalar for all three), r_b) packed
    in a 2-vector: the function enclosed is any one within those radii (interval coefficients)."""
    lo = np.inf; hi = -np.inf
    for v in range(nV[s]):
        vm = Vm[s, v]; vr = Vr[s, v]
        m = a0 * vm[0] + a1 * vm[1] + a2 * vm[2] + b
        ab = abs(a0 * vm[0]) + abs(a1 * vm[1]) + abs(a2 * vm[2]) + abs(b)
        rad = ((abs(a0) + r_coef[0]) * vr[0] + (abs(a1) + r_coef[0]) * vr[1] + (abs(a2) + r_coef[0]) * vr[2]
               + r_coef[0] * (abs(vm[0]) + abs(vm[1]) + abs(vm[2])) + r_coef[1]) * INFL + G4 * ab * INFL + TINY
        if m - rad < lo:
            lo = m - rad
        if m + rad > hi:
            hi = m + rad
    return dn(lo), up(hi)
