"""TPE v3: tau-pencil engine with receiver-side (window-side) aspect classes.

Differences to v2 (code/tpl/engine_v2.py, kept unchanged):
* Lemma 2W (docs/METHOD.md): for a receiver x seen from a fixed receiver-side aspect of C (front faces F(x),
  silhouette edges E(x)), and a source s with f_i(s) < 0 for every i in F(x) (s behind C as seen from x),
      x in S_C(s)  <=>  coord_e(s) <= coord_e(x) for every e in E(x),   coord_e(p) = -f_b(p) / f_f(p);
  if f_i(s) > 0 for some i in F(x), x is never in S_C(s).
  The receiver-side aspect depends only on the receiver.  The window's receivers are partitioned once into
  *classes* by the face planes of C that cross the window (bit j of the class = sign of crossing plane j); items
  straddling a crossing plane are pre-split (code/tpl/run3.py), so every item has one class per component.
  Source regions then need splitting only on the front faces of the receivers' classes (the boundary of
  G = {s : f_i(s) <= 0, i in F}), never on the other face planes of C.
* Components with a receiver within 1e-12 of a crossing plane (exact tangency) or with more than 6 crossing planes
  use the v2 source-side form (face splits) instead.
* Logical tau bounds are keyed by (edge, orientation).
Everything else (stage 1, collapses, BSP, emission, records, budgets) is as in v2.
"""
import numpy as np
from numba import njit

from rpl.predicates import plane_box_range, hull_misses_component, AMARG
from .poly import set_box, volume, referenced
from .engine_v2 import (stage1, item_box, fval, aspect_front, sil_edges, tau_range, prep_comp, eval_comp2,
                        _receiver_limited, _bbox, _copy_slot, _clip_slot, _cell_range, emit_region,
                        COVER, NOCOVER, PARTIAL, AMBG, DARK, LIT, IN_REC, UNDET_REC, CELL_DEPTH, STAT_NAMES,
                        S_NODES, S_OUT, S_IN, S_UNDET, S_FACE, S_TAU, S_WIN, S_GEO, S_EVALS, S_MAXE, S_PASS,
                        S_CLIP, S_OVF, S_AMB, S_BUDGET, S_CELLS, S_CELLS_OUT, S_CELLS_IN, S_COMP, S_EXP)

XUNK = 4


@njit(cache=True)
def _class_status(p, c, kap, K, cfront, Sasp, fmn, fmx, cst, cxf):
    """Gating status of the region for receiver class kap of component c: 0 inside G, 1 outside, 2 crossing."""
    gx = -1
    for i in range(K[c]):
        if not cfront[c, kap, i]:
            continue
        a = Sasp[p, c, i]
        if a == 2:
            continue
        if a == 1:
            cst[c, kap] = 1
            return 1
        if fmn[c, i] > 1e-11:
            cst[c, kap] = 1
            return 1
        if fmx[c, i] >= -1e-11 and gx < 0:
            gx = i
    if gx >= 0:
        cst[c, kap] = 2
        cxf[c, kap] = gx
        return 2
    cst[c, kap] = 0
    return 0


@njit(cache=True)
def _item_class(qlo, qhi, P, c, nxp, xp):
    kap = 0
    for j in range(nxp[c]):
        i = xp[c, j]
        mn, mx = plane_box_range(P[c, i, 0], P[c, i, 1], P[c, i, 2], P[c, i, 3], qlo, qhi)
        if mn > AMARG:
            kap |= 1 << j
        elif mx >= -AMARG:
            return -1
    return kap


@njit(cache=True)
def stage2(P, K, clo, chi, adj, aoff, o, N, L, items_code, items_cls, roots, nroots,
           max_nodes_cell, max_nodes_total, max_depth, max_pass, tau_first, dual, rho,
           wmode, nxp, xp, cvalid, cfront, cnsil, csilf, csilb, ckey, cnuf, cuf, csilfi, csilbi,
           rec_cell, rec_lo, rec_hi, rec_kind, rec_voff, rec_vn, vpool, vtop, stats, cell_nodes, cell_evals,
           face_by_comp, face_surv):
    C = P.shape[0]
    kmax = P.shape[1]
    KC = cvalid.shape[1]
    emax = csilf.shape[2]
    etot = adj.shape[0]
    NSCR = 64
    S = 2 * max_depth + 8 + NSCR
    SCR = S - NSCR
    MV = 160; MF = 72; MPV = 48
    SV = np.empty((S, MV, 3)); SFc = np.empty((S, MF, MPV), np.int32); SFn = np.empty((S, MF), np.int32)
    Snv = np.zeros(S, np.int64); Snf = np.zeros(S, np.int64)
    Sasp = np.zeros((S, C, kmax), np.int8)
    SLb = np.empty((S, max(2 * etot, 1))); SUb = np.empty((S, max(2 * etot, 1)))
    Sdep = np.zeros(S, np.int64)
    Soff = np.zeros(S, np.int64); Slen = np.zeros(S, np.int64)
    NE = 1 << 24
    Pc = np.empty(NE, np.int64); Pk = np.empty(NE, np.int8); Pm = np.empty(NE, np.int64); Pa = np.empty(NE, np.int64)
    NQ = 1 << 23
    Wc = np.empty(NQ, np.int64); Wk = np.empty(NQ, np.int8); Wm = np.empty(NQ, np.int64); Wa = np.empty(NQ, np.int64)
    Qc = np.empty(NQ, np.int64); Qk = np.empty(NQ, np.int8); Qm = np.empty(NQ, np.int64); Qa = np.empty(NQ, np.int64)
    RING = 64
    ring = np.full(RING, -1, np.int64)
    nring = 0
    inring = np.zeros(max(items_code.shape[0], 1), np.bool_)
    Vr = np.empty((MV, 3))
    # source-side (S-mode) region context, as v2
    front = np.zeros((C, kmax), np.bool_)
    nsil = np.zeros(C, np.int64)
    silf = np.zeros((C, emax), np.int64); silb = np.zeros((C, emax), np.int64); silt = np.zeros((C, emax), np.int64)
    uf = np.zeros((C, kmax), np.int64); nuf = np.zeros(C, np.int64); nfrc = np.zeros(C, np.int64)
    silfi = np.zeros((C, emax), np.int64); silbi = np.zeros((C, emax), np.int64)
    skey = np.zeros((C, emax), np.int64)
    # per (component, class) ranges / accumulators; S-mode components use class 0
    tlo = np.zeros((C, KC, emax)); thi = np.zeros((C, KC, emax))
    Unew = np.zeros((C, KC, emax)); Lnew = np.zeros((C, KC, emax))
    cst = np.full((C, KC), -1, np.int64); cxf = np.zeros((C, KC), np.int64)
    fmn = np.zeros((C, kmax)); fmx = np.zeros((C, kmax))
    tmn = np.zeros((C, emax)); tmx = np.zeros((C, emax))
    ekap = np.zeros(C, np.int64)
    xface = np.zeros(C, np.int64)
    fbuf = np.zeros((kmax, 8))
    qlo = np.empty(3); qhi = np.empty(3)
    Rlo = np.empty(3); Rhi = np.empty(3)
    wlo = np.empty(3); whi = np.empty(3)
    Wqlo = np.empty(3); Wqhi = np.empty(3)
    for k in range(3):
        wlo[k] = o[k] / N; whi[k] = (o[k] + L) / N
        Wqlo[k] = (o[k] + 0.5) / N; Wqhi[k] = (o[k] + L - 0.5) / N
    allmask = (np.int64(1) << C) - 1
    nitems = items_code.shape[0]
    nrec = 0
    i0 = np.empty(3, np.int64); i1 = np.empty(3, np.int64)
    G = 1 << CELL_DEPTH
    for ci in range(nroots):
        rd = roots[ci, 0]; ix = roots[ci, 1]; iy = roots[ci, 2]; iz = roots[ci, 3]
        stats[S_CELLS] += 1
        rec_start = nrec
        ev0 = stats[S_EVALS]
        hh = 1.0 / (1 << rd)
        qlo[0] = ix * hh; qlo[1] = iy * hh; qlo[2] = iz * hh
        qhi[0] = qlo[0] + hh; qhi[1] = qlo[1] + hh; qhi[2] = qlo[2] + hh
        nv, nf = set_box(SV[0], SFc[0], SFn[0], qlo, qhi)
        Snv[0] = nv; Snf[0] = nf
        Sasp[0, :, :] = 0
        SLb[0, :] = -np.inf; SUb[0, :] = np.inf
        Sdep[0] = 0
        npush = 0
        for t in range(nring):
            a = ring[t]
            inring[a] = True
            Pc[npush] = items_code[a]; Pk[npush] = items_cls[a]; Pm[npush] = allmask; Pa[npush] = a
            npush += 1
        for t in range(nitems):
            if inring[t]:
                continue
            Pc[npush] = items_code[t]; Pk[npush] = items_cls[t]; Pm[npush] = allmask; Pa[npush] = t
            npush += 1
        for t in range(nring):
            inring[ring[t]] = False
        Soff[0] = 0; Slen[0] = nitems
        ptop = nitems
        sp = 1
        nodes_cell = 0
        while sp > 0:
            sp -= 1
            p = sp
            if Snf[p] == 0:
                ptop = Soff[p]
                continue
            if stats[S_NODES] >= max_nodes_total:
                nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                   rec_kind, nrec, rec_voff, rec_vn, vpool, vtop, P, K, clo, chi, 0.0)
                stats[S_UNDET] += 1
                stats[S_BUDGET] += 1
                ptop = Soff[p]
                continue
            stats[S_NODES] += 1
            nodes_cell += 1
            ne = Slen[p]
            off = Soff[p]
            for t in range(ne):
                Wc[t] = Pc[off + t]; Wk[t] = Pk[off + t]; Wm[t] = Pm[off + t]; Wa[t] = Pa[off + t]
            ptop = off
            nr = referenced(SV[p], SFc[p], SFn[p], Snv[p], Snf[p], Vr)
            _bbox(Vr, nr, Rlo, Rhi)
            split_kind = -1
            sa0 = 0.0; sa1 = 0.0; sa2 = 0.0; sb = 0.0
            sc = -1; si = -1; skap = 0; st_t = -1; sv = 0.0
            # ---- window box
            inside = True
            disjoint = False
            for k in range(3):
                if Rlo[k] < wlo[k] - 1e-12 or Rhi[k] > whi[k] + 1e-12:
                    inside = False
                if Rhi[k] <= wlo[k] + 1e-12 or Rlo[k] >= whi[k] - 1e-12:
                    disjoint = True
            if inside:
                stats[S_OUT] += 1
                continue
            if not disjoint:
                for k in range(3):
                    if split_kind >= 0:
                        break
                    for side in range(2):
                        v = wlo[k] if side == 0 else whi[k]
                        if Rlo[k] < v - 1e-12 and Rhi[k] > v + 1e-12:
                            split_kind = 0
                            sa0 = 1.0 if k == 0 else 0.0
                            sa1 = 1.0 if k == 1 else 0.0
                            sa2 = 1.0 if k == 2 else 0.0
                            sb = -v
                            break
            relmask = np.int64(0)
            Xmask = np.int64(0)
            out = False
            ns = 0
            if split_kind < 0:
                for c in range(C):
                    if not hull_misses_component(Rlo, Rhi, Wqlo, Wqhi, P, K, c, clo, chi):
                        relmask |= np.int64(1) << c
                for c in range(C):
                    if not (relmask >> c) & 1:
                        continue
                    if wmode[c]:
                        for i in range(K[c]):
                            mn = np.inf; mx = -np.inf
                            for v in range(nr):
                                fv = fval(P, c, i, Vr[v, 0], Vr[v, 1], Vr[v, 2])
                                if fv < mn:
                                    mn = fv
                                if fv > mx:
                                    mx = fv
                            fmn[c, i] = mn; fmx[c, i] = mx
                        for kap in range(KC):
                            cst[c, kap] = -1
                        continue
                    cross, nfr = aspect_front(Vr, nr, P, K, c, Sasp[p, c], front[c])
                    if cross >= 0:
                        Xmask |= np.int64(1) << c
                        xface[c] = cross
                        continue
                    if nfr == 0:
                        out = True
                        break
            if out:
                stats[S_OUT] += 1
                continue
            if split_kind < 0:
                for c in range(C):
                    if not (relmask >> c) & 1 or wmode[c] or (Xmask >> c) & 1:
                        continue
                    nsil[c] = sil_edges(adj, aoff, c, front[c], silf[c], silb[c], silt[c])
                    nuf[c], nfrc[c] = prep_comp(c, front[c], K, nsil[c], silf[c], silb[c], uf[c], silfi[c], silbi[c])
                    for k in range(nsil[c]):
                        t = silt[c, k]
                        skey[c, k] = 2 * t + (0 if silf[c, k] == adj[t, 1] else 1)
                        tlo[c, 0, k], thi[c, 0, k] = tau_range(Vr, nr, P, c, silf[c, k], silb[c, k],
                                                               SLb[p, skey[c, k]], SUb[p, skey[c, k]], 1.0)
                xneed = -1
                xneed_face = -1
                for it in range(max_pass):
                    stats[S_PASS] += 1
                    for c in range(C):
                        if (relmask >> c) & 1:
                            for kap in range(KC):
                                for k in range(emax):
                                    Unew[c, kap, k] = np.inf; Lnew[c, kap, k] = -np.inf
                    nq = ne
                    ns = 0
                    i = 0
                    overflow = False
                    xneed = -1
                    while i < nq:
                        code = Wc[i]; kls = Wk[i]; mask = Wm[i] & relmask; anc = Wa[i]
                        i += 1
                        stats[S_EVALS] += 1
                        lvl = item_box(code, o, N, qlo, qhi)
                        ispt = lvl == 0
                        keep = False
                        newmask = np.int64(0)
                        xunk = False
                        rexp = not dual
                        xc = -1
                        xcf = -1
                        ncand = 0
                        uc = -1
                        amb = False
                        cover = False
                        for c in range(C):
                            if not (mask >> c) & 1:
                                continue
                            stats[S_COMP] += 1
                            # ---- status of item w.r.t. component c
                            st = AMBG; nund = 0; kund = -1; kap = 0
                            if wmode[c]:
                                kap = _item_class(qlo, qhi, P, c, nxp, xp)
                                if kap < 0 or not cvalid[c, kap]:
                                    st = AMBG
                                else:
                                    s_ = cst[c, kap]
                                    if s_ < 0:
                                        s_ = _class_status(p, c, kap, K, cfront, Sasp, fmn, fmx, cst, cxf)
                                        if s_ == 0:
                                            for k in range(cnsil[c, kap]):
                                                key = ckey[c, kap, k]
                                                tlo[c, kap, k], thi[c, kap, k] = tau_range(
                                                    Vr, nr, P, c, csilf[c, kap, k], csilb[c, kap, k],
                                                    SLb[p, key], SUb[p, key], -1.0)
                                    if s_ == 1:
                                        st = NOCOVER
                                    elif s_ == 2:
                                        st = XUNK
                                    else:
                                        st, nund, kund = eval_comp2(qlo, qhi, ispt, P, c, cnuf[c, kap], 0, cuf[c, kap],
                                                                    cnsil[c, kap], csilfi[c, kap], csilbi[c, kap],
                                                                    tlo[c, kap], thi[c, kap], tmn[c], tmx[c], fbuf, -1.0)
                            elif (Xmask >> c) & 1:
                                st = XUNK
                            else:
                                st, nund, kund = eval_comp2(qlo, qhi, ispt, P, c, nuf[c], nfrc[c], uf[c], nsil[c],
                                                            silfi[c], silbi[c], tlo[c, 0], thi[c, 0], tmn[c], tmx[c],
                                                            fbuf, 1.0)
                            if st == XUNK:
                                if hull_misses_component(Rlo, Rhi, qlo, qhi, P, K, c, clo, chi):
                                    st = NOCOVER
                            ekap[c] = kap
                            # ---- combine
                            if kls == DARK:
                                if st == COVER:
                                    cover = True
                                    break
                                if st == NOCOVER:
                                    continue
                                if st == XUNK:
                                    xunk = True
                                    amb = True
                                    if xc < 0:
                                        xc = c
                                        xcf = cxf[c, kap] if wmode[c] else xface[c]
                                elif st == AMBG:
                                    amb = True
                                    rexp = True
                                    stats[S_AMB] += 1
                                elif not ispt and not rexp:
                                    rexp = _receiver_limited(cnsil[c, kap] if wmode[c] else nsil[c],
                                                             tlo[c, kap], thi[c, kap], tmn[c], tmx[c], rho)
                                newmask |= np.int64(1) << c
                                ncand += 1
                                uc = c
                            else:
                                if st == COVER:
                                    out = True
                                    break
                                if st == NOCOVER:
                                    continue
                                if st == PARTIAL and nund == 1:
                                    if tmx[c, kund] > Lnew[c, kap, kund]:
                                        Lnew[c, kap, kund] = tmx[c, kund]
                                    continue
                                if st == XUNK:
                                    xunk = True
                                    if xc < 0:
                                        xc = c
                                        xcf = cxf[c, kap] if wmode[c] else xface[c]
                                elif st == AMBG:
                                    rexp = True
                                    stats[S_AMB] += 1
                                elif not ispt and not rexp:
                                    rexp = _receiver_limited(cnsil[c, kap] if wmode[c] else nsil[c],
                                                             tlo[c, kap], thi[c, kap], tmn[c], tmx[c], rho)
                                newmask |= np.int64(1) << c
                        if out:
                            break
                        if kls == DARK:
                            if cover:
                                continue
                            if ncand == 0:
                                out = True
                                break
                            if ncand == 1 and not amb:
                                kp = ekap[uc]
                                nk = cnsil[uc, kp] if wmode[uc] else nsil[uc]
                                for k in range(nk):
                                    if tmn[uc, k] < thi[uc, kp, k] and tmn[uc, k] < Unew[uc, kp, k]:
                                        Unew[uc, kp, k] = tmn[uc, k]
                                continue
                            keep = True
                        else:
                            if newmask == 0:
                                continue
                            keep = True
                        if keep:
                            if xc >= 0 and xneed < 0:
                                xneed = xc
                                xneed_face = xcf
                            if not ispt and not xunk and rexp:
                                if nq + 8 > NQ:
                                    overflow = True
                                    break
                                stats[S_EXP] += 1
                                l1 = lvl - 1
                                bx = (code >> 32) & 0xFFFF; by = (code >> 16) & 0xFFFF; bz = code & 0xFFFF
                                for c8 in range(8):
                                    Wc[nq] = (np.int64(l1) << 48) | (np.int64(2 * bx + (c8 & 1)) << 32) | \
                                             (np.int64(2 * by + ((c8 >> 1) & 1)) << 16) | np.int64(2 * bz + ((c8 >> 2) & 1))
                                    Wk[nq] = kls
                                    Wm[nq] = newmask
                                    Wa[nq] = anc
                                    nq += 1
                            else:
                                if ns >= NQ:
                                    overflow = True
                                    break
                                Qc[ns] = code; Qk[ns] = kls; Qm[ns] = newmask; Qa[ns] = anc
                                ns += 1
                    if nq > stats[S_MAXE]:
                        stats[S_MAXE] = nq
                    if out:
                        pos = -1
                        for t in range(nring):
                            if ring[t] == anc:
                                pos = t
                                break
                        if pos < 0:
                            if nring < RING:
                                nring += 1
                            pos = nring - 1
                        for t in range(pos, 0, -1):
                            ring[t] = ring[t - 1]
                        ring[0] = anc
                        break
                    if overflow:
                        stats[S_OVF] += 1
                        break
                    # ---- apply collapsed half-spaces: coord(s) <= u  <=>  f_b(s) - (m u) f_f(s) <= 0
                    nclip = 0
                    for c in range(C):
                        if out or overflow:
                            break
                        if not (relmask >> c) & 1:
                            continue
                        for kap in range(KC if wmode[c] else 1):
                            if out or overflow:
                                break
                            if wmode[c]:
                                if not cvalid[c, kap] or cst[c, kap] != 0:
                                    continue
                                nk = cnsil[c, kap]; m = -1.0
                            else:
                                if (Xmask >> c) & 1:
                                    continue
                                nk = nsil[c]; m = 1.0
                            for k in range(nk):
                                if wmode[c]:
                                    f = csilf[c, kap, k]; b = csilb[c, kap, k]; key = ckey[c, kap, k]
                                else:
                                    f = silf[c, k]; b = silb[c, k]; key = skey[c, k]
                                u = Unew[c, kap, k]
                                if u < thi[c, kap, k]:
                                    if u < tlo[c, kap, k]:
                                        out = True
                                        break
                                    mu = m * u
                                    r = _clip_slot(p, P[c, b, 0] - mu * P[c, f, 0], P[c, b, 1] - mu * P[c, f, 1],
                                                   P[c, b, 2] - mu * P[c, f, 2], P[c, b, 3] - mu * P[c, f, 3],
                                                   SV, SFc, SFn, Snv, Snf)
                                    stats[S_CLIP] += 1
                                    nclip += 1
                                    if r == 0:
                                        out = True
                                        break
                                    if r == 2:
                                        overflow = True
                                        break
                                    if u < SUb[p, key]:
                                        SUb[p, key] = u
                                lv = Lnew[c, kap, k]
                                if lv > tlo[c, kap, k]:
                                    if lv > thi[c, kap, k]:
                                        out = True
                                        break
                                    ml = m * lv
                                    r = _clip_slot(p, ml * P[c, f, 0] - P[c, b, 0], ml * P[c, f, 1] - P[c, b, 1],
                                                   ml * P[c, f, 2] - P[c, b, 2], ml * P[c, f, 3] - P[c, b, 3],
                                                   SV, SFc, SFn, Snv, Snf)
                                    stats[S_CLIP] += 1
                                    nclip += 1
                                    if r == 0:
                                        out = True
                                        break
                                    if r == 2:
                                        overflow = True
                                        break
                                    if lv > SLb[p, key]:
                                        SLb[p, key] = lv
                    if out or overflow:
                        break
                    if nclip == 0:
                        break
                    nr = referenced(SV[p], SFc[p], SFn[p], Snv[p], Snf[p], Vr)
                    _bbox(Vr, nr, Rlo, Rhi)
                    for c in range(C):
                        if not (relmask >> c) & 1:
                            continue
                        if wmode[c]:
                            for ii in range(K[c]):
                                mn = np.inf; mx = -np.inf
                                for v in range(nr):
                                    fv = fval(P, c, ii, Vr[v, 0], Vr[v, 1], Vr[v, 2])
                                    if fv < mn:
                                        mn = fv
                                    if fv > mx:
                                        mx = fv
                                fmn[c, ii] = mn; fmx[c, ii] = mx
                            for kap in range(KC):
                                cst[c, kap] = -1
                        elif not (Xmask >> c) & 1:
                            for k in range(nsil[c]):
                                tlo[c, 0, k], thi[c, 0, k] = tau_range(Vr, nr, P, c, silf[c, k], silb[c, k],
                                                                       SLb[p, skey[c, k]], SUb[p, skey[c, k]], 1.0)
                    for t in range(ns):
                        Wc[t] = Qc[t]; Wk[t] = Qk[t]; Wm[t] = Qm[t]; Wa[t] = Qa[t]
                    ne = ns
                if out:
                    stats[S_OUT] += 1
                    continue
                if overflow:
                    nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                       rec_kind, nrec, rec_voff, rec_vn, vpool, vtop, P, K, clo, chi, 0.0)
                    stats[S_UNDET] += 1
                    continue
                if ns == 0:
                    nrec = emit_region(p, IN_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                       rec_kind, nrec, rec_voff, rec_vn, vpool, vtop, P, K, clo, chi, 0.0)
                    stats[S_IN] += 1
                    continue
                # ---- a survivor needs a component whose gating/aspect is not fixed on R: split on that face
                if xneed >= 0:
                    split_kind = 1
                    sc = xneed; si = xneed_face
                    sa0 = P[sc, si, 0]; sa1 = P[sc, si, 1]; sa2 = P[sc, si, 2]; sb = P[sc, si, 3]
                elif Sdep[p] >= max_depth or nodes_cell >= max_nodes_cell:
                    nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                       rec_kind, nrec, rec_voff, rec_vn, vpool, vtop, P, K, clo, chi, 0.0)
                    stats[S_UNDET] += 1
                    stats[S_BUDGET] += 1
                    continue
                if split_kind != 1:
                    best = 2.0
                    for q in range(ns):
                        code = Qc[q]; mask = Qm[q]
                        if (code >> 48) & 0xFFFF:
                            continue
                        item_box(code, o, N, qlo, qhi)
                        for c in range(C):
                            if not (mask >> c) & 1:
                                continue
                            if wmode[c]:
                                kap = _item_class(qlo, qhi, P, c, nxp, xp)
                                if kap < 0 or not cvalid[c, kap] or cst[c, kap] != 0:
                                    continue
                                st, nund, kund = eval_comp2(qlo, qhi, True, P, c, cnuf[c, kap], 0, cuf[c, kap],
                                                            cnsil[c, kap], csilfi[c, kap], csilbi[c, kap],
                                                            tlo[c, kap], thi[c, kap], tmn[c], tmx[c], fbuf, -1.0)
                                nk = cnsil[c, kap]
                            else:
                                if (Xmask >> c) & 1:
                                    continue
                                kap = 0
                                st, nund, kund = eval_comp2(qlo, qhi, True, P, c, nuf[c], nfrc[c], uf[c], nsil[c],
                                                            silfi[c], silbi[c], tlo[c, 0], thi[c, 0], tmn[c], tmx[c],
                                                            fbuf, 1.0)
                                nk = nsil[c]
                            if st != PARTIAL:
                                continue
                            for k in range(nk):
                                v = tmn[c, k]
                                lo_ = tlo[c, kap, k]; hi_ = thi[c, kap, k]
                                if not (v < hi_ and lo_ < v):
                                    continue
                                if np.isfinite(lo_) and np.isfinite(hi_) and hi_ > lo_:
                                    score = abs((v - lo_) / (hi_ - lo_) - 0.5)
                                else:
                                    score = 0.45
                                if score < best:
                                    best = score
                                    sc = c; skap = kap; st_t = k; sv = v
                    _cell_range(Vr, nr, i0, i1)
                    gax = -1
                    gext = 0
                    for k in range(3):
                        if i1[k] - i0[k] > gext:
                            gext = i1[k] - i0[k]
                            gax = k
                    use_tau = sc >= 0 and (tau_first or gax < 0)
                    if use_tau:
                        split_kind = 2
                        if wmode[sc]:
                            f = csilf[sc, skap, st_t]; b = csilb[sc, skap, st_t]; msv = -sv
                        else:
                            f = silf[sc, st_t]; b = silb[sc, st_t]; msv = sv
                        sa0 = P[sc, b, 0] - msv * P[sc, f, 0]; sa1 = P[sc, b, 1] - msv * P[sc, f, 1]
                        sa2 = P[sc, b, 2] - msv * P[sc, f, 2]; sb = P[sc, b, 3] - msv * P[sc, f, 3]
                    elif gax >= 0:
                        mid = 0.5 * (Rlo[gax] + Rhi[gax])
                        j = int(np.floor(mid * G + 0.5))
                        if j < i0[gax] + 1:
                            j = i0[gax] + 1
                        if j > i1[gax]:
                            j = i1[gax]
                        split_kind = 3
                        sa0 = 1.0 if gax == 0 else 0.0
                        sa1 = 1.0 if gax == 1 else 0.0
                        sa2 = 1.0 if gax == 2 else 0.0
                        sb = -j / G
                    else:
                        ext = 0.0
                        ax = 0
                        for k in range(3):
                            if Rhi[k] - Rlo[k] > ext:
                                ext = Rhi[k] - Rlo[k]
                                ax = k
                        if ext < 1e-9:
                            nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo,
                                               rec_hi, rec_kind, nrec, rec_voff, rec_vn, vpool, vtop, P, K, clo, chi, 0.0)
                            stats[S_UNDET] += 1
                            continue
                        split_kind = 3
                        sa0 = 1.0 if ax == 0 else 0.0
                        sa1 = 1.0 if ax == 1 else 0.0
                        sa2 = 1.0 if ax == 2 else 0.0
                        sb = -0.5 * (Rlo[ax] + Rhi[ax])
            # ---- perform the split: child A {a.x+b <= 0} at slot p, child B {a.x+b >= 0} at slot p+1
            if split_kind < 0:
                continue
            if split_kind == 0:
                stats[S_WIN] += 1
                ns = ne
                for t in range(ne):
                    Qc[t] = Wc[t]; Qk[t] = Wk[t]; Qm[t] = Wm[t]; Qa[t] = Wa[t]
            elif split_kind == 1:
                stats[S_FACE] += 1
                face_by_comp[sc] += 1
                face_surv[sc] += ns
            elif split_kind == 2:
                stats[S_TAU] += 1
            else:
                stats[S_GEO] += 1
            if ptop + 2 * ns > NE or p + 2 > SCR - 1:
                stats[S_OVF] += 1
                nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                   rec_kind, nrec, rec_voff, rec_vn, vpool, vtop, P, K, clo, chi, 0.0)
                stats[S_UNDET] += 1
                continue
            _copy_slot(p, p + 1, SV, SFc, SFn, Snv, Snf, Sasp, SLb, SUb, Sdep)
            rB = _clip_slot(p + 1, -sa0, -sa1, -sa2, -sb, SV, SFc, SFn, Snv, Snf)
            rA = _clip_slot(p, sa0, sa1, sa2, sb, SV, SFc, SFn, Snv, Snf)
            if rA == 2 or rB == 2:
                stats[S_OVF] += 1
                if rB != 0:
                    nrec = emit_region(p + 1, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                       rec_kind, nrec, rec_voff, rec_vn, vpool, vtop, P, K, clo, chi, 0.0)
                if rA != 0:
                    nrec = emit_region(p, UNDET_REC, SV, SFc, SFn, Snv, Snf, SCR, Vr, rec_cell, rec_lo, rec_hi,
                                       rec_kind, nrec, rec_voff, rec_vn, vpool, vtop, P, K, clo, chi, 0.0)
                Snf[p + 1] = 0
                stats[S_UNDET] += 1
                continue
            Sdep[p] += 1
            Sdep[p + 1] += 1
            if split_kind == 1:
                Sasp[p, sc, si] = 2
                Sasp[p + 1, sc, si] = 1
            elif split_kind == 2:
                key = ckey[sc, skap, st_t] if wmode[sc] else skey[sc, st_t]
                if sv < SUb[p, key]:
                    SUb[p, key] = sv
                if sv > SLb[p + 1, key]:
                    SLb[p + 1, key] = sv
            Soff[p] = ptop; Slen[p] = ns
            for t in range(ns):
                Pc[ptop + t] = Qc[t]; Pk[ptop + t] = Qk[t]; Pm[ptop + t] = Qm[t]; Pa[ptop + t] = Qa[t]
            ptop += ns
            Soff[p + 1] = ptop; Slen[p + 1] = ns
            for t in range(ns):
                Pc[ptop + t] = Qc[t]; Pk[ptop + t] = Qk[t]; Pm[ptop + t] = Qm[t]; Pa[ptop + t] = Qa[t]
            ptop += ns
            sp = p + 2
        cell_nodes[ci] = nodes_cell
        cell_evals[ci] = stats[S_EVALS] - ev0
        if nrec == rec_start:
            stats[S_CELLS_OUT] += 1
        else:
            stats[S_CELLS_IN] += 1
    return nrec
