"""Certified tile culling for exact point tests: a 64^3 tile needs no rendering for sources in box Q if
* every FREE receiver of the tile is observed lit and no component meets conv(Q u tile) (all_visible), or
* every FREE receiver is observed dark and one convex component blocks every segment from Q to the tile (any_blocked),
* or the tile has no FREE receiver.
The predicates are the label engine's float-filtered certified predicates (rpl.predicates, margins 1e-11)."""
import numpy as np
from numba import njit
from rind3d_posterior.rpl.predicates import all_visible, any_blocked
from rind3d_posterior.rpl.engine2 import MAXROWS


def tile_info(lit_zyx, solid_zyx, origin, N, offs):
    boxes = np.zeros((len(offs), 6)); code = np.zeros(len(offs), np.int8)
    for t, (a, b, c) in enumerate(offs):
        sl = (slice(c * 64, (c + 1) * 64), slice(b * 64, (b + 1) * 64), slice(a * 64, (a + 1) * 64))
        lit = lit_zyx[sl]; sol = solid_zyx[sl]; free = ~sol
        nl = int((lit & free).sum()); nd = int((~lit & free).sum())
        code[t] = 3 if nl + nd == 0 else (0 if nd == 0 else (1 if nl == 0 else 2))
        lo = (np.asarray(origin) + np.array([a, b, c]) * 64 + 0.5) / N
        boxes[t, :3] = lo; boxes[t, 3:] = lo + 63.0 / N
    return boxes, code


@njit(cache=True)
def tiles_to_render(qlo, qhi, boxes, code, P, K, clo, chi):
    buf = np.empty((MAXROWS, 4))
    need = np.ones(boxes.shape[0], np.bool_)
    for t in range(boxes.shape[0]):
        if code[t] == 3:
            need[t] = False
        elif code[t] == 0:
            if all_visible(qlo, qhi, boxes[t, :3], boxes[t, 3:], P, K, clo, chi):
                need[t] = False
        elif code[t] == 1:
            if any_blocked(qlo, qhi, boxes[t, :3], boxes[t, 3:], P, K, clo, chi, buf):
                need[t] = False
    return need


from rind3d_posterior.rpl.predicates import classify_component, aabb_disjoint, EXCL, CONTAINS


@njit(cache=True)
def tiles_to_render_point(s, boxes, code, P, K, clo, chi):
    """Per-point culling with the exact shadow polyhedra of the point source s (rpl.predicates.classify_component,
    float-filtered with margins):
    * a tile whose FREE receivers are all observed lit is certified iff every component is EXCL (tile box disjoint
      from S_C(s));
    * a tile observed all dark is certified iff some component is CONTAINS (tile box inside S_C(s));
    * mixed tiles are always rendered."""
    buf = np.empty((MAXROWS, 4)); cut = np.empty(MAXROWS, np.int64)
    need = np.ones(boxes.shape[0], np.bool_)
    ulo = np.empty(3); uhi = np.empty(3)
    for t in range(boxes.shape[0]):
        if code[t] == 3:
            need[t] = False; continue
        if code[t] == 2:
            continue
        lo = boxes[t, :3]; hi = boxes[t, 3:]
        for k in range(3):
            ulo[k] = min(lo[k], s[k]); uhi[k] = max(hi[k], s[k])
        if code[t] == 0:
            ok = True
            for c in range(P.shape[0]):
                if aabb_disjoint(ulo, uhi, clo[c], chi[c]):
                    continue
                st, _ = classify_component(lo, hi, s, P, K, c, buf, cut)
                if st != EXCL:
                    ok = False; break
            if ok:
                need[t] = False
        else:
            for c in range(P.shape[0]):
                if aabb_disjoint(ulo, uhi, clo[c], chi[c]):
                    continue
                st, _ = classify_component(lo, hi, s, P, K, c, buf, cut)
                if st == CONTAINS:
                    need[t] = False; break
    return need
