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
