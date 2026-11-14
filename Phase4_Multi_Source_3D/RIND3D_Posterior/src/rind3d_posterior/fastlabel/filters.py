"""Vectorised twophase.build_filter (identical output arrays) and items_from_vox."""
import numpy as np


def build_filter2(res):
    G = 128; cs = 1.0 / G
    cell = np.asarray(res["cell"], np.int64); voff = res["voff"]; vn = res["vn"]; vp = res["vpool"]
    n = len(cell)
    hasv = (voff >= 0) & (vn > 0)
    fine = np.zeros(n, bool); lo = np.zeros((n, 3)); hi = np.zeros((n, 3))
    idx = np.flatnonzero(hasv)
    if len(idx):
        cnt = vn[idx].astype(np.int64)
        seg_start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        pos = np.repeat(voff[idx].astype(np.int64) - seg_start, cnt) + np.arange(cnt.sum())
        Gv = vp[pos]
        lo[idx] = np.minimum.reduceat(Gv, seg_start, axis=0); hi[idx] = np.maximum.reduceat(Gv, seg_start, axis=0)
        fine[idx] = np.max(hi[idx] - lo[idx], axis=1) < 0.25 * cs
    full_cells = np.unique(cell[~fine])
    cells = np.unique(cell)
    kind = np.where(np.isin(cells, full_cells), 2, 1).astype(np.int8)
    # boxes of fine records whose cell is not full, grouped by cell in record order
    keep = fine & ~np.isin(cell, full_cells)
    ki = np.flatnonzero(keep)
    ki = ki[np.argsort(cell[ki], kind="stable")]
    boxes = np.concatenate([lo[ki] - 1e-13, hi[ki] + 1e-13], axis=1) if len(ki) else np.zeros((0, 6))
    ln = np.zeros(len(cells), np.int64)
    if len(ki):
        uc, c = np.unique(cell[ki], return_counts=True)
        ln[np.searchsorted(cells, uc)] = c
    start = np.concatenate([[0], np.cumsum(ln)[:-1]]).astype(np.int64)
    bb = np.ascontiguousarray(boxes, float).reshape(-1, 6) if len(boxes) else np.zeros((1, 6))
    return (cells.astype(np.int64), kind, start, ln, bb)


def items_from_vox2(win, vox):
    c = win.cls
    vox = np.asarray(vox, np.int64)
    k = c[vox[:, 0], vox[:, 1], vox[:, 2]]; keep = k != 2
    v = vox[keep]
    codes = (v[:, 0] << 32) | (v[:, 1] << 16) | v[:, 2]          # level 0: (0 << 48)
    return codes.astype(np.int64), k[keep].astype(np.int8)
