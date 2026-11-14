"""Pooled Student input features of one window channel (numba; module file so the cache works).

Input crops are the stored ZYX bool arrays. Output: uint8 [4, 32, 32, 32] in XYZ order with the fractions (x 255) of
lit-FREE, dark-FREE, SOLID, and lit/dark-boundary FREE voxels (a FREE voxel with a 6-neighbour FREE voxel of the
other colour inside the crop) per pooled cell, plus exact counts: lit, dark, solid, boundary-pairs."""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def pooled(sol, lit, P):
    L = sol.shape[0]
    f = L // P
    acc = np.zeros((4, P, P, P), np.int64)
    npairs = 0
    nlit = 0
    ndark = 0
    nsol = 0
    bnd = np.zeros((L, L, L), np.uint8)
    for z in range(L):
        for y in range(L):
            for x in range(L):
                if sol[z, y, x]:
                    continue
                c = lit[z, y, x]
                if x + 1 < L and not sol[z, y, x + 1] and lit[z, y, x + 1] != c:
                    bnd[z, y, x] = 1; bnd[z, y, x + 1] = 1; npairs += 1
                if y + 1 < L and not sol[z, y + 1, x] and lit[z, y + 1, x] != c:
                    bnd[z, y, x] = 1; bnd[z, y + 1, x] = 1; npairs += 1
                if z + 1 < L and not sol[z + 1, y, x] and lit[z + 1, y, x] != c:
                    bnd[z, y, x] = 1; bnd[z + 1, y, x] = 1; npairs += 1
    for z in range(L):
        for y in range(L):
            for x in range(L):
                i, j, k = x // f, y // f, z // f
                if sol[z, y, x]:
                    acc[2, i, j, k] += 1; nsol += 1
                elif lit[z, y, x]:
                    acc[0, i, j, k] += 1; nlit += 1
                else:
                    acc[1, i, j, k] += 1; ndark += 1
                if bnd[z, y, x]:
                    acc[3, i, j, k] += 1
    out = np.empty((4, P, P, P), np.uint8)
    vol = f * f * f
    for c in range(4):
        for i in range(P):
            for j in range(P):
                for k in range(P):
                    out[c, i, j, k] = (acc[c, i, j, k] * 255 + vol // 2) // vol
    return out, nlit, ndark, nsol, npairs
