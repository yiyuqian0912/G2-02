"""Component AABBs from vertex enumeration (copied verbatim from rpl.data.Scene.component_aabbs)."""
import numpy as np


def component_aabbs(P, K, margin=1e-9):
    """Float AABB of each component from vertex enumeration (triples of planes), inflated by `margin`."""
    m = P.shape[0]
    lo = np.full((m, 3), np.inf)
    hi = np.full((m, 3), -np.inf)
    for c in range(m):
        rows = P[c, :K[c]]
        k = len(rows)
        for i in range(k):
            for j in range(i + 1, k):
                for l in range(j + 1, k):
                    A = rows[[i, j, l], :3]
                    if abs(np.linalg.det(A)) < 1e-12:
                        continue
                    x = np.linalg.solve(A, -rows[[i, j, l], 3])
                    if np.all(rows[:, :3] @ x + rows[:, 3] <= 1e-9):
                        lo[c] = np.minimum(lo[c], x)
                        hi[c] = np.maximum(hi[c], x)
    return lo - margin, hi + margin
