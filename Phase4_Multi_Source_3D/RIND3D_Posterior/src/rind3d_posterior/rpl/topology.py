"""Per-component face adjacency (silhouette candidates) from float vertex enumeration.

Two faces i, j are recorded as adjacent if at least two *distinct* enumerated vertices lie on both planes
(|f| <= TOL).  The generous tolerance can only add non-adjacent pairs, whose Fourier-Motzkin rows are valid but
redundant inequalities of S_C(r); a true edge is never missed because both of its endpoints are enumerated
vertices lying (to rounding) on both planes.  `tests/test_shadow.py` checks the reduced H-representation
against the full Fourier-Motzkin one."""
import numpy as np

TOL = 1e-9


def vertices(rows):
    k = len(rows)
    out = []
    for i in range(k):
        for j in range(i + 1, k):
            for l in range(j + 1, k):
                A = rows[[i, j, l], :3]
                if abs(np.linalg.det(A)) < 1e-14:
                    continue
                x = np.linalg.solve(A, -rows[[i, j, l], 3])
                if np.all(rows[:, :3] @ x + rows[:, 3] <= TOL):
                    out.append(x)
    return np.array(out)


def adjacency(P, K):
    """Returns (pairs (n,3) int32 rows (component, i, j) with i < j, offsets per component)."""
    pairs = []
    off = [0]
    for c in range(P.shape[0]):
        rows = P[c, :K[c]]
        V = vertices(rows)
        on = np.abs(V @ rows[:, :3].T + rows[:, 3]) <= TOL     # (nv, k)
        cnt = 0
        for i in range(K[c]):
            for j in range(i + 1, K[c]):
                both = V[on[:, i] & on[:, j]]
                if len(both) >= 2 and np.max(np.linalg.norm(both - both[0], axis=1)) > 0:
                    pairs.append((c, i, j))
                    cnt += 1
        off.append(off[-1] + cnt)
    return np.array(pairs, np.int32).reshape(-1, 3), np.array(off, np.int64)
