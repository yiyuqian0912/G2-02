"""Scene geometry for the tau-pencil engine.

Components C_c = {x : n_k.x + d_k <= 0} with *unit* normals (float64 roundings of the canonical rational rows,
renormalised), face adjacency (silhouette-edge candidates, rpl/topology.py), component AABBs.
"""
import numpy as np

from rpl.data import Scene
from rpl.topology import adjacency


class Geom:
    def __init__(self, P_raw, K, rows_exact=None):
        P_raw = np.asarray(P_raw, float)
        K = np.asarray(K, np.int32)
        P = P_raw.copy()
        for c in range(P.shape[0]):
            for k in range(K[c]):
                P[c, k] /= np.linalg.norm(P[c, k, :3])
        self.P = np.ascontiguousarray(P)
        self.K = K
        self.clo, self.chi = Scene.component_aabbs(P_raw, K)
        self.adj, self.aoff = adjacency(P_raw, K)
        self.rows_exact = rows_exact
        self.C = P.shape[0]
        self.max_edges = int(np.max(np.diff(self.aoff))) if self.C else 0
        if self.C > 62:
            raise ValueError('component bitmask supports at most 62 components')

    @classmethod
    def from_scene(cls, scene):
        P, K, _, _ = scene.geometry_arrays()
        return cls(P, K, scene.rows_exact)

    @classmethod
    def from_bodies(cls, bodies):
        m = len(bodies)
        kmax = max(len(b) for b in bodies)
        P = np.zeros((m, kmax, 4))
        K = np.zeros(m, np.int32)
        for c, b in enumerate(bodies):
            P[c, :len(b)] = b
            K[c] = len(b)
        return cls(P, K)

    def tup(self):
        return self.P, self.K, self.clo, self.chi, self.adj, self.aoff

    def surface_area(self):
        """Total boundary area of all components (float; for the clearance-shell term)."""
        from rpl.topology import vertices
        tot = 0.0
        for c in range(self.C):
            rows = self.P[c, :self.K[c]]
            V = vertices(rows)
            for i in range(self.K[c]):
                on = V[np.abs(V @ rows[i, :3] + rows[i, 3]) <= 1e-9]
                if len(on) < 3:
                    continue
                n = rows[i, :3]
                ctr = on.mean(0)
                u = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u)
                w = np.cross(n, u)
                ang = np.arctan2((on - ctr) @ w, (on - ctr) @ u)
                p = on[np.argsort(ang)]
                a = 0.0
                for j in range(1, len(p) - 1):
                    a += 0.5 * np.dot(np.cross(p[j] - p[0], p[j + 1] - p[0]), n)
                tot += abs(a)
        return tot
