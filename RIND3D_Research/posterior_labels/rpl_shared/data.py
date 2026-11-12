"""Scene / window access for RIND3D-Lite v0.1 (read-only).

Geometry: finite union of closed convex polyhedra C_c = {x : n_k.x + d_k <= 0}.  The canonical rows are exact
rationals (fractions.Fraction) normalised by their largest |coefficient|; `rows_f64` holds their float64
roundings (relative error <= 2^-53), which the numba predicates use with explicit safety margins.

Window classes (XYZ indexing, cls[x, y, z]):  0 = dark FREE, 1 = lit FREE, 2 = SOLID.
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import rindenv  # noqa: F401  (sets sys.path for the canonical packages)

DARK, LIT, SOLID = 0, 1, 2
N = 1024


class Scene:
    def __init__(self, scene_id, root=None):
        from rind3d.geometry import load_geometry
        from rind3d.rendering.context import GeometryContext
        self.id = scene_id
        self.dir = Path(root or rindenv.SCENES) / scene_id
        meta = json.loads((self.dir / "verification.json").read_text())
        self.ctx = GeometryContext(load_geometry(self.dir / "geometry.npz"), meta["numerical_policy"])
        self.rows_exact = [tuple(tuple(Fraction(c) for c in row) for row in body) for body in self.ctx.bodies]
        self.truth = np.fromfile(self.dir / "driver.f64", "<f8").copy()
        rec = np.fromfile(self.dir / "window_records.bin", "<u2").reshape(-1, 4)
        self.windows = rec.astype(np.int64)  # (x, y, z, L), half-open [o, o+L)
        self._pack = {}

    # -- geometry as padded float64 arrays for numba ------------------------------------------------
    def geometry_arrays(self):
        m = len(self.rows_exact)
        kmax = max(len(b) for b in self.rows_exact)
        P = np.zeros((m, kmax, 4))
        K = np.zeros(m, np.int32)
        for c, body in enumerate(self.rows_exact):
            K[c] = len(body)
            for k, row in enumerate(body):
                P[c, k] = [float(v) for v in row]
        lo, hi = self.component_aabbs(P, K)
        return P, K, lo, hi

    @staticmethod
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

    # -- stored fields -------------------------------------------------------------------------------
    def _rindpack(self, name):
        if name not in self._pack:
            from rind3d.pack import RINDPack
            self._pack[name] = RINDPack(self.dir / f"{name}.rindb")
        return self._pack[name]

    def window_classes(self, wid):
        x, y, z, L = (int(v) for v in self.windows[wid])
        return self.region_classes((x, y, z), L), (x, y, z), L

    def region_classes(self, origin, L):
        """uint8 class grid cls[x, y, z] for the half-open cube origin + [0, L)^3."""
        resp = self._read(origin, L, "response")
        sol = self._read(origin, L, "solid")
        cls = np.where(sol, SOLID, np.where(resp, LIT, DARK)).astype(np.uint8)
        return cls

    def _read(self, origin, L, name):
        pk = self._rindpack(name)
        x, y, z = (int(v) for v in origin)
        a = pk.read_window(x, y, z, int(L))  # ZYX bool
        return np.ascontiguousarray(np.transpose(a, (2, 1, 0)))

    def selection_table(self):
        import pyarrow.parquet as pq
        return pq.read_table(self.dir / "window_selection.parquet").to_pandas() if _has_pandas() else \
            pq.read_table(self.dir / "window_selection.parquet").to_pydict()


def _has_pandas():
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


def scene_ids(root=None):
    return sorted(p.name for p in Path(root or rindenv.SCENES).iterdir() if p.is_dir())


def splits(root=None):
    """scene_id -> split ('train' / 'val' / 'test') from the immutable campaign catalog."""
    cat = Path(root or rindenv.SCENES).parent / "catalog" / "scenes"
    out = {}
    for f in cat.glob("*.json"):
        d = json.loads(f.read_text())
        out[d["scene_id"]] = d["split"]
    return out
