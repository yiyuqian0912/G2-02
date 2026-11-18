"""Hausdorff gap between the 1K `faceted_spheres` polyhedra and the ball they circumscribe (docs/CURVED_ASSESSMENT.md §3.2).

rind3d_1k.families: n unit normals on a Fibonacci sphere, every plane tangent to the unit sphere (offset 1), scaled by
the radius r ~ LU(0.02, 0.12), i.e. the same radius law as the curved `spheres` family. The polyhedron contains the
ball; its farthest vertex lies (max_gap/r) r beyond the ball. Voxel units at N = 1024. CPU only."""
import numpy as np
from scipy.spatial import HalfspaceIntersection


def fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def gap(n):
    V = HalfspaceIntersection(np.c_[fib(n), -np.ones(n)], np.zeros(3)).intersections
    d = np.linalg.norm(V, axis=1)
    return d.max() - 1, np.median(d) - 1, len(V)


if __name__ == "__main__":
    radii = (0.02 * 1024, np.sqrt(0.02 * 0.12) * 1024, 0.12 * 1024)          # min, median (log-uniform), max
    print("planes  max_gap/r  median_vertex_gap/r  vertices  max gap in voxels at r = %.0f / %.0f / %.0f" % radii)
    for n in (20, 32, 48, 100, 200, 400, 800, 1600):
        g, gm, nv = gap(n)
        print(f"{n:6d}  {g:.4f}     {gm:.4f}               {nv:5d}     " + " / ".join(f"{g * r:.2f}" for r in radii))
    for r in radii:
        n = next(n for n in range(20, 5000, 10) if gap(n)[0] * r <= 0.5)
        print(f"r = {r:.0f} voxels: n >= {n} planes for a max gap <= 0.5 voxel")
