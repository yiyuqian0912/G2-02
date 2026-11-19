"""Development-only per-cell GMP differential oracle."""
from collections import Counter
from time import perf_counter
from kernel import SCALE

def grid_credit_native(self, parts, native):
    """GMP octree (native/cells.cpp), same algorithm as _grid_credit_python and exactly
    equal per-cell credits (differential test). Conservation is re-checked here against
    the independent Python rational volume of each piece."""
    started = perf_counter()
    lo, hi = Counter(), Counter()
    try:
        for p in parts:
            self.budget.check()
            credits, boundary, interior = native.grid_credit(p.faces, 128, 160)
            subtotal = 0
            for cid, low, high in credits:
                if low:
                    lo[cid] += low
                hi[cid] += high
                subtotal += low
            total = p.volume * SCALE
            ok = (subtotal <= total < subtotal + boundary) if boundary else (subtotal == total)
            if not ok:
                raise ArithmeticError("native grid integration failed volume conservation")
            self.stats["grid_intersections"] += boundary
            self.stats["cells_interior"] += interior
        return lo, hi
    finally:
        self.stats["integration_seconds"] += perf_counter() - started

