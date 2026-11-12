import sys, numpy as np
sys.path.insert(0, 'code')
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from rpl.polytope import clipped_box_volume

def ref_volume(lo, hi, planes):
    H = [np.r_[-np.eye(3)[k], lo[k]] for k in range(3)] + [np.r_[np.eye(3)[k], -hi[k]] for k in range(3)]
    H = np.array(H + [p for p in planes])
    # interior point via Chebyshev centre LP
    A = H[:, :3]; b = -H[:, 3]; nrm = np.linalg.norm(A, axis=1)
    res = linprog(np.r_[0, 0, 0, -1], A_ub=np.c_[A, nrm], b_ub=b, bounds=[(None, None)] * 3 + [(0, None)])
    if not res.success or res.x[3] < 1e-12:
        return 0.0
    hs = HalfspaceIntersection(H, res.x[:3])
    return ConvexHull(hs.intersections).volume

def test_random_clips():
    rng = np.random.default_rng(1)
    worst = 0
    for t in range(400):
        lo = rng.random(3) * 0.5; hi = lo + 0.05 + rng.random(3) * 0.4
        n = rng.integers(1, 40)
        c = lo + (hi - lo) * rng.random(3)
        pl = []
        for _ in range(n):
            a = rng.normal(size=3); a /= np.linalg.norm(a)
            off = rng.random() * 0.3
            pl.append(np.r_[a, -(a @ c) - off])          # keeps a neighbourhood of c
        pl = np.array(pl)
        v, st = clipped_box_volume(lo, hi, pl, len(pl))
        r = ref_volume(lo, hi, pl)
        assert st in (0, 1)
        err = abs(v - r) / max(r, 1e-300) if r > 0 else abs(v)
        worst = max(worst, err)
    print('worst relative error', worst)
    assert worst < 1e-9

def test_thin_slab():
    lo = np.zeros(3); hi = np.ones(3) * 2 ** -7
    for th in [1e-4, 1e-6, 1e-8, 1e-10]:
        a = np.array([1.0, 2.0, 3.0]); a /= np.linalg.norm(a)
        m = (lo + hi) / 2
        pl = np.array([np.r_[a, -(a @ m) - th / 2], np.r_[-a, (a @ m) - th / 2]])
        v, st = clipped_box_volume(lo, hi, pl, 2)
        r = ref_volume(lo, hi, pl)
        print('thickness', th, 'vol', v, 'ref', r, 'rel', abs(v - r) / r)
        assert abs(v - r) / r < 1e-6

if __name__ == '__main__':
    test_random_clips(); test_thin_slab()
