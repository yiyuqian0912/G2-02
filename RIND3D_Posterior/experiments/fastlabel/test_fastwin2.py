"""fastwin v3 helpers are identical to the originals (class tensor, boundary mask, top items)."""
import sys, time, numpy as np
sys.path[:0] = [".", "../twophase"]
import twophase as T
from fastwin import fast_cls, fast_cls2, LazyWindow, LazyWindow2
from rind3d_posterior.tpl.window import Window
specs = sys.argv[1:]
for spec in specs:
    sc, w, s = spec.split(":"); w = int(w); s = int(s)
    S = T.SceneObs(T.R / sc)
    a = fast_cls(S, w, s); t = time.perf_counter(); b = fast_cls2(S, w, s); tb = time.perf_counter() - t
    assert np.array_equal(a, b), "cls"
    o = np.asarray(S.origin(w), np.int64)
    W = Window(a, o, 1024); W2 = LazyWindow2(b, o, 1024)
    assert np.array_equal(W.boundary_mask(), W2.boundary_mask()), "boundary"
    c1, k1 = W.top_items(); t = time.perf_counter(); c2, k2 = W2.top_items(); ti = time.perf_counter() - t
    assert np.array_equal(c1, c2) and np.array_equal(k1, k2), "top_items"
    x1 = W.sentinels(512, 64, 0); x2 = W2.sentinels(512, 64, 0)
    assert np.array_equal(x1[0], x2[0]) and np.array_equal(x1[1], x2[1]), "sentinels"
    p1 = T.point_items(W, 1024, 512, 7); p2 = T.point_items(W2, 1024, 512, 7)
    assert np.array_equal(p1[0], p2[0]) and np.array_equal(p1[1], p2[1]), "point_items"
    print(spec, W.L, "identical", len(c1), f"cls2 {tb:.3f}s items2 {ti:.3f}s")
