"""1K wrapper of rind3d_posterior.fastlabel.window: fast_cls3(S, w, s) from a SceneObs."""
from rind3d_posterior.fastlabel import window as _core
from rind3d_posterior.fastlabel.window import *  # noqa: F401,F403
from rind3d_posterior.fastlabel.window import LazyWindow, LazyWindow2, cls_from_packed, window_classes  # noqa: F401


def __getattr__(name):
    return getattr(_core, name)


def fast_cls3(S, w, s):
    """Class tensor cls[x, y, z] of a SceneObs window (= SceneObs.cls(w, s))."""
    def raw(j):
        o, n = S._pos[(w, j)]
        return S.dz.decompress(S.blob[o:o + n])
    import numpy as np
    return cls_from_packed(np.frombuffer(raw(s), np.uint8), np.frombuffer(raw(-1), np.uint8), int(S.W[w]["L"]))
