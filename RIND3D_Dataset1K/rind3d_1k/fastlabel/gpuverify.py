"""1K wrapper of rind3d_posterior.fastlabel.gpuverify: WindowVerifier(fac, ctx, S, w, s, cls) with a SceneObs."""
from rind3d_posterior.fastlabel import gpuverify as _core
from rind3d_posterior.fastlabel.gpuverify import *  # noqa: F401,F403
from rind3d_posterior.fastlabel.gpuverify import leaf_codes, leaf_codes_ref, coarse_codes, GeometryRows, DevBuf  # noqa: F401


def __getattr__(name):
    return getattr(_core, name)


class WindowVerifier(_core.WindowVerifier):
    def __init__(self, fac, ctx, S, w, s, cls, **kw):
        from rind3d_posterior.fastlabel.inputs import window_from_1k
        super().__init__(fac, ctx, window_from_1k(S.dir, w, s), cls, **kw)
