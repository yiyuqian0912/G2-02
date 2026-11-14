"""Re-export of rind3d_posterior.fastlabel.sampler (1K compatibility)."""
from rind3d_posterior.fastlabel import sampler as _core
from rind3d_posterior.fastlabel.sampler import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_core, name)
