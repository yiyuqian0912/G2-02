"""Re-export of rind3d_posterior.fastlabel.filters (1K compatibility)."""
from rind3d_posterior.fastlabel import filters as _core
from rind3d_posterior.fastlabel.filters import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_core, name)
