"""Re-export of rind3d_posterior.fastlabel.fastplanes (1K compatibility)."""
from rind3d_posterior.fastlabel import fastplanes as _core
from rind3d_posterior.fastlabel.fastplanes import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_core, name)
