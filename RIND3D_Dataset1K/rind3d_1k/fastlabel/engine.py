"""Re-export of rind3d_posterior.fastlabel.engine (1K compatibility)."""
from rind3d_posterior.fastlabel import engine as _core
from rind3d_posterior.fastlabel.engine import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_core, name)
