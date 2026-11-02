"""Offline RIND3D dataset. No source or global-response labels in model views."""
from .pack import RINDPack, RINDPackWriter, PackedWindow

__version__='0.3.0'

def __getattr__(name):
    if name=='RINDDataset':
        from .data import RINDDataset
        return RINDDataset
    raise AttributeError(name)
