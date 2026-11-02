import os
from pathlib import Path
import pytest


@pytest.fixture
def dataset_root():
    value = os.environ.get("RIND3D_DATA_ROOT")
    if not value:
        pytest.skip("Set RIND3D_DATA_ROOT to the verified public RIND3D v0.1 release")
    return Path(value)


@pytest.fixture
def box():
    pytest.importorskip("rind3d._convex")
    from rind.contracts import Geometry, RootKind, make_leaf
    from rind.storage import CanonicalBundle
    import numpy as np

    b = CanonicalBundle(3)
    leaf = make_leaf("affine_box_v1", 3, center=[0.5] * 3, matrix=np.diag([0.125] * 3))
    b.add_leaf(leaf)
    b.add_geometry(Geometry(3, RootKind.REF, leaf.content_hash))
    return b
