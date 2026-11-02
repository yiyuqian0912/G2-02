import numpy as np
import pytest
from rind_phase3.coordinates import Window, voxel_centres, world_to_voxel


def test_xyz_dyadic_centres_and_half_open_bounds():
    w = Window((17, 29, 43), 64)
    p = w.centres(np.array([[0, 1, 2], [63, 62, 61]]))
    np.testing.assert_array_equal(
        p, np.array([[17.5, 30.5, 45.5], [80.5, 91.5, 104.5]]) / 1024
    )
    np.testing.assert_array_equal(world_to_voxel(p), [[17, 30, 45], [80, 91, 104]])
    assert w.contains(w.world_origin).tolist() == [True]
    assert not w.contains(w.world_origin + [w.world_side, 0, 0])[0]


@pytest.mark.parametrize("side", [32, 96, 110, 127, 192, 512])
def test_invalid_observation_sizes(side):
    with pytest.raises(ValueError):
        Window((0, 0, 0), side)


def test_coordinates_are_validated_without_clamping():
    with pytest.raises(ValueError):
        Window((1000, 0, 0), 64)
    with pytest.raises(ValueError):
        voxel_centres([[1024, 0, 0]])
    with pytest.raises(ValueError):
        voxel_centres([[0.5, 0, 0]])
    with pytest.raises(ValueError):
        world_to_voxel([[1.0, 0, 0]])
