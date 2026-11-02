import numpy as np
from rind3d_reader.pack import PackedWindow
from rind_phase3.coordinates import Window
from rind_phase3.oracle import PhysicalOracle
from rind_phase3.physics import sample_probes, SparsePhysicalCost, full_window_cost


def packed(a):
    return PackedWindow(np.packbits(a, axis=-1, bitorder="little"), a.shape)


def test_sampled_and_full_cost_distinguished_even_at_zero(box):
    w = Window((701, 720, 480), 64)
    with PhysicalOracle(box) as oracle:
        rendered = oracle.rerender_window([0.0625, 0.5, 0.5], w)
        observation = {
            "origin": np.array(w.origin),
            "side": w.side,
            "observed_response": packed(rendered.response),
            "local_solid_mask": packed(rendered.solid),
        }
        probes = sample_probes(observation, count=32, seed=9)
        sampled, result = SparsePhysicalCost(oracle, probes).evaluate(
            [[0.0625, 0.5, 0.5], [0.5, 0.5, 0.5]]
        )
        assert sampled[0] == 0 and np.isnan(sampled[1]) and result.evaluated_pairs == 32
        full = full_window_cost(oracle, observation, [0.0625, 0.5, 0.5])
        assert full["cost"] == 0 and not full["equivalent_source_certification"]
        assert len(np.unique(probes.world_xyz, axis=0)) == 32
