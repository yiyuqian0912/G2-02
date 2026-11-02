import numpy as np
import pytest
from rind_phase3.oracle import PhysicalOracle, OracleFailure, QueryBudgetExceeded
from rind_phase3.coordinates import Window


def test_closed_contact_tangency_and_canonical_agreement(box):
    source = [0.0625, 0.5, 0.5]
    points = np.array(
        [
            [0.375, 0.5, 0.5],
            [0.6875, 0.75, 0.5],
            [0.6875, 0.75, 0.75],
            [0.6875, np.nextafter(0.75, 1.0), 0.5],
        ]
    )
    with PhysicalOracle(box) as a, PhysicalOracle(box, backend="cpu_reference") as b:
        result = a.evaluate_points(source, points)
        reference = b.evaluate_points(source, points)
        np.testing.assert_array_equal(result.response, reference.response)
        assert result.response.tolist() == [[0, 0, 0, 1]]
        assert result.solid.tolist() == [True, False, False, False]
        assert result.certified and result.evaluated_pairs == 4


def test_invalid_sources_never_become_response_labels(box):
    w = Window((64, 64, 64), 64)
    with PhysicalOracle(box) as oracle:
        p = np.array(
            [
                [0, 0.2, 0.2],
                [0.5, 0.5, 0.5],
                [0.375 - 2**-26, 0.5, 0.5],
                [0.1, 0.1, 0.1],
                [0.0625, 0.5, 0.5],
            ]
        )
        result = oracle.evaluate_points(p, [[0.9, 0.8, 0.7], [0.8, 0.2, 0.3]], window=w)
        assert result.validity.valid.tolist() == [False, False, False, False, True]
        assert np.all(result.response[:4] == -1) and result.evaluated_pairs == 2
        assert result.validity.reasons[2] == "insufficient_clearance"


def test_cartesian_order_bounded_chunks_and_no_label_separation(box):
    p = np.array([[0.0625, 0.5, 0.5], [0.0625 + 2**-20, 0.5, 0.5]])
    q = np.array([[0.8, 0.5, 0.5], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
    with PhysicalOracle(box, receiver_batch=1, max_pairs=6) as oracle:
        r = oracle.evaluate_points(p, q)
        assert r.response.shape == (2, 3) and r.evaluated_pairs == 6
        assert r.validity.valid.all()  # teacher candidates need no 32/1024 separation
        assert oracle.stats["source_receiver_pairs"] == 6
        with pytest.raises(QueryBudgetExceeded):
            oracle.evaluate_points(p, np.tile(q, (2, 1)))
        with pytest.raises(ValueError):
            oracle.evaluate_points(p, [[1.1, 0, 0]])


def test_unknown_fails_closed_and_invalid_rerender_refused(box, monkeypatch):
    with PhysicalOracle(box) as oracle:

        def fail(*args):
            raise RuntimeError("unresolved fixture")

        monkeypatch.setattr(oracle.context, "clearance", fail)
        with pytest.raises(OracleFailure) as error:
            oracle.source_valid([[0.1, 0.1, 0.1]])
        assert error.value.status == "UNRESOLVED"
        with pytest.raises(ValueError):
            oracle.rerender_window([[0.5] * 3], Window((64, 64, 64), 64))


def test_incomplete_raster_batch_fails_closed(box, monkeypatch):
    class IncompleteRaster:
        backend = "cuda_certified"

        def packed_batch(self, *args):
            return []

        def close(self):
            pass

    with PhysicalOracle(box, raster_backend="auto") as oracle:
        monkeypatch.setattr(oracle.factory, "source", lambda *args: IncompleteRaster())
        with pytest.raises(OracleFailure) as failure:
            oracle.rerender_window([0.0625, 0.5, 0.5], Window((701, 401, 401), 64))
        assert failure.value.status == "UNRESOLVED"


@pytest.mark.parametrize("side", [64, 128, 256])
def test_arbitrary_window_matches_exact_point_queries(box, side):
    w = Window((701, 401, 401), side)
    with PhysicalOracle(box) as oracle:
        rendered = oracle.rerender_window([0.0625, 0.5, 0.5], w)
        local = np.array(
            [[0, 0, 0], [side - 1, side - 1, side - 1], [63, 31, 7], [2, 60, 63]],
            dtype=np.int64,
        )
        points = oracle.evaluate_points([0.0625, 0.5, 0.5], w.centres(local))
        x, y, z = local.T
        np.testing.assert_array_equal(rendered.response[z, y, x], points.response[0])
        assert rendered.certified and not rendered.equivalence_certification


@pytest.mark.skipif(
    not __import__("os").environ.get("RIND_TEST_CUDA"),
    reason="Explicit CUDA device test",
)
def test_real_cuda_unaligned_window_matches_cpu(box):
    w = Window((701, 401, 401), 64)
    with (
        PhysicalOracle(box) as cpu,
        PhysicalOracle(box, raster_backend="cuda_certified") as gpu,
    ):
        a = cpu.rerender_window([0.0625, 0.5, 0.5], w)
        b = gpu.rerender_window([0.0625, 0.5, 0.5], w)
        assert b.backend == "cuda_certified"
        np.testing.assert_array_equal(a.response, b.response)
        np.testing.assert_array_equal(a.solid, b.solid)
