import numpy as np
from rind_phase3 import RIND3D, make_dataloader


def test_real_reader_shapes_statistics_splits(dataset_root):
    with RIND3D(dataset_root, split=None) as data:
        assert len(data) == 256000 and data.metadata()["selected_scenes"] == 1000
        for i in (0, 85, 170):
            sample = data[i]
            side = sample["side"]
            assert side in (64, 128, 256)
            y = sample["observed_response"].unpack()
            m = sample["local_solid_mask"].unpack()
            counts = data.statistics(i)
            assert y.shape == (side,) * 3 and m.shape == y.shape
            assert counts.lit == np.count_nonzero(
                y & ~m
            ) and counts.shadow == np.count_nonzero(~y & ~m)
            assert counts.informative
    for split, n in (("train", 776), ("val", 116), ("test", 108)):
        with RIND3D(dataset_root, split=split) as data:
            assert len(data) == n * 256


def test_real_student_multiworker_loader(dataset_root):
    with RIND3D(dataset_root, split=None, window_sizes=[64]) as data:
        loader = make_dataloader(data, batch_size=2, num_workers=2, seed=7)
        iterator = iter(loader)
        try:
            batch = next(iterator)
            assert batch["observed_response"].shape == (2, 64, 64, 8)
            assert batch["local_solid_mask"].shape == batch["observed_response"].shape
            assert "structure" not in batch and "candidate_sources" not in batch
        finally:
            iterator.close()
