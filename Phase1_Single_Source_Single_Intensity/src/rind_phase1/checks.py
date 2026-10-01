"""Verify installed data access; research handoff checks remain to be added."""

import argparse
import pickle
import numpy as np

from rind_phase1.data import Phase1Dataset, SizeBucketBatchSampler, make_dataloader, split_scene_ids
from rind_phase1.install_data import validate_installed


def check_data(root=None, *, check_torch=False, workers=0):
    ds = Phase1Dataset(root)
    counts = validate_installed(ds.root)
    sample = ds[0]
    assert set(sample) == {"scene_id", "view_id", "response", "window"}
    assert sample["response"].dtype == np.float32
    assert sample["window"].dtype == np.int64
    assert np.array_equal(pickle.loads(pickle.dumps(ds))[0]["response"], sample["response"])
    split = split_scene_ids(ds.num_scenes, ratios=(0.8, 0.1, 0.1))
    sets = [set(ids) for ids in split.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(range(ds.num_scenes))
    selected = sorted({0, ds.num_scenes - 1})
    subset = Phase1Dataset(ds.root, scene_ids=selected)
    batches = list(SizeBucketBatchSampler(subset, 4))
    assert sorted(i for batch in batches for i in batch) == list(range(len(subset)))
    for batch in batches:
        assert len({int(subset[i]["window"][2]) for i in batch}) == 1
    for scene_id in selected:
        leaves = list(ds.iter_view_leaves(scene_id))
        assert len(leaves) == len(ds.get_scene(scene_id)["local_views"])
        assert all(node["kind"] == "view" for node in leaves)
        leaf = min(leaves, key=lambda node: node["size"])
        observed = ds.get_observation(scene_id, leaf["view_id"])
        references = ds.get_candidates(scene_id, leaf["view_id"])
        assert len(references) == counts["references_per_view"]
        for reference in references:
            assert reference.shape == (1, 3) and reference[0, 2] == 1
            fresh = ds.rerender(scene_id, observed["window"], reference[0, :2])
            assert np.array_equal(fresh, observed["response"])
    if check_torch:
        loader = make_dataloader(subset, batch_size=4, num_workers=workers)
        for batch in loader:
            assert str(batch["response"].dtype) == "torch.float32"
            assert batch["window"].shape[1] == 3
            assert batch["response"].shape[1] == batch["response"].shape[2]
            assert bool((batch["window"][:, 2] == batch["response"].shape[1]).all())
    print(f"PASS: {counts['num_scenes']} scenes, {counts['num_views']} views; "
          "observation fields, scene subsets, batching, tree, references, rerender, and worker serialization.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    check_data(args.data_root, check_torch=args.torch, workers=args.workers)


if __name__ == "__main__":
    main()
