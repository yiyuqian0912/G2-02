"""Verify installed data and the uniform physics -> teacher handoff."""

import argparse
import pickle
import numpy as np

from rind_phase1.data import Phase1Dataset, SizeBucketBatchSampler, make_dataloader, split_scene_ids
from rind_phase1.install_data import validate_installed


def check_data(root=None, *, check_torch=False, workers=0):
    ds = Phase1Dataset(root)
    counts = validate_installed(ds.root)
    sample = ds[0]
    assert set(sample) == {"scene_id", "view_id", "response", "obstacle", "window"}
    assert sample["response"].dtype == np.float32
    assert sample["obstacle"].dtype == np.dtype(bool)
    assert sample["obstacle"].shape == sample["response"].shape
    assert not sample["response"][sample["obstacle"]].any()
    assert sample["window"].dtype == np.int64
    assert np.array_equal(pickle.loads(pickle.dumps(ds))[0]["response"], sample["response"])
    # Audit the generator prior for every stored view, with continuous sources.
    world = ds.manifest["global_size"]
    for start in range(0, ds.num_scenes, 256):
        views = np.asarray(ds._base.local_views[start:start + 256], dtype=np.int64)
        view_counts = ds._base.view_counts[start:start + 256]
        active = np.arange(views.shape[1])[None, :] < view_counts[:, None]
        lengths = np.where(active, views[..., 2], 1)
        source = ds._base.drivers[start:start + 256, 0, :][:, None, :]
        parent_origin = (views[..., :2] // (2 * lengths[..., None])) * (2 * lengths[..., None])

        def inside(origin, size):
            end = origin + size[..., None]
            upper = (source < end) | ((end == world) & (source <= end))
            return ((source >= origin) & upper).all(axis=-1)

        assert inside(parent_origin, 2 * lengths)[active].all()
        assert not inside(views[..., :2], lengths)[active].any()
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
            assert str(batch["obstacle"].dtype) == "torch.bool"
            assert batch["obstacle"].shape == batch["response"].shape
            assert batch["window"].shape[1] == 3
            assert batch["response"].shape[1] == batch["response"].shape[2]
            assert bool((batch["window"][:, 2] == batch["response"].shape[1]).all())
    print(f"PASS: {counts['num_scenes']} scenes, {counts['num_views']} views; "
          "all-view generation prior, local masks, subsets, batching, tree, references, rerender, and worker serialization.")


def check_teacher(root=None, *, scene_id=0):
    """Six sizes, prior on/off, all witnesses, actual cost alignment, and record loading.

    World/8 spacing and tau=.05 are explicitly smoke-check settings, not an
    experiment protocol. The unit tests exercise the remaining algebraic gates.
    """
    import tempfile
    from pathlib import Path
    from rind_phase1.diagnostics import inspect_teacher, representative_view_ids
    from rind_phase1.physics import PhysicalEvaluator, response_disagreement
    from rind_phase1.search import CandidateDomain, uniform_grid
    from rind_phase1.teacher import generate_uniform_teacher, load_teacher_record, save_teacher_record

    ds = Phase1Dataset(root)
    spacing = ds.manifest["global_size"] / 8
    references_checked = 0
    for view_id in representative_view_ids(ds, scene_id):
        observation = ds.get_observation(scene_id, view_id)
        side = int(observation["window"][2])
        record = generate_uniform_teacher(ds, scene_id, view_id, spacing=spacing, temperature=0.05)
        domain = CandidateDomain(tuple(observation["window"]), ds.manifest["global_size"])
        assert domain.name == "world_minus_window"
        assert record["metadata"]["settings"]["support"]["source_location_scope"] == "outside_observation_window"
        expected = uniform_grid(domain, spacing)
        assert np.array_equal(record["candidate_xy"], expected.candidate_xy)
        assert domain.contains(record["candidate_xy"]).all()
        assert np.isclose(record["teacher_prob"].sum(), 1)
        assert not record["teacher_prob"][~record["valid"]].any()
        for index in np.flatnonzero(record["valid"])[[0, -1]]:
            rendered = ds.rerender(scene_id, observation["window"], record["candidate_xy"][index])
            assert record["physical_cost"][index] == response_disagreement(rendered, observation["response"])
        report = inspect_teacher(ds, record)
        assert report["reference_cost"] == [0.0] * 10
        assert report["reference_domain_coverage_fraction"] == 1
        references_checked += 10
        print(f"PASS size={side}: {len(record['candidate_xy'])} grid points, "
              f"{report['zero_cost_candidates']} zero-cost points; ten references remain separate.", flush=True)

    view_id = representative_view_ids(ds, scene_id)[0]
    sample = ds.get_observation(scene_id, view_id)
    world = record
    parent = generate_uniform_teacher(ds, scene_id, view_id, spacing=int(sample["window"][2]) / 2,
                                      temperature=0.05, quadtree_prior=True)
    assert parent["metadata"]["settings"]["support"]["candidate_domain"] == "quadtree_parent_minus_window"
    evaluator = PhysicalEvaluator(ds, scene_id, view_id)
    truth = ds.get_scene(scene_id)["drivers"][0]
    first, cached = evaluator.evaluate(truth), evaluator.evaluate(truth)
    assert first.physical_cost == cached.physical_cost == 0
    assert first.num_physics_evaluations == 1 and cached.num_physics_evaluations == 0 and cached.cache_hit
    invalid = evaluator.evaluate([-1, 0])
    assert not invalid.valid and invalid.num_physics_evaluations == 0
    for candidate_view in range(len(ds.get_scene(scene_id)["local_views"])):
        observed = ds.get_observation(scene_id, candidate_view)
        pixels = np.argwhere(observed["obstacle"])
        if len(pixels):
            row, column = pixels[0]
            x, y, _ = observed["window"]
            obstacle_point = evaluator.evaluate([x + column + .5, y + row + .5])
            assert not obstacle_point.valid and obstacle_point.num_physics_evaluations == 0
            break
    with tempfile.TemporaryDirectory(prefix="rind-teacher-check-") as directory:
        path = save_teacher_record(world, Path(directory) / "teacher.npz")
        loaded = load_teacher_record(path)
        for key in ("candidate_xy", "valid", "physical_cost", "teacher_prob", "grid_index"):
            assert np.array_equal(world[key], loaded[key])
        assert loaded["metadata"] == world["metadata"]
    print(f"PASS: uniform teacher across six sizes, {references_checked} exact witnesses, "
          "world-domain baseline, optional parent prior, physical cache/counts, and portable records. "
          f"Smoke tau=.05; world spacing={spacing:g}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--data-only", action="store_true")
    modes.add_argument("--teacher", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument("--scene-id", type=int, default=0)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.teacher:
        if args.torch or args.workers:
            parser.error("--torch/--workers apply to --data-only")
        check_teacher(args.data_root, scene_id=args.scene_id)
    else:
        check_data(args.data_root, check_torch=args.torch, workers=args.workers)


if __name__ == "__main__":
    main()
