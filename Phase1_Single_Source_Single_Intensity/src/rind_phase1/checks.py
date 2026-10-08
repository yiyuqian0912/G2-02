"""Verify installed data access; research handoff checks remain to be added."""

import argparse
import pickle
import numpy as np

from rind_phase1.data import Phase1Dataset, SizeBucketBatchSampler, make_dataloader, split_scene_ids
from rind_phase1.install_data import validate_installed
from rind_phase1.physics import edge_cost, evaluate_candidate, evaluate_candidates


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


def check_physics(root=None):
    """Verify Pass 1 and Pass 2 physical supervision."""
    ds = Phase1Dataset(root)

    sample = ds[0]

    scene_id = int(
        sample["scene_id"]
    )

    view_id = int(
        sample["view_id"]
    )

    response = sample["response"]
    window = sample["window"]

    # Smoke-test values only.
    # Official experiment hyperparameters remain unresolved.
    boundary_lambda = 1.0
    boundary_sigma = 1.0

    references = ds.get_candidates(
        scene_id,
        view_id,
    )

    candidate_xy = np.stack(
        [
            reference[0, :2]
            for reference in references
        ]
    )

    # ---------------------------------------------------------------
    # Pass 1: response-only baseline
    # ---------------------------------------------------------------

    response_only = evaluate_candidates(
        ds,
        scene_id,
        response,
        window,
        candidate_xy,
        boundary_lambda=boundary_lambda,
        boundary_sigma=boundary_sigma,
        alpha=1.0,
        beta=0.0,
    )

    assert np.array_equal(
        response_only["candidate_xy"],
        candidate_xy,
    )

    assert response_only[
        "valid"
    ].all()

    assert np.allclose(
        response_only["L_resp"],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    assert np.allclose(
        response_only["physical_cost"],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    assert (
        response_only["L_edge"]
        is None
    )

    # ---------------------------------------------------------------
    # Pass 2: response + edge cost
    #
    # Every supplied compatible reference reproduces the exact same
    # response, so both L_resp and L_edge must be zero.
    # beta=1 is only a smoke-test value here.
    # ---------------------------------------------------------------

    response_plus_edge = evaluate_candidates(
        ds,
        scene_id,
        response,
        window,
        candidate_xy,
        boundary_lambda=boundary_lambda,
        boundary_sigma=boundary_sigma,
        alpha=1.0,
        beta=1.0,
    )

    assert response_plus_edge[
        "valid"
    ].all()

    assert np.allclose(
        response_plus_edge["L_resp"],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    assert np.allclose(
        response_plus_edge["L_edge"],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    assert np.allclose(
        response_plus_edge["physical_cost"],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )

    # ---------------------------------------------------------------
    # Synthetic edge sanity checks
    # ---------------------------------------------------------------

    observed = np.zeros(
        (8, 8),
        dtype=np.float32,
    )

    observed[:, :4] = 1.0

    identical = observed.copy()

    shifted = np.zeros(
        (8, 8),
        dtype=np.float32,
    )

    shifted[:, :5] = 1.0

    no_boundary = np.zeros(
        (8, 8),
        dtype=np.float32,
    )

    # Identical boundary geometry.
    assert edge_cost(
        observed,
        identical,
    ) == 0.0

    # Moving the boundary should create a positive edge cost.
    assert edge_cost(
        observed,
        shifted,
    ) > 0.0

    # A missing candidate boundary must be penalized.
    assert edge_cost(
        observed,
        no_boundary,
    ) > 0.0

    # An extra candidate boundary must also be penalized.
    assert edge_cost(
        no_boundary,
        observed,
    ) > 0.0

    # If neither response contains a boundary, they agree.
    assert edge_cost(
        no_boundary,
        no_boundary,
    ) == 0.0

    # ---------------------------------------------------------------
    # Generating-source rerender
    # ---------------------------------------------------------------

    generating = evaluate_candidate(
        ds,
        scene_id,
        response,
        window,
        candidate_xy[0],
        boundary_lambda=boundary_lambda,
        boundary_sigma=boundary_sigma,
        alpha=1.0,
        beta=1.0,
    )

    assert generating["valid"]

    assert np.array_equal(
        generating[
            "candidate_response"
        ],
        response,
    )

    assert (
        generating["L_resp"]
        == 0.0
    )

    assert (
        generating["L_edge"]
        == 0.0
    )

    assert (
        generating["physical_cost"]
        == 0.0
    )

    # ---------------------------------------------------------------
    # Invalid in-window candidate
    # ---------------------------------------------------------------

    x, y, size = map(
        float,
        window,
    )

    in_window_xy = np.array(
        [
            x + size / 2.0,
            y + size / 2.0,
        ],
        dtype=np.float64,
    )

    invalid = evaluate_candidate(
        ds,
        scene_id,
        response,
        window,
        in_window_xy,
        boundary_lambda=boundary_lambda,
        boundary_sigma=boundary_sigma,
        alpha=1.0,
        beta=1.0,
    )

    assert not invalid["valid"]

    assert (
        invalid["candidate_response"]
        is None
    )

    assert np.isinf(
        invalid["L_resp"]
    )

    assert np.isinf(
        invalid["L_edge"]
    )

    assert np.isinf(
        invalid["physical_cost"]
    )

    print(
        "PASS: physics Pass 1/2; response-only compatibility, "
        "symmetric edge cost, generating/reference reconstruction, "
        "candidate ordering, and invalid-candidate handling verified."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    check_data(args.data_root, check_torch=args.torch, workers=args.workers)
    if args.physics:
        check_physics(args.data_root)


if __name__ == "__main__":
    main()
