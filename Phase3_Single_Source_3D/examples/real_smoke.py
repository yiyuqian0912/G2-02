"""Bounded real-data acceptance: nine families × three sizes, no hidden labels."""

import argparse, json, resource, time
from pathlib import Path
import numpy as np
from rind_phase3 import RIND3D
from rind_phase3.oracle import PhysicalOracle
from rind_phase3.physics import sample_probes, SparsePhysicalCost, full_window_cost
from rind_phase3.teacher import OctreeTeacher, VariationRefinement
from rind3d.geometry import load_geometry
from rind.contracts import Leaf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--reference-probes", type=int, default=8)
    a = p.parse_args()
    if not 1 <= a.reference_probes <= 32:
        p.error("--reference-probes must be in [1,32]")
    if a.output.exists():
        raise FileExistsError(a.output)
    started = time.perf_counter()
    cpu = time.process_time()
    reports = []
    selected = {}
    rng = np.random.default_rng(20261102)
    with RIND3D(a.root, split=None) as data:
        pool = {}
        for i, row in enumerate(data._reader.rows):
            entries = pool.setdefault(row["families"][0], [])
            if len(entries) < 8:
                bundle = load_geometry(
                    data.teacher_scene(row["scene_id"]).geometry_path
                )
                complexity = sum(isinstance(x, Leaf) for x in bundle.references)
                entries.append((complexity, i, row))
        # A bounded correctness smoke, not a production throughput benchmark.
        # Avoid paying arbitrary high-complexity CGAL union initialization here.
        for family, entries in pool.items():
            _, i, row = min(entries, key=lambda entry: (entry[0], entry[1]))
            selected[family] = (i, row)
        for family, (scene_index, row) in sorted(selected.items()):
            print("Starting family:", family, "scene ordinal:", scene_index, flush=True)
            with (
                PhysicalOracle(data.teacher_scene(row["scene_id"])) as native,
                PhysicalOracle(
                    data.teacher_scene(row["scene_id"]), backend="cpu_reference"
                ) as canonical,
            ):
                sources = rng.uniform(0.01, 0.99, (4, 3))
                for side, target in ((64, 0.1), (128, 0.5), (256, 0.9)):
                    candidates = (
                        range(0, 85)
                        if side == 64
                        else range(85, 170)
                        if side == 128
                        else range(170, 256)
                    )
                    # Metadata-only ratio selection of one descriptive window per size.
                    choices = []
                    for offset in list(candidates)[::8]:
                        idx = scene_index * 256 + offset
                        counts = data.statistics(idx)
                        choices.append(
                            (abs(counts.lit_ratio - target), idx, counts.lit_ratio)
                        )
                    _, index, ratio = min(choices)
                    observation = data[index]
                    probes = sample_probes(observation, count=32, seed=index)
                    costs, result = SparsePhysicalCost(native, probes).evaluate(sources)
                    print("Native probes complete:", family, side, flush=True)
                    reference = canonical.evaluate_points(
                        sources,
                        probes.world_xyz[: a.reference_probes],
                        window=probes.window,
                    )
                    np.testing.assert_array_equal(
                        result.response[:, : a.reference_probes], reference.response
                    )
                    np.testing.assert_array_equal(
                        result.solid[: a.reference_probes], reference.solid
                    )
                    print("Canonical probes agree:", family, side, flush=True)
                    entry = {
                        "family": family,
                        "side": side,
                        "window_id": observation["window_id"],
                        "lit_ratio": ratio,
                        "valid_sources": int(result.validity.valid.sum()),
                        "candidate_attempts": len(sources),
                        "evaluated_pairs": result.evaluated_pairs,
                        "native_seconds": result.wall_seconds,
                        "canonical_seconds": reference.wall_seconds,
                        "canonical_compared_receivers": min(
                            a.reference_probes, len(probes.world_xyz)
                        ),
                        "canonical_compared_pairs": reference.evaluated_pairs,
                        "canonical_geometry_initialization_seconds": canonical.context.canonical_init_seconds,
                        "all_point_labels_equal": True,
                    }
                    if family == sorted(selected)[0]:
                        valid = np.flatnonzero(result.validity.valid)
                        if not len(valid):
                            raise RuntimeError("No valid deterministic smoke source")
                        full = full_window_cost(native, observation, sources[valid[0]])
                        print("CPU local raster complete:", family, side, flush=True)
                        entry.update(
                            full_cost=full["cost"],
                            full_window_voxels=side**3,
                            full_raster_seconds=full["response"].wall_seconds,
                        )
                        if a.cuda:
                            with PhysicalOracle(
                                data.teacher_scene(row["scene_id"]),
                                raster_backend="cuda_certified",
                            ) as gpu:
                                other = gpu.rerender_window(
                                    sources[valid[0]], probes.window
                                )
                                np.testing.assert_array_equal(
                                    other.response, full["response"].response
                                )
                                np.testing.assert_array_equal(
                                    other.solid, full["response"].solid
                                )
                                entry.update(
                                    cuda_full_raster_seconds=other.wall_seconds,
                                    cuda_cpu_both_fields_equal=True,
                                )
                    reports.append(entry)
            print("Accepted oracle comparisons:", family, flush=True)
        observation = data[0]
        probes = sample_probes(observation, count=64, seed=0)
        with PhysicalOracle(data.teacher_scene(observation["scene_id"])) as oracle:
            tree = OctreeTeacher(
                SparsePhysicalCost(oracle, probes),
                refinement=VariationRefinement(),
                max_candidates=64,
                max_pairs=4096,
            ).run(1)
        assert (
            tree["cells_visited"] == 16
            and tree["cells_refined"] == 1
            and tree["physical_source_receiver_pairs"] <= 4096
        )
    result = {
        "complete": True,
        "dataset_name": "RIND3D",
        "real_observations": reports,
        "families": len(selected),
        "scene_selection": "minimum component count among first eight scenes of each family; ties by ordinal; correctness smoke only",
        "sizes": [64, 128, 256],
        "no_hidden_sources_or_reference_pools_read": True,
        "oracle_scope": "certified point labels; full-window mismatch on three selected windows; no equivalent-source search/certification",
        "octree": tree,
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu,
        "peak_self_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("real_observations", "octree")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
