"""Full structural and sampled physical audit for informative unit-source data."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from rind_phase2.data import Phase2Dataset
from rind_phase2.window_checks import response_integrals, validate_windows
from rind_phase2.unit_schema import validate_arrays
from rind_dataset.geometry import is_driver_valid_mixed


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def audit(root, *, physical_samples=24, progress_file=None, write_reports=True,
          scene_range=None, check_layout=True):
    from numba import set_num_threads
    set_num_threads(1)
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    ids = range(*(scene_range or (0, manifest["num_scenes"])))
    if check_layout:
        validate_arrays(root)
    dataset = Phase2Dataset(root, scene_ids=list(ids))
    summary = dict(num_scenes=len(ids), num_views=len(dataset), references_per_view=manifest["candidate_capacity"])
    base = dataset._base
    world = base.global_size
    selected = set(np.random.default_rng(42).choice(base.num_scenes,
        min(physical_samples, base.num_scenes), replace=False).tolist())
    histogram, source_histogram = Counter(), Counter()
    size_availability = np.zeros(4, dtype=np.int64)
    valid_counts = np.load(root / "valid_origin_counts.npy", mmap_mode="r")
    reference_count = base.manifest["candidate_capacity"]
    physical_references = 0
    started = time.time()
    for scene_id in ids:
        scene = dataset.get_scene(scene_id)
        sources = scene["drivers"]
        source_histogram[str(len(sources))] += 1
        for source in sources:
            assert is_driver_valid_mixed(source, scene["obstacle_types"], scene["obstacle_params"],
                                         scene["obstacle_count"], domain_size=world), scene_id
        bits = base.visibility_bits[scene_id, :len(sources)]
        assert not np.any(bits & base.obstacle_bits[scene_id][None, :]), scene_id
        channels = np.unpackbits(bits, axis=1, bitorder="little")[:, :world * world].reshape(-1, world, world)
        obstacle = np.unpackbits(base.obstacle_bits[scene_id], bitorder="little")[:world * world].reshape(world, world).astype(bool)
        response = channels.sum(axis=0, dtype=np.uint8)
        views = scene["local_views"]
        assert len(np.unique(views, axis=0)) == len(views), scene_id
        assert validate_windows(*response_integrals(response, obstacle), sources, views), scene_id
        sizes, counts = np.unique(views[:, 2], return_counts=True)
        histogram.update(dict(zip(map(str, sizes.tolist()), map(int, counts))))
        size_availability += valid_counts[scene_id] > 0
        for index, side in enumerate(base.manifest["window_sizes"]):
            assert np.count_nonzero(views[:, 2] == side) <= valid_counts[scene_id, index], scene_id
        references = base.reference_sources[scene_id, :, :len(sources)]
        assert len(np.unique(references.reshape(reference_count, -1), axis=0)) == reference_count, scene_id
        assert np.array_equal(references[0], sources), scene_id
        cell = np.minimum(np.floor(sources), world - 1)
        assert np.all(references >= cell[None]) and np.all(references < (cell + 1)[None]), scene_id
        if scene_id in selected:
            original = dataset.rerender(scene_id, np.array([0, 0, world]), dataset.get_source_params(scene_id),
                                        backend="numba", return_channels=True)
            assert np.array_equal(original["obstacle"], obstacle), scene_id
            assert np.array_equal(original["channels"], channels), scene_id
            for reference in dataset.get_candidates(scene_id, 0):
                rendered = dataset.rerender(scene_id, np.array([0, 0, world]), reference, backend="numba")
                assert np.array_equal(rendered, response), scene_id
                physical_references += 1
            # Region and full-world render paths must agree at both extremes.
            for view_id in {int(np.argmin(views[:, 2])), int(np.argmax(views[:, 2]))}:
                observation = dataset.get_observation(scene_id, view_id)
                assert np.array_equal(dataset.rerender(scene_id, observation["window"],
                    dataset.get_source_params(scene_id), backend="numba"), observation["response"])
        if scene_range is None and ((scene_id + 1) % 1000 == 0 or scene_id + 1 == base.num_scenes):
            progress = dict(scenes_checked=scene_id + 1, total_scenes=base.num_scenes,
                            elapsed_seconds=time.time() - started)
            if progress_file:
                write_json(progress_file, progress)
            print(f"Validated {scene_id + 1}/{base.num_scenes} scenes", flush=True)
    stats = dict(scene_count=len(ids), total_views=len(dataset),
                 views_per_scene=base.manifest["view_capacity"], view_size_histogram=dict(histogram),
                 source_count_histogram=dict(source_histogram),
                 scenes_with_valid_origins_by_size=dict(zip(map(str, base.manifest["window_sizes"]), size_availability.tolist())),
                 constant_windows=0, source_containing_windows=0,
                 references_per_view=reference_count,
                 reference_storage="one exact full-world source-set list per scene, shared across all windows")
    report = dict(complete=True, **summary, all_windows_informative=True, all_windows_source_free=True,
                  all_windows_unique_within_scene=True, all_sources_unit_strength=True,
                  physical_scene_ids=sorted(selected.intersection(ids)), physical_references_checked=physical_references,
                  sampled_physical_disagreement=0,
                  elapsed_seconds=time.time() - started)
    if write_reports:
        write_json(root / "generation_stats.json", stats)
        write_json(root / "validation_report.json", report)
    if scene_range is not None:
        report["statistics"] = stats
    return report


def _audit_part(task):
    root, bounds, physical_samples = task
    return audit(root, physical_samples=physical_samples, scene_range=bounds,
                 write_reports=False, check_layout=False)


def parallel_audit(root, *, workers, physical_samples=24, progress_file=None, chunk_size=1000):
    """Run the identical per-scene assertions on disjoint ranges, then merge."""
    root = Path(root)
    summary = validate_arrays(root)
    chunks = [(a, min(a + chunk_size, summary["num_scenes"]))
              for a in range(0, summary["num_scenes"], chunk_size)]
    histogram, source_histogram, availability = Counter(), Counter(), Counter()
    physical_ids, references, checked = [], 0, 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=min(workers, len(chunks)),
            mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_audit_part, (str(root), bounds, physical_samples)): bounds for bounds in chunks}
        try:
            for future in as_completed(futures):
                part = future.result()
                start, stop = futures[future]
                assert part["num_scenes"] == stop - start
                checked += stop - start
                statistics = part.pop("statistics")
                histogram.update(statistics["view_size_histogram"])
                source_histogram.update(statistics["source_count_histogram"])
                availability.update(statistics["scenes_with_valid_origins_by_size"])
                physical_ids.extend(part["physical_scene_ids"])
                references += part["physical_references_checked"]
                progress = dict(scenes_checked=checked, total_scenes=summary["num_scenes"],
                                workers=workers, elapsed_seconds=time.time() - started)
                if progress_file:
                    write_json(progress_file, progress)
                print(f"Validated {checked}/{summary['num_scenes']} scenes", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    assert checked == summary["num_scenes"]
    assert len(physical_ids) == len(set(physical_ids)) == min(physical_samples, summary["num_scenes"])
    manifest = json.loads((root / "manifest.json").read_text())
    statistics = dict(scene_count=checked, total_views=summary["num_views"],
        views_per_scene=manifest["view_capacity"], view_size_histogram=dict(sorted(histogram.items(), key=lambda row: int(row[0]))),
        source_count_histogram=dict(sorted(source_histogram.items())),
        scenes_with_valid_origins_by_size=dict(sorted(availability.items(), key=lambda row: int(row[0]))),
        constant_windows=0, source_containing_windows=0, references_per_view=manifest["candidate_capacity"],
        reference_storage="one exact full-world source-set list per scene, shared across all windows")
    report = dict(complete=True, **summary, all_windows_informative=True, all_windows_source_free=True,
        all_windows_unique_within_scene=True, all_sources_unit_strength=True,
        physical_scene_ids=sorted(physical_ids), physical_references_checked=references,
        sampled_physical_disagreement=0,
        audit_workers=workers, elapsed_seconds=time.time()-started)
    write_json(root / "generation_stats.json", statistics)
    write_json(root / "validation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--physical-samples", type=int, default=24)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if min(args.physical_samples, args.workers) <= 0:
        parser.error("physical-samples and workers must be positive")
    if args.workers > 1:
        report = parallel_audit(args.root, workers=args.workers, physical_samples=args.physical_samples,
                                progress_file=args.progress_file)
    else:
        report = audit(args.root, physical_samples=args.physical_samples, progress_file=args.progress_file)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
