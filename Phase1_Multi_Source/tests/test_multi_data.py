"""Small physical fixtures exercise format, student routing and ZIP cleanup."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import zipfile

import numpy as np

from rind_phase1_multi.checks import audit_metadata
from rind_phase1_multi.data import Phase1MultiDataset, SizeBucketBatchSampler, split_scene_ids
from rind_phase1_multi.install_data import ARRAYS, install_data, validate_installed
from rind_phase1_multi.physics import REFERENCE_ATOL, evaluate_sources, response_disagreement


def fixture(root):
    root.mkdir(parents=True)
    world, scenes, maximum, capacity, witnesses = 64, 2, 4, 20, 6
    manifest = {"schema_version": 3, "num_scenes": scenes, "global_size": world,
                "min_view_size": 16, "max_drivers": maximum, "view_capacity": capacity,
                "candidate_capacity": witnesses, "response_type": "additive-driver-intensity",
                "bit_order": "little", "cell_sampling": "cell-center", "array_indexing": "yx"}
    (root / "manifest.json").write_text(json.dumps(manifest))
    arrays = {
        "drivers": np.zeros((scenes, maximum, 2)),
        "driver_counts": np.array([1, 4], dtype=np.uint8),
        "driver_strengths": np.zeros((scenes, maximum)),
        "obstacle_counts": np.zeros(scenes, dtype=np.uint8),
        "obstacle_types": np.zeros((scenes, 1), dtype=np.uint8),
        "obstacle_params": np.zeros((scenes, 1, 32)),
        "obstacle_group_ids": np.zeros((scenes, 1), dtype=np.uint16),
        "obstacle_bits": np.zeros((scenes, world * world // 8), dtype=np.uint8),
        "visibility_bits": np.zeros((scenes, maximum, world * world // 8), dtype=np.uint8),
        "local_views": np.zeros((scenes, capacity, 3), dtype=np.uint16),
        "view_counts": np.zeros(scenes, dtype=np.uint16),
        "view_candidates": np.zeros((scenes, capacity, witnesses, maximum + 1, 3)),
        "candidate_counts": np.zeros((scenes, capacity, witnesses), dtype=np.uint8),
        "candidate_numbers": np.zeros((scenes, capacity), dtype=np.uint8),
    }
    for sid, count in enumerate((1, 4)):
        points = np.array([[8, 8], [56, 8], [8, 56], [56, 56]], dtype=np.float64)[:count]
        strengths = np.array([.2, .3, .4, .5])[:count]
        arrays["drivers"][sid, :count] = points
        arrays["driver_strengths"][sid, :count] = strengths
        arrays["visibility_bits"][sid, :count] = 255
        views = []

        def partition(x, y, side):
            inside = ((points >= [x, y]) & (points < [x + side, y + side])).all(axis=1)
            if not inside.any():
                views.append((x, y, side))
            elif side > 16:
                half = side // 2
                for dx, dy in ((0, 0), (half, 0), (0, half), (half, half)):
                    partition(x + dx, y + dy, half)

        partition(0, 0, world)
        arrays["local_views"][sid, :len(views)] = views
        arrays["view_counts"][sid] = len(views)
        true = np.column_stack((points, strengths))
        references = [true]
        if count > 1:
            references.append(true[::-1])
        index = int(np.argmax(strengths))
        for fraction in (.2, .4, .6, .8):
            split = np.vstack((true, true[index]))
            split[index, 2] = true[index, 2] * fraction
            split[-1, 2] = true[index, 2] - split[index, 2]
            references.append(split)
        for vid in range(len(views)):
            arrays["candidate_numbers"][sid, vid] = len(references)
            for cid, reference in enumerate(references):
                arrays["candidate_counts"][sid, vid, cid] = len(reference)
                arrays["view_candidates"][sid, vid, cid, :len(reference)] = reference
    for name, array in arrays.items():
        np.save(root / f"{name}.npy", array)
    return root


def release(root, path, *, corrupt=False):
    lines = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("manifest.json", *(f"{name}.npy" for name in ARRAYS)):
            contents = (root / name).read_bytes()
            digest = hashlib.sha256(contents).hexdigest()
            if corrupt and name == "drivers.npy":
                digest = "0" * 64
            member = f"data/multi/{name}"
            archive.writestr(member, contents)
            lines.append(f"{digest}  {member}\n")
        archive.writestr("SHA256SUMS", "".join(lines))
        archive.writestr("downloaded-code.py", "raise RuntimeError('Must not install this code')")
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name)
    return sidecar


class MultiDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rind-multi-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = fixture(Path(self.temp.name) / "fixture")
        self.ds = Phase1MultiDataset(self.root)

    def test_student_has_only_local_inputs_and_ids(self):
        sample = self.ds[0]
        self.assertEqual(set(sample), {"scene_id", "view_id", "response", "obstacle", "window"})
        self.assertEqual(sample["response"].dtype, np.float32)
        self.assertEqual(sample["obstacle"].dtype, np.dtype(bool))
        self.assertEqual(sample["response"].shape, sample["obstacle"].shape)
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(self.ds))[0]["response"], sample["response"])

    def test_exact_channels_add_without_clipping(self):
        channels = self.ds.get_channels(1, 0)
        self.assertEqual(channels.shape[0], 4)
        self.assertEqual(channels.dtype, np.float64)
        window = self.ds.get_observation(1, 0)["window"]
        raw = self.ds.get_region(1, *map(int, window))["response"]
        np.testing.assert_array_equal(channels.sum(axis=0), raw)
        self.assertGreater(float(raw.min()), 1)
        np.testing.assert_array_equal(raw.astype(np.float32), self.ds.get_observation(1, 0)["response"])

    def test_render_explicit_strengths_and_all_witnesses(self):
        for sid in range(2):
            sample = self.ds.get_observation(sid, 0)
            refs = self.ds.get_candidates(sid, 0)
            self.assertEqual(len(refs), 5 if sid == 0 else 6)
            self.assertEqual(len(refs[-1]), int(self.ds._base.driver_counts[sid]) + 1)
            for reference in refs:
                result = evaluate_sources(self.ds, sid, 0, reference, backend="reference")
                self.assertTrue(result["valid"])
                self.assertLessEqual(result["physical_cost"], REFERENCE_ATOL)
            with self.assertRaises(ValueError):
                self.ds.rerender(sid, sample["window"], refs[0][:, :2])
        response = self.ds.rerender(0, [16, 0, 16], [[8, 8, .7]], backend="reference")
        np.testing.assert_array_equal(response, np.full((16, 16), .7))

    def test_world_outside_window_has_no_parent_restriction(self):
        # Scene 0's first window is a small leaf; [56,56] lies outside its parent.
        self.assertTrue(evaluate_sources(self.ds, 0, 0, [[56, 56, .2]], backend="reference")["valid"])
        x, y, _ = self.ds.get_observation(0, 0)["window"]
        invalid = evaluate_sources(self.ds, 0, 0, [[x + .5, y + .5, .2]])
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["num_source_renders"], 0)
        for proposal in ([[-1, 0, .5]], [[56, 56, 2]], [[np.nan, 0, .5]]):
            self.assertFalse(evaluate_sources(self.ds, 0, 0, proposal)["valid"])

    def test_variable_sizes_use_mean_intensity_error(self):
        for side in (16, 32):
            self.assertAlmostEqual(response_disagreement(np.full((side, side), 1.4),
                                                        np.ones((side, side))), .4)
        with self.assertRaises(ValueError):
            response_disagreement(np.ones((2, 2)), np.ones((4, 4)))

    def test_subsets_size_buckets_and_tree(self):
        subset = Phase1MultiDataset(self.root, scene_ids=[1, 0])
        batches = list(SizeBucketBatchSampler(subset, 3))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(subset))))
        for batch in batches:
            self.assertEqual(len({int(subset[i]["window"][2]) for i in batch}), 1)
        for sid in range(2):
            self.assertEqual(len(list(self.ds.iter_view_leaves(sid))), int(self.ds._base.view_counts[sid]))
        report = audit_metadata(self.ds)
        self.assertEqual(report["all_scenes_audited"], 2)
        self.assertEqual(report["all_views_audited"], len(self.ds))
        split = split_scene_ids(10, ratios=(.8, .1, .1))
        self.assertEqual(len(set(sum(split.values(), []))), 10)
        with self.assertRaises(IndexError):
            Phase1MultiDataset(self.root, scene_ids=[0]).get_scene(1)

    def test_installer_deletes_only_successful_archive_and_sidecar(self):
        archive = Path(self.temp.name) / "multi.zip"
        sidecar = release(self.root, archive)
        destination = Path(self.temp.name) / "installed"
        with redirect_stdout(io.StringIO()):
            install_data(archive, destination)
        self.assertFalse(archive.exists())
        self.assertFalse(sidecar.exists())
        self.assertFalse((destination / "downloaded-code.py").exists())
        self.assertTrue((destination / "installation.json").is_file())
        self.assertEqual(validate_installed(destination)["num_scenes"], 2)
        release(self.root, archive)
        with redirect_stdout(io.StringIO()):
            install_data(archive, destination)
        self.assertTrue(archive.exists(), "Existing installations must retain a newly supplied ZIP")

    def test_keep_archive_and_failed_checksum_retention(self):
        archive = Path(self.temp.name) / "keep.zip"
        sidecar = release(self.root, archive)
        with redirect_stdout(io.StringIO()):
            install_data(archive, Path(self.temp.name) / "kept", keep_archive=True)
        self.assertTrue(archive.exists() and sidecar.exists())
        bad = Path(self.temp.name) / "bad.zip"
        bad_sidecar = release(self.root, bad, corrupt=True)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "checksum mismatch"):
            install_data(bad, Path(self.temp.name) / "failed")
        self.assertTrue(bad.exists() and bad_sidecar.exists())
        self.assertFalse((Path(self.temp.name) / "failed").exists())
        self.assertFalse(list(Path(self.temp.name).glob(".rind-install-*")))

    def test_single_source_release_rejected(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest.update(max_drivers=1, driver_strength_mode="fixed-one", candidate_layout="single-position-list")
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            Phase1MultiDataset(self.root)
        with self.assertRaises(ValueError):
            validate_installed(self.root)


if __name__ == "__main__":
    unittest.main()
