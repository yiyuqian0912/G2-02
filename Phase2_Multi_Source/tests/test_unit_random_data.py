"""Portable physical fixture for the new format and its installation path."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from rind_dataset.constants import OBSTACLE_RECTANGLE
from rind_dataset.render import render_global
from rind_phase2.data import Phase2Dataset
from rind_phase2.install_data import install_data, validate_installed, _valid_manifest
from rind_phase2.unit_schema import ARRAYS, array_specs


def fixture_windows(response, obstacle, sources, *, count_per_size=4):
    """Build physical fixture metadata independently of production checks."""
    free = ~obstacle
    values = np.where(free, response, 0).astype(np.int64)
    integrals = []
    for image in (free, values, values * values):
        integral = np.zeros((1025, 1025), dtype=np.int64)
        integral[1:, 1:] = image.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)
        integrals.append(integral)
    views, capacities = [], []
    for side in (64, 128, 256, 512):
        n, total, squared = [a[side:, side:] - a[:-side, side:] - a[side:, :-side] + a[:-side, :-side] for a in integrals]
        valid = (n > 1) & (n * squared > total * total)
        axis = np.arange(1024 - side + 1)
        for x, y in sources:
            valid &= ~(((axis <= x) & (x < axis + side))[None, :] &
                       ((axis <= y) & (y < axis + side))[:, None])
        origins = np.argwhere(valid)
        capacities.append(len(origins))
        assert len(origins) >= count_per_size
        selected = origins[np.linspace(0, len(origins) - 1, count_per_size, dtype=int)]
        views.extend((int(x), int(y), side) for y, x in selected)
    return np.asarray(views, dtype=np.uint16), np.asarray(capacities, dtype=np.uint32)


def native_fixture(root):
    root.mkdir()
    manifest = dict(schema_version=3, variant="multi-source-unit-random", num_scenes=2,
        max_drivers=4, driver_count_range=[1, 4], driver_strength_mode="fixed-one",
        view_sampling="random-informative", source_free=True, view_capacity=16,
        window_sizes=[64, 128, 256, 512], min_view_size=64, global_size=1024,
        candidate_capacity=10, candidate_layout="scene-source-sets", obstacle_capacity=1,
        response_type="additive-driver-intensity", bit_order="little", cell_sampling="cell-center", array_indexing="yx")
    (root / "manifest.json").write_text(json.dumps(manifest))
    arrays = {name: np.zeros(shape, dtype=dtype) for name, (dtype, shape) in array_specs(manifest).items()}
    for sid, count in enumerate((2, 3)):
        sources = np.array([[100.123456789, 200.987654321], [101.314159265, 800.271828182],
                            [102.141421356, 500.173205081]])[:count]
        params = np.zeros((1, 32));params[0, :5] = [512, 512, 70, 200, 0]
        types = np.array([OBSTACLE_RECTANGLE], dtype=np.uint8)
        fields = [render_global(source, params, obs_types=types, backend="numba") for source in sources]
        channels = np.array([field == 1 for field in fields])
        obstacle = fields[0] == 2
        response = channels.sum(axis=0, dtype=np.uint8)
        arrays["drivers"][sid, :count] = sources
        arrays["driver_counts"][sid] = count
        arrays["driver_strengths"][sid, :count] = 1
        arrays["obstacle_counts"][sid] = 1
        arrays["obstacle_types"][sid] = types
        arrays["obstacle_params"][sid] = params
        arrays["obstacle_bits"][sid] = np.packbits(obstacle.ravel(), bitorder="little")
        arrays["visibility_bits"][sid, :count] = np.packbits(channels.reshape(count, -1), axis=1, bitorder="little")
        views, capacities = fixture_windows(response, obstacle, sources)
        arrays["local_views"][sid] = views
        arrays["view_counts"][sid] = len(views)
        arrays["valid_origin_counts"][sid] = capacities
        candidate = sources.copy()
        for index in range(10):
            arrays["reference_sources"][sid, index, :count] = candidate
            candidate[0, 0] = np.nextafter(candidate[0, 0], np.inf)
    for name, values in arrays.items():
        np.save(root / f"{name}.npy", values)
    return root


def archive_fixture(root, output, *, corrupt=False):
    hashes = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for name in ("manifest.json", *(f"{name}.npy" for name in ARRAYS)):
            data = (root / name).read_bytes()
            relative = f"data/unit/{name}"
            with z.open(f"release/{relative}", "w", force_zip64=True) as f:
                f.write(data)
            digest = "0" * 64 if corrupt and name == "drivers.npy" else hashlib.sha256(data).hexdigest()
            hashes.append(f"{digest}  {relative}")
        z.writestr("release/SHA256SUMS", "\n".join(hashes) + "\n")
    return output


class UnitRandomDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="rind-unit-test-")
        cls.root = native_fixture(Path(cls.temp.name) / "native")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_observations_channels_and_implicit_reference_strengths(self):
        dataset = Phase2Dataset(self.root)
        self.assertEqual((dataset.num_scenes, len(dataset)), (2, 32))
        item = dataset[-1]
        self.assertEqual(set(item), {"scene_id", "view_id", "response", "obstacle", "window"})
        self.assertEqual(item["scene_id"], 1)
        channels = dataset.get_channels(1, 15)
        np.testing.assert_array_equal(channels.sum(axis=0), item["response"])
        self.assertEqual(channels.shape[0], 3)
        self.assertEqual(len(dataset.get_candidates(1, 15)), 10)
        self.assertTrue(all(np.all(candidate[:, 2] == 1) for candidate in dataset.get_candidates(1, 15)))

    def test_random_windows_have_no_fabricated_tree(self):
        dataset = Phase2Dataset(self.root)
        self.assertEqual(len(list(dataset.iter_windows(1))), 16)
        with self.assertRaisesRegex(ValueError, "no quadtree"):
            dataset.get_view_tree(1)

    def test_joint_teacher_uses_unit_prior_without_truth_or_reference_labels(self):
        from rind_phase2.teacher import generate_joint_teacher
        dataset = Phase2Dataset(self.root)

        class LabelsForbidden:
            root, manifest = dataset.root, dataset.manifest

            def get_observation(self, *args):
                return dataset.get_observation(*args)

            def get_scene(self, *args):
                scene = dataset.get_scene(*args)
                return {key: scene[key] for key in ("obstacle_types", "obstacle_params", "obstacle_count")}

            def get_region(self, *args):
                region = dataset.get_region(*args)
                return {"response": region["response"], "obstacle": region["obstacle"]}

            def rerender(self, *args, **kwargs):
                return dataset.rerender(*args, **kwargs)

            def get_candidates(self, *_):
                raise AssertionError("Truth and reference labels must not enter the teacher")

            get_channels = get_source_params = get_candidates

        record = generate_joint_teacher(LabelsForbidden(), 0, 0, spacing=512,
            temperature=.05, source_counts=[1, 2], samples_per_count=12, count_prior=[.5, .5],
            seed=11, backend="numba")
        self.assertEqual(record["metadata"]["settings"]["support"]["strength_mode"], "fixed-one")
        self.assertEqual(set(record["source_counts"].tolist()), {1, 2})
        for sources, count in zip(record["sources"], record["source_counts"]):
            self.assertTrue(np.all(sources[:count, 2] == 1))
        self.assertAlmostEqual(record["teacher_prob"].sum(), 1)

    def test_all_reference_sets_physically_reproduce_windows(self):
        dataset = Phase2Dataset(self.root)
        for sid in (0, 1):
            scene = dataset.get_scene(sid)
            vid = int(np.argmax(scene["local_views"][:, 2]))
            observed = dataset.get_observation(sid, vid)
            for sources in dataset.get_candidates(sid, vid):
                np.testing.assert_array_equal(dataset.rerender(sid, observed["window"], sources, backend="numba"), observed["response"])

    def test_zip64_install_cleanup_and_no_old_data_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            archive = archive_fixture(self.root, path / "release.zip")
            with redirect_stdout(io.StringIO()):
                install_data(archive, path / "installed")
            self.assertFalse(archive.exists())
            self.assertEqual(validate_installed(path / "installed")["num_views"], 32)
            manifest = json.loads((path / "installed/manifest.json").read_text())
            manifest["release_id"] = "different"
            (path / "installed/manifest.json").write_text(json.dumps(manifest))
            archive = archive_fixture(self.root, path / "new.zip")
            with self.assertRaisesRegex(ValueError, "--replace"):
                install_data(archive, path / "installed")
            self.assertTrue(archive.exists())

    def test_bad_checksum_retains_archive_and_previous_installation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            original = archive_fixture(self.root, path / "first.zip")
            with redirect_stdout(io.StringIO()):
                install_data(original, path / "installed")
            bad = archive_fixture(self.root, path / "bad.zip", corrupt=True)
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "checksum"):
                install_data(bad, path / "installed", replace=True)
            self.assertTrue(bad.exists())
            self.assertEqual(validate_installed(path / "installed")["num_scenes"], 2)
            self.assertFalse(list(path.glob(".rind-install-*")))

    def test_active_strength_not_silently_treated_as_random(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["driver_strength_mode"] = "uniform-0-1"
        with self.assertRaises(ValueError):
            _valid_manifest(manifest)

    def test_browser_uses_sampled_world_reference_scope_and_nonbinary_scale(self):
        try:
            from rind_phase2.browser import create_app
        except ImportError:
            self.skipTest("Browser extra is not installed")
        with create_app(self.root).test_client() as client:
            self.assertEqual(client.get("/api/manifest").get_json()["view_sampling"], "random-informative")
            scene = client.get("/api/scene/1").get_json()
            self.assertEqual(scene["response_scale"], 3)
            self.assertTrue(all(v["is_informative"] and v["is_mixed"] and
                                not v["is_all_response"] and not v["is_no_response"]
                                for v in scene["view_diagnostics"]))
            response = client.get("/api/scene/1/view/0/candidates")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["equivalence"], "sampled-world")
            self.assertEqual(len(response.get_json()["candidates"]), 10)
            self.assertEqual(client.get("/api/scene/1/view/0.png").status_code, 200)


if __name__ == "__main__":
    unittest.main()
