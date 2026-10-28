import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from dataset_update.dataset_union import ComponentDataset, open_dataset
from dataset_update.manage_dataset import build_union, install_archive, randomize_strengths, sha256_file


def make_dataset(root: Path, geometry_ids, unit=False):
    root.mkdir(parents=True)
    count, world, max_sources, views = len(geometry_ids), 8, 2, 1
    manifest = {
        "schema_version": 3, "num_scenes": count, "global_size": world,
        "max_drivers": max_sources, "view_capacity": views, "candidate_capacity": 1,
        "driver_strength_mode": "fixed-one" if unit else "uniform-0-1",
        "candidate_layout": "scene-source-sets" if unit else "padded-source-sets",
        "response_type": "additive-driver-intensity", "bit_order": "little",
        "cell_sampling": "cell-center", "array_indexing": "yx",
        "window_sizes": [4],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    arrays = {
        "drivers": np.zeros((count, max_sources, 2), dtype=np.float64),
        "driver_counts": np.ones(count, dtype=np.uint8),
        "driver_strengths": np.zeros((count, max_sources), dtype=np.float64),
        "obstacle_counts": np.ones(count, dtype=np.uint8),
        "obstacle_types": np.zeros((count, 1), dtype=np.uint8),
        "obstacle_params": np.zeros((count, 1, 32), dtype=np.float64),
        "obstacle_group_ids": np.zeros((count, 1), dtype=np.uint16),
        "obstacle_bits": np.zeros((count, 8), dtype=np.uint8),
        "visibility_bits": np.zeros((count, max_sources, 8), dtype=np.uint8),
        "local_views": np.zeros((count, views, 3), dtype=np.uint16),
        "view_counts": np.ones(count, dtype=np.uint16),
    }
    arrays["driver_strengths"][:, 0] = 1.0 if unit else 0.3
    arrays["drivers"][:, 0] = [7.0, 7.0]
    arrays["local_views"][:, 0] = [0, 0, 4]
    arrays["obstacle_params"][:, 0, 0] = geometry_ids
    visible = np.zeros((8, 8), dtype=bool)
    visible[:, 2:] = True
    arrays["visibility_bits"][:, 0] = np.packbits(visible.ravel(), bitorder="little")
    if unit:
        arrays["reference_sources"] = np.zeros((count, 1, max_sources, 2), dtype=np.float64)
        arrays["reference_sources"][:, 0] = arrays["drivers"]
        arrays["valid_origin_counts"] = np.ones((count, 1), dtype=np.uint32)
    else:
        arrays["view_candidates"] = np.zeros((count, views, 1, max_sources + 1, 3), dtype=np.float64)
        arrays["candidate_counts"] = np.ones((count, views, 1), dtype=np.uint8)
        arrays["candidate_numbers"] = np.ones((count, views), dtype=np.uint8)
    for name, value in arrays.items():
        np.save(root / f"{name}.npy", value)


class DatasetUnionTests(unittest.TestCase):
    def test_random_strength_overlay_exact_addition_and_original_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit, derived = root / "unit", root / "derived"
            make_dataset(unit, [1.0, 2.0], unit=True)
            original_hash = sha256_file(unit / "driver_strengths.npy")
            randomize_strengths(unit, derived, seed=123)
            value = open_dataset(derived)
            active = np.arange(2)[None] < value.driver_counts[:, None]
            self.assertTrue(np.all(value.driver_strengths[active] > 0))
            self.assertTrue(np.all(value.driver_strengths[active] < 1))
            self.assertTrue(np.all(value.driver_strengths[~active] == 0))
            original = ComponentDataset(unit)
            self.assertTrue(np.array_equal(value.get_channels(0, 0),
                original.get_channels(0, 0) * value.get_source_params(0)[:, 2, None, None]))
            self.assertTrue(np.array_equal(value.get_region(0, 0, 0, 4)["response"],
                                           value.get_channels(0, 0).sum(axis=0)))
            self.assertEqual(original_hash, sha256_file(unit / "driver_strengths.npy"))
            self.assertEqual(value.geometry_group(0), original.geometry_group(0))
            self.assertEqual(value.manifest["reference_policy"], "generating-configuration-only")
            self.assertTrue(np.array_equal(value.get_candidates(0, 0)[0], value.get_source_params(0)))
            first = np.array(value.driver_strengths)
            randomize_strengths(unit, derived, seed=123)
            self.assertTrue(np.array_equal(first, open_dataset(derived).driver_strengths))
            with self.assertRaisesRegex(ValueError, "different seed/base"):
                randomize_strengths(unit, derived, seed=124)

    def test_random_overlay_preserves_duplicate_geometry_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, unit, derived, teacher = (root / name for name in ("old", "unit", "derived", "teacher"))
            make_dataset(old, [1.0, 2.0])
            make_dataset(unit, [1.0, 3.0], unit=True)
            randomize_strengths(unit, derived, seed=99)
            teacher.mkdir()
            (teacher / "protocol.json").write_text(json.dumps({
                "scene_splits": {"train": [0], "validation": [1], "test": []},
            }))
            build_union(old, derived, root / "union", teacher)
            value = open_dataset(root / "union")
            self.assertIn(2, value.scene_splits["train"])
            self.assertEqual(value.source_metadata(2)["driver_strength_mode"], "uniform-0-1")
            self.assertEqual(value.source_metadata(2)["strength_seed"], 99)

    def test_preserves_old_ids_strength_variants_and_geometry_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new, union, teacher = (root / name for name in ("old", "new", "union", "teacher"))
            make_dataset(old, [1.0, 2.0])
            make_dataset(new, [1.0, 2.0, 3.0], unit=True)
            teacher.mkdir()
            (teacher / "protocol.json").write_text(json.dumps({
                "scene_splits": {"train": [0], "validation": [1], "test": []},
            }))
            build_union(old, new, union, teacher)
            dataset = open_dataset(union)
            self.assertEqual(dataset.num_scenes, 5)
            self.assertEqual(len(dataset), 5)
            self.assertIn(0, dataset.scene_splits["train"])
            self.assertIn(2, dataset.scene_splits["train"])
            self.assertIn(1, dataset.scene_splits["validation"])
            self.assertIn(3, dataset.scene_splits["validation"])
            self.assertEqual(dataset.geometry_group(0), dataset.geometry_group(2))
            self.assertAlmostEqual(dataset.get_observation(0, 0)["response"].max(), 0.3, places=6)
            self.assertEqual(dataset.get_observation(2, 0)["response"].max(), 1.0)
            self.assertEqual(dataset.get_observation(2, 0)["scene_id"], 2)
            self.assertEqual(dataset.get_channels(2, 0).dtype, np.float64)
            self.assertEqual(list(dataset.iter_scene_views(4)), [(0, 4)])
            self.assertEqual(int(dataset._base.view_counts[3]), 1)
            self.assertEqual(dataset._base.local_views[3, 0, 2], 4)
            # Repeating exactly the same operation is idempotent.
            build_union(old, new, union, teacher)

    def test_rejects_old_geometry_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new, teacher = (root / name for name in ("old", "new", "teacher"))
            make_dataset(old, [1.0, 1.0])
            make_dataset(new, [3.0], unit=True)
            teacher.mkdir()
            (teacher / "protocol.json").write_text(json.dumps({
                "scene_splits": {"train": [0], "validation": [1], "test": []},
            }))
            with self.assertRaisesRegex(ValueError, "leaks an identical geometry"):
                build_union(old, new, root / "union", teacher)

    def test_installs_verified_unit_release_without_deleting_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "source"
            make_dataset(data, [1.0], unit=True)
            archive = root / "release.zip"
            checksums = []
            with zipfile.ZipFile(archive, "w") as stream:
                for file in sorted(data.iterdir()):
                    content = file.read_bytes()
                    relative = f"data/unit/{file.name}"
                    stream.writestr(f"release/{relative}", content)
                    checksums.append(f"{hashlib.sha256(content).hexdigest()}  {relative}")
                stream.writestr("release/SHA256SUMS", "\n".join(checksums))
            result = install_archive(archive, root / "installed")
            self.assertEqual(result["num_scenes"], 1)
            self.assertTrue(archive.exists())
            self.assertEqual(ComponentDataset(root / "installed").get_source_params(0)[0, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
