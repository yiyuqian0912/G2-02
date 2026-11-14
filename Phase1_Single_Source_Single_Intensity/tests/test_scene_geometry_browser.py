"""Check geometry conventions, additive exports and separation from inference."""
import base64
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'scene_geometry_browser', Path(__file__).resolve().parents[1] / 'tools/phase2_v4_browser.py')
browser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(browser)


def fixture_scene(scene_id=0):
    params = np.zeros((4, 32), dtype=np.float64)
    params[0, :5] = [4, 3, 2, 1, np.pi / 2]
    params[1, :5] = [6, 5, 1, 2, np.pi / 7]
    params[2, :6] = [0, 0, 2, 0, 0, 2]
    params[3, :9] = [4, 1, 1, 3, 1, 3, 3, 1, 3]
    return dict(scene_id=scene_id, drivers=np.array([[1.25, 2.5], [6.5, 7.125]]),
                driver_strengths=np.array([.6, .8]), obstacle_types=np.arange(4),
                obstacle_params=params, obstacle_group_ids=np.array([5, 5, 8, 9]),
                local_views=np.array([[0, 0, 2], [2, 2, 2]]))


class FakeDataset:
    manifest = {'global_size': 8}

    def __init__(self):
        self.region_calls = 0

    def get_scene(self, scene_id):
        return fixture_scene(scene_id)

    def get_region(self, scene_id, x, y, width, height):
        self.region_calls += 1
        assert (x, y, width, height) == (0, 0, 8, 8)
        # A bright overlap must remain 1.4, not OR visibility or clipped to 1.
        response = np.full((8, 8), 1.4, dtype=np.float64)
        obstacle = np.zeros((8, 8), dtype=bool)
        obstacle[0, 0] = True
        response[0, 0] = 0
        return {'response': response, 'obstacle': obstacle}


def explorer_fixture():
    explorer = browser.Explorer.__new__(browser.Explorer)
    explorer.dataset = FakeDataset()
    explorer.scene_cache = {}
    explorer.student_run = 'example'
    explorer.entries = [dict(index=0, scene_id=0, view_id=0, size=2),
                        dict(index=1, scene_id=0, view_id=1, size=2),
                        dict(index=2, scene_id=1, view_id=0, size=2)]
    # Deliberately no model, checkpoint, teacher directory or inference cache.
    return explorer


class SceneGeometryTests(unittest.TestCase):
    def test_original_precision_shape_layout_and_rotation(self):
        original = fixture_scene()
        before = copy.deepcopy(original)
        value = browser.scene_geometry(original, 8)
        np.testing.assert_allclose(value['obstacles'][0]['vertices'],
                                   [[5, 1], [5, 5], [3, 5], [3, 1]], atol=1e-14)
        self.assertEqual(value['obstacles'][1]['theta_radians'], float(np.pi / 7))
        self.assertEqual(value['obstacles'][1]['radii'], [1, 2])
        self.assertEqual(value['obstacles'][2]['vertices'], [[0, 0], [2, 0], [0, 2]])
        self.assertEqual(value['obstacles'][3]['vertices'], [[1, 1], [3, 1], [3, 3], [1, 3]])
        self.assertEqual(value['logical_obstacle_count'], 3)
        self.assertEqual(value['obstacle_params'], original['obstacle_params'].tolist())
        for key in ('obstacle_params', 'obstacle_types', 'drivers'):
            np.testing.assert_array_equal(original[key], before[key])

    def test_scene_cache_and_window_identity_do_not_require_inference(self):
        explorer = explorer_fixture()
        first, second = explorer.scene(0), explorer.scene(1)
        self.assertEqual(first['metadata']['window'], [0, 0, 2])
        self.assertEqual(second['metadata']['window'], [2, 2, 2])
        self.assertEqual(explorer.dataset.region_calls, 1)
        values = np.frombuffer(base64.b64decode(first['response']['data']), dtype='<f8').reshape(8, 8)
        self.assertEqual(values[1, 1], 1.4)
        self.assertEqual(values[0, 0], 0)
        for bad in (-1, 3):
            with self.assertRaises(ValueError):
                explorer.scene(bad)

    def test_collection_deduplicates_scenes_and_does_not_decode_full_fields(self):
        explorer = explorer_fixture()
        collection = explorer.geometry_collection()
        self.assertEqual((collection['window_count'], collection['scene_count']), (3, 2))
        self.assertEqual([x['scene_id'] for x in collection['scenes']], [0, 1])
        self.assertEqual(len(collection['scenes'][0]['browser_views']), 2)
        self.assertEqual(explorer.dataset.region_calls, 0)

    def test_scene_and_download_endpoints(self):
        client = browser.create_app(explorer_fixture(), Path('/tmp')).test_client()
        self.assertEqual(client.get('/api/scene?index=0').status_code, 200)
        self.assertEqual(client.get('/api/scene?index=-1').status_code, 400)
        self.assertEqual(client.get('/api/geometry?scope=unknown').status_code, 400)
        current = client.get('/api/geometry?index=1&download=1')
        self.assertEqual(current.json['scene_id'], 0)
        self.assertIn('rind-scene-0-geometry.json', current.headers['Content-Disposition'])
        all_scenes = client.get('/api/geometry?scope=all&download=1')
        self.assertEqual(all_scenes.json['scene_count'], 2)
        self.assertIn('attachment', all_scenes.headers['Content-Disposition'])


if __name__ == '__main__':
    unittest.main()
