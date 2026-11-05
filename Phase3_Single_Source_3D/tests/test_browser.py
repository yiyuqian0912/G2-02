"""Display correctness, source isolation and read-only viewer regression checks."""

from http.server import ThreadingHTTPServer
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import zlib

import numpy as np
import pytest

from rind_phase3.browser import ViewerStore, display_components, make_handler, png_labels, private_source, sampled_volume, stored_plane
from rind3d_reader.durable import read_sealed, seal


class PatternPack:
    def read_brick_packed(self, xyz):
        x, y, z = (np.arange(64) + v * 64 for v in xyz)
        bits = (x[None, None, :] + 2 * y[None, :, None] + 3 * z[:, None, None]) % 7 < 3
        return np.packbits(bits, axis=-1, bitorder="little").tobytes()


class VolumePatternPack:
    shape = (1024,) * 3

    def __init__(self, solid=False):
        z, y, x = np.indices((64,) * 3)
        self.tiles = []
        self.calls = set()
        for phase in range(7):
            values = (x + 2*y + 3*z + phase) % 7
            bits = values == 2 if solid else values >= 3
            self.tiles.append(np.packbits(bits, axis=-1, bitorder='little').tobytes())

    def read_brick_packed(self, xyz):
        self.calls.add(xyz)
        bx, by, bz = xyz
        return self.tiles[(13*bx + by + 5*bz) % 7]


@pytest.mark.parametrize('resolution', [64, 128, 256])
def test_volume_exact_nearest_samples_zyx_and_solid_separation(resolution):
    response, solid = VolumePatternPack(), VolumePatternPack(solid=True)
    actual = sampled_volume(response, solid, resolution)
    assert actual.shape == (resolution,)*3 and actual.dtype == np.uint8
    assert actual.flags.c_contiguous and set(np.unique(actual)) == {0, 1, 2}
    assert len(response.calls) == len(solid.calls) == 4096
    # Independent XYZ point formula checks unaligned samples across all bricks.
    xyz = np.random.default_rng(981).integers(0, resolution, size=(4096, 3))
    xyz = np.vstack((xyz, [[0,0,0], [resolution-1]*3]))
    original = xyz*(1024//resolution) + (512//resolution)
    local, brick = original % 64, original // 64
    values = (local @ [1,2,3] + brick @ [13,1,5]) % 7
    expected = np.where(values == 2, 2, values >= 3)
    np.testing.assert_array_equal(actual[xyz[:,2], xyz[:,1], xyz[:,0]], expected)
    with pytest.raises(ValueError):
        sampled_volume(response, solid, 110)


def source_fixture(tmp_path):
    """Entirely synthetic coordinates and file contents, never real labels."""
    sid, gid = 'a'*64, 'b'*64
    campaign = tmp_path/'campaign'; private = campaign/'scenes'/sid
    private.mkdir(parents=True); public = tmp_path/'public'; public.mkdir()
    position = [.125, .25, .75]
    driver = struct.pack('<3d', *position)
    (private/'driver.f64').write_bytes(driver)
    (private/'sources.private.json').write_text(json.dumps({'truth':{
        'canonical_scene_id':sid, 'position':position}}))
    hashes = {}
    for name in ('response.rindb', 'solid.rindb', 'geometry.npz', 'window_records.bin'):
        (public/name).write_bytes(('synthetic '+name).encode())
        hashes[name] = hashlib.sha256((public/name).read_bytes()).hexdigest()
    metadata = {'scene_id':sid, 'geometry_id':gid, 'hashes':hashes,
                'source_id':hashlib.sha256(bytes.fromhex(gid)+driver).hexdigest()}
    (private/'verification.json').write_text(json.dumps(metadata))
    files = {name:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
             for name in ('driver.f64','sources.private.json','verification.json')
             for path in [private/name]}
    files.update({name:{'sha256':h} for name, h in hashes.items()})
    payload = {'scene_id':sid,'geometry_id':gid,'status':'READY_LITE','files':files}
    (private/'scene.seal.json').write_text(json.dumps(seal(payload)))
    return campaign, public, sid, gid, position, private


def test_private_source_requires_exact_original_identity(tmp_path):
    campaign, public, sid, gid, position, _ = source_fixture(tmp_path)
    assert private_source(campaign, public, sid, gid) == position
    with pytest.raises(ValueError, match='identity mismatch'):
        private_source(campaign, public, sid, 'c'*64)


@pytest.mark.parametrize('changed', ['seal','driver','source_id','coordinates','field','metadata_hash'])
def test_private_source_rejects_corrupted_and_mismatched_records(tmp_path, changed):
    campaign, public, sid, gid, _, private = source_fixture(tmp_path)
    payload = read_sealed(private/'scene.seal.json')
    if changed == 'seal':
        record = json.loads((private/'scene.seal.json').read_text()); record['payload']['status'] = 'READY'
        (private/'scene.seal.json').write_text(json.dumps(record))
    elif changed == 'field':
        (public/'response.rindb').write_bytes(b'wrong public response')
    else:
        name = 'driver.f64' if changed == 'driver' else 'sources.private.json' if changed == 'coordinates' else 'verification.json'
        path = private/name
        if changed == 'driver':
            path.write_bytes(struct.pack('<3d', .25, .25, .75))
        else:
            record = json.loads(path.read_text())
            if changed == 'coordinates': record['truth']['position'][0] = .5
            elif changed == 'metadata_hash': record['hashes']['solid.rindb'] = '0'*64
            else: record['source_id'] = '0'*64
            path.write_text(json.dumps(record))
        # Resealing cannot conceal a cross-record or scientific-file mismatch.
        payload['files'][name]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (private/'scene.seal.json').write_text(json.dumps(seal(payload)))
    with pytest.raises(ValueError):
        private_source(campaign, public, sid, gid)


def test_private_source_cli_refuses_nonloopback_host():
    result = subprocess.run([sys.executable, '-m', 'rind_phase3.browser', '--private-root', '/unused',
                             '--host', '0.0.0.0'], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and 'loopback-only' in result.stderr


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("index", [0, 63, 64, 1023])
def test_plane_orientation_and_brick_boundaries(axis, index):
    row, col = np.indices((1024, 1024))
    expected = ((index + 2 * col + 3 * row) if axis == 0 else
                (col + 2 * index + 3 * row) if axis == 1 else (col + 2 * row + 3 * index)) % 7 < 3
    np.testing.assert_array_equal(stored_plane(PatternPack(), axis, index), expected)


def test_png_orientation_and_lossless_classes():
    labels = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.uint8)
    value = png_labels(labels)
    assert value[:8] == b"\x89PNG\r\n\x1a\n"
    pos, compressed = 8, b""
    while pos < len(value):
        size = struct.unpack(">I", value[pos:pos + 4])[0]
        kind, raw = value[pos + 4:pos + 8], value[pos + 8:pos + 8 + size]
        assert zlib.crc32(kind + raw) == struct.unpack(">I", value[pos + 8 + size:pos + 12 + size])[0]
        if kind == b"IDAT":
            compressed += raw
        pos += size + 12
    rows = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(2, 4)
    assert (rows[:, 0] == 0).all()
    np.testing.assert_array_equal(rows[:, 1:], labels[::-1])


def box_arrays():
    return {
        "geometry/dimension": np.array([3]), "geometry/root_kind": np.array([1]),
        "geometry/root_ref": np.array([0]), "geometry_ref/ref_kind": np.array([0]),
        "geometry_ref/codec_id": np.array([1]), "geometry_ref/codec_row": np.array([0]),
        "geometry_codecs/affine_box/center": np.array([[.5, .5, .5]]),
        "geometry_codecs/affine_box/matrix": np.array([np.diag([.1, .2, .3])]),
    }


def test_display_mesh_is_only_supported_canonical_convex_geometry():
    components = display_components(box_arrays())
    assert len(components) == 1
    v = np.array(components[0]["triangles"]).reshape(-1, 3)
    np.testing.assert_allclose(v.min(0), [.4, .3, .2])
    np.testing.assert_allclose(v.max(0), [.6, .7, .8])
    assert components[0]["display_faces"] == 6
    assert len(components[0]["edges"]) == 12 * 2 * 3
    bad = box_arrays(); bad["geometry_ref/codec_id"][0] = 8
    with pytest.raises(ValueError, match="Unsupported display codec"):
        display_components(bad)


def test_real_viewer_counts_fields_and_http_privacy(dataset_root):
    store = ViewerStore(dataset_root)
    row = next(r for r in store.catalog()["scenes"] if r["id"].startswith("cd9f325f1b"))
    sid = row["id"]
    try:
        metadata = store.scene(sid)
        assert len(metadata["windows"]) == 256 and len(metadata["components"]) == 8
        assert metadata["source_position"] is None
        assert metadata['source_status'] == 'not_configured'
        assert sum(metadata["global_counts"].values()) == 1024**3
        w = store.window(sid, 198)
        assert w["informative"] and w["lit"] + w["shadow"] + w["solid"] == 256**3
        assert w["origin"] == (184, 706, 503)
        # Check unaligned ROI planes against the independent reader ROI path.
        item = store.dataset._open(sid)
        for axis in (0, 1, 2):
            index = w["origin"][axis] + 127
            y = stored_plane(item["response"], axis, index)
            m = stored_plane(item["solid"], axis, index)
            roi = store.dataset.scene(sid).roi(w["origin"], 256, packed=False)
            expected = roi[:, :, 127] if axis == 0 else roi[:, 127, :] if axis == 1 else roi[127]
            u, v = ((1, 2), (0, 2), (0, 1))[axis]
            np.testing.assert_array_equal(y[w["origin"][v]:w["origin"][v] + 256,
                                          w["origin"][u]:w["origin"][u] + 256], expected)
            assert not np.any(y & m)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(url + "/api/catalog") as response:
                catalog = json.load(response)
            assert catalog["scene_count"] == 1000 and catalog["window_count"] == 256000
            assert not catalog["private_sources_exposed"]
            with urllib.request.urlopen(url + '/api/volume?scene='+sid+'&resolution=128') as response:
                assert response.headers['Content-Encoding'] == 'gzip'
                assert response.headers['Cache-Control'] == 'no-store'
                assert response.headers['X-Sample-Step'] == '8'
                assert response.headers['X-Sample-Offset'] == '4'
                labels = np.frombuffer(gzip.decompress(response.read()), dtype=np.uint8).reshape((128,)*3)
            for x, y, z in np.random.default_rng(319).integers(0, 128, size=(512,3)):
                xyz = (int(x)*8+4,int(y)*8+4,int(z)*8+4)
                expected = 2 if item['solid'].read_point(*xyz) else int(item['response'].read_point(*xyz))
                assert labels[z,y,x] == expected
            with urllib.request.urlopen(url + '/assets/volume.js') as response:
                assert response.status == 200 and b'Data3DTexture' in response.read()
            for path, status in [("/api/source", 404), ("/api/plane?scene=" + sid + "&axis=z&index=1024", 400),
                                 ("/assets/../data/manifest.json", 404), ("/api/window?scene=" + sid + "&id=-1", 400),
                                 ('/api/volume?scene='+sid+'&resolution=110', 400),
                                 ('/.rind3d-viewer.local.json', 404)]:
                with pytest.raises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(url + path)
                assert error.value.code == status
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(url + "/api/scene", data=b"{}", method="POST"))
            assert error.value.code == 405
        finally:
            server.shutdown(); worker.join(); server.server_close()
    finally:
        store.close()


def test_configured_missing_private_record_stays_hidden(dataset_root, tmp_path):
    store = ViewerStore(dataset_root, private_root=tmp_path)
    try:
        value = store.scene(store.rows[0]['scene_id'])
        assert value['source_position'] is None and value['source_status'] == 'record_missing'
    finally:
        store.close()


def test_opt_in_real_private_source_binding(dataset_root):
    private = os.environ.get('RIND3D_PRIVATE_ROOT')
    if not private:
        pytest.skip('Set RIND3D_PRIVATE_ROOT for authorized original-source binding checks')
    store = ViewerStore(dataset_root, private_root=private)
    try:
        # One actual scene per family, without serializing any private coordinates.
        seen = set()
        for row in store.rows:
            family = row['families'][0]
            if family in seen: continue
            seen.add(family)
            value = store.scene(row['scene_id'])
            assert value['source_status'] == 'verified_original'
            assert np.isfinite(value['source_position']).all()
            assert value['source_scope'] == 'local_authorized_inspection'
        assert len(seen) == 9
    finally:
        store.close()


def test_viewer_import_and_scene_without_native_oracle(dataset_root):
    code = '''
import builtins,sys
original=builtins.__import__
def guarded(name,*a,**k):
    if name.split('.')[0] in {'rind','rind3d','torch','matplotlib'}:
        raise ImportError('Forbidden viewer dependency: '+name)
    return original(name,*a,**k)
builtins.__import__=guarded
from rind_phase3.browser import ViewerStore
s=ViewerStore(sys.argv[1]);m=s.scene(s.rows[0]['scene_id'])
assert m['components'] and m['source_position'] is None
s.close()
'''
    subprocess.run([sys.executable, "-c", code, str(Path(dataset_root).resolve())], check=True, timeout=30)
