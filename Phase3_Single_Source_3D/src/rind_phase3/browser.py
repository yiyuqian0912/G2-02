"""Read-only local scene explorer; stored fields are the only response truth.

Display meshes and sampled volumes never define physical ground truth.
Private sources are optional, explicitly configured, and identity checked.
"""

import argparse
from collections import OrderedDict
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
import mimetypes
import os
from pathlib import Path
import re
import struct
import threading
from urllib.parse import parse_qs, urlsplit
import zlib

import numpy as np
from rind3d_reader import RINDDataset
from rind3d_reader.durable import read_sealed


WEB = Path(__file__).with_name("web")
PALETTE = bytes((23, 51, 79, 246, 206, 105, 146, 159, 170))


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def private_source(private_root, public_scene_path, sid, geometry_id):
    """Opt-in local inspection, never a source recovery or public SDK path."""
    private_scene = Path(private_root) / 'scenes' / sid
    seal = read_sealed(private_scene / 'scene.seal.json')
    if seal['scene_id'] != sid or seal['geometry_id'] != geometry_id or seal['status'] not in ('READY', 'READY_LITE'):
        raise ValueError('Private scene identity mismatch')
    files = seal['files']
    def checked(name):
        path = private_scene / name
        if file_sha(path) != files[name]['sha256']:
            raise ValueError('Private record checksum mismatch')
        return path
    meta = json.loads(checked('verification.json').read_text())
    if meta['scene_id'] != sid or meta['geometry_id'] != geometry_id:
        raise ValueError('Private verification identity mismatch')
    truth = json.loads(checked('sources.private.json').read_text())['truth']
    driver = checked('driver.f64').read_bytes()
    if len(driver) != 24 or truth['canonical_scene_id'] != sid:
        raise ValueError('Invalid true-source record')
    source = np.asarray(truth['position'], dtype='<f8')
    if source.shape != (3,) or not np.isfinite(source).all() or np.any(source <= 0) or np.any(source >= 1):
        raise ValueError('Source outside open unit world')
    if source.tobytes() != driver or hashlib.sha256(bytes.fromhex(geometry_id) + driver).hexdigest() != meta['source_id']:
        raise ValueError('Exact source identity mismatch')
    for name in ('response.rindb', 'solid.rindb', 'geometry.npz', 'window_records.bin'):
        expected = files[name]['sha256']
        if meta['hashes'].get(name) != expected or file_sha(Path(public_scene_path) / name) != expected:
            raise ValueError('Private source does not match public scientific files')
    return source.tolist()


def sampled_volume(response, solid, resolution):
    """Display-only nearest stored-voxel samples, ZYX uint8 class labels.

    The original 1024³ fields remain unchanged. At display index i the exact
    sampled original voxel is i*step+step//2. Solids are a distinct class.
    """
    if resolution not in (64, 128, 256) or response.shape != (1024,)*3 or solid.shape != (1024,)*3:
        raise ValueError('Supported display volumes are 64/128/256 for 1024³ fields')
    step = 1024 // resolution
    local = np.arange(step // 2, 64, step)
    per_brick = len(local)
    iz, iy, ix = np.ix_(local, local, local)
    out = np.empty((resolution,)*3, dtype=np.uint8)
    for bz in range(16):
        for by in range(16):
            for bx in range(16):
                classes = []
                for pack in (response, solid):
                    raw = np.frombuffer(pack.read_brick_packed((bx, by, bz)), dtype=np.uint8).reshape(64, 64, 8)
                    classes.append((raw[iz, iy, ix // 8] >> (ix % 8)) & 1)
                y, m = classes
                if np.any(y & m):
                    raise ValueError('Stored lit/solid inconsistency')
                labels = y.copy(); labels[m.astype(bool)] = 2
                z0, y0, x0 = np.array((bz, by, bx)) * per_brick
                out[z0:z0+per_brick, y0:y0+per_brick, x0:x0+per_brick] = labels
    return out


def png_labels(labels):
    """Lossless palette PNG: PNG top row is largest world row coordinate."""
    values = np.asarray(labels, dtype=np.uint8)
    height, width = values.shape
    if np.any(values > 2):
        raise ValueError("Unknown stored response class")
    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value))
    raw = np.concatenate((np.zeros((height, 1), dtype=np.uint8), values[::-1]), axis=1).tobytes()
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0))
            + chunk(b"PLTE", PALETTE) + chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))


def stored_plane(pack, axis, index):
    """Full-resolution plane; XYZ axes, rows Y/Z, columns X/Y."""
    if axis not in (0, 1, 2) or not 0 <= index < 1024:
        raise ValueError("Axis or plane index out of range")
    out = np.empty((1024, 1024), dtype=bool)
    for row in range(16):
        for col in range(16):
            brick = ((index // 64, col, row) if axis == 0 else
                     (col, index // 64, row) if axis == 1 else (col, row, index // 64))
            raw = np.frombuffer(pack.read_brick_packed(brick), dtype=np.uint8).reshape(64, 64, 8)
            if axis == 2:
                plane = np.unpackbits(raw[index % 64], axis=-1, bitorder="little")
            elif axis == 1:
                plane = np.unpackbits(raw[:, index % 64], axis=-1, bitorder="little")
            else:
                plane = (raw[:, :, index % 64 // 8] >> (index % 8)) & 1
            out[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = plane
    return out


def display_components(arrays):
    """Decode only the public SoA convex-union schema for DISPLAY.

    This is not an alternative physics decoder: float64 Qhull meshes never
    decide membership, visibility, response bits or scientific acceptance.
    """
    try:
        from scipy.spatial import ConvexHull, HalfspaceIntersection
    except ImportError as exc:
        raise RuntimeError("Install the viewer extra: uv sync --extra viewer") from exc
    if list(arrays["geometry/dimension"]) != [3] or int(arrays["geometry/root_kind"][0]) != 1:
        raise ValueError("Viewer requires the supported nonempty 3D convex-union schema")
    root = int(arrays["geometry/root_ref"][0])
    leaves = []
    def walk(ref):
        kind = int(arrays["geometry_ref/ref_kind"][ref])
        row = int(arrays["geometry_ref/codec_row"][ref])
        if kind == 1:
            if int(arrays["geometry_csg/opcode"][row]) != 0:
                raise ValueError("Unsupported display CSG: only certified top-level unions")
            a, b = map(int, arrays["geometry_csg/child_offsets"][row:row + 2])
            for child in arrays["geometry_csg/child_ref"][a:b]:
                if not 0 <= int(child) < ref:
                    raise ValueError("Invalid canonical reference ordering")
                walk(int(child))
        elif kind == 0:
            leaves.append((ref, int(arrays["geometry_ref/codec_id"][ref]), row))
        else:
            raise ValueError("Unsupported geometry reference")
    walk(root)
    components = []
    for ref, codec, row in leaves:
        if codec not in (1, 4):
            raise ValueError("Unsupported display codec; no voxel approximation is substituted")
        base = "geometry_codecs/" + ("affine_box/" if codec == 1 else "convex_halfspaces/")
        center, matrix = arrays[base + "center"][row], arrays[base + "matrix"][row]
        if codec == 1:
            local = np.array(list(itertools.product((-1., 1.), repeat=3)))
            planes = 6
        else:
            a, b = map(int, arrays[base + "object_plane_offsets"][row:row + 2])
            normals, offsets = arrays[base + "normal"][a:b], arrays[base + "plane_offset"][a:b]
            if np.any(offsets <= 0):
                raise ValueError("Missing strict interior display anchor")
            local = HalfspaceIntersection(np.column_stack((normals, -offsets)), np.zeros(3)).intersections
            planes = len(normals)
        vertices = np.asarray(center + local @ matrix.T, dtype=np.float64)
        if not np.isfinite(vertices).all():
            raise ValueError("Unresolved display vertices")
        hull = ConvexHull(vertices)
        groups = {}
        for triangle, equation in zip(hull.simplices, hull.equations):
            # Coplanar face merging is only cosmetic for the edge display.
            groups.setdefault(tuple(np.round(equation, 9)), set()).update(map(int, triangle))
        edges = set()
        for equation, ids in groups.items():
            ids = np.array(sorted(ids));points = vertices[ids];origin = points.mean(axis=0)
            u = points[0] - origin;u /= np.linalg.norm(u)
            v = np.cross(np.array(equation[:3]), u)
            order = np.argsort(np.arctan2((points - origin) @ v, (points - origin) @ u))
            ordered = ids[order]
            for a, b in zip(ordered, np.roll(ordered, -1)):
                edges.add(tuple(sorted((int(a), int(b)))))
        components.append({
            "id": ref, "codec": "affine_box_v1" if codec == 1 else "convex_halfspaces_v1",
            "canonical_planes": planes, "display_faces": len(groups),
            "triangles": vertices[hull.simplices].reshape(-1).tolist(),
            "edges": vertices[np.array(sorted(edges))].reshape(-1).tolist(),
        })
    return components


class ViewerStore:
    def __init__(self, root, private_root=None):
        self.dataset = RINDDataset(root, split=None, cache_scenes=2, cache_bricks=32)
        self.lock = threading.RLock()
        self.meshes = OrderedDict()
        self.planes = OrderedDict()
        self.windows = OrderedDict()
        self.volumes = OrderedDict()
        self.sources = OrderedDict()
        self.private_root = Path(private_root).resolve() if private_root else None
        self.rows = list(self.dataset.rows)
        self.lookup = {r["scene_id"]: r for r in self.rows}

    def close(self):
        with self.lock:
            self.dataset.close()

    def _scene(self, sid):
        if not re.fullmatch("[0-9a-f]{64}", sid) or sid not in self.lookup:
            raise KeyError("Unknown public scene ID")
        return self.dataset._open(sid)

    @staticmethod
    def _cache(cache, key, value, limit):
        cache[key] = value;cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def catalog(self):
        return {
            "dataset_name": "RIND3D", "version": "0.1", "resolution": 1024,
            "release_id": self.dataset.manifest.get("release_id"),
            "scene_count": len(self.rows), "window_count": sum(r["window_count"] for r in self.rows),
            "scenes": [{"id": r["scene_id"], "geometry_id": r["geometry_id"], "family": r["families"][0],
                        "split": r["split"], "windows": r["window_count"], "ordinal": i + 1}
                       for i, r in enumerate(self.rows)],
            "private_sources_exposed": self.private_root is not None,
        }

    def scene(self, sid):
        with self.lock:
            item = self._scene(sid)
            if sid not in self.meshes:
                self._cache(self.meshes, sid, display_components(item["structure"]["arrays"]), 8)
            else:
                self.meshes.move_to_end(sid)
            lit = sum(record[7] for record in item["response"].index)
            solid = sum(record[7] for record in item["solid"].index)
            source = None
            source_status = 'not_configured'
            if self.private_root:
                if sid not in self.sources:
                    try:
                        value = (private_source(self.private_root, item['path'], sid, self.lookup[sid]['geometry_id']), 'verified_original')
                    except FileNotFoundError:
                        value = (None, 'record_missing')
                    except (ValueError, KeyError, TypeError):
                        value = (None, 'identity_rejected')
                    self._cache(self.sources, sid, value, 16)
                source, source_status = self.sources[sid]
            return {
                "id": sid, "geometry_id": self.lookup[sid]["geometry_id"],
                "family": self.lookup[sid]["families"][0], "split": self.lookup[sid]["split"],
                "components": self.meshes[sid],
                "windows": [{"id": i, "origin": list(map(int, r[:3])), "side": int(r[3])}
                            for i, r in enumerate(item["windows"])],
                "global_counts": {"lit": lit, "shadow": 1024**3 - lit - solid, "solid": solid},
                "geometry_display_only": True, "source_position": source,
                "source_status": source_status,
                "source_scope": 'local_authorized_inspection' if self.private_root else 'public_reader',
            }

    def window(self, sid, wid):
        with self.lock:
            key = (sid, wid)
            item = self._scene(sid)
            if not 0 <= wid < len(item["windows"]):
                raise ValueError("Window ID out of range")
            if key not in self.windows:
                record = item["windows"][wid];origin = tuple(map(int, record[:3]));side = int(record[3])
                counts = self.dataset.scene(sid).counts(origin, side)
                value = {"id": wid, "origin": origin, "side": side, "lit": counts.lit,
                         "shadow": counts.shadow, "solid": counts.solid, "free": counts.free,
                         "lit_ratio": counts.lit_ratio, "informative": counts.informative}
                self._cache(self.windows, key, value, 512)
            else:
                self.windows.move_to_end(key)
            return self.windows[key]

    def plane(self, sid, axis, index):
        with self.lock:
            if axis not in (0, 1, 2) or not 0 <= index < 1024:
                raise ValueError("Plane outside 1024-cubed world")
            item = self._scene(sid);key = (sid, axis, index)
            if key not in self.planes:
                lit = stored_plane(item["response"], axis, index)
                solid = stored_plane(item["solid"], axis, index)
                if np.any(lit & solid):
                    raise ValueError("Stored lit/solid inconsistency")
                labels = lit.astype(np.uint8);labels[solid] = 2
                counts = np.bincount(labels.ravel(), minlength=3)
                self._cache(self.planes, key, (png_labels(labels), {
                    "X-Shadow": str(counts[0]), "X-Lit": str(counts[1]), "X-Solid": str(counts[2]),
                    "X-Index": str(index), "X-Axis": "xyz"[axis],
                }), 48)
            else:
                self.planes.move_to_end(key)
            return self.planes[key]

    def volume(self, sid, resolution):
        with self.lock:
            if resolution not in (64, 128, 256):
                raise ValueError('Unsupported display volume resolution')
            item = self._scene(sid); key = (sid, resolution)
            if key not in self.volumes:
                labels = sampled_volume(item['response'], item['solid'], resolution)
                self._cache(self.volumes, key, gzip.compress(labels.tobytes(), compresslevel=1, mtime=0), 2)
            else:
                self.volumes.move_to_end(key)
            return self.volumes[key], {
                'Content-Encoding': 'gzip', 'X-Volume-Resolution': str(resolution),
                'X-Source-Resolution': '1024', 'X-Sample-Step': str(1024 // resolution),
                'X-Sample-Offset': str(512 // resolution),
                'X-Volume-Scope': 'display-only-nearest-stored-voxel-samples',
            }


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
                super().log_message(fmt, *args)

        def send(self, code, content, kind, headers=None):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store" if self.path.startswith('/api/') else "no-cache")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            self.send(405, b'{"error":"Read-only viewer"}', "application/json")

        def do_GET(self):
            path = urlsplit(self.path);query = parse_qs(path.query)
            try:
                if path.path.startswith("/api/"):
                    def arg(key):
                        values = query.get(key)
                        if not values or len(values) != 1:
                            raise ValueError("Missing or repeated argument: " + key)
                        return values[0]
                    if path.path == "/api/catalog":
                        value = store.catalog()
                    elif path.path == "/api/scene":
                        value = store.scene(arg("scene"))
                    elif path.path == "/api/window":
                        value = store.window(arg("scene"), int(arg("id")))
                    elif path.path == "/api/plane":
                        axis = arg("axis")
                        if axis not in ("x", "y", "z"):
                            raise ValueError("Unknown axis")
                        content, headers = store.plane(arg("scene"), "xyz".index(axis), int(arg("index")))
                        self.send(200, content, "image/png", headers)
                        return
                    elif path.path == '/api/volume':
                        content, headers = store.volume(arg('scene'), int(arg('resolution')))
                        self.send(200, content, 'application/octet-stream', headers)
                        return
                    else:
                        raise KeyError("Unknown API route")
                    self.send(200, json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8")
                else:
                    relative = "index.html" if path.path == "/" else path.path.removeprefix("/assets/")
                    allowed = {"index.html", "style.css", "app.js", "volume.js", "vendor/three.module.js",
                               "vendor/OrbitControls.js", "vendor/THREE-LICENSE.txt"}
                    if relative not in allowed:
                        raise KeyError("Unknown asset")
                    kind = "text/javascript" if relative.endswith(".js") else mimetypes.guess_type(relative)[0] or "application/octet-stream"
                    self.send(200, (WEB / relative).read_bytes(), kind + "; charset=utf-8")
            except KeyError as exc:
                self.send(404, json.dumps({"error": str(exc)}).encode(), "application/json")
            except (ValueError, IndexError) as exc:
                self.send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
            except Exception as exc:
                self.log_error("Viewer request failed: %s", exc)
                self.send(500, json.dumps({"error": type(exc).__name__, "message": str(exc)}).encode(), "application/json")
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("RIND3D_DATA_ROOT", "data/RIND3D-v0.1"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument('--private-root', default=os.environ.get('RIND3D_PRIVATE_ROOT'),
                        help='Explicit original campaign root for authorized local true-source inspection')
    args = parser.parse_args()
    config = Path('.rind3d-viewer.local.json')
    if args.private_root is None and config.exists():
        args.private_root = json.loads(config.read_text()).get('private_root')
    if args.private_root and args.host not in ('127.0.0.1', 'localhost', '::1'):
        parser.error('Private source inspection requires a loopback-only host')
    store = ViewerStore(args.root, private_root=args.private_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    server.daemon_threads = True
    print(f"RIND3D Explorer: http://{args.host}:{server.server_port}", flush=True)
    print(f"Read-only: {len(store.rows)} scenes. Ctrl+C stops the viewer.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close();store.close()


if __name__ == "__main__":
    main()
