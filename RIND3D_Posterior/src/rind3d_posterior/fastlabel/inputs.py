"""Dataset-independent input of one FastLabel task, the exact legality predicate, and scene-format adapters.

A task is one (scene geometry, window, source channel):
* `FastLabelWindow` carries everything the solver and the verifier need: the canonical geometry bundle (exact
  predicates), its float64 half-space rows P/K (tau-pencil engine), the window origin / side L / resolution N, the stored
  observation as packed bits (np.packbits of bool [Z, Y, X], big-endian bit order; lit response of the channel and the
  SOLID mask), the RNG key (scene index, window, source) that makes every label reproducible, a cache key for per-scene
  objects, and a FREE point used only to bind the native evaluator for the SOLID-consistency check.
* Adapters build it from the canonical scene formats: `window_from_1k` (RIND3D-1K scene directory) and
  `window_from_100k` (RIND3D-100K scene container `.npz`, branch yushu/3d-100k-h20). Both formats store crops with the
  same encoding, so the solver sees identical inputs for identical physical windows.
`legal_source` is the exact source-domain predicate D(G) of the RIND3D contracts (open cube, outside the closed solid,
certified clearance > 2^-26), identical to rind3d_1k.physical.legal_source."""
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# canonical-geometry context policy of the RIND3D contracts (rind3d_1k.physical.POLICY; same values in the 100K)
POLICY = {"max_precision_bits": 256, "max_subdivisions_per_item": 4096,
          "max_arrangement_events_per_item": 1048576, "qfnra_rlimit_per_item": 4194304}


def legal_source(context, p):
    """Exact legality of a source position: open cube, not in the closed solid, certified clearance > 2^-26."""
    from rind3d.lite_acceptance import DRIVER_RADIUS, verify_clearance
    p = np.asarray(p, dtype="<f8").reshape(3)
    if not np.all((p > 0) & (p < 1)):
        return None
    if context.contains(p[None])[0]:
        return None
    proof = context.clearance(p, DRIVER_RADIUS)
    if proof["status"] != "GT":
        return None
    verify_clearance(context, p, proof)
    return proof


@dataclass
class FastLabelWindow:
    P: np.ndarray                      # [n_bodies, max_rows, 4] float64 half-space rows a.x + d <= 0
    K: np.ndarray                      # [n_bodies] number of rows per body
    bundle: object                     # canonical geometry bundle (rind.storage.CanonicalBundle)
    origin: tuple                      # window origin (x, y, z) in voxels
    L: int                             # window side in voxels
    N: int                             # resolution (voxels per axis)
    lit_bits: np.ndarray               # uint8 packed bits of the channel's lit response, [Z, Y, X], big-endian
    solid_bits: np.ndarray             # uint8 packed bits of the SOLID mask, [Z, Y, X], big-endian
    key: tuple                         # (scene_index, window, source): keyed RNG streams and certificate seed
    scene_key: str                     # cache key of per-scene objects (geometry context, engine geometry)
    bind_point: np.ndarray             # a legal source position (binds the native evaluator; membership is source-free)
    policy: dict = field(default_factory=lambda: dict(POLICY))


# ------------------------------------------------------------------ adapters
_ADAPT = {}


def _cached(key, make):
    v = _ADAPT.get(key)
    if v is None:
        if len(_ADAPT) >= 2:
            _ADAPT.pop(next(iter(_ADAPT)))
        v = _ADAPT[key] = make()
    return v


def window_from_1k(scene_dir, w, s):
    """RIND3D-1K scene directory (public/observations.bin + index, privileged/geometry*.npz, scene.json)."""
    import zstandard
    from rind3d.geometry import load_geometry
    scene_dir = Path(scene_dir)

    def make():
        W = np.load(scene_dir / "public" / "windows.npy")
        idx = np.load(scene_dir / "public" / "observations_index.npy")
        g = np.load(scene_dir / "privileged" / "geometry_rows.npz")
        src = json.loads((scene_dir / "privileged" / "sources.json").read_text())["sources_hex"]
        return dict(W=W, pos={(int(a), int(b)): (int(o), int(n)) for a, b, o, n in idx},
                    blob=(scene_dir / "public" / "observations.bin").read_bytes(), P=g["P"], K=g["K"],
                    N=int(json.loads((scene_dir / "scene.json").read_text())["resolution"]),
                    bundle=load_geometry(scene_dir / "privileged" / "geometry.npz"),
                    sources=np.array([[float.fromhex(v) for v in q] for q in src]), dz=zstandard.ZstdDecompressor())
    c = _cached(("1k", str(scene_dir)), make)
    return _make(c, w, s, scene_index=int(scene_dir.name), scene_key=str(scene_dir))


def window_from_100k(container, w, s, scene_index=None):
    """RIND3D-100K scene container (.npz written by rind3d_100k.physical; read like rind3d_100k.storage.SceneFile).
    scene_index defaults to the container's slot (meta["slot"])."""
    import zstandard
    from rind3d.geometry import load_geometry
    container = Path(container)

    def make():
        with np.load(container, allow_pickle=False) as z:
            meta = json.loads(z["meta"].tobytes().decode())
            W = z["windows"]; idx = z["obs_index"]; blob = z["obs"].tobytes()
            P, K, src = z["P"], z["K"], z["sources_hex"]; geo = z["geometry"].tobytes()
        return dict(W=W, pos={(int(a), int(b)): (int(o), int(n)) for a, b, o, n in idx}, blob=blob, P=P, K=K,
                    N=int(meta["resolution"]), bundle=load_geometry(io.BytesIO(geo)), slot=int(meta["slot"]),
                    sources=np.array([[float.fromhex(str(v)) for v in q] for q in src]), dz=zstandard.ZstdDecompressor())
    c = _cached(("100k", str(container)), make)
    return _make(c, w, s, scene_index=c["slot"] if scene_index is None else int(scene_index), scene_key=str(container))


def _make(c, w, s, scene_index, scene_key):
    w, s = int(w), int(s)

    def raw(j):
        o, n = c["pos"][(w, j)]
        return np.frombuffer(c["dz"].decompress(c["blob"][o:o + n]), np.uint8)
    r = c["W"][w]
    return FastLabelWindow(P=c["P"], K=c["K"], bundle=c["bundle"], origin=(int(r["x"]), int(r["y"]), int(r["z"])),
                           L=int(r["L"]), N=c["N"], lit_bits=raw(s), solid_bits=raw(-1), key=(scene_index, w, s),
                           scene_key=scene_key, bind_point=c["sources"][s])
