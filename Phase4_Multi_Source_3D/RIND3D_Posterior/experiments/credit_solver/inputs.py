"""Canonical v3 scene/channel inputs. The loader never reads source coordinates."""
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import zstandard
from rind3d.geometry import load_geometry
from rind3d.rendering.polyhedra import decode_convex

EXPERIMENTS = Path(__file__).resolve().parent.parent
for directory in ("mixed_patch", "global_inverse", "xlf_assurance"):
    sys.path.insert(0, str(EXPERIMENTS / directory))

from certificates import integer_row
from solver import Observation
from validated_input import integer
import evidence


def load(path, window, channel):
    window, channel = integer(window), integer(channel)
    with np.load(path, allow_pickle=False) as container:
        windows = container["windows"]
        if not 0 <= window < len(windows) or not 0 <= channel < 4:
            raise ValueError("invalid window/channel")
        w = windows[window]
        side = integer(w["L"])
        origin = tuple(integer(w[k]) for k in "xyz")
        meta = json.loads(container["meta"].tobytes())
        if (meta["resolution"] != 1024 or side not in (64,128,256)
                or not integer(w["mask"]) & (1 << channel)
                or any(a < 0 or a+side > 1024 for a in origin)):
            raise ValueError("unsupported or ineligible v3 observation")
        blob = container["obs"].tobytes()
        entries = container["obs_index"]
        if entries.dtype.kind not in "iu" or entries.ndim != 2 or entries.shape[1] != 4:
            raise ValueError("expected integer crop index with four columns")
        width = side**3 // 8
        def bits(index):
            matches = [(int(start),int(size)) for i,j,start,size in entries
                       if int(i) == window and int(j) == index]
            if len(matches) != 1:
                raise ValueError("missing/duplicate crop")
            start,size = matches[0]
            if start < 0 or size <= 0 or start+size > len(blob):
                raise ValueError("invalid crop extent")
            packed = zstandard.ZstdDecompressor().decompress(blob[start:start+size],max_output_size=width)
            if len(packed) != width:
                raise ValueError("incorrect crop length")
            return np.unpackbits(np.frombuffer(packed,np.uint8)).reshape((side,)*3).T
        geometry = container["geometry"].tobytes()
        cls = bits(channel)
        solid = bits(-1).astype(bool)
        if np.any(cls[solid]):
            raise ValueError("active solid")
        cls[solid] = 2
    bodies = tuple(tuple(integer_row(r) for r in c.rows)
                   for c in decode_convex(load_geometry(io.BytesIO(geometry))))
    cls.flags.writeable = False
    obs = Observation(bodies,cls,origin,1024)
    identity = evidence.problem(obs)
    return obs, dict(**identity, schema="rind3d-v3-free-channel/world-volume-128/v1",
                     geometry_sha256=hashlib.sha256(geometry).hexdigest(),
                     window=window,
                     channel=channel, family=str(meta["family"]), slot=int(meta["slot"]))
