"""Environment wiring: producer checkout (certified physics, read-only) and optional mesh/BVH engines.

Run every entry point with the producer interpreter:
    /home/yushu/Projects/RIND/G2-02-3d/RIND3D_Generator/.venv/bin/python -m rind3d_production.<module>
with PYTHONPATH=<this directory's parent>. Mesh/BVH engines (Embree 4, Warp) come from the pilot mesh
environment and are appended to sys.path only when requested (`mesh_engines()`).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

PRODUCER = Path(os.environ.get("RIND3D_PRODUCER", "/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Generator"))
MESH_SITE = Path("/home/yushu/Projects/RIND/G2-02-gaussian/RIND3D_Generator/outputs/gaussian-mesh-dataset-v1/venv/lib/python3.12/site-packages")
MESH_EXPERIMENTS = Path("/home/yushu/Projects/RIND/G2-02-gaussian/RIND3D_Generator/experiments")
ROOT = Path(__file__).resolve().parent.parent
CERTIFICATE = PRODUCER/"outputs"/"integration"/"scientific-certification.json"
POLICY = {"max_precision_bits": 256, "max_subdivisions_per_item": 4096,
          "max_arrangement_events_per_item": 1048576, "qfnra_rlimit_per_item": 4194304}


def mesh_engines():
    """Embree/Warp wrappers and the exact triangle oracle of the mesh studies (imported read-only)."""
    for p in (str(MESH_SITE), str(MESH_EXPERIMENTS)):
        if p not in sys.path:
            sys.path.append(p)
    from gaussian_mesh_dataset import engines, oracle            # noqa: E402
    return engines, oracle


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def certificate():
    """The producer build's existing scientific certification, re-validated against the current code."""
    from rind3d.lite_certification import scientific_identity
    cert = json.loads(CERTIFICATE.read_text())
    current = scientific_identity()
    ok = bool(cert.get("complete") and cert.get("hard_cases_passed") and cert.get("cpu_native_certified")
              and current["sha256"] == cert["scientific_identity"]["sha256"])
    return {"path": str(CERTIFICATE), "file_SHA256": sha256_file(CERTIFICATE), "scientific_identity": current["sha256"],
            "matches_current_build": ok, "cuda_certified": bool(cert.get("cuda_certified"))}
