"""Paths, versions and private-secret handling for the v0.3 training-ready candidate.

Read-only inputs (never written by v0.3):
  CANARY      accepted v0.2 canary (216 scenes, frozen contract v0.2)
  V01         restricted v0.1 lite-core 1K campaign (private labels: driver.f64 and *.private.json)
  V01_PUBLIC  public v0.1 snapshot used by the Phase 3 Teacher (no labels)
  PHASE3      Phase 3 Teacher source tree (imported read-only by the adapter)
Everything v0.3 writes lives under ROOT/outputs (derived indexes and exports) and ROOT/private (secrets).
"""
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = Path("/home/yushu/Projects/RIND/G2-02-gaussian/RIND3D_Production")
CANARY = PRODUCTION/"outputs"/"canary"
V02_CONTRACT = PRODUCTION/"RIND3D_PRODUCTION_CONTRACT.json"
V01 = Path("/home/yushu/Projects/RIND/RIND3D/outputs/lite-core-1k-v01/campaign")
V01_PUBLIC = Path("/home/yushu/Projects/RIND/G2-02-3d/Phase3_Single_Source_3D/data/RIND3D-v0.1")
PHASE3 = Path("/home/yushu/Projects/RIND/G2-02-3d/Phase3_Single_Source_3D")
PRODUCER = Path("/home/yushu/Projects/RIND/G2-02-3d/RIND3D_Generator")
OUT = ROOT/"outputs"
PRIVATE = ROOT/"private"

TRAINING_CONTRACT = "rind3d-training-contract-v0.3"
INDEX_POLICY = "obs-index-v0.3"
STUDENT_SCHEMA = "rind3d-student-window-v0.3"
PRIVILEGED_SCHEMA = "rind3d-privileged-window-v0.3"


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def private_secret(name, create=False):
    """256-bit secret kept in ROOT/private (mode 0600, directory 0700). Never exported, never in the review ZIP.
    Index seeds, opaque-id keys and dynamic-window keys are all derived from such secrets with HMAC."""
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    p = PRIVATE/f"{name}.secret"
    if not p.exists():
        if not create:
            raise FileNotFoundError(f"private secret {name} missing (create explicitly)")
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32) + "\n")
    key = bytes.fromhex(p.read_text().strip())
    if len(key) < 32:
        raise ValueError("private secret shorter than 256 bits")
    return key


def keyed_int(key, *parts, bits=64):
    """Deterministic integer derived from a private key and public parts (HMAC-SHA256)."""
    msg = "\x1f".join(str(p) for p in parts).encode()
    return int.from_bytes(hmac.new(key, msg, hashlib.sha256).digest()[: bits // 8], "little")
