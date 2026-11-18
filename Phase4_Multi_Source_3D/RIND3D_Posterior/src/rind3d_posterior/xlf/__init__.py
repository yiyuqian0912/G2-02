"""XLF: exact local-first posterior labeller (research).

A self-certifying solver: every region it creates is an exact plane-based polytope (certify.poly), every claim it makes
about a region is proven by rigorous interval evaluation of the closed segment-polytope criterion (xlf.pred) before it is
recorded, and the posterior is integrated over the resulting exact partition (docs/XLF.md). The trusted computing base
is certify.ar + certify.poly + certify.scene + certify.tvb + xlf.pred + the leaf/partition logic of xlf.solve; the
search heuristics (split and clip planes, extreme-receiver selection) are untrusted and cannot affect correctness.

Numba cache: keyed by the SHA-256 of every jitted module this package uses (certify kernel + xlf)."""
import hashlib
import os
from pathlib import Path

_here = Path(__file__).resolve().parent
_cert = _here.parent / "certify"
_h = hashlib.sha256()
for _p in [_cert / n for n in ("ar.py", "poly.py")] + sorted(_here.glob("*.py")):
    _h.update(_p.name.encode()); _h.update(_p.read_bytes())
SOURCE_HASH = _h.hexdigest()
XLF_VERSION = "rind3d-xlf-0.1+" + SOURCE_HASH[:12]

from .. import certify as _certify  # noqa: E402,F401  (sets numba's cache dir; overridden below)
import numba  # noqa: E402

_base = os.environ.get("RIND3D_XLF_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "rind3d_xlf"))
numba.config.CACHE_DIR = os.path.join(_base, SOURCE_HASH[:16])
