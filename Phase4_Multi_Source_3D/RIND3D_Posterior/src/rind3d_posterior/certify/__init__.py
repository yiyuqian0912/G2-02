"""Independent verifier of deterministic posterior certificates (docs/CERTIFIED_LABELS.md).

Numba's on-disk cache is invalidated only when a function's own source file changes, not when a jitted callee in
another module changes. To make the verifier's compiled code a function of its exact source, the cache directory is
keyed by the SHA-256 of every source file of this package (set before any submodule is compiled)."""
import hashlib
import os
from pathlib import Path

_here = Path(__file__).resolve().parent
_h = hashlib.sha256()
for _p in sorted(_here.glob("*.py")):
    _h.update(_p.name.encode()); _h.update(_p.read_bytes())
SOURCE_HASH = _h.hexdigest()
VERIFIER_VERSION = "rind3d-certify-1.0+" + SOURCE_HASH[:12]

import numba  # noqa: E402

_base = os.environ.get("RIND3D_CERTIFY_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "rind3d_certify"))
numba.config.CACHE_DIR = os.path.join(_base, SOURCE_HASH[:16])
