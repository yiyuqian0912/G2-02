"""Versioned evidence on the rationally decoded little-endian wire distribution."""
from functools import lru_cache
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import llvmlite
import numba
import numpy as np


HERE = Path(__file__).resolve().parent
VERSION = "astra-xlf-cbc/1"
WIRE = "sparse-128-xyz/little-u32-binary64/dyadic52/v1"


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def crop_digest(cls):
    """Same XYZ byte stream for C/Fortran/strided crops, with bounded transpose storage."""
    if cls.flags.c_contiguous:
        return hashlib.sha256(cls).hexdigest()
    size = cls.shape[0]
    chunk = np.empty((min(8,size), *cls.shape[1:]), dtype=cls.dtype)
    h = hashlib.sha256()
    for x in range(0,size,8):
        out = chunk[:min(8,size-x)]
        for z in range(0,cls.shape[2],8):
            out[:,:,z:z+8] = cls[x:x+8,:,z:z+8]
        h.update(out)
    return h.hexdigest()


def problem(obs):
    # Hex encodes unbounded integers without CPython's decimal conversion limit.
    rows = [[[hex(v) for v in r] for r in b] for b in obs.bodies]
    return dict(halfspaces_sha256=digest(canonical(rows)), origin=list(obs.origin),
                side=obs.side, resolution=1024, observation_xyz_sha256=crop_digest(obs.cls))


@lru_cache(maxsize=1)
def _build_bytes():
    from rind3d import geometry
    from rind3d.rendering import polyhedra
    import certificates, envelopes, interface_chart, kernel, ledger, penumbra, poly, raster, solver, whole
    import zstandard
    from rind import storage, contracts
    from llvmlite.binding import ffi
    from numba import _helperlib
    from numpy._core import _multiarray_umath
    files = {name: HERE/name for name in (
        "evidence.py","grid_accounting.py","cbc_kernel.py","screen2.py","polycheck.py","validated_input.py","inputs.py",
        "cbc_search.py","cbc_run.py","run.py","shadow_facets.py","hull_proposals.py")}
    for module in (storage,contracts,geometry,polyhedra,certificates,envelopes,interface_chart,kernel,ledger,penumbra,poly,raster,solver,whole):
        files[module.__name__] = Path(module.__file__)
    runtime = {}
    for key, path in (("llvm", ffi.lib._name), ("numba", _helperlib.__file__),
                      ("numpy", _multiarray_umath.__file__), ("python", sys.executable)):
        with open(path, "rb") as stream:
            runtime[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    result = subprocess.run(["git","-C",str(HERE),"rev-parse","HEAD"],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError("research certification requires a versioned checkout")
    return canonical(dict(kernel=VERSION,backend="python-integer-fraction", commit=result.stdout.strip(),
        source_sha256={k:digest(path.read_bytes()) for k,path in files.items()},
        python=sys.version, runtime_sha256=runtime,
        machine=platform.machine(), numpy=np.__version__, numba=numba.__version__,
        llvmlite=llvmlite.__version__,zstandard=zstandard.__version__,
        semantics="binary64-nearest-even; no fastmath; signed int64 floor; exact Python int/Fraction"))


def seal(problem_identity, payload, bound, tolerance):
    data = dict(build=json.loads(_build_bytes()), problem=problem_identity, wire_format=WIRE,
                wire_sha256=digest(payload), tv=str(bound), tolerance=str(tolerance))
    return canonical(dict(statement=data,statement_sha256=digest(canonical(data))))


def check_seal(binding, payload, bound, tolerance, identity):
    if digest(canonical(binding["statement"])) != binding["statement_sha256"]:
        raise ArithmeticError("changed certificate statement")
    st=binding["statement"]
    if (st["wire_format"]!=WIRE or st["wire_sha256"]!=digest(payload) or
        st["tv"]!=str(bound) or st["tolerance"]!=str(tolerance) or
        any(identity[k]!=v for k,v in st["problem"].items())):
        raise ArithmeticError("certificate disagrees with problem, wire or bound")


def artifact_digest(identity, binding):
    return digest(canonical([identity, binding["statement_sha256"]]))


def check_artifact(rec, expected_identity):
    """Verify an extracted new-format envelope against independently expected input metadata."""
    import base64
    from fractions import Fraction
    from ledger import decode
    if (rec["format"] != "astra-xlf-certified-interface/1" or rec["status"] != "CERTIFIED"
            or rec["solver"] != VERSION or rec["problem"] != expected_identity):
        raise ValueError("unexpected artifact, status or input identity")
    wire = base64.b64decode(rec["wire"], validate=True)
    binding = rec["certificate"]
    bound, tolerance = Fraction(rec["tv"]), Fraction(rec["tolerance"])
    check_seal(binding, wire, bound, tolerance, expected_identity)
    if (rec["artifact_sha256"] != artifact_digest(expected_identity, binding)
            or digest(wire) != rec["wire_sha256"] or not 0 <= bound <= tolerance <= 1
            or len(decode(wire)) != rec["cells"]):
        raise ArithmeticError("changed artifact binding or invalid output")
    return decode(wire)
