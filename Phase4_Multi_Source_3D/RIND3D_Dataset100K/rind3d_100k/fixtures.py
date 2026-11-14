"""Reference fixtures shipped with the handoff (built by scripts/make_fixtures.py)."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import zstandard

HERE = Path(__file__).resolve().parent.parent
FIX = HERE / "handoff" / "fixtures"

SMOKE = ["000000_w63_s0", "000000_w60_s2", "000000_w35_s3", "000000_w06_s1", "000000_w11_s2"]


def index():
    return json.loads((FIX / "labels" / "INDEX.json").read_text())


def digest(cell, lo, hi, kind):
    h = hashlib.sha256()
    for a in (np.ascontiguousarray(cell, np.int64), np.ascontiguousarray(lo), np.ascontiguousarray(hi),
              np.ascontiguousarray(kind, np.int8)):
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def load(name):
    z = np.load(FIX / "labels" / f"{name}.npz")
    L = int(z["L"]); dz = zstandard.ZstdDecompressor()
    lit = np.unpackbits(np.frombuffer(dz.decompress(z["lit"].tobytes()), np.uint8), count=L ** 3).astype(bool)
    sol = np.unpackbits(np.frombuffer(dz.decompress(z["solid"].tobytes()), np.uint8), count=L ** 3).astype(bool)
    cls = np.where(sol, 2, np.where(lit, 1, 0)).astype(np.uint8).reshape(L, L, L)
    return z["P"], z["K"], cls, z["origin"], int(z["N"])


def fixture_task(t, engine, kw):
    """Pool entry: t = fixture name. Returns digest, tv_bound and timings (labeller kwargs from kw)."""
    try:
        from rind3d_posterior import label_window
        P, K, cls, origin, N = load(t)
        a = dict(kw.get("tier1", {}))
        c0 = time.process_time(); t0 = time.perf_counter()
        if engine == "gpu":
            r = label_window(P, K, cls, origin, N, engine="gpu", gpu_min=1024, gpu_prefix=64, **a, **kw.get("gpu", {}))
        else:
            r = label_window(P, K, cls, origin, N, engine="cpu", **a)
        return dict(name=t, engine=engine, digest=digest(r["cell"], r["lo"], r["hi"], r["kind"]), tv_bound=float(r["tv_bound"]),
                    n_records=int(len(r["cell"])), cpu=time.process_time() - c0, wall=time.perf_counter() - t0,
                    gpu=(dict((k, v) for k, v in r["gpu"].items() if k != "tt") if r.get("gpu") else None))
    except Exception as e:
        import traceback
        return dict(name=t, engine=engine, error=repr(e)[:400], trace=traceback.format_exc()[-1200:])
