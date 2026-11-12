"""Compact posterior-label side-car (format rind3d-posterior-labels-v2), per scene, for offline caching.

Per window: status, certified TV bound of the stored label against the exact natural posterior (including the
storage quantisation error, computed exactly below), engine/version metadata, cost.  Cells: sorted 128^3 linear
indices, delta-encoded as LEB128 varints; masses as float16 (renormalised on load).  The exact float64 cell masses
are what the engine certifies; the stored label's TV error is the rounding TV, which is computed and recorded.

Training needs no physics: `to_hierarchical` converts a label into the Student's 16^3 coarse + 8^3 fine targets.
"""
import json
import zlib

import numpy as np

FORMAT = 'rind3d-posterior-labels-v2'
G = 128


def _varint(a):
    out = bytearray()
    for v in a.tolist():
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def _unvarint(buf, n):
    out = np.empty(n, np.int64)
    i = 0; k = 0
    while k < n:
        v = 0; sh = 0
        while True:
            b = buf[i]; i += 1
            v |= (b & 0x7F) << sh
            sh += 7
            if not b & 0x80:
                break
        out[k] = v; k += 1
    return out


def encode(cells, masses, truncate_tv=0.0):
    """cells (int), masses (>= 0, any scale).  Optional truncation: drop the smallest cells whose total mass is
    <= 2*truncate_tv / (1 + ...) so that TV(stored, exact) <= truncate_tv (the dropped mass d gives TV = d)."""
    cells = np.asarray(cells, np.int64); q = np.asarray(masses, float)
    q = q / q.sum()
    if truncate_tv > 0:
        o = np.argsort(q)
        c = np.cumsum(q[o])
        drop = o[c <= truncate_tv]
        keep = np.ones(len(q), bool); keep[drop] = False
        cells, q = cells[keep], q[keep]
    o = np.argsort(cells)
    cells, q = cells[o], q[o]
    h = q.astype(np.float16)
    if not np.all(np.isfinite(h)) or h.sum() == 0:
        h = (q / q.max()).astype(np.float16)
    blob = _varint(np.diff(np.r_[0, cells])) + h.tobytes()
    return dict(n=int(len(cells)), blob=zlib.compress(blob, 6))


def decode(rec):
    raw = zlib.decompress(rec['blob'])
    n = rec['n']
    # varints end where float16 payload begins (2n bytes at the end)
    idx = _unvarint(raw[:len(raw) - 2 * n], n)
    cells = np.cumsum(idx)
    q = np.frombuffer(raw[len(raw) - 2 * n:], np.float16).astype(float)
    return cells, q / q.sum()


def storage_tv(cells, masses, rec):
    """Exact TV between the exact label and the decoded stored label."""
    c2, q2 = decode(rec)
    q = np.asarray(masses, float); q = q / q.sum()
    d = dict(zip(np.asarray(cells).tolist(), q.tolist()))
    e = dict(zip(c2.tolist(), q2.tolist()))
    return 0.5 * sum(abs(d.get(k, 0) - e.get(k, 0)) for k in set(d) | set(e))


def to_hierarchical(cells, q):
    """(coarse 16^3 index, coarse mass) and per-coarse-cell fine (8^3) distributions."""
    ix, iy, iz = np.unravel_index(np.asarray(cells), (G, G, G))
    cc = ((ix // 8) * 16 + iy // 8) * 16 + iz // 8
    fc = ((ix % 8) * 8 + iy % 8) * 8 + iz % 8
    coarse = {}
    for a, f, m in zip(cc.tolist(), fc.tolist(), np.asarray(q).tolist()):
        coarse.setdefault(a, {}).setdefault(f, 0.0)
        coarse[a][f] += m
    out = {a: (sum(v.values()), {f: m / sum(v.values()) for f, m in v.items()}) for a, v in coarse.items()}
    return out


def pack_scene(path, scene_id, windows, meta):
    """windows: list of dicts with keys window, status, tv_bound, cells, masses, cost (+ any extras)."""
    recs = []
    for w in windows:
        e = encode(w['cells'], w['masses'])
        st = storage_tv(w['cells'], w['masses'], e)
        recs.append(dict({k: v for k, v in w.items() if k not in ('cells', 'masses')}, n=e['n'],
                         blob=e['blob'].hex(), storage_tv=st, tv_bound_stored=min(1.0, w['tv_bound'] + st)))
    json.dump(dict(format=FORMAT, scene_id=scene_id, meta=meta, windows=recs), open(path, 'w'))
