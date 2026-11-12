"""Posterior-label side-car format (prototype, `rind3d-posterior-labels-v1`) and conversion to the Student's
hierarchical 16^3 x 8^3 target.  One file per scene: <scene>/posterior_labels.v1.npz (immutable, next to the
existing v0.1 files; nothing existing is modified).

Per window (arrays of length 256, window order = window_records.bin):
  status      u1   0 CERT_ONEHOT  (all other 128^3 cells certified empty; label = truth cell, TV error 0)
                   1 CERT_SPARSE  (certified, tv_bound <= tolerance)
                   2 UNCERTIFIED  (budget exhausted or tv_bound > tolerance: use true-source NLL instead)
  tv_bound    f4   deterministic TV bound of the stored masses (NaN for UNCERTIFIED)
  volume_lo   f8   certified feasible volume L;  volume_und  f8  undetermined volume U
  n_cells     u4 , cell_offset u8     slice into the flat cell arrays
  nodes, receiver_tests (u8), seconds (f4)   cost diagnostics
Flat arrays: cell (u4, fine index (ix*128+iy)*128+iz), mass (f4, normalised lower bounds L_c / L).
File attributes (JSON string `meta`): format, engine version + code SHA256, engine parameters, tolerance,
geometry_id, source_id, physics contract ('closed-segment, voxel-centre receivers, v0.1'), prior
('uniform legal domain, natural source-excluded selection'), clearance-shell bound.
"""
import json
import numpy as np

CERT_ONEHOT, CERT_SPARSE, UNCERTIFIED = 0, 1, 2
FORMAT = "rind3d-posterior-labels-v1"


def classify(summary, cells, lo, tolerance):
    """Decide the status of one window result (summary dict from code/batch.py)."""
    if not summary.get("complete") or summary.get("L", 0) <= 0:
        return UNCERTIFIED
    tvb = summary["tv_bound"]
    if tvb is None or tvb > tolerance:
        return UNCERTIFIED
    pos = cells[lo > 0]
    if len(pos) == 1 and tvb == 0.0:
        return CERT_ONEHOT
    return CERT_SPARSE


def pack_scene(window_results, meta, tolerance):
    """window_results: list over window ids of (summary, cells, lo) or None (not computed)."""
    n = len(window_results)
    status = np.full(n, UNCERTIFIED, np.uint8)
    tvb = np.full(n, np.nan, np.float32)
    vlo = np.zeros(n); vund = np.zeros(n)
    ncell = np.zeros(n, np.uint32); off = np.zeros(n, np.uint64)
    nodes = np.zeros(n, np.uint64); rtests = np.zeros(n, np.uint64); secs = np.zeros(n, np.float32)
    flat_c = []; flat_m = []; pos = 0
    for w, item in enumerate(window_results):
        off[w] = pos
        if item is None:
            continue
        s, cells, lo = item
        st = classify(s, cells, lo, tolerance)
        status[w] = st
        nodes[w] = s["stats"]["nodes"]; rtests[w] = s["stats"]["receiver_tests"]; secs[w] = s["seconds"]
        vlo[w] = s.get("L", 0.0); vund[w] = s.get("U", 0.0)
        if st == UNCERTIFIED:
            continue
        tvb[w] = s["tv_bound"]
        m = lo > 0
        c = cells[m].astype(np.uint32); q = (lo[m] / lo[m].sum()).astype(np.float32)
        flat_c.append(c); flat_m.append(q); ncell[w] = len(c); pos += len(c)
    cell = np.concatenate(flat_c) if flat_c else np.zeros(0, np.uint32)
    mass = np.concatenate(flat_m) if flat_m else np.zeros(0, np.float32)
    meta = dict(meta, format=FORMAT, tolerance=tolerance)
    return dict(status=status, tv_bound=tvb, volume_lo=vlo, volume_und=vund, n_cells=ncell, cell_offset=off,
                nodes=nodes, receiver_tests=rtests, seconds=secs, cell=cell, mass=mass,
                meta=np.array(json.dumps(meta, sort_keys=True)))


def save(path, packed):
    np.savez_compressed(path, **packed)


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def window_label(d, w):
    a = int(d["cell_offset"][w]); n = int(d["n_cells"][w])
    return int(d["status"][w]), d["cell"][a:a + n], d["mass"][a:a + n]


def to_hierarchical(cells, mass):
    """Sparse fine masses -> (coarse_index[k], coarse_mass[k], fine_local_index[k][], fine_conditional[k][])
    for the 16^3 coarse x 8^3 fine Student.  Fine local index = (fx*8+fy)*8+fz within the coarse cell."""
    cells = np.asarray(cells, np.int64)
    ix, rem = np.divmod(cells, 128 * 128); iy, iz = np.divmod(rem, 128)
    ck = ((ix // 8) * 16 + iy // 8) * 16 + iz // 8
    fl = ((ix % 8) * 8 + iy % 8) * 8 + iz % 8
    out = []
    for k in np.unique(ck):
        m = ck == k
        qk = float(mass[m].sum())
        out.append((int(k), qk, fl[m].astype(np.int16), (mass[m] / qk).astype(np.float32)))
    return out


def hierarchical_ce(label_h, log_p_coarse, log_p_fine):
    """CE of a hierarchical label against model log-probabilities: log_p_coarse[4096], log_p_fine[4096, 512]."""
    ce = 0.0
    for k, qk, fl, qf in label_h:
        ce -= qk * log_p_coarse[k]
        ce -= qk * float(np.sum(qf * log_p_fine[k, fl]))
    return ce
