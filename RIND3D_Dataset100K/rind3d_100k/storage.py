"""RIND3D-100K storage: one atomic container per scene, append-only label journals, consolidated per-scene labels.

**Root layout:**
* `CONTRACT.json`
* `plans/epoch_<e:05d>.json`: controller plans
* `scenes/<slot//1000:04d>/<slot:06d>.npz`: scene container (written by a physical worker; tmp, fsync, rename)
* `manifest/physical.jsonl`: one record per attempt (written by the coordinator only)
* `labels/journal/<writer>.r3lj`: append-only binary label journal (coordinator only)
* `labels/journal/<writer>.idx`: task keys of the journal, for fast resume
* `labels/scenes/<shard>/<slot:06d>.labels.npz`: consolidated labels of a scene
* `labels/failures.jsonl`: crash / timeout / error / memory records (quarantine)
* `STOP`: safe stop request

**Scene container (npz):**
* `meta`: utf-8 JSON;
* `windows`: structured (window, x, y, z, L, Lp, m, mask, k);
* `obs` and `obs_index`: rows (window, source or -1 = SOLID, offset, nbytes); packed bits, zstd, (Z, Y, X);
* `geometry`: canonical bundle npz bytes;
* `P`, `K`: float rows for the labeller;
* `sources_hex`;
* `acceptance`;
* `log_L64`, `log_L128`, `log_L256`: selection logs.

**Label journal record:**
* `b"R3LJ"`
* u32 header length
* header JSON: key, sha256 of the payload, payload length
* payload: npz bytes of cell, lo, hi, kind, meta

A torn tail (crash while appending) is detected by the length and sha256 checks and ignored."""
import hashlib
import io
import json
import os
import struct
from pathlib import Path

import numpy as np
import zstandard

MAGIC = b"R3LJ"


def shard_of(slot):
    return int(slot) // 1000


def scene_path(root, slot):
    return Path(root) / "scenes" / f"{shard_of(slot):04d}" / f"{int(slot):06d}.npz"


def labels_path(root, slot):
    return Path(root) / "labels" / "scenes" / f"{shard_of(slot):04d}" / f"{int(slot):06d}.labels.npz"


def atomic_write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


class SceneFile:
    """Read access to one scene container. Responses stay source-separated; nothing is flattened."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            self.meta = json.loads(z["meta"].tobytes().decode())
            self.W = z["windows"]; self.index = z["obs_index"]; self.blob = z["obs"].tobytes()
            self.P = z["P"]; self.K = z["K"]; self.sources_hex = z["sources_hex"]
            self._geometry = z["geometry"].tobytes()
            self.logs = {kk: z[kk] for kk in z.files if kk.startswith("log_")}
        self._pos = {(int(w), int(s)): (int(o), int(n)) for w, s, o, n in self.index}
        self.dz = zstandard.ZstdDecompressor()

    @property
    def k(self):
        return int(self.meta["k"])

    def sources(self):
        return np.array([[float.fromhex(str(v)) for v in s] for s in self.sources_hex])

    def geometry(self):
        from rind3d.geometry import load_geometry
        return load_geometry(io.BytesIO(self._geometry))

    def crop(self, w, s):
        """bool [Z, Y, X]: s = -1 for SOLID, else the lit response of source s."""
        o, n = self._pos[(int(w), int(s))]
        L = int(self.W[w]["L"])
        bits = np.frombuffer(self.dz.decompress(self.blob[o:o + n]), np.uint8)
        return np.unpackbits(bits, count=L ** 3).astype(bool).reshape(L, L, L)

    def cls(self, w, s):
        """Labeller classes cls[x, y, z]: 0 dark FREE, 1 lit FREE, 2 SOLID (same convention as rind3d_1k)."""
        sol = self.crop(w, -1); lit = self.crop(w, s)
        c = np.where(sol, 2, np.where(lit, 1, 0)).astype(np.uint8)
        return np.ascontiguousarray(np.transpose(c, (2, 1, 0)))

    def origin(self, w):
        r = self.W[w]
        return np.array([r["x"], r["y"], r["z"]], np.int64)


# ------------------------------------------------------------------ label journal
def encode_record(key, arrays, meta):
    buf = io.BytesIO()
    np.savez_compressed(buf, meta=np.frombuffer(json.dumps(meta, default=str).encode(), np.uint8), **arrays)
    payload = buf.getvalue()
    hdr = json.dumps(dict(key=key, sha256=hashlib.sha256(payload).hexdigest(), n=len(payload))).encode()
    return MAGIC + struct.pack("<I", len(hdr)) + hdr + payload


def iter_journal(path):
    """Yield (key, payload_bytes) of complete, checksum-valid records; stops at a torn tail."""
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    while i + 8 <= len(data):
        if data[i:i + 4] != MAGIC:
            break
        hl = struct.unpack("<I", data[i + 4:i + 8])[0]
        if i + 8 + hl > len(data):
            break
        hdr = json.loads(data[i + 8:i + 8 + hl])
        j = i + 8 + hl
        if j + hdr["n"] > len(data):
            break
        payload = data[j:j + hdr["n"]]
        if hashlib.sha256(payload).hexdigest() != hdr["sha256"]:
            break
        yield hdr["key"], payload
        i = j + hdr["n"]


def decode_payload(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        out = {kk: z[kk] for kk in z.files if kk != "meta"}
        out["meta"] = json.loads(z["meta"].tobytes().decode())
    return out


class Journal:
    """Single-writer append-only journal plus key index (one per coordinator run)."""

    def __init__(self, root, writer):
        d = Path(root) / "labels" / "journal"; d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{writer}.r3lj"; self.idx = d / f"{writer}.idx"
        self.f = open(self.path, "ab"); self.fi = open(self.idx, "a")

    def append(self, key, arrays, meta):
        rec = encode_record(key, arrays, meta)
        off = self.f.tell()
        self.f.write(rec); self.f.flush(); os.fsync(self.f.fileno())
        self.fi.write(f"{key}\t{off}\t{len(rec)}\n"); self.fi.flush()
        return (str(self.path), off, len(rec))

    def close(self):
        self.f.close(); self.fi.close()


def completed_label_keys(root, verify=False):
    """Keys of completed label records. Fast path: the .idx files. An idx line is written only after its journal record
    is flushed and fsynced, so idx is a subset of the journal. verify=True scans the journals themselves (authoritative;
    also finds a record whose idx line was lost in a crash; a duplicate recomputation is harmless, because
    consolidation keeps the first valid record of a key)."""
    keys = set()
    d = Path(root) / "labels" / "journal"
    if not d.exists():
        return keys
    if verify:
        for p in sorted(d.glob("*.r3lj")):
            for key, _ in iter_journal(p):
                keys.add(key)
    else:
        keys.update(journal_index(root))
    return keys


def journal_index(root):
    """{key: (journal_path, offset, length)} from the .idx files (first record of a key wins)."""
    out = {}
    d = Path(root) / "labels" / "journal"
    if not d.exists():
        return out
    for p in sorted(d.glob("*.idx")):
        jp = str(p.with_suffix(".r3lj"))
        for line in p.read_text().splitlines():
            f = line.split("\t")
            if len(f) == 3 and f[0] not in out:
                out[f[0]] = (jp, int(f[1]), int(f[2]))
    return out


def read_record(loc):
    """Payload bytes of the journal record at (path, offset, length), checksum-verified."""
    path, off, n = loc
    with open(path, "rb") as f:
        f.seek(off); data = f.read(n)
    if data[:4] != MAGIC:
        raise IOError("bad journal record")
    hl = struct.unpack("<I", data[4:8])[0]
    hdr = json.loads(data[8:8 + hl]); payload = data[8 + hl:8 + hl + hdr["n"]]
    if hashlib.sha256(payload).hexdigest() != hdr["sha256"]:
        raise IOError("journal record checksum mismatch")
    return hdr["key"], payload


def task_key(slot, w, s):
    return f"{int(slot):06d}:{int(w):02d}:{int(s)}"
