"""Student-side reader, deterministic scene-balanced sampler and size-bucketed batching (NumPy + zstd only).

This module is importable without the producer checkout (it runs in the benchmark-only CUDA venv). It reads
ONLY a student export directory (outputs/student/NAME/SPLIT). It returns model inputs and, separately, the opaque
sample id used to join privileged targets in the training process:

    feats, sample_id = shards.read(i)          # feats keys == FEATURE_KEYS exactly
    batch = shards.batch(indices, packed=True) # {"features": {...}, "sample_id": uint64[B]}

The per-scene group token is used by SceneBalancedSampler and is never returned by read()/batch().
"""
import json
import os
import zlib
from pathlib import Path

import numpy as np
import zstandard as zstd

FEATURE_KEYS = ("observed_response", "observed_solid", "window_origin", "window_size", "modality")
IDX_FIELDS = ("sample_id", "group", "modality", "L", "x", "y", "z", "resp_offset", "resp_bytes", "npal", "nplanes",
              "solid_offset", "solid_bytes", "resp_crc", "solid_crc")


def unpack_bits(packed, L):
    return np.unpackbits(np.frombuffer(packed, np.uint8), bitorder="little").reshape(L, L, L).astype(bool)


class StudentShards:
    def __init__(self, root, modalities=None, sides=None, verify_crc=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root/"STUDENT_MANIFEST.json").read_text())
        rows, files = [], []
        for p in sorted((self.root/"shards").glob("*.idx.npy")):
            idx = np.load(p)
            if set(idx.dtype.names) != set(IDX_FIELDS):
                raise ValueError(f"unexpected student index fields in {p.name}")
            sel = np.ones(len(idx), bool)
            if modalities is not None:
                sel &= np.isin(idx["modality"], list(modalities))
            if sides is not None:
                sel &= np.isin(idx["L"], list(sides))
            if not sel.any():
                continue
            fid = len(files)
            files.append(p.with_name(p.name.replace(".idx.npy", ".bin")))
            rows.append((fid, idx[sel]))
        self.files = files
        self.index = np.concatenate([r for _, r in rows]) if rows else np.zeros(0)
        self.file_of = np.concatenate([np.full(len(r), f, np.int32) for f, r in rows]) if rows else np.zeros(0, np.int32)
        self.verify_crc = verify_crc
        self._fds = None
        self._pid = None
        self._dz = None

    def __len__(self):
        return len(self.index)

    def _io(self):
        if self._pid != os.getpid():                         # lazily (re)open after fork: DataLoader workers
            self._fds = [os.open(f, os.O_RDONLY) for f in self.files]
            self._dz = zstd.ZstdDecompressor()
            self._pid = os.getpid()
        return self._fds, self._dz

    def raw(self, i):
        """Decompressed payloads (bytes) of sample i: response payload, SOLID packed bits."""
        fds, dz = self._io()
        r = self.index[i]
        fd = fds[self.file_of[i]]
        a = os.pread(fd, int(r["resp_bytes"]), int(r["resp_offset"]))
        b = os.pread(fd, int(r["solid_bytes"]), int(r["solid_offset"]))
        pa, pb = dz.decompress(a), dz.decompress(b)
        if self.verify_crc and (zlib.crc32(pa) != r["resp_crc"] or zlib.crc32(pb) != r["solid_crc"]):
            raise IOError(f"payload CRC mismatch for sample {i}")
        return r, pa, pb

    def decode_response(self, r, pa):
        L = int(r["L"])
        if r["modality"] == 0:
            return unpack_bits(pa, L)
        npal, nb = int(r["npal"]), int(r["nplanes"])
        pal = np.frombuffer(pa[:8*npal], "<f8")
        codes = np.zeros(L**3, np.uint8)
        plane = L**3//8
        for b in range(nb):
            codes |= np.unpackbits(np.frombuffer(pa[8*npal+b*plane: 8*npal+(b+1)*plane], np.uint8), bitorder="little") << b
        return pal[codes.reshape(L, L, L)]

    def read(self, i):
        r, pa, pb = self.raw(i)
        L = int(r["L"])
        feats = {"observed_response": self.decode_response(r, pa), "observed_solid": unpack_bits(pb, L),
                 "window_origin": np.array([r["x"], r["y"], r["z"]], np.int64), "window_size": L,
                 "modality": int(r["modality"])}
        return feats, np.uint64(r["sample_id"])

    def batch(self, indices, packed=True):
        """Collate same-size samples. packed=True keeps bits packed (uint8 [B,L,L,L/8]) for on-GPU unpacking;
        MA samples carry a padded palette [B,16] and code planes [B,4,L,L,L/8]."""
        rs = [self.raw(int(i)) for i in indices]
        Ls = {int(r["L"]) for r, _, _ in rs}
        if len(Ls) != 1:
            raise ValueError("batch() requires a single window size (use size-bucketed batches)")
        L = Ls.pop()
        B = len(rs)
        mods = np.array([int(r["modality"]) for r, _, _ in rs], np.uint8)
        feats = {"window_origin": np.array([[r["x"], r["y"], r["z"]] for r, _, _ in rs], np.int64),
                 "window_size": np.full(B, L, np.int64), "modality": mods}
        if packed:
            solid = np.stack([np.frombuffer(pb, np.uint8).reshape(L, L, L//8) for _, _, pb in rs])
            if np.all(mods == 0):
                resp = np.stack([np.frombuffer(pa, np.uint8).reshape(L, L, L//8) for _, pa, _ in rs])
                feats.update(observed_response_packed=resp, observed_solid_packed=solid)
            else:
                pal = np.zeros((B, 16))
                pl = np.zeros((B, max(1, max(int(r["nplanes"]) for r, _, _ in rs)), L, L, L//8), np.uint8)
                plane = L**3//8
                for b, (r, pa, _) in enumerate(rs):
                    npal, nb = int(r["npal"]), int(r["nplanes"])
                    pal[b, :npal] = np.frombuffer(pa[:8*npal], "<f8")
                    for k in range(nb):
                        pl[b, k] = np.frombuffer(pa[8*npal+k*plane: 8*npal+(k+1)*plane], np.uint8).reshape(L, L, L//8)
                feats.update(observed_palette=pal, observed_code_planes=pl, observed_solid_packed=solid)
        else:
            feats["observed_response"] = np.stack([self.decode_response(r, pa) for r, pa, _ in rs])
            feats["observed_solid"] = np.stack([unpack_bits(pb, L) for _, _, pb in rs])
        return {"features": feats, "sample_id": np.array([r["sample_id"] for r, _, _ in rs], np.uint64)}


class SceneBalancedSampler:
    """Deterministic, scene-balanced, size-bucketed (profile-aware) batch sampler.

    Epoch e: every scene (group) contributes min(m_L, available) windows per side L, drawn without replacement
    by a keyed permutation (seed, e); windows are then shuffled across scenes, bucketed by (modality, L) and cut
    into batches of batch_size[L]; the batch order is shuffled. With m_L = None every stored window is used once.
    The epoch plan depends only on (seed, epoch, index), never on labels or weights.
    """

    def __init__(self, index, batch_size, seed=0, per_scene=None, drop_last=False, mixed=False):
        self.idx = index
        self.bs = batch_size if isinstance(batch_size, dict) else {L: batch_size for L in (64, 128, 256)}
        self.seed, self.per_scene, self.drop_last, self.mixed = int(seed), per_scene, drop_last, mixed
        order = np.lexsort((index["L"], index["group"]))
        self.groups = {}
        for i in order:
            self.groups.setdefault((int(index["group"][i]), int(index["L"][i])), []).append(int(i))

    def epoch(self, e):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(e), 0x5ce7e]))
        chosen = []
        for (g, L), members in sorted(self.groups.items()):
            m = len(members) if self.per_scene is None else min(len(members), self.per_scene.get(L, 0))
            perm = rng.permutation(len(members))[:m]
            chosen += [members[j] for j in perm]
        chosen = np.array(chosen, np.int64)[rng.permutation(len(chosen))]
        batches = []
        if self.mixed:
            b = min(self.bs.values())
            for s in range(0, len(chosen), b):
                batches.append(chosen[s:s+b])
        else:
            for key in sorted({(int(self.idx["modality"][i]), int(self.idx["L"][i])) for i in chosen}):
                sel = chosen[(self.idx["modality"][chosen] == key[0]) & (self.idx["L"][chosen] == key[1])]
                b = self.bs[key[1]]
                for s in range(0, len(sel), b):
                    if self.drop_last and len(sel[s:s+b]) < b:
                        continue
                    batches.append(sel[s:s+b])
        return [batches[j] for j in rng.permutation(len(batches))]


def gpu_unpack(packed, torch):
    """uint8 [..., L/8] packed little-endian along X -> bool [..., L] on the device of `packed`."""
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*packed.shape[:-1], packed.shape[-1]*8).bool()
