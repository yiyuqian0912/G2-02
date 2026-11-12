"""Random-access read-only loader over published shards, with explicit access views.

  ProductionDataset(root, split=None, view="student_local", subsets=("natural",), profiles=None,
                    research_access=False, sparse=None)
Views (contract data_access):
  student_local      scene_id, window_id, profile, window_origin, window_size, observed_response, observed_solid
  student_geometry   + geometry (canonical arrays)
  teacher            + sources, strengths, K, channels (per-source window lit), subset, counts, physics()
                     (requires research_access=True)
observed_response: SB -> bool [L,L,L] (lit FREE receivers); MA -> float64 [L,L,L] combined response decoded from
the window's LOCAL palette. observed_solid: bool [L,L,L]. Subset selection uses private sampling records
inside the loader; the tags are never returned by student views. sparse=("uniform", k) adds a declared
uniform FREE receiver sample (local indices, world coordinates, values, inclusion probability, weight).
The index (one row per window) is built once from the shard indexes; every read opens only the files of
the requested scene (LRU-cached RINDPack readers).
"""
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import zstandard as zstd

from rind3d.geometry import load_geometry
from rind3d.pack import RINDPack

from .windows import SUBSET_CODES

VIEWS = ("student_local", "student_geometry", "teacher")
STUDENT_KEYS = ("scene_id", "window_id", "profile", "window_origin", "window_size", "observed_response", "observed_solid")
SPARSE_KEYS = ("sparse_design", "sparse_local_indices", "sparse_world_xyz", "sparse_values", "sparse_inclusion_probability",
               "sparse_weight")
GEOMETRY_KEYS = ("geometry",)
TEACHER_KEYS = ("sources", "strengths", "K", "channels", "subset", "counts", "scene_dir")


class AccessError(PermissionError):
    pass


def allowed_keys(view, sparse=False):
    keys = set(STUDENT_KEYS) | (set(SPARSE_KEYS) if sparse else set())
    if view in ("student_geometry", "teacher"):
        keys |= set(GEOMETRY_KEYS)
    if view == "teacher":
        keys |= set(TEACHER_KEYS)
    return keys


class ProductionDataset:
    def __init__(self, root, split=None, view="student_local", subsets=("natural",), profiles=None,
                 research_access=False, sparse=None, cache=8):
        if view not in VIEWS:
            raise ValueError(view)
        if view == "teacher" and not research_access:
            raise AccessError("teacher view requires research_access=True")
        self.root, self.view, self.sparse = Path(root), view, sparse
        self.cache = OrderedDict()
        self.cache_size = cache
        codes = {SUBSET_CODES[s] for s in subsets}
        rows = []
        self.scenes = []
        for shard in sorted((self.root/"shards").glob("shard-*")):
            scenes = [json.loads(l) for l in (shard/"index"/"scenes.jsonl").read_text().splitlines()]
            for sc in scenes:
                if (split is not None and sc["split"] != split) or (profiles is not None and sc["profile"] not in profiles):
                    continue
                d = shard/"scenes"/sc["scene_id"]
                tags = np.load(d/"private"/"window_records.npy")         # sampling metadata: loader-internal only
                keep = np.flatnonzero(np.isin(tags["subset"], list(codes)))
                si = len(self.scenes)
                self.scenes.append({**sc, "dir": d})
                rows += [(si, int(w)) for w in tags["window"][keep]]
        self.index = np.array(rows, dtype=np.int64).reshape(-1, 2)

    def __len__(self):
        return len(self.index)

    def _scene(self, si):
        if si in self.cache:
            self.cache.move_to_end(si)
            return self.cache[si]
        sc = self.scenes[si]
        d = sc["dir"]
        item = {"meta": sc, "public_windows": np.load(d/"public"/"windows.npy"),
                "solid": RINDPack(d/"public"/"solid.rindb", cache_bricks=32)}
        if sc["profile"].endswith("single-binary-v0.2"):
            item["response"] = RINDPack(d/"public"/"response.rindb", cache_bricks=32)
        else:
            item["obs_index"] = np.load(d/"public"/"observations_index.npy")
            item["obs_file"] = open(d/"public"/"observations.bin", "rb")
        self.cache[si] = item
        while len(self.cache) > self.cache_size:
            _, old = self.cache.popitem(last=False)
            old["solid"].close()
            if "response" in old:
                old["response"].close()
            if "obs_file" in old:
                old["obs_file"].close()
        return item

    def _observation(self, item, w, origin, L):
        if "response" in item:
            return item["response"].read_window(*origin, L)
        offset, npal, nbytes = item["obs_index"][w]
        f = item["obs_file"]
        f.seek(int(offset))
        palette = np.frombuffer(f.read(8*int(npal)), "<f8")
        codes = np.frombuffer(zstd.ZstdDecompressor().decompress(f.read(int(nbytes))), np.uint8).reshape(L, L, L)
        return palette[codes]

    def __getitem__(self, i):
        si, w = map(int, self.index[i])
        item = self._scene(si)
        rec = item["public_windows"][w]
        origin, L = (int(rec["x"]), int(rec["y"]), int(rec["z"])), int(rec["L"])
        sc = item["meta"]
        out = {"scene_id": sc["scene_id"], "window_id": w, "profile": sc["profile"], "window_origin": origin, "window_size": L,
               "observed_response": self._observation(item, w, origin, L), "observed_solid": item["solid"].read_window(*origin, L)}
        if self.sparse is not None:
            design, k = self.sparse
            if design != "uniform":
                raise AccessError("student sparse views are response-independent (uniform) only")
            free = np.flatnonzero(~out["observed_solid"].ravel())
            rng = np.random.default_rng(np.random.SeedSequence([int.from_bytes(bytes.fromhex(sc["scene_id"][:8]), "little"), w, k]))
            take = np.sort(rng.choice(free, min(k, len(free)), replace=False)) if len(free) else np.zeros(0, np.int64)
            a = np.arange(L)+.5
            z, y, x = np.unravel_index(take, (L, L, L))
            out.update(sparse_design=f"uniform-{k}", sparse_local_indices=take,
                       sparse_world_xyz=np.column_stack([(origin[0]+a[x])/1024, (origin[1]+a[y])/1024, (origin[2]+a[z])/1024]),
                       sparse_values=out["observed_response"].ravel()[take],
                       sparse_inclusion_probability=np.full(len(take), len(take)/max(1, len(free))),
                       sparse_weight=np.full(len(take), len(free)/max(1, len(take))))
        if self.view in ("student_geometry", "teacher"):
            from rind.storage import pack_arrays
            out["geometry"] = pack_arrays(load_geometry(sc["dir"]/"public"/"geometry.npz"))
        if self.view == "teacher":
            src = json.loads((sc["dir"]/"private"/"sources.json").read_text())
            tags = np.load(sc["dir"]/"private"/"window_records.npy")[w]
            chans = []
            for k in range(src["K"]):
                p = sc["dir"]/("public/response.rindb" if src["K"] == 1 else f"private/src{k}/response.rindb")
                with RINDPack(p) as r:
                    chans.append(r.read_window(*origin, L))
            out.update(sources=np.array([[float.fromhex(v) for v in s] for s in src["sources_hex"]]),
                       strengths=np.array([float.fromhex(v) for v in src["strengths_hex"]]), K=src["K"],
                       channels=np.stack(chans), subset=int(tags["subset"]),
                       counts={"free": int(tags["free"]), "solid": int(tags["solid"]), "lit": tags["lit"][:src["K"]].tolist()},
                       scene_dir=str(sc["dir"]))
        return out

    def physics(self, i, **kw):
        if self.view != "teacher":
            raise AccessError("physics() is a privileged Teacher interface")
        from .teacher import TeacherPhysics
        return TeacherPhysics(self.scenes[int(self.index[i][0])]["dir"], **kw)


def collate_numpy(batch):
    """Group-by-size friendly collation for DataLoader workers (NumPy payloads, no tensors in workers)."""
    return {k: [b[k] for b in batch] for k in batch[0]}
