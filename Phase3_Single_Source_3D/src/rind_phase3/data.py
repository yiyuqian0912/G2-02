"""Restricted student samples and explicit full-geometry teacher access."""

from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np
from rind3d_reader import RINDDataset
from .coordinates import Window, RESOLUTION


@dataclass(frozen=True)
class TeacherScene:
    """Privileged public geometry; never a value in a student sample."""

    scene_id: str
    geometry_id: str
    split: str
    geometry_path: Path


class RIND3D:
    def __init__(self, root=None, split="train", *, window_sizes=None):
        root = root or os.environ.get("RIND3D_DATA_ROOT")
        if not root:
            raise ValueError("Pass dataset root or set RIND3D_DATA_ROOT")
        self._reader = RINDDataset(root, split=split, window_sizes=window_sizes)
        self.root = Path(root)

    def __len__(self):
        return len(self._reader)

    def __getitem__(self, index):
        raw = self._reader[index]
        side = raw["window_size"]
        origin = raw["window_origin"]
        mask = self._reader.scene(raw["scene_id"]).roi(
            tuple(origin), side, field="solid", packed=True
        )
        # Whitelist, rather than copying metadata and removing known secrets.
        return {
            "scene_id": raw["scene_id"],
            "window_id": raw["window_id"],
            "origin": origin.copy(),
            "side": side,
            "observed_response": raw["observed_response"],
            "local_solid_mask": mask,
        }

    def window(self, index):
        sample = self[index]
        return Window(tuple(sample["origin"]), sample["side"])

    def statistics(self, index):
        raw = self._reader[index]
        return self._reader.scene(raw["scene_id"]).counts(
            tuple(raw["window_origin"]), raw["window_size"]
        )

    def teacher_scene(self, scene_id):
        row = self._reader.scenes[scene_id]
        return TeacherScene(
            scene_id,
            row["geometry_id"],
            row["split"],
            self.root / "scenes" / scene_id / "geometry.npz",
        )

    def metadata(self):
        m = self._reader.metadata()
        return {
            "dataset_name": "RIND3D",
            "release_version": "0.1",
            "release_id": m.get("release_id"),
            "internal_provenance_id": m.get("name"),
            "selected_scenes": len(self._reader.scenes),
            "selected_windows": len(self),
            "split_counts": m.get("split_counts"),
            "resolution": RESOLUTION,
            "window_sides": [64, 128, 256],
        }

    def close(self):
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def collate(samples):
    return {
        "scene_id": [s["scene_id"] for s in samples],
        "window_id": np.asarray([s["window_id"] for s in samples]),
        "origin": np.stack([s["origin"] for s in samples]),
        "side": samples[0]["side"],
        "observed_response": np.stack([s["observed_response"].data for s in samples]),
        "local_solid_mask": np.stack([s["local_solid_mask"].data for s in samples]),
        "format": "packed-LSB-X; arrays ZYX",
    }


class Loader:
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        import torch

        for batch in self.loader:
            for key in ("window_id", "origin", "observed_response", "local_solid_mask"):
                batch[key] = torch.from_numpy(batch[key])
            yield batch


def make_dataloader(data, batch_size=2, num_workers=0, seed=0):
    import torch

    if batch_size < 1 or num_workers < 0:
        raise ValueError("Positive batch size and nonnegative worker count required")
    rng = np.random.default_rng(seed)
    groups = {64: [], 128: [], 256: []}
    for j, row in enumerate(data._reader.rows):
        records = np.fromfile(
            data.root / "scenes" / row["scene_id"] / "window_records.bin", dtype="<u2"
        ).reshape(-1, 4)
        for i, wid in enumerate(data._reader.windows[row["scene_id"]]):
            groups[int(records[wid, 3])].append(int(data._reader.offsets[j]) + i)
    batches = []
    for ids in groups.values():
        rng.shuffle(ids)
        batches.extend(ids[i : i + batch_size] for i in range(0, len(ids), batch_size))
    rng.shuffle(batches)
    kwargs = (
        {"multiprocessing_context": "spawn", "prefetch_factor": 1}
        if num_workers
        else {}
    )
    return Loader(
        torch.utils.data.DataLoader(
            data,
            batch_sampler=batches,
            collate_fn=collate,
            num_workers=num_workers,
            **kwargs,
        )
    )
