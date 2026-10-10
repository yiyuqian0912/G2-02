"""Multi-source observations, separate truth channels, and size-aware loading."""

import json
import math
import os
from pathlib import Path

import numpy as np
from rind_dataset import RINDDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_data_root():
    configured = os.environ.get("RIND_MULTI_DATA_ROOT")
    return Path(configured).expanduser().resolve() if configured else (
        PROJECT_ROOT / "data/raw/phase1-multi-source")


class Phase1MultiDataset:
    """Lazy additive observations; source labels use separate teacher interfaces.

    dataset[i] returns IDs, response [L,L] float32, obstacle [L,L] bool,
    and window [3] int64. The response is not clipped. Per-source contributions,
    source count, strengths and full geometry are teacher/evaluation-only.
    scene_ids selects whole scenes. Disk arrays stay memory mapped.
    """

    def __init__(self, root=None, *, scene_ids=None):
        self.root = Path(root).expanduser().resolve() if root is not None else default_data_root()
        if not (self.root / "manifest.json").is_file():
            raise FileNotFoundError(
                f"RIND data not installed at {self.root}. Place the multi-source ZIP "
                "in the project root and run ./scripts/install_data.sh, or pass root=...")
        m = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if (not isinstance(m, dict) or m.get("schema_version") != 3 or
                not isinstance(m.get("max_drivers"), int) or m["max_drivers"] < 2 or
                m.get("response_type") != "additive-driver-intensity" or
                m.get("driver_strength_mode", "uniform-0-1") != "uniform-0-1" or
                m.get("candidate_layout", "padded-source-sets") != "padded-source-sets"):
            raise ValueError("Multi-source Phase I requires schema-v3 additive random-strength data")
        self._base = RINDDataset(self.root)
        if scene_ids is None:
            self.scene_ids = np.arange(self.num_scenes, dtype=np.int64)
        else:
            ids = np.asarray(scene_ids)
            if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
                raise ValueError("scene_ids must be a one-dimensional integer sequence")
            self.scene_ids = ids.astype(np.int64)
            if (np.any(self.scene_ids < 0) or np.any(self.scene_ids >= self.num_scenes) or
                    len(np.unique(self.scene_ids)) != len(self.scene_ids)):
                raise ValueError("scene_ids must be unique and within the dataset")
        self._selected_scenes = set(self.scene_ids.tolist())
        self._offsets = np.concatenate(([0], np.cumsum(
            self._base.view_counts[self.scene_ids], dtype=np.int64)))

    @property
    def num_scenes(self):
        """Total release scenes; len(scene_ids) is the selected scene count."""
        return self._base.num_scenes

    @property
    def manifest(self):
        return dict(self._base.manifest)

    def __len__(self):
        return int(self._offsets[-1])

    def _identity(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Observation index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Observation index out of range")
        selected = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return int(self.scene_ids[selected]), int(index - self._offsets[selected])

    def __getitem__(self, index):
        return self.get_observation(*self._identity(index))

    def _check_selected(self, scene_id):
        if scene_id not in self._selected_scenes:
            raise IndexError(f"Scene {scene_id} is outside this dataset subset")

    def get_observation(self, scene_id, view_id):
        self._check_selected(scene_id)
        item = self._base.get_view(scene_id, view_id)
        return {"scene_id": int(scene_id), "view_id": int(view_id),
                "response": item["response"].astype(np.float32),
                "obstacle": item["obstacle"].astype(bool),
                "window": item["view"].astype(np.int64)}

    def get_candidates(self, scene_id, view_id):
        """Incomplete reference witnesses [K,3] float64: (x,y,strength).

        These are not search proposals or probability labels. Some witnesses
        use K+1 co-located sources, even though generating scenes use at most K.
        """
        self._check_selected(scene_id)
        return self._base.get_candidates(scene_id, view_id)

    def get_scene(self, scene_id):
        """Continuous truth for teacher/evaluation use."""
        self._check_selected(scene_id)
        return self._base.get_scene(scene_id)

    def get_region(self, scene_id, x, y, width, height=None):
        """Teacher/diagnostic response channels and obstacle mask for any rectangle."""
        self._check_selected(scene_id)
        return self._base.get_region(scene_id, x, y, width, height)

    def get_channels(self, scene_id, view_id):
        """Exact float64 [K,L,L] source contributions, teacher/evaluation-only."""
        self._check_selected(scene_id)
        return self._base.get_view(scene_id, view_id)["channels"]

    def get_source_params(self, scene_id):
        """Generating source set [K,3] float64, teacher/evaluation-only."""
        scene = self.get_scene(scene_id)
        return np.column_stack((scene["drivers"], scene["driver_strengths"]))

    def get_view_tree(self, scene_id):
        self._check_selected(scene_id)
        return self._base.get_view_tree(scene_id)

    def iter_view_leaves(self, scene_id, *, include_occupied=False):
        self._check_selected(scene_id)
        return self._base.iter_view_leaves(scene_id, include_occupied=include_occupied)

    def rerender(self, scene_id, window, sources, *, backend="auto", return_channels=False):
        """Render explicit [K,3] source parameters, returning float64 [L,L].

        The CPU geometry renderer is not differentiable. Invalid source
        positions (inside obstacles or outside the world) are rejected.
        """
        self._check_selected(scene_id)
        window = np.asarray(window)
        if window.shape != (3,) or window.dtype.kind not in "iu":
            raise ValueError("window must contain three integers (x,y,L)")
        sources = np.asarray(sources, dtype=np.float64)
        if sources.ndim != 2 or sources.shape[1] != 3 or not len(sources):
            raise ValueError("Multi-source rerender requires explicit [K,3] (x,y,strength) rows")
        x, y, side = map(int, window)
        result = self._base.rerender_region(scene_id, x, y, side,
                                           drivers=sources,
                                           backend=backend)
        return result if return_channels else result["response"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_base")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._base = RINDDataset(self.root)


def split_scene_ids(num_scenes, *, ratios, seed=20260923):
    """Deterministically split whole scenes; ratios are an explicit experiment choice."""
    weights = np.asarray(ratios, dtype=np.float64)
    if (num_scenes <= 0 or weights.shape != (3,) or not np.isfinite(weights).all() or
            np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10)):
        raise ValueError("Provide positive num_scenes and three nonnegative ratios summing to 1")
    ids = np.random.default_rng(seed).permutation(num_scenes)
    a = int(num_scenes * weights[0])
    b = a + int(num_scenes * weights[1])
    return {"train": ids[:a].tolist(), "validation": ids[a:b].tolist(),
            "test": ids[b:].tolist()}


class SizeBucketBatchSampler:
    """Batch equal-size views without decoding response images."""

    def __init__(self, dataset, batch_size, *, shuffle=True, seed=20260923, drop_last=False):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size, self.shuffle, self.seed = batch_size, shuffle, seed
        self.drop_last, self.epoch = drop_last, 0
        sizes = np.concatenate([
            dataset._base.local_views[s, :int(dataset._base.view_counts[s]), 2]
            for s in dataset.scene_ids]) if len(dataset.scene_ids) else np.empty(0)
        self.buckets = [np.flatnonzero(sizes == size) for size in np.unique(sizes)]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return sum(len(b) // self.batch_size if self.drop_last else
                   math.ceil(len(b) / self.batch_size) for b in self.buckets)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for bucket in self.buckets:
            indices = rng.permutation(bucket) if self.shuffle else bucket
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size].tolist()
                if not self.drop_last or len(batch) == self.batch_size:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def make_dataloader(dataset, *, batch_size=8, shuffle=True, seed=20260923,
                    num_workers=0, drop_last=False, pin_memory=False):
    """Return response/obstacle [B,L,L], window [B,3], and IDs [B].

    Variable source counts do not require padding: source labels/channels are
    excluded from these student batches. They are accessed separately.
    """
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is optional. Run uv sync --locked --extra train") from exc
    sampler = SizeBucketBatchSampler(dataset, batch_size, shuffle=shuffle,
                                     seed=seed, drop_last=drop_last)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                      pin_memory=pin_memory)
