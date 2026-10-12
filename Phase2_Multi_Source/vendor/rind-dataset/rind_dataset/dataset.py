"""Memory-mapped loader for the canonical adaptive additive RIND dataset."""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union
import numpy as np


class RINDDataset:
    """Variable-size source-free views with exact separate source channels."""

    def __init__(self, root: Union[str, Path], return_scene: bool = False):
        self.root = Path(root)
        self.return_scene = return_scene
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 3:
            raise ValueError("RINDDataset requires canonical schema v3")
        self.num_scenes = int(self.manifest["num_scenes"])
        self.global_size = int(self.manifest["global_size"])
        names = ("drivers", "driver_counts", "driver_strengths", "obstacle_types",
                 "obstacle_params", "obstacle_counts", "obstacle_group_ids",
                 "obstacle_bits", "visibility_bits", "local_views", "view_counts")
        for name in names:
            setattr(self, name, np.load(self.root / f"{name}.npy", mmap_mode="r"))
        self.candidate_layout = self.manifest.get("candidate_layout", "padded-source-sets")
        if self.candidate_layout == "single-position-list":
            self.reference_positions = np.load(self.root / "reference_positions.npy", mmap_mode="r")
            if self.reference_positions.shape != (self.num_scenes,
                    self.manifest["view_capacity"], self.manifest["candidate_capacity"], 2):
                raise ValueError("Invalid single-source reference position array")
        elif self.candidate_layout == "padded-source-sets":
            for name in ("view_candidates", "candidate_counts", "candidate_numbers"):
                setattr(self, name, np.load(self.root / f"{name}.npy", mmap_mode="r"))
        else:
            raise ValueError(f"Unknown candidate layout: {self.candidate_layout}")
        self.view_offsets = np.concatenate(([0], np.cumsum(self.view_counts, dtype=np.int64)))

    def __len__(self) -> int:
        return int(self.view_offsets[-1])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"View index {index} out of range")
        scene_id = int(np.searchsorted(self.view_offsets, index, side="right") - 1)
        view_id = int(index - self.view_offsets[scene_id])
        return self.get_view(scene_id, view_id, return_scene=self.return_scene)

    def _check_scene(self, scene_id: int):
        if scene_id < 0 or scene_id >= self.num_scenes:
            raise IndexError(f"Scene {scene_id} out of range")

    def get_region(self, scene_id: int, x: int, y: int, width: int, height: Optional[int] = None) -> Dict[str, Any]:
        """Decode stored channels, response and obstacle mask for any rectangle."""
        from rind_dataset.runtime import unpack_region_bits
        self._check_scene(scene_id)
        rows = width if height is None else height
        if (not all(isinstance(v, (int, np.integer)) for v in (x, y, width, rows)) or
                x < 0 or y < 0 or width <= 0 or rows <= 0 or
                x + width > self.global_size or y + rows > self.global_size):
            raise ValueError("Region must be an integer rectangle inside the world")
        m = int(self.driver_counts[scene_id])
        visible = unpack_region_bits(self.visibility_bits[scene_id, :m],
                                     self.global_size, x, y, width, rows)
        obstacle = unpack_region_bits(self.obstacle_bits[scene_id],
                                      self.global_size, x, y, width, rows)
        strengths = np.asarray(self.driver_strengths[scene_id, :m], dtype=np.float64)
        channels = visible.astype(np.float64) * strengths[:, None, None]
        return {"channels": channels, "response": channels.sum(axis=0),
                "obstacle": obstacle}

    def get_view(self, scene_id: int, view_id: int, return_scene: bool = False) -> Dict[str, Any]:
        """Get one adaptive (x,y,l) view and its independent response planes."""
        self._check_scene(scene_id)
        if view_id < 0 or view_id >= int(self.view_counts[scene_id]):
            raise IndexError("View ID out of range")
        x, y, side = map(int, self.local_views[scene_id, view_id])
        item = self.get_region(scene_id, x, y, side)
        m = int(self.driver_counts[scene_id])
        item.update({"scene_id": scene_id, "view_id": view_id, "size": side,
                     "view": np.array([x, y, side], dtype=np.uint16),
                     "origin": np.array([x, y], dtype=np.uint16),
                     "drivers": np.array(self.drivers[scene_id, :m], dtype=np.float64),
                     "driver_strengths": np.array(self.driver_strengths[scene_id, :m], dtype=np.float64)})
        if return_scene:
            item.update(self.get_scene(scene_id))
        return item

    def get_view_tree(self, scene_id: int) -> Dict[str, Any]:
        """Rebuild the scene's quadtree from continuous driver positions.

        Nodes have ``kind`` = ``split``, ``view``, or ``occupied``. A ``view``
        leaf has a ``view_id`` for ``get_view``; an ``occupied`` leaf is a
        source-containing minimum-size tile and has no stored local view.
        Children of a split are ordered top-left, top-right, bottom-left,
        bottom-right. Only metadata is built; response rasters stay lazy.
        """
        self._check_scene(scene_id)
        count = int(self.driver_counts[scene_id])
        points = np.asarray(self.drivers[scene_id, :count], dtype=np.float64)
        min_size = int(self.manifest["min_view_size"])
        views = self.local_views[scene_id, :int(self.view_counts[scene_id])]
        view_ids = {tuple(map(int, row)): i for i, row in enumerate(views)}

        def visit(x: int, y: int, size: int, driver_ids: list, depth: int) -> Dict[str, Any]:
            node = {"x": x, "y": y, "size": size, "depth": depth,
                    "driver_ids": driver_ids}
            if not driver_ids:
                node["kind"] = "view"
                try:
                    node["view_id"] = view_ids.pop((x, y, size))
                except KeyError as exc:
                    raise ValueError("Stored local views do not match the driver quadtree") from exc
            elif size == min_size:
                node["kind"] = "occupied"
            else:
                node["kind"] = "split"
                half = size // 2
                child_ids = [[], [], [], []]
                for driver_id in driver_ids:
                    px, py = points[driver_id]
                    quadrant = 2 * int(py >= y + half) + int(px >= x + half)
                    child_ids[quadrant].append(driver_id)
                node["children"] = [
                    visit(x + (index % 2) * half, y + (index // 2) * half,
                          half, child_ids[index], depth + 1)
                    for index in range(4)
                ]
            return node

        tree = visit(0, 0, self.global_size, list(range(count)), 0)
        if view_ids:
            raise ValueError("Stored local views contain entries outside the driver quadtree")
        return tree

    def iter_view_leaves(self, scene_id: int, *, include_occupied: bool = False) -> Iterator[Dict[str, Any]]:
        """Visit quadtree leaves in top-left, top-right, bottom-left, bottom-right order."""
        stack = [self.get_view_tree(scene_id)]
        while stack:
            node = stack.pop()
            if node["kind"] == "split":
                stack.extend(reversed(node["children"]))
            elif include_occupied or node["kind"] == "view":
                yield node

    def get_candidates(self, scene_id: int, view_id: int):
        """Return view-equivalent source parameter sets, each [M,3]."""
        self._check_scene(scene_id)
        if view_id < 0 or view_id >= int(self.view_counts[scene_id]):
            raise IndexError("View ID out of range")
        if self.candidate_layout == "single-position-list":
            positions = self.reference_positions[scene_id, view_id]
            return [np.array([[position[0], position[1], 1.0]], dtype=np.float64)
                    for position in positions]
        n = int(self.candidate_numbers[scene_id, view_id])
        return [np.array(self.view_candidates[scene_id, view_id, c,
                         :int(self.candidate_counts[scene_id, view_id, c])], dtype=np.float64)
                for c in range(n)]

    def get_scene(self, scene_id: int) -> Dict[str, Any]:
        """Return the continuous ground-truth parameters of one scene."""
        self._check_scene(scene_id)
        m = int(self.driver_counts[scene_id])
        k = int(self.obstacle_counts[scene_id])
        return {"scene_id": scene_id, "drivers": np.array(self.drivers[scene_id, :m], dtype=np.float64),
                "driver_strengths": np.array(self.driver_strengths[scene_id, :m], dtype=np.float64),
                "driver_count": m, "obstacle_count": k,
                "obstacle_types": np.array(self.obstacle_types[scene_id, :k], dtype=np.uint8),
                "obstacle_params": np.array(self.obstacle_params[scene_id, :k], dtype=np.float64),
                "obstacle_group_ids": np.array(self.obstacle_group_ids[scene_id, :k], dtype=np.uint16),
                "local_views": np.array(self.local_views[scene_id, :int(self.view_counts[scene_id])], dtype=np.uint16)}

    def rerender_region(self, scene_id, x, y, side, drivers=None, *, strengths=None,
                        height=None, backend="auto"):
        """Evaluate continuous geometry for candidate source positions/strengths."""
        from rind_dataset.runtime import render_scene_region
        return render_scene_region(self, scene_id, x, y, side, drivers,
                                   strengths=strengths, height=height, backend=backend)


try:
    import torch
    from torch.utils.data import Dataset, Sampler

    class RINDTorchDataset(Dataset):
        def __init__(self, root: Union[str, Path], return_scene: bool = False):
            self.base_dataset = RINDDataset(root, return_scene=return_scene)

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, index):
            item = self.base_dataset[index]
            result = {"scene_id": item["scene_id"], "view_id": item["view_id"],
                      "size": item["size"],
                      "view": torch.from_numpy(item["view"].astype(np.int64)),
                      "channels": torch.from_numpy(item["channels"]),
                      "response": torch.from_numpy(item["response"]),
                      "obstacle": torch.from_numpy(item["obstacle"]),
                      "drivers": torch.from_numpy(item["drivers"]),
                      "driver_strengths": torch.from_numpy(item["driver_strengths"])}
            if "obstacle_params" in item:
                result["obstacle_params"] = torch.from_numpy(item["obstacle_params"])
                result["obstacle_types"] = torch.from_numpy(item["obstacle_types"].astype(np.int64))
            return result

    class SceneGroupedSampler(Sampler[int]):
        """Shuffle adaptive views while keeping views from each scene together."""

        def __init__(self, view_counts, shuffle_scenes=True,
                     shuffle_within_scene=True, seed=None):
            self.view_counts = np.asarray(view_counts, dtype=np.int64)
            self.offsets = np.concatenate(([0], np.cumsum(self.view_counts)))
            self.shuffle_scenes = shuffle_scenes
            self.shuffle_within_scene = shuffle_within_scene
            self.seed = seed
            self.epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self) -> Iterator[int]:
            rng = np.random.default_rng(None if self.seed is None else self.seed + self.epoch)
            scenes = np.arange(len(self.view_counts))
            if self.shuffle_scenes:
                rng.shuffle(scenes)
            for scene in scenes:
                views = np.arange(self.view_counts[scene])
                if self.shuffle_within_scene:
                    rng.shuffle(views)
                for view in views:
                    yield int(self.offsets[scene] + view)

        def __len__(self):
            return int(self.offsets[-1])

except ImportError:
    pass
