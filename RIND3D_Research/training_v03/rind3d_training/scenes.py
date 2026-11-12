"""Read-only scene adapters over the two existing certified layouts (no physics is regenerated).

  canary_scenes()   accepted v0.2 canary (SB + MA), public/private layout of rind3d_production
  v01_scenes()      restricted v0.1 lite-core 1K campaign (SB only; labels = private driver.f64)
A SceneRef exposes field paths, exact counts and - only through privileged_labels() - the true sources and
strengths. Student code never receives a SceneRef.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .env import CANARY, V01


@dataclass
class SceneRef:
    release: str                     # "canary-v0.2" | "v0.1-1k"
    scene_id: str
    geometry_id: str
    split: str
    family: str
    profile: str                     # "SB" | "MA"
    K: int
    dir: Path
    geometry: Path
    solid: list                      # per source (identical SOLID bits; CountIndex needs the matching identity)
    responses: list                  # per source binary lit fields
    counts: list                     # per source counts.rindc
    v02_records: Path | None = None  # canary: private window_records.npy (obs-policy-v0.2)
    v02_slot: int | None = None
    extra: dict = field(default_factory=dict)

    def privileged_labels(self):
        """True sources [K,3] float64 and strengths [K] (privileged target channel)."""
        if self.release == "v0.1-1k":
            return np.fromfile(self.dir/"driver.f64", "<f8").reshape(1, 3).copy(), np.ones(1)
        src = json.loads((self.dir/"private"/"sources.json").read_text())
        s = np.array([[float.fromhex(v) for v in p] for p in src["sources_hex"]], np.float64)
        a = np.array([float.fromhex(v) for v in src["strengths_hex"]], np.float64)
        return s, a

    def open_counts(self):
        from rind3d.counts import CountIndex
        from rind3d.pack import RINDPack
        packs = [RINDPack(p, cache_bricks=64) for p in self.responses]
        sols = [RINDPack(p, cache_bricks=64) for p in self.solid]
        cis = [CountIndex.load(c, packs[k], sols[k]) for k, c in enumerate(self.counts)]
        return packs, sols, cis


def canary_scenes(root=CANARY):
    out = []
    seals = {}
    for shard in sorted((Path(root)/"shards").glob("shard-*")):
        for line in (shard/"index"/"scenes.jsonl").read_text().splitlines():
            sc = json.loads(line)
            d = shard/"scenes"/sc["scene_id"]
            seal = json.loads((d/"SEAL.json").read_text())
            seals[sc["scene_id"]] = seal
            K = int(seal["K"])
            prof = seal["profile"]
            resp = [d/"public"/"response.rindb"] if prof == "SB" else [d/"private"/f"src{k}"/"response.rindb" for k in range(K)]
            sol = [d/"public"/"solid.rindb"] + [d/"private"/f"src{k}"/"solid.rindb" for k in range(1, K)]
            out.append(SceneRef("canary-v0.2", sc["scene_id"], seal["geometry_id"], seal["split"], seal["family"], prof, K, d,
                                d/"public"/"geometry.npz", sol, resp, [d/"private"/f"src{k}"/"counts.rindc" for k in range(K)],
                                d/"private"/"window_records.npy", int(seal["slot"])))
    return out


def v01_scenes(root=V01, limit=None):
    out = []
    for d in sorted((Path(root)/"scenes").iterdir()):
        v = json.loads((d/"verification.json").read_text())
        out.append(SceneRef("v0.1-1k", v["scene_id"], v["geometry_id"], v["split"], v["families"][0], "SB", 1, d,
                            d/"geometry.npz", [d/"solid.rindb"], [d/"response.rindb"], [d/"counts.rindc"],
                            extra={"v01_windows": d/"window_records.bin"}))
        if limit and len(out) >= limit:
            break
    return out
