"""Freeze disjoint B2 geometry/source inputs before measuring new methods."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..b2_angular.preservation import sha
from ..generators import DEFAULT, generate
from ..models import record, save
from ..queries import FieldOracle
from ..study import atomic_json, quality, sample_source

CONFIG = Path(__file__).parents[1] / "configs/b2-independent-v1.json"


def freeze(output, config=CONFIG):
    cfg = json.loads(config.read_text())
    output.mkdir(parents=True, exist_ok=False)
    seed = cfg["heldout_master_seed"]
    cases = []
    for ordinal, target in cfg["heldout"].items():
        i = int(ordinal)
        name = f"gaussian_sum-{i:03d}"
        scene = generate("gaussian_sum", seed, i, target, DEFAULT)
        folder = output / name
        save(folder / "geometry.json", scene)
        oracle = FieldOracle(scene)
        containment = oracle.containment(DEFAULT["world_margin"])
        source, source_sampling = sample_source(oracle, seed + i)
        diagnostics = quality(scene, 64)
        peak = np.array(diagnostics["approx_peak_xyz"])
        interior = oracle.classify_box(peak - 2**-20, peak + 2**-20).status
        accepted = containment["status"] == "CERTIFIED" and source is not None and interior == "SOLID"
        row = {
            "case": name,
            "geometry_id": record(scene)["geometry_id"],
            "seed": seed,
            "ordinal": i,
            "occupancy_target": target,
            "style": scene.metadata["style"],
            "kernels": len(scene.kernels),
            "source_xyz_synthetic": source.tolist() if source is not None else None,
            "source_hex": [float(v).hex() for v in source] if source is not None else None,
            "source_sampling": source_sampling,
            "containment": containment,
            "interior": interior,
            "status": "EXPERIMENTAL_INPUT_ONLY" if accepted else "INPUT_ACCEPTANCE_INCOMPLETE",
            "quality": diagnostics,
            "geometry_file_SHA256": sha(folder / "geometry.json"),
            "ready_capable": False,
        }
        atomic_json(folder / "result.json", row)
        cases.append(row | {"result_file_SHA256": sha(folder / "result.json")})
    atomic_json(output / "manifest.json", {
        "config_SHA256": sha(config),
        "freeze_source_SHA256": sha(Path(__file__)),
        "selection": "All predeclared geometry/source attempts, with no runtime screening or replacement",
        "cases": cases,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.output)
