import argparse, json, os
import numpy as np
from rind_phase3 import RIND3D
from rind_phase3.oracle import PhysicalOracle
from rind_phase3.physics import sample_probes, SparsePhysicalCost, full_window_cost


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", default=os.environ.get("RIND3D_DATA_ROOT", "data/RIND3D-v0.1")
    )
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--rerender", action="store_true")
    p.add_argument(
        "--raster-backend",
        choices=("cpu_native_certified", "cuda_certified", "auto"),
        default="cpu_native_certified",
    )
    a = p.parse_args()
    with RIND3D(a.root, split=None) as data:
        observation = data[a.index]
        probes = sample_probes(observation, count=64, seed=7)
        sources = np.random.default_rng(11).uniform(0.01, 0.99, (4, 3))
        with PhysicalOracle(
            data.teacher_scene(observation["scene_id"]), raster_backend=a.raster_backend
        ) as oracle:
            costs, result = SparsePhysicalCost(oracle, probes).evaluate(sources)
            print("Source validity:", result.validity.status)
            print(
                "Sparse costs:",
                costs,
                "point backend:",
                result.backend,
                "pairs:",
                result.evaluated_pairs,
            )
            if a.rerender:
                valid = np.flatnonzero(result.validity.valid)
                if not len(valid):
                    raise RuntimeError(
                        "No valid illustrative candidate; use another index"
                    )
                full = full_window_cost(oracle, observation, sources[valid[0]])
                print(
                    "Full-window cost:",
                    full["cost"],
                    "backend:",
                    full["response"].backend,
                    "equivalent-source certification:",
                    full["equivalent_source_certification"],
                )
            print(json.dumps(dict(oracle.stats), indent=2))


if __name__ == "__main__":
    main()
