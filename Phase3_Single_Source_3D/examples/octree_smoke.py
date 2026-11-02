import argparse, json, os
from pathlib import Path
from rind_phase3 import RIND3D
from rind_phase3.oracle import PhysicalOracle
from rind_phase3.physics import sample_probes, SparsePhysicalCost
from rind_phase3.teacher import OctreeTeacher, UniformRefinement, VariationRefinement


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", default=os.environ.get("RIND3D_DATA_ROOT", "data/RIND3D-v0.1")
    )
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("results/octree-smoke.json"))
    p.add_argument("--strategy", choices=("uniform", "variation"), default="uniform")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--probes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-candidates", type=int, default=64)
    p.add_argument("--max-pairs", type=int, default=4096)
    p.add_argument("--refinements", type=int, default=1)
    a = p.parse_args()
    with RIND3D(a.root, split=None) as data:
        observation = data[a.index]
        probes = sample_probes(observation, count=a.probes, seed=a.seed)
        with PhysicalOracle(
            data.teacher_scene(observation["scene_id"]), max_pairs=a.max_pairs
        ) as oracle:
            policy = (
                UniformRefinement()
                if a.strategy == "uniform"
                else VariationRefinement()
            )
            result = OctreeTeacher(
                SparsePhysicalCost(oracle, probes),
                k=a.k,
                seed=a.seed,
                max_candidates=a.max_candidates,
                max_pairs=a.max_pairs,
                refinement=policy,
            ).run(a.refinements)
            result.update(
                release_id=data.metadata()["release_id"],
                scene_id=observation["scene_id"],
                window_id=observation["window_id"],
                observation_origin=observation["origin"].tolist(),
                observation_side=observation["side"],
                probe_policy=probes.policy,
                oracle_statistics=dict(oracle.stats),
            )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError("Preserve diagnostic output; choose a new filename")
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "dataset_name",
                    "execution_complete",
                    "teacher_support_complete",
                    "candidate_attempts",
                    "physical_source_receiver_pairs",
                    "cells_visited",
                    "cells_refined",
                    "wall_seconds",
                )
            },
            indent=2,
        )
    )
    print(a.output)


if __name__ == "__main__":
    main()
