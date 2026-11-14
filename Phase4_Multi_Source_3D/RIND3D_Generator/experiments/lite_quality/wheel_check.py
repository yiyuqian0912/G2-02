"""Build and read the final wheel without using the editable RIND3D source.

This is a local artifact check. Canonical sibling dependencies remain read-only;
it does not claim a standalone production-generator distribution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

OUT = Path("outputs/lite-v02-research/wheel-final")


def binary_hashes():
    import importlib
    paths = [Path(importlib.import_module("rind3d." + name).__file__)
             for name in ("_native", "_convex", "_research")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def assert_wheel_worker(worker_id):
    import rind3d
    import rind3d.data
    from rind3d import _research
    root = Path(os.environ["RIND_WHEEL_CHECK_ROOT"])
    for module in (rind3d, rind3d.data, _research):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise AssertionError(f"Worker {worker_id} imported editable code: {module.__file__}")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.glob("*.whl")):
        raise FileExistsError("Preserve an existing wheel; use a new artifact version")
    before = binary_hashes()
    cache = Path.home() / ".cache/uv/archive-v0"
    paths = []
    for name in ("scikit_build_core", "pathspec"):
        matches = sorted(cache.glob(f"*/{name}/__init__.py"))
        if not matches:
            raise RuntimeError(f"Offline build dependency unavailable: {name}")
        paths.append(str(matches[-1].parent.parent))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", UV_CACHE_DIR="/tmp/rind-quality-uv-cache")
    env["PYTHONPATH"] = os.pathsep.join([*paths, env.get("PYTHONPATH", "")])
    subprocess.run(["uv", "build", "--offline", "--wheel", "--no-build-isolation",
                    "--config-setting=build-dir=build/lite-quality-wheel", "--out-dir", str(OUT)],
                   check=True, env=env)
    after = binary_hashes()
    if before != after:
        raise AssertionError("Wheel build replaced a source-tree numerical binary")
    wheel = next(OUT.glob("*.whl"))
    (OUT / "build-info.json").write_text(json.dumps({"wheel": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "bytes": wheel.stat().st_size,
        "offline_build": True, "source_binaries_not_replaced": True,
        "source_binary_hashes_before_and_after": before}, indent=2) + "\n")


def read(base):
    wheel = next(OUT.glob("*.whl"))
    with tempfile.TemporaryDirectory(prefix="rind-final-wheel-") as directory:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(directory)
        # Source-independent import must happen before importing any RIND3D
        # module. Leave the separately installed canonical rind dependency.
        sys.meta_path[:] = [f for f in sys.meta_path if "rind3d" not in type(f).__module__.lower()]
        sys.path.insert(0, directory)
        # The normal DataLoader uses spawn. Its fresh interpreters also need
        # to disable the checkout redirector, rather than silently testing the
        # wheel in the parent and editable source in workers.
        Path(directory, "sitecustomize.py").write_text(
            "import sys\nsys.meta_path[:] = [f for f in sys.meta_path if 'rind3d' not in type(f).__module__.lower()]\n"
        )
        os.environ["PYTHONPATH"] = os.pathsep.join([directory, os.environ.get("PYTHONPATH", "")])
        os.environ["RIND_WHEEL_CHECK_ROOT"] = directory
        import rind3d
        import rind3d.data
        from rind3d import _research
        from rind3d.data import RINDDataset, make_dataloader
        root = Path(directory).resolve()
        for module in (rind3d, rind3d.data, _research):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise AssertionError("Editable source contaminated the wheel test")
        expected = json.loads((Path(base) / "report.json").read_text())
        with RINDDataset(base, split=None) as dataset:
            assert len(dataset) == expected["ready_scenes"] * 256
            sample = dataset[0]
            assert set(sample) == {"scene_id", "window_id", "window_origin", "window_size", "structure", "observed_response"}
            oracle = dataset.private_oracle()
            refs = oracle.get_candidate_sources(sample["scene_id"], sample["window_id"])
            assert len(refs) == 1
            counts = oracle.get_window_counts(sample["scene_id"], sample["window_id"])
            assert counts.informative and counts.lit + counts.shadow + counts.solid == sample["window_size"] ** 3
            loader = make_dataloader(dataset, batch_size=1, num_workers=2, seed=7)
            loader.loader.worker_init_fn = assert_wheel_worker
            iterator = iter(loader)
            batch = next(iterator)
            assert "candidate_sources" not in batch
            del iterator
        with RINDDataset(base, split=None, view="candidate_conditioned") as private:
            loader = make_dataloader(private, batch_size=1, num_workers=2, seed=7)
            loader.loader.worker_init_fn = assert_wheel_worker
            iterator = iter(loader)
            batch = next(iterator)
            assert "candidate_sources" in batch
            del iterator
        (OUT / "loader-test.json").write_text(json.dumps({"passed": True, "base": str(base),
            "windows": expected["ready_scenes"] * 256, "scenes": expected["ready_scenes"],
            "normal_and_private_pytorch": True, "exact_count_index_from_wheel": True,
            "editable_rind3d_finder_removed": True, "wheel": wheel.name,
            "spawned_workers_asserted_wheel_import_paths": True,
            "source_modules_not_used": ["rind3d", "rind3d.data", "rind3d._research"]}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "read"))
    parser.add_argument("--base", default="outputs/lite-v02-research/core-fast-mixed-300")
    args = parser.parse_args()
    if args.action == "build":
        build()
    else:
        read(args.base)
