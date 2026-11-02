# RIND3D — 3D Physical Reasoning Dataset

**RIND3D v0.1 — Initial release containing 1,000 certified 3D scenes and 256,000 observation windows.** This Phase 3 workspace makes the existing data usable and provides certified physical queries and a small adaptive-teacher experiment. It does not implement a 3D student network or launch training.

The permanent dataset name is **RIND3D**. Scene count and scientific profile are versioned metadata. `Lite-Core-1K` is the initial internal production profile, and the historical manifest identifier is retained unchanged as provenance. The public archive filename is `RIND3D-v0.1.tar.zst`; renaming the outer file changes none of its contents or hashes.

[Download RIND3D-v0.1.tar.zst](https://drive.google.com/file/d/1PmMp39En-uc4QTgACd_EpS4mTZ3wO7yx/view?usp=drivesdk). Drive permission is **Anyone with the link — Viewer**, verified by API readback and a complete unauthenticated download. Size: **633,273,091 bytes**. SHA256: `1f5a2ff2bd9ac4a479a074b33e0d06f160706e4369b199144ee117693913ee1e`. The installer below downloads this exact file and checks both archive and extracted scientific integrity.

## Install and load

The tested platform is Linux x86_64 with Python 3.12; the fresh-checkout reader/native/CUDA adapter checks also passed on Python 3.13. This does not establish support on other operating systems. Reading needs no CUDA, C++ build or generator. Start from this directory:

```bash
uv sync --extra dev --extra train
uv run python examples/download_dataset.py --output data
export RIND3D_DATA_ROOT="$PWD/data/RIND3D-v0.1"
uv run python examples/inspect_dataset.py
uv run python examples/load_window.py --index 85
uv run python examples/pytorch_dataloader.py --workers 2
```

PyTorch is optional: omit `--extra train` when reading with NumPy only. To install an already downloaded archive, use `uv run python examples/download_dataset.py --output data --archive /path/to/RIND3D-v0.1.tar.zst`. The installer streams downloads, handles Drive confirmation forms, verifies size/SHA256, rejects unsafe archive entries, verifies the immutable manifest and atomically publishes a new directory. Partial downloads restart safely; a valid installed dataset is never deleted or overwritten.

```python
from rind_phase3 import RIND3D, make_dataloader

with RIND3D("data/RIND3D-v0.1", split="train") as data:
    sample = data[0]
    response_zyx = sample["observed_response"].unpack()
    solid_zyx = sample["local_solid_mask"].unpack()
    print(sample["origin"], sample["side"], data.statistics(0).lit_ratio)
    batch = next(iter(make_dataloader(data, batch_size=2, num_workers=0)))
```

The default student sample has exactly six fields: scene ID, window ID, integer XYZ origin, side, local response and local solid mask. It contains no full geometry, full response target, source coordinates, reference solutions or witnesses. The archive itself uses `full-field-training-v1`: it publicly includes full geometry and response/solid fields. Consequently the adapter enforces the local-only student view; the corpus is not a blind hidden-field benchmark. See [data and coordinates](docs/DATASET.md).

All scenes have complete 1024³ response/solid fields and 256 unique informative windows: 85/85/86 at sides 64/128/256, with arbitrary integer origins. Both lit and shadowed FREE voxels occur in every window and truth lies outside. The nine certified convex-union families and mixed natural/uniform/extreme observation policy are preserved. Geometry-group splits contain 776/116/108 train/validation/test scenes. Source references stay private; one reference never proves uniqueness.

## Enable exact teacher physics

```bash
uv sync --extra dev --extra train --extra physics
RIND_PYTHON="$PWD/.venv/bin/python" ../RIND3D_Generator/scripts/build.sh
uv run python examples/physical_query.py --index 0
uv run python examples/physical_query.py --index 0 --rerender
uv run python examples/octree_smoke.py --output results/octree.json
```

Native build prerequisites and dependency licenses are in [RIND3D_Generator](../RIND3D_Generator/README.md). Arbitrary source/receiver point queries use certified native CPU or the original canonical reference, in bounded Cartesian batches. They do **not** generate a complete field. Whole observation windows can additionally use the certified CUDA renderer with `--raster-backend cuda_certified --rerender`; this requires both an NVIDIA driver and NVRTC, separately from CPU PyTorch. Arbitrary-point CUDA is not provided in this adapter. Source acceptance, clearance, closed-segment contact and tangency semantics are preserved; unresolved physical results fail closed. See [PhysicalOracle](docs/PHYSICAL_ORACLE.md).

The octree starts with exactly eight 512³ source-space cubes, samples k=4 points per visited cube, evaluates sparse physical mismatch, then splits one selected cube into eight 256³ children. Defaults cap the diagnostic at 64 candidate attempts and 4,096 source/receiver pairs. Both breadth-first and exploratory cost-variation policies are available. Unvisited/uncertain cells are retained; no homogeneity or complete probability target is claimed. [Mathematical design](docs/ADAPTIVE_TEACHER.md) explains volume weighting and the distinction between physical cost variation and neural gradients.

## Checks and collaboration

```bash
uv run pytest -q
RIND_TEST_CUDA=1 uv run pytest -q tests/test_oracle.py
uv run python examples/real_smoke.py --root "$RIND3D_DATA_ROOT" --cuda --output results/real-smoke.json
```

Native/data tests require the corresponding extras/build/data; unavailable optional capabilities are explicitly skipped. CUDA tests must run outside environments that hide the device. The real smoke covers nine families, all three window sizes, original-reference comparisons and a bounded octree. It selects the least-component scene among the first eight of each family and compares eight original-reference receivers per window by default (`--reference-probes 32` enables the larger diagnostic). This bounds an integration check, not a production performance benchmark or a change to scientific acceptance. Original CGAL comparisons can still take minutes; use the native backend for routine teacher queries. Keep outputs outside the immutable dataset. See [development boundaries](docs/DEVELOPMENT.md).

License and citation authors remain pending. The download URL and archive checksum are also packaged in `rind_phase3/release.json`; no permissive license is inferred from public access.
