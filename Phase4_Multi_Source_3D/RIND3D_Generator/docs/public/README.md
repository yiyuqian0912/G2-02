# RIND3D — 3D Physical Reasoning Dataset

RIND3D is a synthetic dataset of three-dimensional point-light visibility. A scene contains continuous obstacle geometry, one point source, a complete sampled response field, and local observations. It supports research on inverse visibility, source ambiguity, nonlocal occlusion, and learning from partial observations. This Core distribution retains all nine certified convex-union families.

The obstacles are continuous canonical binary64 geometry in the unit cube. They are not voxel obstacles. A free voxel is lit exactly when its closed segment to the source does not meet any closed obstacle. Endpoint contact and tangency block light. Samples are at voxel centers `(index + 0.5) / 1024`. The complete 1024³ response and solid fields are compressed losslessly. A zero response can mean solid or shadow; use the solid mask to distinguish them.

A **geometry** is a canonical obstacle arrangement. A **source** is one accepted free-space point with certified clearance. A **scene** is one geometry/source pair. A **window** is a local cubic observation of that scene. Core uses one independently generated geometry per scene. Future sources sharing a geometry must stay in the same split.

Each READY_LITE scene has exactly 256 unique informative windows: 85 of side 64, 85 of side 128, and 86 of side 256. Origins are arbitrary legal integer voxel coordinates, not aligned to a grid. Windows exclude the true source and contain both lit and shadowed FREE voxels. Mixed sampling uses soft targets of 30% natural, 50% ten-bin lit/free coverage, and 20% extreme-ratio coverage. Unavailable bins have recorded deficits and bounded fallback. These are selected observations, not unbiased uniform accepted samples. Natural sampling remains a separate reproducible baseline.

Geometry families are rotated boxes, thin slabs, prisms, wedges, random convex polyhedra, multiscale convex mixtures, corridors, barriers, and occluder clusters. Family assignment is balanced cyclically. Complete source/configuration policy is in `manifest.json`. The original 4096-window full profile is separate and unchanged.

**Read the manifest for actual counts.** `scene_count`, `window_count`, `family_counts`, and `split_counts` describe this specific immutable snapshot. The original internally named 1K release is valid only with 1,000 unique geometries/scenes and 256,000 windows. Diagnostic subsets are explicitly marked and are not the 1K release.

The initial RIND3D v0.1 release contains **1,000 independent geometries/scenes and 256,000 windows**. Its geometry-group split is 776 train / 116 validation / 108 test scenes, corresponding to 198,656 / 29,696 / 27,648 windows. Measured distributions, throughput, restart behavior and scientific limits are in `REPORT.md`; any subset or later release must use its own manifest counts.

The default reader returns scene/window IDs, integer origin and side, full canonical geometry (`structure`), and a packed local binary response. Full response and solid fields are also public targets under `full-field-training-v1`. Source coordinates, candidate coordinates, private witnesses and private validation records are withheld. This is a full-field research corpus, **not a blind hidden-field benchmark**: public geometry and fields can support inverse inference. Default inputs must be adapted by researchers if their task permits only local geometry.

Restricted Core production retains exactly one fully verified true-source reference per window. This never proves uniqueness. Finite binary observations can admit many, continuously many, equivalent sources. Additional-source search is disabled for Core. Authorized restricted campaigns provide `private_oracle()`; this public snapshot refuses that interface.

All scenes pass geometry/source/clearance acceptance, complete-field integrity, exact window tests, and independent parametric-clipping A and rational-shadow-projection B physical verification. UNKNOWN and ABORTED fail closed. Unsupported functional geometry is not made READY by a fast renderer. Public checks verify retained file integrity; they cannot rerender hidden-source physical proofs without authorized restricted evidence.

## Install and verify

Use Python 3.12 on Linux x86_64 for the tested installation. The SDK is pure Python with NumPy and Zstandard. Reading requires no CUDA, NVIDIA device, CGAL, GMP extension, generator checkout, or editable install. Python 3.13 is declared compatible but must not be interpreted as a tested universal-platform guarantee. PyTorch is optional; legacy parquet catalogs additionally require PyArrow.

After extracting a snapshot, install the included wheel (or the identical wheel supplied next to the archive):

```bash
python -m pip install ./sdk/rind3d_reader-0.1.0-py3-none-any.whl
rind3d-reader verify .
rind3d-reader info .
```

Before extracting, check the `.tar.zst.sha256` sidecar with `sha256sum -c`. Once the SDK is installed, `rind3d-reader extract ARCHIVE DESTINATION` verifies that sidecar, rejects unsafe archive entries, extracts into a fresh directory, and verifies the release manifest. `verify` checks file SHA and decoded brick CRC; `--metadata-only` is faster and has a narrower guarantee.

```python
from rind3d_reader import RINDDataset, make_dataloader

with RINDDataset("/path/to/snapshot", split="train") as data:
    print(len(data), data.metadata()["split_counts"])
    observation = data[0]
    response_zyx = observation["observed_response"].unpack()
    scene = data.scene(observation["scene_id"])
    counts = scene.counts(tuple(observation["window_origin"]), observation["window_size"])
    print(counts.lit, counts.shadow, counts.solid, counts.lit_ratio)
    roi = scene.roi((17, 29, 43), 79, field="solid", packed=False)
```

Install PyTorch separately for your platform, then use the included `examples/torch_batches.py`. Its `if __name__ == "__main__"` guard is required for multiple worker processes. `make_dataloader` groups equal-sized windows; geometry is a variable-length list of canonical arrays. The packed tensor shape is `[batch, L, L, ceil(L/8)]`, with X as the least-significant-bit axis. A dense 256³ bool window is 16 MiB, so keep worker prefetch and batch size bounded.

Runnable examples from the extracted snapshot:

```bash
python examples/basic.py .
python examples/torch_batches.py . --workers 2
python -m pip install matplotlib
python examples/visualize.py . --index 87 --output ../window.png
```

The visualization displays an actual Z slice, choosing one with the most minority-class FREE voxels by default. Use `--slice` to choose a local Z index. A three-dimensional informative window can still have some completely lit or shadowed slices. Save generated outputs outside the immutable snapshot; integrity verification rejects unexpected added files.

See `FORMAT.md` for exact layouts/APIs, `DATASET_CARD.md` for bias and limitations, `PRODUCTION.md` for restricted campaign operation, and `RELEASES.md` for immutable full/delta snapshots.

**License: pending. Citation authors and bibliographic metadata: pending.** No redistribution license is implied by file availability. The verified public v0.1 archive is linked from the G2-02 root and Phase 3 README; this template does not assert an upload for future snapshots. Do not invent attribution or infer a license from availability.
