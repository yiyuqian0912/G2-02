# RIND / G2-02

[中文说明](README.zh-CN.md) · [Project status](docs/PROJECT_STATUS.zh-CN.md)

RIND studies source localization from local responses in a two-dimensional occluded environment. Sources have positive intensities, binary visibility and additive contributions without distance attenuation. The main student does not receive full-scene obstacle geometry.

- **Phase1:** a physics-supervised conditional energy model predicts a full-field single-contribution location distribution. The current encoder combines multiscale boundaries, response regions, global context and relative candidate geometry.
- **Phase2:** a slot model predicts overlapping contribution masks, strengths and existence from the original multi-source observation. Current training focuses on decomposition; Phase1 localization can be connected later.
- **Data:** the old 10k configurations and the new 100k multi-source configurations are joined without moving repeated geometries across splits. Training responses use the installed random-strength variant.

The 20k-window decomposition run is complete. The 100k-window expansion workflow is available and reuses the old training prefix and locked validation/test sets. Completion of that larger experiment and the final multi-view localization system have not yet been established.

## RIND3D — 3D Physical Reasoning Dataset

**RIND3D v0.1 — Initial release containing 1,000 certified 3D scenes and 256,000 observation windows.** [Phase 3](Phase3_Single_Source_3D/README.md) adds restricted local-observation loading, certified physical queries and a bounded adaptive-octree demonstration. The completed dataset is available now; the 3D teacher/student design remains research in progress.

[Download RIND3D-v0.1.tar.zst from Google Drive](https://drive.google.com/file/d/1PmMp39En-uc4QTgACd_EpS4mTZ3wO7yx/view?usp=drivesdk) · Anyone with the link — Viewer · 633,273,091 bytes.

SHA256: `1f5a2ff2bd9ac4a479a074b33e0d06f160706e4369b199144ee117693913ee1e`

```bash
cd Phase3_Single_Source_3D
uv sync --extra dev --extra train
uv run python examples/download_dataset.py --output data
export RIND3D_DATA_ROOT="$PWD/data/RIND3D-v0.1"
uv run python examples/inspect_dataset.py
uv run python examples/pytorch_dataloader.py --workers 2
```

Reading requires no GPU or generator build. Use the separate Phase 3 environment: its NumPy/native physics dependencies differ from the 2D stack. The public training archive includes canonical geometry and complete response/solid fields; the default Phase 3 student adapter exposes only window identity, origin/size, local response and local mask. Hidden sources and private references are excluded from the archive. This is the `full-field-training-v1` policy, rather than a blind hidden-field benchmark.

After installing the dataset, launch the [interactive RIND3D viewer](Phase3_Single_Source_3D/docs/VIEWER.md) from `Phase3_Single_Source_3D`:

```bash
uv sync --extra viewer
./scripts/browser.sh
```

Open **http://127.0.0.1:8765** to explore all scenes, global 3D lit/shadow volumes, observation windows and exact XYZ slices. Volume previews have adjustable 64³/128³/256³ display resolution; the stored 1024³ fields stay unchanged. The tool requires no CUDA or native geometry build and works offline. True-source markers require explicitly configured, matching local original production records; public downloads keep those coordinates hidden.

The dataset's permanent name is **RIND3D**. Historical `Lite-Core-1K` identifiers remain internal scientific provenance. All nine certified geometry families, mixed lit/shadow sampling, 85/85/86 size quotas at 64/128/256 and stable 776/116/108 geometry-group splits are preserved. One private reference does not imply a unique source. License and citation metadata remain pending. See [coordinates and data](Phase3_Single_Source_3D/docs/DATASET.md), [PhysicalOracle](Phase3_Single_Source_3D/docs/PHYSICAL_ORACLE.md), [teacher design](Phase3_Single_Source_3D/docs/ADAPTIVE_TEACHER.md) and [integration acceptance](docs/RIND3D_INTEGRATION.md).

## Start

```bash
./setup.sh
export RIND_PYTHON="$PWD/.venv/bin/python"
# Install data separately; see the phase-specific guides.
./run_masks_tmux.sh start all --gpus auto
./run_masks_tmux.sh status
```

To expand an existing `masks_only_v1` run:

```bash
./run_masks_expanded_tmux.sh start all --gpus auto
```

See [the expansion guide](Phase2_Multi_Source/docs/MASKS_EXPANDED.zh-CN.md) for prerequisites and outputs. Training runs in tmux, saves checkpoints and supports resuming after disconnection. Existing environments can be selected with `RIND_PYTHON`.

## Repository

| Path | Purpose |
|---|---|
| `Phase1_Single_Source_Single_Intensity/` | Single-contribution physics, models, training, evaluation and visualization |
| `Phase2_Multi_Source/` | Multi-source data, residual labels, slot decomposition, training and evaluation |
| `Phase3_Single_Source_3D/` | RIND3D data adapter, interactive scene viewer, certified physical queries and exploratory single-source octree teacher |
| `RIND3D_Generator/` | Self-contained canonical geometry, CPU/CUDA rendering, independent verification and resumable generation |
| `tools/` | Repository checks and reproducible server packages |
| `docs/` | Cross-phase status, compact results and maintenance notes |
| `Literature_Review/` | Team literature reviews |

Raw datasets, caches, environments, checkpoints and transfer archives remain local and are excluded from Git. Phase1's `part_e/` and legacy student adapter are still runtime dependencies and intentionally retained.

```bash
"$RIND_PYTHON" tools/check_repo.py
"$RIND_PYTHON" tools/package_update.py
```

The package is written to `dist/RIND_Phase2_Update.tar.gz`. It contains source and configuration, with no raw data, caches or model weights. The extraction prefix remains `RIND_Phase2_Training_Minimal` for existing server installations.
