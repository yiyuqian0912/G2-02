# RIND / G2-02

[中文说明](README.zh-CN.md) · [Project status](docs/PROJECT_STATUS.zh-CN.md)

RIND studies source localization from local responses in a two-dimensional occluded environment. Sources have positive intensities, binary visibility and additive contributions without distance attenuation. The main student does not receive full-scene obstacle geometry.

- **Phase1:** a physics-supervised conditional energy model predicts a full-field single-contribution location distribution. The current encoder combines multiscale boundaries, response regions, global context and relative candidate geometry.
- **Phase2:** a slot model predicts overlapping contribution masks, strengths and existence from the original multi-source observation. Current training focuses on decomposition; Phase1 localization can be connected later.
- **Data:** the old 10k configurations and the new 100k multi-source configurations are joined without moving repeated geometries across splits. Training responses use the installed random-strength variant.

The 20k-window decomposition run is complete. The 100k-window expansion workflow is available and reuses the old training prefix and locked validation/test sets. Completion of that larger experiment and the final multi-view localization system have not yet been established.

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
| `tools/` | Repository checks and reproducible server packages |
| `docs/` | Cross-phase status, compact results and maintenance notes |
| `Literature_Review/` | Team literature reviews |

Raw datasets, caches, environments, checkpoints and transfer archives remain local and are excluded from Git. Phase1's `part_e/` and legacy student adapter are still runtime dependencies and intentionally retained.

```bash
"$RIND_PYTHON" tools/check_repo.py
"$RIND_PYTHON" tools/package_update.py
```

The package is written to `dist/RIND_Phase2_Update.tar.gz`. It contains source and configuration, with no raw data, caches or model weights. The extraction prefix remains `RIND_Phase2_Training_Minimal` for existing server installations.
