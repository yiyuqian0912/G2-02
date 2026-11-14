# RIND Phase2

[中文说明](README.zh-CN.md) · [Research chain](../docs/RIND_RESEARCH_CHAIN.zh-CN.md) · [Documentation index](docs/README.zh-CN.md)

The current v6 student reads the original multi-source local response and window coordinates. It predicts the effective contribution count and autonomously extracts binary layers through recurrent subtraction. Only the final count and unordered teacher mask set supervise training; teacher strengths, residual trajectories and ordering are not model inputs.

Count accuracy and per-layer hard-mask IoU ≥0.90 accuracy are the two equally weighted tasks. Response conservation is a feasibility constraint rather than a reconstruction loss. Final v6 test performance is not established yet.

From the repository root, reuse the previous completed v5 cache and best checkpoint:

```bash
unset RIND_MASKS_CONFIG RIND_DECOMPOSITION_CONFIG RIND_TMUX_SESSION
./run_implicit_masks_tmux.sh start all \
  --source-root outputs/masks_disentangled_v5_complex245746 \
  --gpus auto --auto-batch
./run_implicit_masks_tmux.sh status
```

Replace the source path with the actual experiment. All original and appended complex windows are preserved. Cache-only training requires no raw dataset or candidate rendering. The new output is `outputs/<source-name>_implicit_v6/`; old weights are retained for paired evaluation. See the [v6 guide](docs/SEQUENTIAL_COUNT_SHAPE.zh-CN.md).

`research/implicit_slots/` contains current and historical decomposition models, labels, training and evaluation. `research/layered_residual/` retains single-layer physical scoring and previous localization browsers. `src/rind_phase2/` provides data and visibility physics; `dataset_update/` installs and joins releases with geometry-safe splits. Active observation remains a separate research environment.

Effective layers are not physical source identities. Local arithmetic label closure does not prove single-source physical realizability. Historical localization browsers need their matching artifacts; they do not automatically display the current mask-only student. Raw data, caches, checkpoints and detailed outputs remain local and are excluded from Git.
