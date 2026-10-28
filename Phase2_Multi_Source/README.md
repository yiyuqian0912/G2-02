# RIND Phase2

[中文说明](README.zh-CN.md) · [Current decomposition training](docs/MASKS_ONLY.zh-CN.md) · [100k-window expansion](docs/MASKS_EXPANDED.zh-CN.md)

The current student reads the original multi-source local response and predicts overlapping effective-contribution masks, positive strengths and slot existence. Supervision uses the existing minimum-positive residual/BFS decomposition. It does not use source identities to decide the labels, and mask-only training does not render candidate positions or instantiate the Phase1 locator.

From the repository root:

```bash
./run_masks_tmux.sh start all --gpus auto
./run_masks_expanded_tmux.sh start all --gpus auto
```

Use the second command only after the base `masks_only_v1` cache and best model exist. It expands the actual training windows from20k to100k while preserving the original2k validation and2k test windows. The20k run is complete; the expanded workflow is available but its full server result has not yet been established.

`research/implicit_slots/` contains the current model, label/cache preparation, trainer and metrics. `src/rind_phase2/` provides physical rendering, geometry and data tools. `dataset_update/` installs the new release and preserves geometry-based split isolation. `research/layered_residual/` keeps prior localization experiments and browsers.

The full localization pipeline remains available through `run_decomposition*_tmux.sh` at repository root. Old localization browsers require the matching original artifacts; they do not automatically visualize the new mask-only student.

Raw data, weights and generated outputs are not stored in Git. Arithmetic decomposition labels are effective contributions, not a guarantee of unique physical-source recovery. See the project status and phase-specific documents for completed experiments and limitations.
