# RIND / G2-02

[中文说明](README.zh-CN.md) · [Full research report (Chinese)](docs/RIND_RESEARCH_CHAIN.zh-CN.md) · [Experiment status](docs/PROJECT_STATUS.zh-CN.md)

RIND studies source localization from local responses in a 1024×1024 occluded world. Positive source intensities add linearly through binary visibility, without distance attenuation. The main student does not receive full-scene obstacle geometry.

## Research workflow

- **Phase1:** infer a full-field single-contribution location distribution from its local response and window coordinates. A multiscale boundary/region/global encoder queries candidate coordinates. Formal test results are available.
- **Phase2:** infer the effective contribution count and overlapping binary masks from the original multi-source observation. The current v6 student performs autonomous recurrent subtraction, supervised only by the final count and unordered mask set. Response conservation is a feasibility constraint. Final v6 performance is not established yet.
- **Next:** connect predicted layers to Phase1, then test multi-window history fusion and budgeted observation selection. The acquisition environment and reference policies exist; end-to-end localization gains remain unverified.

The two current primary metrics are exact effective-layer count accuracy and the fraction of teacher layers recovered with hard-mask IoU ≥0.90. Effective layers are not physical source identities. The 32-step capacity preserves existing labels and does not imply 32 physical sources.

## Run the current experiment

From the repository root, install the environment or select an existing interpreter:

```bash
./setup.sh
export RIND_PYTHON="$PWD/.venv/bin/python"
```

The current workflow requires the previous completed v5 mask cache and best checkpoint. Replace the source experiment below with its actual location. It preserves the entire cache and fixed evaluation cohorts; raw data and candidate teacher rendering are unnecessary for cache-only training.

```bash
unset RIND_MASKS_CONFIG RIND_DECOMPOSITION_CONFIG RIND_TMUX_SESSION
./run_implicit_masks_tmux.sh start all \
  --source-root outputs/masks_disentangled_v5_complex245746 \
  --gpus auto --auto-batch
./run_implicit_masks_tmux.sh status
```

Long jobs run in tmux and save logs/checkpoints. Resume is at saved epoch boundaries. For evaluation without retraining:

```bash
./run_implicit_masks_tmux.sh start evaluate --gpus auto --auto-batch
```

See the [v6 guide](Phase2_Multi_Source/docs/SEQUENTIAL_COUNT_SHAPE.zh-CN.md), [Phase2 document index](Phase2_Multi_Source/docs/README.zh-CN.md), and [data installation guide](Phase2_Multi_Source/dataset_update/README.zh-CN.md). Historical entry points remain for reproducibility.

## Three-dimensional phases

**Phase3: RIND3D v0.1, single source.** [`Phase3_Single_Source_3D/`](Phase3_Single_Source_3D/README.md) reads the released RIND3D v0.1 dataset: 1,000 certified scenes and 256,000 observation windows.
* It provides restricted local-observation loading, certified physical queries, an interactive scene viewer and an exploratory adaptive-octree teacher, in its own `uv` environment.
* [Download RIND3D-v0.1.tar.zst](https://drive.google.com/file/d/1PmMp39En-uc4QTgACd_EpS4mTZ3wO7yx/view?usp=drivesdk): 633,273,091 bytes, SHA256 `1f5a2ff2bd9ac4a479a074b33e0d06f160706e4369b199144ee117693913ee1e`. Setup and viewer instructions are in the Phase3 README.

**Phase4: RIND3D-100K, multiple sources.** RIND3D-100K studies global source localization from limited local observations in three-dimensional multi-source worlds.
* The pipeline includes certified physical rendering, efficient probabilistic labelling with FastLabel, and Student training.
* See [`Phase4_Multi_Source_3D/README.md`](Phase4_Multi_Source_3D/README.md). Handoff: [`HANDOFF.md`](Phase4_Multi_Source_3D/HANDOFF.md); operator commands: [`PRODUCTION_RUNBOOK.md`](Phase4_Multi_Source_3D/PRODUCTION_RUNBOOK.md); contract: [`RIND3D_100K_V3_CONTRACT.md`](Phase4_Multi_Source_3D/RIND3D_100K_V3_CONTRACT.md).
* Status: the v3 production pipeline (FastLabel v3.4 statistical labels) is validated on a laptop; the 100K dataset has not been generated yet.

## Repository

| Path | Purpose |
|---|---|
| `Phase1_Single_Source_Single_Intensity/` | Single-contribution physics, models, training, evaluation and visualization |
| `Phase2_Multi_Source/` | Multi-source data, residual labels, decomposition models, evaluation and active observation research |
| `Phase3_Single_Source_3D/` | RIND3D v0.1 data adapter, scene viewer, certified physical queries and exploratory single-source teacher |
| `Phase4_Multi_Source_3D/` | RIND3D-100K: generator, certified posterior labels (tau-pencil, FastLabel), 100K dataset pipeline, Student research |
| `docs/` | Research chain, status and compact numerical evidence |
| `tools/` | Repository checks and reproducible server packages |
| `Literature_Review/` | Team literature reviews |

Phase1's `part_e/` and legacy student adapter remain runtime dependencies. Raw datasets, environments, caches, model weights and detailed outputs remain local and are excluded from Git.

```bash
"$RIND_PYTHON" tools/check_repo.py
"$RIND_PYTHON" tools/package_update.py
```

The source-only package is `dist/RIND_Phase2_Update.tar.gz`, with the stable extraction prefix `RIND_Phase2_Training_Minimal`.
