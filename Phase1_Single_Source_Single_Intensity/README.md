# RIND Phase I — Responsibilities and Deliverables

English | [简体中文](README.zh-CN.md)

The Chinese README adopts the latest supplied revision, organized as inputs, processing, outputs, and purpose. This English guide summarizes the same responsibilities and interface conventions. The [Chinese module guide](README.zh-CN.md) now explains every input by meaning, purpose, and use; the [interface reference](docs/interfaces.md) also records input routing and configuration ownership.

**Goal: predict a distribution of possible 2D source locations from a local response and its window metadata, preserving ambiguity and multiple solutions.**

The physical teacher rerenders candidates to judge whether they explain the observation. The student learns those judgments and predicts without candidate-by-candidate rerendering. The project contains **seven research files and one shared verification file**.

**Status:** ZIP installation, data reading, the local browser, scene subsets, size-aware batching, and reference rerendering are implemented. The physical costs, search, teacher, student, training, and evaluation remain research placeholders. Data verification does not establish completion of the research pipeline.

**Thomas’s student baseline:** [Standalone implementation and verification](student_baseline/README.md) are available separately. It follows the supplied fixed-size, window-relative student specification and does not implement the newer shared interface described below.

Supplementary references: [method and mathematical definitions](docs/method.md) · [module data contracts](docs/interfaces.md). Both follow the current flat module layout.

## Dataset: download, install, and use

The prepared **single-source, unit-intensity RIND dataset** contains 10,000 scenes and 180,000 local observations (18 per scene). Each scene is a continuous 1024 × 1024 world containing obstacles and one source of strength 1. Response is 1 at an unobstructed pixel center and 0 otherwise; obstacle cells also have response 0 and a separate obstacle mask. The grid sets sampling precision; source positions and obstacle boundaries remain continuous.

### First installation

1. Download the ZIP from [Google Drive](https://drive.google.com/file/d/1_zerwmggBskUtPtkhWTb6Xqs4kIU5L5I/view?usp=sharing) and put it **beside this README and `pyproject.toml`**. Keep it compressed.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available.
3. From this project directory, run:

```bash
./scripts/install_data.sh
```

The project declares **Python >=3.10** and does not pin a minor version. uv chooses a compatible interpreter and the lockfile's matching dependency versions; NumPy, Numba, and optional PyTorch must support that interpreter/platform. Core data access and rerendering have been tested with Python 3.12 and 3.13. An existing `.venv` keeps its current interpreter. To choose another version explicitly, for example, run `UV_PYTHON=3.11 ./scripts/install_data.sh` (uv may recreate the environment).

The script installs the project and its reader into this project's `.venv` using `uv.lock`, verifies the ZIP's internal SHA256 checksums, extracts only the data to `data/raw/phase1-single-source/`, checks array shapes and source strengths, then deletes the ZIP after success. On verification failure it keeps the ZIP and removes its temporary extraction. The downloaded package's Python code and environment are not installed. Repeating the command checks the existing data and does not extract again. Allow about 3 GB of free space for the current data and installation; the basic reader environment is additional. Large arrays are memory mapped, and windows are decoded on demand.

Successful extraction also removes an adjacent `.zip.sha256` file if present. Installed data, `.venv`, and uv's shared dependency cache are retained. If the dataset is already installed, the script skips extraction and leaves any newly supplied ZIP untouched; `--keep-archive` retains the ZIP on a new installation.

If several ZIPs are present, specify the intended file. To retain the archive:

```bash
./scripts/install_data.sh ./RIND-adaptive-v3-10000scenes-single.zip --keep-archive
```

For data on another disk, set the location once before installation and use the same setting when running the model:

```bash
export RIND_DATA_ROOT=/mnt/data/rind-phase1-single
./scripts/install_data.sh
```

Alternatively use `--data-root /path/to/data` during installation and `Phase1Dataset(root="/path/to/data")` in code. The script accepts the supplied single-source release with reference positions; a multi-source ZIP is rejected. The bash entry point is for Linux/macOS. The equivalent Python entry point is `uv sync --locked` followed by `uv run --no-sync rind-install-data`.

### Browse the installed data

```bash
./scripts/browser.sh
```

The script installs the optional Flask/Pillow dependencies and opens the original RIND browser at `http://127.0.0.1:8766`. It reads the same `data/raw/phase1-single-source/` directory, or `RIND_DATA_ROOT`, as `Phase1Dataset`. No second dataset or separate dataset environment is needed.

The page includes scene thumbnails, continuous obstacle/source overlays, variable-size local crops, response layers, diagnostic filters, and exact pixel response values. In this single-source release the total response and the sole source channel agree, with source intensity fixed at 1.

```bash
./scripts/browser.sh --no-open --port 8770
./scripts/browser.sh --data-root /path/to/data
# Equivalent Python entry point after installing the browser extra:
uv run --extra browser rind-browser
```

### Read an observation

Run Python through `uv run python`, or select this project's `.venv` in your IDE:

```python
from rind_phase1.data import Phase1Dataset

ds = Phase1Dataset()  # default installed path, or RIND_DATA_ROOT
print(ds.num_scenes, len(ds))  # 10000 scenes, 180000 observations
sample = ds[0]
R = sample["response"]
window = sample["window"]
s, v = sample["scene_id"], sample["view_id"]
print(R.shape, R.dtype)       # (L, L), float32
print(window)                # [x, y, L] in world coordinates
```

| Observation field | Meaning |
|---|---|
| `scene_id`, `view_id` | Integer identifiers for matching observations to teacher records; not model features |
| `response` | NumPy `float32 [L,L]`; values 0/1 at the original sampling resolution |
| `window` | NumPy `int64 [3]`: top-left world position `(x,y)` and side length `L` |

Only these four fields are returned by `ds[i]`. The student receives response and window; source truth, reference solutions, and geometry are accessed separately. Array indexing is `[row,column]`; x points right and y points down. Pixel `(row,column)` samples `(x+column+0.5, y+row+0.5)`. The six view sizes are 16, 32, 64, 128, 256, and 512. Keep this scale and position even if an encoder resizes images internally.

### Scene splits and PyTorch DataLoader

Choose and save scene splits before training. The ratios below are an example, not a fixed research protocol:

```python
import json
from pathlib import Path
from rind_phase1.data import Phase1Dataset, split_scene_ids, make_dataloader

all_data = Phase1Dataset()
splits = split_scene_ids(all_data.num_scenes, ratios=(0.8, 0.1, 0.1), seed=20260923)
Path("data/splits").mkdir(parents=True, exist_ok=True)
for name, ids in splits.items():
    Path(f"data/splits/{name}.json").write_text(json.dumps(ids))
train_data = Phase1Dataset(scene_ids=splits["train"])
loader = make_dataloader(train_data, batch_size=8, num_workers=0)
for batch in loader:
    R = batch["response"]      # torch.float32 [B,L,L]
    window = batch["window"]   # torch.int64 [B,3]
    break
```

PyTorch is optional; install it with `uv sync --locked --extra train`, and use `uv run --extra train python` when running this example. The default installation installs the reader and CPU renderer without PyTorch. `make_dataloader` groups equal-size windows so batches can stack without padding or changing sampling precision. Batch size can be smaller at the end of each size group. Call `loader.batch_sampler.set_epoch(epoch)` for reproducible new shuffles. When increasing `num_workers` on a platform using spawn, construct the loader under `if __name__ == "__main__":`. All windows of a scene must remain in the same split. Reload saved scene lists for later runs instead of changing the split per experiment.

### Reference sources, rerendering, and the view tree

```python
import numpy as np

references = ds.get_candidates(s, v)  # 10 arrays, each float64 [1,3]
source_xy = references[1][0, :2]      # (x,y); column 2 is strength 1
fresh = ds.rerender(s, sample["window"], source_xy)
assert np.array_equal(fresh, sample["response"])
scene = ds.get_scene(s)              # true source and continuous obstacle geometry
region = ds.get_region(s, 0, 0, 32)  # channels, total response, obstacle mask
root = ds.get_view_tree(s)           # metadata only; no image decoding
for leaf in ds.iter_view_leaves(s):
    observation = ds.get_observation(s, leaf["view_id"])
```

Reference 0 is the generating source. The other nine distinct positions stay in its occupied 16 × 16 tile and were verified to reproduce this window's sampled response exactly. They are examples of compatible sources, not exhaustive coverage or a teacher probability distribution; some differences are very small. They need not match responses elsewhere or at every point between sampled pixel centers. Teacher search must still explore the declared candidate domain.

`rerender` evaluates one hypothetical source at strength 1 in the original scene. It returns NumPy `float32 [L,L]` and rejects sources outside the world or inside obstacles. The caller's search code must also exclude the observation window according to the protocol. This CPU NumPy/Numba renderer is for teacher costs and evaluation; it does not backpropagate gradients to source coordinates. The first call may take longer while Numba compiles. Scene geometry and reference coordinates retain `float64`; the adapter converts observation responses to `float32`.

Tree nodes have `kind="split"`, `"view"`, or `"occupied"`. A view leaf has a `view_id`; an occupied minimum-size leaf contains the source and has no observation. Children are ordered top-left, top-right, bottom-left, bottom-right. `iter_view_leaves` yields observation leaves by default; pass `include_occupied=True` to include occupied leaves.

### Verify the installation and update it later

```bash
uv run python -m rind_phase1.checks --data-only
# Optional: verify actual tensor batches and worker loading
uv run --extra train python -m rind_phase1.checks --data-only --torch --workers 2
```

These commands check data access, scene subsets, size grouping, tree recovery, and sample reference rerendering. They do not claim that teacher generation or training is implemented. `installation.json` records the archive hash and environment lock hash. The reader snapshot lives in `vendor/rind-dataset/`; runtime updates are made in this repository and locked with uv. A future data version should be installed into a new data root and selected explicitly, preserving experiment traceability. The supplied ZIP's Word guide can be read separately if a fuller description of generation and geometry is needed.

## 1. Alignment with the proposal

Retain one source, fixed intensity, continuous 2D occlusion, and scene-disjoint train/validation/test splits. Exclude distance decay, reflections, noise, and material differences.

Two later decisions update the original proposal:

- **Adaptive quadtree windows:** subdivide source-containing regions into four; retain source-free regions immediately. The minimum size is 16; occupied minimum-size leaves are excluded.
- **Student conditioning:** response plus window position and size `(x, y, size)`. Candidate source coordinates are the locations being queried.

| Proposal section | File | Required outcome |
|---|---|---|
| §1 Problem and dataset separation | `data.py` | Provided single-source data, correct windows, scene-disjoint splits |
| §2 Forward rendering for inverse supervision | `physics.py` | Candidate responses comparable to the observation |
| §3 Physical supervision | `physics.py` | Boundary-weighted response cost and optional geometric edge cost |
| §4.1 Costs to supervision | `teacher.py` | Teacher distributions preserving compatible alternatives |
| §4.2 Conditional energy and teacher fitting | `model.py`, `train.py` | Candidate scoring and distribution fitting |
| §4.3 Fourier coordinates | `model.py` | Continuous candidate representation with fine spatial detail |
| §5 Adaptive search | `search.py` | Coarse-to-fine physical evaluation with cost and missed-region analysis |
| §6 Evaluation | `evaluate.py` | Agreement, uncertainty, generalization, and edge-term ablation |
| §7 Integrated method | `checks.py` | Verified handoffs across the complete pipeline |

Adaptive search remains part of the full proposal: establish the uniform baseline first, then complete the adaptive comparison. Edge cost may be disabled, but its benefit must be evaluated. Fourier features belong to the student, not the renderer.

## 2. Minimal structure

```text
.
├── README.md
├── README.zh-CN.md
├── docs/
│   ├── method.md
│   └── interfaces.md
├── configs/
│   └── phase1.json
├── src/rind_phase1/
│   ├── __init__.py
│   ├── install_data.py         # Verified extraction and ZIP cleanup
│   ├── browser.py              # Original browser using the installed Phase I data
│   ├── data.py                 # Implemented observations, subsets, and size-aware loading
│   ├── physics.py              # Rerendering and physical costs
│   ├── search.py               # Uniform candidates and adaptive search
│   ├── teacher.py              # Teacher distributions and saved supervision
│   ├── model.py                # Conditional energy model and prediction
│   ├── train.py                # Training and validation
│   ├── evaluate.py             # Metrics, figures, ablations, and report
│   └── checks.py               # Small end-to-end verification
├── scripts/install_data.sh     # Environment setup and verified ZIP installation
├── scripts/browser.sh          # Optional dependencies and local browser startup
├── vendor/rind-dataset/        # Reader and renderer snapshot, installed by uv
├── uv.lock
├── data/
│   ├── raw/
│   └── splits/
├── outputs/
│   ├── teachers/
│   ├── checkpoints/
│   ├── figures/
│   └── reports/
├── pyproject.toml
├── .gitignore
└── archive/                    # Previous planning and web prototype; inactive
```

`src/rind_phase1/` contains code. Root `data/` and `outputs/` hold datasets and experiment artifacts. Data installation, loading, and data-only checks are implemented. The research files remain responsibility descriptions.

## 3. What each file must accomplish

### 3.1 `data.py` — Supply the Phase I dataset

**Inputs:** the installed data root, explicit scene lists or split ratios, and loader settings. The prepared release has 10,000 scenes, one source of intensity 1, and 18 source-free windows per scene.

**Implemented:** `Phase1Dataset`, `split_scene_ids`, `SizeBucketBatchSampler`, and `make_dataloader`; separate methods expose reference sources, scene geometry, the view tree, and rerendering. `install_data.py` handles checksum verification, extraction, and archive cleanup. Runtime code comes from this repository's reader snapshot.

**Remaining responsibilities:** agree on and save scene-disjoint experiment splits in `data/splits/`, supply observations to the teacher and student stages, and declare any separately generated evaluation scenes. Use the supplied data for the initial integration; additional generation is needed only when an experiment requires new scenes or protocols.

**Completion goal:** observations exclude the source, saved partitions do not overlap, and every observation is traceable. The prepared single-source rule produces 18 source-free windows per scene.

### 3.2 `physics.py` — Judge whether a candidate explains the observation

**Inputs:** original scene, observation, window metadata, candidate coordinates, and fixed intensity.

**Responsibilities:**

1. Place a candidate source in the original scene and rerender the same window.
2. Compute boundary-weighted response disagreement `Lresp`. Weights depend only on the observed boundary; use uniform weights when it is absent.
3. Provide optional symmetric boundary distance `Ledge`, covering displacement, missing boundaries, and extra boundaries.
4. Return `C = αLresp + βLedge`, including the response-only baseline with `β=0`.

**Deliverables:** candidate responses, component costs, and final physical costs for `search.py` and `teacher.py`.

**Completion goal:** the generating source reproduces its observation; all candidates use consistent cost definitions; empty/missing boundaries have defined outcomes. This file produces costs, not probability targets or trained models.

### 3.3 `search.py` — Find candidates worth evaluating

**Inputs:** allowed source domain, window metadata, physical evaluation capability, and candidate budget.

**Responsibilities:**

1. Provide uniform candidates for the baseline and a small reference evaluation.
2. Implement coarse-to-fine search: cover the domain, screen with response consistency, and refine low-cost regions or regions identified by an explicit uncertainty rule.
3. When using edge enhancement, complete the required evaluation of final candidates so the teacher receives comparable costs.
4. Record candidate counts, physical evaluation counts, and final sampling spacing; inspect missed disconnected feasible regions.

**Deliverables:** candidate coordinates, validity information, available physical costs, and search records. Refine retained regions toward a common final spacing and deduplicate coordinates to limit density-induced mass differences. Region bounds may remain internal search metadata but are not required teacher inputs. Candidates lie inside the world and outside the observation window; the physical teacher also excludes obstacle interiors.

**Completion goal:** adaptive search reduces computation relative to uniform reference evaluation and provides evidence that plausible regions are retained. It changes teacher cost, not student inputs or architecture. A design note alone is not final completion.

### 3.4 `teacher.py` — Convert costs into supervision

**Inputs:** observation identifiers, candidate coordinates, final physical costs, temperature, and final sampling spacing.

**Responsibilities:**

1. Coordinate `search.py` and `physics.py` to obtain candidate evaluations, reusing computed results.
2. Construct the proposal's soft distribution: on retained valid candidates, `qᵢ ∝ exp(−Cᵢ/τ)`.
3. Retain probability on compatible alternatives instead of saving only one minimum-cost coordinate.
4. Save complete supervision records so every target probability is associated with the correct candidate.

**Deliverables:** records in `outputs/teachers/` containing sample identifiers, candidates, physical costs, teacher probabilities, and relevant settings.

**Completion goal:** targets normalize correctly and remain traceable to physical evaluations; coordinates, ordering, and validity survive handoff. Apply softmax over retained valid candidates without area factors. Temperature must be positive; invalid candidates receive zero mass; an empty valid set is a failure. Retain raw costs because normalization alone does not establish a good explanation. The target describes relative candidate compatibility, not a calibrated continuous spatial posterior.

### 3.5 `model.py` — Score candidates from the observation

**Inputs:** local response, window `(x,y,size)`, and queried candidate coordinates.

**Responsibilities:**

1. Support different observation sizes while retaining world location and extent.
2. Provide Fourier features for candidate coordinates.
3. Output conditional energy `Eθ(O,v,s)`, with lower energy indicating greater compatibility.
4. Produce a student distribution on the same valid candidates using `pᵢ ∝ exp(−Eᵢ)`. `valid` controls normalization only; it is not an energy-encoder feature.

**Deliverables:** a callable student and example scores/distributions for different window sizes.

**Completion goal:** new continuous coordinates can be queried without physical rerendering. Ground-truth sources, full geometry, obstacle masks, and the complete quadtree are not student inputs. A raw-coordinate comparison may be added within this file without creating another module.

### 3.6 `train.py` — Fit the student to the teacher

**Inputs:** training/validation observations, matching teacher targets, student model, and training settings.

**Responsibilities:**

1. Match observations to the correct targets and compare distributions over the same candidates.
2. Train for distribution agreement rather than unique-coordinate regression or numerical energy-to-cost equality.
3. Record training/validation behavior and save models with their configuration.
4. Establish learning on a small subset before scaling; use validation data for model selection.

**Deliverables:** checkpoints, settings, and training records in `outputs/checkpoints/`.

**Completion goal:** demonstrate that the student learns teacher compatibility structure, beyond merely executing a loop. The result can be restored and evaluated on unseen scenes by `evaluate.py`.

### 3.7 `evaluate.py` — Test the proposal's research claims

**Inputs:** held-out observations, teacher results, trained student, and experiment settings.

**Responsibilities:**

1. Compare teacher and student distributions on common support and report physical compatibility.
2. Inspect retained alternatives in ambiguous observations; show representative one-boundary and two-independent-boundary examples.
3. Assess unseen scenes and unseen obstacle combinations, with results separated by window size.
4. Complete the required `Lresp` versus `Lresp + Ledge` ablation and uniform versus adaptive search quality/cost comparison.
5. Produce figures and a concise report with settings, failure cases, and conclusions. Fourier and window-metadata-only comparisons are supplementary and do not block the first pipeline milestone.

**Deliverables:** observation–teacher–student figures in `outputs/figures/`, metrics and findings in `outputs/reports/`.

**Completion goal:** answer whether source-space constraints and uncertainty are recovered, whether edge cost adds independent value, and whether adaptive search saves computation. Use a common uniform evaluation grid and label spacing. Adaptive candidate probabilities are discrete masses, not continuous densities. Count pruned reference locations as uncovered when measuring search recall. Coordinate error cannot replace this evidence; a second boundary is not guaranteed to concentrate every scene's posterior.

### 3.8 `checks.py` — Verify the handoffs

Data-only checks are implemented via `--data-only`; research handoff checks below remain to be added by their owners.

**Inputs:** a few real RIND samples and the modules above.

**Responsibilities:** check windows and splits, generating-source rerendering, candidate/probability alignment, variable-size model inputs, and a complete observation-to-teacher-to-student handoff.

**Deliverable:** a short pass/fail result identifying the failing stage when applicable.

**Completion goal:** one shared small-scale verification entry point. Each owner adds checks for their stage; no separate testing workstream or per-function test-file plan is required.

## 4. Supporting files and artifact destinations

| File or directory | Responsibility / expected contents |
|---|---|
| `configs/phase1.json` | Shared data, observation, cost, search, model, and training settings; `null` means unresolved, not a runnable default |
| `README.md`, `README.zh-CN.md` | Goals, file responsibilities, and ownership; synchronize assignment changes between versions |
| `src/rind_phase1/__init__.py` | Package marker; no separate research task |
| `pyproject.toml`, `uv.lock` | Project dependencies and locked reader environment; optional `train` extra adds PyTorch |
| `scripts/install_data.sh`, `install_data.py` | Install the environment, verify and extract a local ZIP, and clean the archive |
| `scripts/browser.sh`, `browser.py` | Install optional browser dependencies and visualize the same installed dataset |
| `vendor/rind-dataset/` | Versioned reader and CPU renderer, independent of the downloaded code |
| `.gitignore` | Keep large data, checkpoints, caches, and temporary files out of routine commits |
| `data/raw/phase1-single-source/` | Installed, memory-mapped single-source data |
| `data/splits/` | `train.json`, `validation.json`, and `test.json` scene lists |
| `outputs/teachers/` | Physical costs and teacher distributions reusable by training |
| `outputs/checkpoints/` | Student models, training state, and run settings |
| `outputs/figures/` | Evaluation and presentation figures |
| `outputs/reports/` | Per-experiment `metrics.json`, `summary.md`, and team `first_review.md` |
| `.gitkeep` | Preserve empty data/output directories; not an experimental result |
| `archive/` | Previous README versions, notes, empty scaffolds, and web prototype; old paths/tasks are inactive and require no further work |

The installation script and data-only verification commands above are runnable. Put future research run entry points in their corresponding core files; teacher generation and training commands are not implemented yet.

## 5. Minimum handoff agreement

| Record | Required content |
|---|---|
| Observation | `scene_id`, `view_id`, `response`, `window=(x,y,size)` |
| Teacher supervision | Sample identifiers, `candidate_xy [N,2]`, `valid [N]`, `physical_cost [N]`, `teacher_prob [N]`, `temperature`, and `config_id` |
| Student result | `candidate_xy`, candidate-aligned `energy` and `probability`, with model/configuration identifiers retained in run records |

World coordinates span `[0,1024]`, with x rightward and y downward; arrays use `[row,column]`. Scene IDs, true sources, and geometry may be retained in records but are not student features.

**Three necessary interpretation notes:**

- Source-driven window selection makes window location and size informative. Declare this observation process. Reproducing the same single-source quadtree leaf requires a source inside its parent and outside the leaf. That restriction is disabled by default: the current teacher is a fixed-window response-compatibility Gibbs target, not the full generative posterior conditioned on window selection.
- The teacher knows hidden scene geometry; the student does not. Identical observable inputs may correspond to different teacher maps, so exact recovery of every scene-specific target is not guaranteed.
- The teacher excludes obstacle-interior candidates using geometry. If student evaluation uses this teacher-provided support, label it explicitly rather than presenting it as geometry-free deployment. Deployment also needs a predeclared public candidate set: teacher-selected adaptive support leaks information even without passing a mask. Training on retained valid candidates does not automatically suppress energies outside that support.

## 6. Self-selected ownership and Thursday milestone

First review: **Thursday, October 1, 2026**. Owners and compute remain pending. Claim a row; one person may own several related files.

| Task | Files | Owner | Minimum first-review deliverable |
|---|---|---|---|
| A Data | `data.py` | Yushu He | Install provided data, show window examples, and save agreed scene splits |
| B Physical supervision | `physics.py`, `teacher.py` | Unclaimed | Generating-source reconstruction plus one response-cost map and teacher distribution |
| C Candidate search | `search.py` | Unclaimed | Working uniform candidates and a defined next adaptive-search deliverable |
| D Model and training | `model.py`, `train.py` | Unclaimed | Scores for unequal-size windows; small training verification when targets are ready |
| E Evaluation and summary | `evaluate.py` | Unclaimed | One teacher figure, evaluation checklist, and completed-stage summary |
| Shared checks | `checks.py` | Each module owner | Verify available stages and identify disconnected handoffs |

Dependencies: **A → B/C → teacher targets → D → E**. B and C first agree on candidate records and physical evaluation inputs. D can verify model inputs/outputs early; E can inspect teacher results before training finishes.

Each handoff states **where the files are, what is implemented, what evidence exists, and what is missing**. Distinguish plans/placeholders from executed results.

## 7. Final completion

Phase I ultimately requires verified single-source data and correct splits, a verified physical teacher, a conditional energy student, distribution-fitting training, held-out evaluation, edge-term ablation, and adaptive-search quality/cost evidence.

For the first milestone, connect **one observation → candidate physical costs → teacher distribution → student scores → visualization**, then scale training and experiments. Keeping the first milestone small does not remove the proposal's later research obligations.
