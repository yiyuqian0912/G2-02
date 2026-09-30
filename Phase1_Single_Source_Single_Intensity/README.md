# RIND Phase I — Responsibilities and Deliverables

English | [简体中文](README.zh-CN.md)

The Chinese README adopts the latest supplied revision, organized as inputs, processing, outputs, and purpose. This English guide summarizes the same responsibilities and interface conventions. The [Chinese module guide](README.zh-CN.md) now explains every input by meaning, purpose, and use; the [interface reference](docs/interfaces.md) also records input routing and configuration ownership.

**Goal: predict a distribution of possible 2D source locations from a local response and its window metadata, preserving ambiguity and multiple solutions.**

The physical teacher rerenders candidates to judge whether they explain the observation. The student learns those judgments and predicts without candidate-by-candidate rerendering. The project contains **seven research files and one shared verification file**.

**Status:** folders, configuration, and responsibility placeholders exist; research functionality is not implemented. The original RIND code and data still need to be connected. A file's presence does not mean its task is complete.

Supplementary references: [method and mathematical definitions](docs/method.md) · [module data contracts](docs/interfaces.md). Both follow the current flat module layout.

## 1. Alignment with the proposal

Retain one source, fixed intensity, continuous 2D occlusion, and scene-disjoint train/validation/test splits. Exclude distance decay, reflections, noise, and material differences.

Two later decisions update the original proposal:

- **Adaptive quadtree windows:** subdivide source-containing regions into four; retain source-free regions immediately. The minimum size is 16; occupied minimum-size leaves are excluded.
- **Student conditioning:** response plus window position and size `(x, y, size)`. Candidate source coordinates are the locations being queried.

| Proposal section | File | Required outcome |
|---|---|---|
| §1 Problem and dataset separation | `data.py` | New single-source data, correct windows, scene-disjoint splits |
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
│   ├── data.py                 # Data generation integration, windows, splits, loading
│   ├── physics.py              # Rerendering and physical costs
│   ├── search.py               # Uniform candidates and adaptive search
│   ├── teacher.py              # Teacher distributions and saved supervision
│   ├── model.py                # Conditional energy model and prediction
│   ├── train.py                # Training and validation
│   ├── evaluate.py             # Metrics, figures, ablations, and report
│   └── checks.py               # Small end-to-end verification
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

`src/rind_phase1/` contains code. Root `data/` and `outputs/` hold datasets and experiment artifacts. The core Python files currently contain responsibility descriptions only.

## 3. What each file must accomplish

### 3.1 `data.py` — Supply the Phase I dataset

**Inputs:** original RIND project location, scene count, fixed intensity, seed, and split settings.

**Responsibilities:**

1. Connect to RIND generation and create new single-source scenes at the agreed intensity without overwriting existing data.
2. Supply source-free quadtree windows. Reuse and verify existing RIND windows when they already satisfy the protocol.
3. Partition whole scenes into training, validation, and test; all windows of a scene belong to the same partition.
4. Expose consistent observations while preserving variable sizes and world-coordinate meaning.

**Deliverables:** new data in `data/raw/`, scene lists in `data/splits/`, and readable samples containing at least `scene_id`, `view_id`, `response`, and `window=(x,y,size)`. Source truth and geometry are supervision/evaluation information only.

**Completion goal:** observations exclude the source, scene partitions do not overlap, and samples are traceable. The single-source subdivision rule from 1024 down to 16 should produce 18 source-free windows per scene; verify this property.

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
4. Record candidate counts, physical evaluation counts, and necessary spatial weights; inspect missed disconnected feasible regions.

**Deliverables:** candidate coordinates, validity information, available physical costs, and search records. Candidates lie inside the world and outside the observation window; the physical teacher also excludes obstacle interiors.

**Completion goal:** adaptive search reduces computation relative to uniform reference evaluation and provides evidence that plausible regions are retained. It changes teacher cost, not student inputs or architecture. A design note alone is not final completion.

### 3.4 `teacher.py` — Convert costs into supervision

**Inputs:** observation identifiers, candidate coordinates, final physical costs, temperature, and necessary spatial weights.

**Responsibilities:**

1. Coordinate `search.py` and `physics.py` to obtain candidate evaluations, reusing computed results.
2. Construct the proposal's soft distribution: for equal-weight candidates, `qᵢ ∝ exp(−Cᵢ/τ)`.
3. Retain probability on compatible alternatives instead of saving only one minimum-cost coordinate.
4. Save complete supervision records so every target probability is associated with the correct candidate.

**Deliverables:** records in `outputs/teachers/` containing sample identifiers, candidates, physical costs, teacher probabilities, and relevant settings.

**Completion goal:** targets normalize correctly and remain traceable to physical evaluations; coordinates, ordering, and weights survive handoff. Use a uniform spatial prior and cell-area weights: teacher mass is proportional to `area_weight × exp(−physical_cost / temperature)`. Save `cell_bounds [N,4]` with coordinates. Refined parent cells must not overlap their children in the final area accounting. For cells crossing window or obstacle boundaries, clip, subdivide, or document valid-area approximations; a valid center does not imply a fully valid cell. Normalize only valid candidates; invalid candidates have zero probability. Valid areas and temperature must be positive. An empty valid set is an explicit failure. If another sampling scheme is introduced, define its weights accordingly. Retain raw costs because normalization alone does not establish a good explanation.

### 3.5 `model.py` — Score candidates from the observation

**Inputs:** local response, window `(x,y,size)`, and queried candidate coordinates.

**Responsibilities:**

1. Support different observation sizes while retaining world location and extent.
2. Provide Fourier features for candidate coordinates.
3. Output conditional energy `Eθ(O,v,s)`, with lower energy indicating greater compatibility.
4. Produce a student distribution on the same valid candidates using `pᵢ ∝ area_weightᵢ × exp(−Eᵢ)`. Equal areas reduce this to the original softmax. `valid` and `area_weight` control normalization only; they are not energy-encoder features.

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

**Completion goal:** answer whether source-space constraints and uncertainty are recovered, whether edge cost adds independent value, and whether adaptive search saves computation. For variable-area heatmaps, display probability divided by represented area as density; region probability is the sum of cell masses. Coordinate error cannot replace this evidence; a second boundary is not guaranteed to concentrate every scene's posterior.

### 3.8 `checks.py` — Verify the handoffs

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
| `pyproject.toml` | Project identity, Python requirement, and actual runtime dependencies |
| `.gitignore` | Keep large data, checkpoints, caches, and temporary files out of routine commits |
| `data/raw/` | New data in RIND's native layout |
| `data/splits/` | `train.json`, `validation.json`, and `test.json` scene lists |
| `outputs/teachers/` | Physical costs and teacher distributions reusable by training |
| `outputs/checkpoints/` | Student models, training state, and run settings |
| `outputs/figures/` | Evaluation and presentation figures |
| `outputs/reports/` | Per-experiment `metrics.json`, `summary.md`, and team `first_review.md` |
| `.gitkeep` | Preserve empty data/output directories; not an experimental result |
| `archive/` | Previous README versions, notes, empty scaffolds, and web prototype; old paths/tasks are inactive and require no further work |

There is no separate scripts hierarchy or detailed test-directory plan. Put a stage's eventual run entry point in its corresponding core file. These entry points are not implemented yet, so no generation/training commands are claimed.

## 5. Minimum handoff agreement

| Record | Required content |
|---|---|
| Observation | `scene_id`, `view_id`, `response`, `window=(x,y,size)` |
| Teacher supervision | Sample identifiers, `candidate_xy [N,2]`, `cell_bounds [N,4]`, `valid [N]`, `area_weight [N]`, `physical_cost [N]`, `teacher_prob [N]`, `temperature`, and `config_id` |
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
| A Data | `data.py` | Unclaimed | Small new single-source dataset, window examples, and scene splits |
| B Physical supervision | `physics.py`, `teacher.py` | Unclaimed | Generating-source reconstruction plus one response-cost map and teacher distribution |
| C Candidate search | `search.py` | Unclaimed | Working uniform candidates and a defined next adaptive-search deliverable |
| D Model and training | `model.py`, `train.py` | Unclaimed | Scores for unequal-size windows; small training verification when targets are ready |
| E Evaluation and summary | `evaluate.py` | Unclaimed | One teacher figure, evaluation checklist, and completed-stage summary |
| Shared checks | `checks.py` | Each module owner | Verify available stages and identify disconnected handoffs |

Dependencies: **A → B/C → teacher targets → D → E**. B and C first agree on candidate records and physical evaluation inputs. D can verify model inputs/outputs early; E can inspect teacher results before training finishes.

Each handoff states **where the files are, what is implemented, what evidence exists, and what is missing**. Distinguish plans/placeholders from executed results.

## 7. Final completion

Phase I ultimately requires new single-source data and correct splits, a verified physical teacher, a conditional energy student, distribution-fitting training, held-out evaluation, edge-term ablation, and adaptive-search quality/cost evidence.

For the first milestone, connect **one observation → candidate physical costs → teacher distribution → student scores → visualization**, then scale training and experiments. Keeping the first milestone small does not remove the proposal's later research obligations.
