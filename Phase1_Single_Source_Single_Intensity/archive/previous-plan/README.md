# RIND Phase I — Project Responsibilities and Deliverables

English | [简体中文](README.zh-CN.md)

This project learns **where a source could be**, given a local response and the window's position. Its output is a distribution of possible locations, including multiple solutions when the observation is ambiguous.

This README tells each team member **which folder to own, what each file is responsible for, and what result to deliver**. Mathematical details are in [docs/method.md](docs/method.md); shared input/output agreements are in [docs/interfaces.md](docs/interfaces.md).

## 1. Scope and current status

| Item | Agreed scope |
|---|---|
| Dataset | Newly generated scenes, each with exactly one source |
| Source strength | Fixed across the dataset; value to be agreed |
| World | 1024 × 1024 continuous coordinate domain |
| Observation windows | Split source-containing regions into four; keep source-free regions without further splitting; exclude source-containing 16 × 16 leaves |
| Student input | Response plus window location and size `(x, y, side)` |
| Model output | Compatibility scores and a distribution over candidate source locations |
| Data separation | Train, validation, and test use different `scene_id` values |
| Team ownership | Members claim tasks in Section 6 |
| First review | Thursday, October 1, 2026 |
| Compute | Pending |

**Current state:** the repository contains documentation, a planning configuration, and empty Python package scaffolds. Research modules and execution scripts remain to be written. The original RIND generator and data have not been found in this folder yet.

**File status used below:**

- **Existing**: present in the repository. A package marker does not mean its module is implemented.
- **Planned**: a proposed file for the assigned owner to create.
- **Generated**: an artifact produced by a completed stage, not a source file to write manually.

The first review is a pilot checkpoint, not the deadline for all final experiments.

## 2. How the work connects

```text
New single-source scenes
        ↓
Source-free quadtree observations + scene-level splits
        ↓
Candidate source locations
        ↓
RIND rerendering + physical compatibility scores
        ↓
Teacher distributions
        ↓
Student training
        ↓
Predicted source distributions + evaluation report
```

| Stage | Responsible folder | Main result | Next recipient |
|---|---|---|---|
| Prepare observations | `src/rind_phase1/data/` | Valid observations with window metadata and split membership | Physics, sampling, and training owners |
| Choose locations to evaluate | `src/rind_phase1/sampling/` | Candidate locations and their support information | Physics and training owners |
| Evaluate physical explanations | `src/rind_phase1/physics/` | Rerendered responses and candidate costs | Teacher-target owner |
| Define the student | `src/rind_phase1/models/` | A model that scores candidate locations | Training owner |
| Teach the student | `src/rind_phase1/training/` | Teacher targets, trained checkpoints, and run records | Evaluation owner |
| Assess the result | `src/rind_phase1/evaluation/` | Metrics, figures, comparisons, and failure cases | Whole team |

## 3. Directory map

This tree shows **existing folders**. Planned files inside each folder are specified in Section 4.

```text
.
├── README.md
├── README.zh-CN.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── phase1.json
├── docs/
│   ├── method.md
│   └── interfaces.md
├── src/
│   └── rind_phase1/
│       ├── data/
│       ├── physics/
│       ├── sampling/
│       ├── models/
│       ├── training/
│       └── evaluation/
├── scripts/
│   └── README.md
├── tests/
│   └── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── outputs/
│   ├── teachers/
│   ├── checkpoints/
│   ├── figures/
│   └── reports/
└── prototypes/
    └── workspace/
```

**Two different uses of “data”:** `src/rind_phase1/data/` contains code that prepares and reads observations. The root `data/` folder stores the actual dataset files.

## 4. Research folders — what each file must achieve

All `.py` files listed in this section are **planned**, except existing `__init__.py` package markers. File boundaries are the initial responsibility split; coordinate changes with the affected owner.

### 4.1 `src/rind_phase1/data/` — Prepare the research dataset

**Responsibility:** provide a new single-source dataset whose observations follow the agreed quadtree rule and whose scenes are separated across splits.

| Planned file | What it must do | Expected result / completion goal |
|---|---|---|
| `generation.py` | Coordinate creation of new single-source scenes using the original RIND generator | A new dataset with recorded generation settings, exactly one source per scene, and one agreed intensity |
| `quadtree.py` | Provide and verify source-free windows under the agreed subdivision rule | Windows have the correct location and size; none contains the source; occupied minimum-size leaves are excluded |
| `splits.py` | Assign whole scenes to training, validation, or test | Saved scene-ID lists with no overlap |
| `dataset.py` | Present observations and their metadata consistently to downstream modules | Every sample has a response, window metadata, and traceable scene/view identifiers; different window sizes remain supported |

**Handoff:** dataset location, split files, a sample inventory by window size, and a few inspectable observations. For a single-source scene, the stated subdivision rule should produce 18 source-free windows. If RIND already supplies correct windows, use and verify them rather than creating a second conflicting dataset definition.

### 4.2 `src/rind_phase1/sampling/` — Define the source locations to consider

**Responsibility:** determine which source hypotheses will be evaluated and ensure the candidate set has a clear spatial meaning.

| Planned file | What it must do | Expected result / completion goal |
|---|---|---|
| `candidates.py` | Produce a uniform candidate set for the initial baseline | A reproducible set of world-coordinate locations that can be shared by teacher and student |
| `support.py` | Identify candidate restrictions from world bounds, observation-window selection, and teacher-only scene geometry | Explicitly labeled validity information; no hidden use of scene geometry in a claimed geometry-free prediction |
| `adaptive.py` | Propose a later candidate set that focuses physical evaluation on promising regions | Fewer physical evaluations with measured retention of plausible regions and documented spatial weights |

**Handoff:** candidate coordinates, validity information, and any spatial weights. The uniform baseline is the first-review priority; adaptive search is a later extension. Existing RIND equivalent-parameter candidates do not replace a search over source positions.

### 4.3 `src/rind_phase1/physics/` — Judge whether a source explains an observation

**Responsibility:** use the known scene and RIND physics to evaluate candidate source locations.

| Planned file | What it must do | Expected result / completion goal |
|---|---|---|
| `renderer.py` | Obtain the local response produced by a specified candidate source in the original scene | The generating source reproduces the stored observation; alternative candidates produce comparable responses |
| `boundaries.py` | Identify meaningful boundaries in observed and candidate responses | Consistent boundary information, including a documented outcome when no boundary exists |
| `response_cost.py` | Measure disagreement between candidate and observed responses, with optional emphasis near observed boundaries | One interpretable response-consistency cost per candidate |
| `edge_cost.py` | Measure spatial mismatch between observed and candidate boundaries | An optional geometric-consistency cost with defined behavior for missing or empty boundaries |
| `cost.py` | Assemble the physical scores used to judge candidates | One consistent final cost per candidate, with response-only and response-plus-edge settings clearly distinguished |

**Handoff:** rerendering verification, candidate costs, and examples of good and poor explanations. This folder supplies physical judgments; it does not train the student. The first baseline uses response cost alone.

### 4.4 `src/rind_phase1/models/` — Define the source-scoring student

**Responsibility:** provide a model that uses the response and window metadata to judge candidate source locations without rerendering.

| Planned file | What it must do | Expected result / completion goal |
|---|---|---|
| `observation_encoder.py` | Represent the local response together with its location and extent | A usable observation representation for every supported window size |
| `coordinate_features.py` | Represent each candidate location, with raw-coordinate and Fourier-feature options | Candidate representations that support a controlled comparison of the two choices |
| `energy_model.py` | Combine the observation and candidate information into compatibility scores | One score per candidate; more compatible candidates receive lower energy |
| `predict.py` | Turn the model's scores on a declared candidate set into a source distribution | A source-space probability map with its coordinates and support recorded |

**Handoff:** a callable model and example scores for observations of different sizes. Ground-truth source coordinates, full scene geometry, obstacle masks, and the complete quadtree are not student inputs.

### 4.5 `src/rind_phase1/training/` — Transfer physical judgments to the student

**Responsibility:** construct supervision from physical costs and produce a trained student that approximates it.

| Planned file | What it must do | Expected result / completion goal |
|---|---|---|
| `teacher_targets.py` | Convert candidate physical costs into a distribution that retains multiple compatible solutions | Saved teacher targets with matching candidate coordinates, support, weights, and provenance |
| `objective.py` | Measure disagreement between teacher and student distributions | A training objective that compares the same locations under the same probability convention |
| `trainer.py` | Coordinate learning and validation, and save run progress | Checkpoints and training records that demonstrate whether the student is learning the teacher targets |

**Ownership boundary:** T3 owns `teacher_targets.py`; T6 owns `objective.py` and `trainer.py`.

**Handoff:** teacher artifacts, a trained checkpoint, its settings, and validation records. First demonstrate a small pilot fit before expanding the experiment.

### 4.6 `src/rind_phase1/evaluation/` — Determine what the model has learned

**Responsibility:** assess distribution quality, ambiguity preservation, physical consistency, generalization, and computational cost.

| Planned file | What it must do | Expected result / completion goal |
|---|---|---|
| `metrics.py` | Quantify agreement with the teacher and physical compatibility | Comparable per-observation and aggregate measurements on a declared common candidate support |
| `visualize.py` | Show observations, teacher distributions, and student distributions together | Figures that reveal plausible regions, missing modes, and overly concentrated predictions |
| `ablations.py` | Compare the proposed components under a consistent evaluation protocol | Evidence for or against edge costs, Fourier features, adaptive search, and the added value of response beyond window metadata |
| `report.py` | Summarize held-out results, window-size differences, timing, and failure cases | A reviewable report connecting each claim to its result and experiment settings |

**Handoff:** metrics, comparison figures, and a concise findings report. Coordinate error alone is insufficient: the model must preserve alternative physically compatible locations when they exist.

## 5. Supporting folders and stored artifacts

### 5.1 Root files and package markers

| Existing file | Purpose | Goal |
|---|---|---|
| `README.md` | Project responsibility map and task board | Each member can identify their files, expected outputs, and handoffs |
| `pyproject.toml` | Python project identity and dependency declaration | One shared package definition for the team |
| `.gitignore` | Separate source-controlled work from generated local files | Large data, checkpoints, and temporary files do not enter routine commits |
| `src/rind_phase1/__init__.py` and each module's `__init__.py` | Package markers; currently documentation-only | Organize the research code into the six named modules |

`src/` holds source code; `src/rind_phase1/` groups the research package. Neither is an experiment-output destination.

### 5.2 `configs/` — Record the agreed experiment

| Existing file | What it must contain | Goal |
|---|---|---|
| `phase1.json` | Dataset scope, observation policy, model inputs, supervision settings, and run settings | Every run can be traced to explicit decisions; unresolved values remain visibly pending |

### 5.3 `docs/` — Keep shared research agreements

| Existing file | What it must contain | Goal |
|---|---|---|
| `method.md` | Detailed method, mathematical formulation, assumptions, and limitations | The team shares the same scientific definition of the task |
| `interfaces.md` | Common records and module input/output agreements | Independently developed modules can exchange data without conflicting meanings |

### 5.4 `scripts/` — Provide complete project actions

Scripts are the team's entry points for running a stage. Research responsibilities remain in the corresponding source modules.

| File | Status | Action and expected output |
|---|---|---|
| `README.md` | Existing | Lists the planned entry points and their status |
| `generate_data.py` | Planned | Produce a new single-source dataset in `data/raw/` and prepared observation records in `data/processed/` |
| `make_splits.py` | Planned | Save scene-level split files in `data/splits/` |
| `build_teacher.py` | Planned | Produce candidate costs and teacher targets in `outputs/teachers/` |
| `train.py` | Planned | Produce student checkpoints and a training run record |
| `evaluate.py` | Planned | Produce metrics, figures, and a findings report for a selected run |

**Goal:** another team member can run a complete stage from its agreed inputs and obtain the documented artifacts.

### 5.5 `tests/` — Establish whether each stage is trustworthy

| File | Status | Question it must answer |
|---|---|---|
| `README.md` | Existing | Which scientific and integration checks must be covered? |
| `test_data.py` | Planned | Are scenes single-source, windows correct, and splits disjoint? |
| `test_physics.py` | Planned | Does rerendering reproduce known observations, and do costs behave sensibly? |
| `test_sampling.py` | Planned | Are candidate restrictions correct and spatial weights accounted for? |
| `test_models.py` | Planned | Can the student score different window sizes consistently without using forbidden inputs? |
| `test_training.py` | Planned | Are targets and predictions aligned, and can a small pilot be learned? |
| `test_evaluation.py` | Planned | Do reported comparisons use consistent support and produce traceable results? |

**Goal:** each module owner supplies evidence for their stage before handing it to the next owner. These test files do not exist yet.

### 5.6 `data/` — Store observations and split membership

The filenames below are **proposed generated artifact conventions**. The original RIND generator retains its native dataset layout.

| Folder | Expected contents | Goal |
|---|---|---|
| `raw/` | Newly generated RIND instance and its native metadata | Preserve the original generated scene truth and responses |
| `processed/` | Prepared sample records plus `observations.json`, an index of their locations, identifiers, and window metadata | Downstream modules can find valid observations without rediscovering dataset structure |
| `splits/` | `train.json`, `validation.json`, `test.json`, each listing scene IDs | Every stage uses the same scene-disjoint partition |

### 5.7 `outputs/` — Store evidence from completed work

Organize artifacts by a run identifier so different experiments remain distinguishable. Names below describe **generated outputs**, not currently present results.

| Folder | Expected artifacts | Goal |
|---|---|---|
| `teachers/` | Candidate coordinates, physical costs, teacher probabilities, and generation metadata | Training supervision is inspectable and traceable to the physical teacher |
| `checkpoints/` | Saved model/training state and its run configuration | A learned student can be restored and identified unambiguously |
| `figures/` | Observation, teacher, student, and comparison images | Reviewers can inspect uncertainty and failure cases visually |
| `reports/` | Per-run `metrics.json`, `run_config.json`, and `summary.md`; a team `first_review.md` | Numerical results, settings, conclusions, and Thursday blockers are recorded together |

`metrics.json` records measured outcomes. `run_config.json` identifies the settings that produced them. `summary.md` explains what the evidence supports and what remains unresolved. `first_review.md` collects each owner's initial deliverable and next step.

Each data/output directory currently contains only a `.gitkeep` placeholder. Its purpose is to preserve the empty folder; it is not a research artifact.

### 5.8 `prototypes/workspace/` — Preserve the earlier interface

| Existing file | Purpose |
|---|---|
| `index.html` | Structure of the archived web workspace |
| `style.css` | Its visual presentation |
| `app.js` | Its content and local assignment interactions |
| `README.md` | Instructions for that prototype |

**Goal:** retain the earlier work as optional reference. No Phase I research task depends on this folder; the root README is the current responsibility map.

## 6. Task ownership and Thursday deliverables

Replace **Unclaimed** with your name. Keep owner assignments synchronized between the English and Chinese README files; these are static documents. Claims apply to the files listed in Sections 4–5. Each owner also supplies the matching checks in `tests/`.

| ID | Files / area to own | Owner | First-review deliverable | Handoff |
|---|---|---|---|---|
| T1 | `src/rind_phase1/data/` | Unclaimed | A small new dataset, verified windows, scene splits, and sample examples | T2, T4, T5, T6 |
| T2 | `physics/renderer.py` | Unclaimed | One generating-source reconstruction and one alternative-source response | T3 |
| T3 | Remaining `physics/` files + `training/teacher_targets.py` | Unclaimed | One response-only physical cost map and its teacher distribution | T6, T7 |
| T4 | `src/rind_phase1/sampling/` | Unclaimed | A uniform candidate set and clearly labeled support restrictions | T2, T3, T6, T7 |
| T5 | `src/rind_phase1/models/` | Unclaimed | Candidate scores for example observations of different sizes | T6 |
| T6 | `training/objective.py`, `training/trainer.py` | Unclaimed | A training-stage prototype; pilot fit if teacher targets are available | T7 |
| T7 | `src/rind_phase1/evaluation/` | Unclaimed | Initial metric outputs or an agreed evaluation specification, plus a teacher figure | T8 |
| T8 | `scripts/`, shared configs/docs, `outputs/reports/first_review.md` | Unclaimed | Combined runnable stages where available, artifact links, owners, and blockers | Whole team |

Paths such as `physics/renderer.py` in this table are relative to `src/rind_phase1/`. Suggested groupings for a smaller team: T1+T2, T3+T4, T5+T6, and T7+T8.

T8 coordinates shared files and run entry points; the module owner remains responsible for the scientific behavior of their stage. T3 depends on T2's rendering and T4's candidate set. T6 depends on T3 and T5. T7 can inspect teacher results before a trained student is ready.

**Each Thursday handoff should answer four questions:**

1. What is available now, and where are the files?
2. What result demonstrates that it works?
3. What does the next owner receive?
4. What is still blocked or undecided?

## 7. What counts as a completed Phase I

The project is complete when the team can trace a held-out observation through a verified physical teacher, a trained student, and a documented evaluation. The final evidence must show whether plausible source regions and their uncertainty are recovered, how results change with window size and unseen scenes, and which optional components provide a measurable benefit.

Before scaling beyond the pilot, resolve the original RIND project location, fixed intensity, scene count, split allocation, and available compute. Scientific settings and thresholds belong in the shared configuration and method notes; they are not implicit decisions for individual folder owners.
