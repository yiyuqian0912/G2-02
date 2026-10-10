# RIND Phase I — Multi-source additive observations

English | [简体中文](README.zh-CN.md)

This version extends the single-source task to several sources with different strengths. It studies **sources outside a fixed observation window**: which complete source configurations can explain its local response? Several configurations may explain the same observation.

**Implemented:** release installation, memory-mapped access, local obstacle masks, separate source channels, source-set rerendering, physical comparison, tree access, size-aware DataLoader, the browser, and a reproducible **joint source-set teacher**. **Pending:** adaptive search, student architecture, training and evaluation. The teacher samples whole configurations and preserves their probabilities; a 2D projection is also available for inspection.

[Design and task semantics](docs/method.md) · [Teacher guide](docs/teacher.md) · [Interfaces](docs/interfaces.md) · [Why 1–4 sources?](docs/source-counts.md) · [Verification record](docs/verification.md)

## 1. What is in this release?

| Property | Current multi-source release |
|---|---|
| Scenes / observations | 10,000 / 367,109 |
| World | Continuous 1024 × 1024; x right, y down |
| Sources per scene | Random integer 1–4; each strength sampled uniformly from [0,1) |
| Response | Sum of visible source strengths; **can exceed 1** |
| Obstacles | Continuous rectangles, ellipses, triangles and polygons, including compound shapes |
| Observation sizes | 16, 32, 64, 128, 256, 512 world units |
| Reference sets per observation | 5 for a one-source scene; 6 for scenes with 2–4 sources |
| Storage | Bit-packed visibility per source plus float64 strengths; decoded on demand |

A sampled point sees a source when their connecting segment is unobstructed. There is no distance decay, reflection or noise. Obstacle pixels have response zero. Pixel `[row,col]` samples `(x+col+0.5,y+row+0.5)`; the grid is sampling precision, not the continuous world geometry.

Quadtree generation splits a source-containing cell into four, records source-free children immediately, and continues to side length 16. Occupied terminal cells are excluded. With several sources, view counts vary; they are not `18 × source_count`.

## 2. Install and browse

Obtain the **multi-source** `RIND-adaptive-v3-10000scenes.zip` from the dataset maintainer. Its download link is not yet listed here; the single-source Drive link is a different release. Put the ZIP beside this README and keep it compressed:

```bash
./scripts/install_data.sh
./scripts/browser.sh
```

Install uv first if necessary. The installer creates this project's `.venv`, verifies release SHA256 checksums, extracts only native data to `data/raw/phase1-multi-source/`, then removes the successfully installed ZIP and its `.zip.sha256` sidecar. It retains both on failure and removes temporary extraction. Repeated installation validates existing data without extracting again; a newly supplied ZIP is then retained. `--keep-archive` preserves a ZIP after successful extraction.

Allow about **7 GB for native data**, plus environment and temporary archive space. Downloaded code and its environment are not installed. This version owns its data directory and uses **`RIND_MULTI_DATA_ROOT`**, independently of the single-source version.

```bash
./scripts/install_data.sh /path/to/RIND-adaptive-v3-10000scenes.zip --keep-archive
export RIND_MULTI_DATA_ROOT=/mnt/data/rind-multi
./scripts/install_data.sh /path/to/RIND-adaptive-v3-10000scenes.zip
./scripts/browser.sh --no-open --port 8767
```

The browser defaults to `http://127.0.0.1:8767`. Switch between total response and individual source layers; pixel inspection reports actual intensity and individual contributions. Display brightness is scaled by total scene strength; numerical values are not clipped or converted to 0/1.

Python is declared **>=3.10**, without a pinned minor version; dependencies must support the interpreter/platform. uv selects a compatible interpreter, and an existing `.venv` keeps its interpreter. Choose explicitly with `UV_PYTHON=3.12 ./scripts/install_data.sh` if needed. Run code through `uv run python` or select this `.venv` in your editor.

## 3. Read an observation

```python
from rind_phase1_multi.data import Phase1MultiDataset

ds = Phase1MultiDataset()  # or root="/path/to/native/data"
print(ds.num_scenes, len(ds))  # 10000, 367109
sample = ds[0]
s, v = sample["scene_id"], sample["view_id"]
R = sample["response"]       # float32 [L,L], additive intensity
M = sample["obstacle"]       # bool [L,L], True = obstacle
W = sample["window"]         # int64 [3]: (x,y,L), world coordinates
```

Only IDs, `response`, `obstacle`, and `window` are returned. Student conditioning is `(R,M,W)`. IDs match records; true source count/positions/strengths, source channels, full geometry and outside-window masks are accessed separately for teacher/evaluation. Keep the original world extent even if an encoder resizes an image.

## 4. Split scenes and load batches

Install PyTorch when training is needed:

```bash
uv sync --locked --extra train
```

Run this example with `uv run --extra train python`:

```python
import json
from pathlib import Path
from rind_phase1_multi.data import Phase1MultiDataset, split_scene_ids, make_dataloader

all_data = Phase1MultiDataset()
# Example ratios; save your team's chosen split once.
splits = split_scene_ids(all_data.num_scenes, ratios=(.8,.1,.1), seed=20260923)
Path("data/splits").mkdir(parents=True, exist_ok=True)
for name, ids in splits.items():
    Path(f"data/splits/{name}.json").write_text(json.dumps(ids))
train_data = Phase1MultiDataset(scene_ids=splits["train"])
loader = make_dataloader(train_data, batch_size=8, num_workers=0)
for batch in loader:
    R = batch["response"]  # torch.float32 [B,L,L]
    M = batch["obstacle"]  # torch.bool [B,L,L]
    W = batch["window"]    # torch.int64 [B,3]
    break
```

Equal-size windows share a batch; different batches can have different L. No resizing or padding is needed. Variable source counts require no padding here because source labels are excluded. Call `loader.batch_sampler.set_epoch(epoch)` each epoch. For worker spawning, create the loader under `if __name__ == "__main__":`.

All views and derived targets of one scene must stay in the same split. Uniform observation sampling gives scenes with more sources more weight because they have more views; use a declared scene-balanced policy or report results separately by source count.

## 5. Teacher/evaluation interfaces

```python
import numpy as np
from rind_phase1_multi.physics import evaluate_sources

sources = ds.get_source_params(s)  # float64 [K,3]: x,y,strength
channels = ds.get_channels(s, v)   # float64 [K,L,L]
raw = ds.get_region(s, *map(int, W))["response"]  # float64 [L,L]
assert np.array_equal(channels.sum(axis=0), raw)

fresh = ds.rerender(s, W, sources)  # float64 [L,L]
parts = ds.rerender(s, W, sources, return_channels=True)
# parts: channels [K,L,L], response [L,L], obstacle [L,L]
result = evaluate_sources(ds, s, v, sources)
print(result["valid"], result["physical_cost"])

scene = ds.get_scene(s)  # continuous geometry and generating sources
references = ds.get_candidates(s, v)  # list of float64 [K_candidate,3] arrays
```

Pass an explicit strength in every source row. A complete source set is one inverse hypothesis. Its physical cost is mean absolute intensity error, compared in float64; the float32 student image is not the renderer sanity target. `evaluate_sources` rejects every source inside the window, outside the world, or inside an obstacle. It places no quadtree-parent restriction. `rerender` is a general forward renderer and can also render window-internal sources; task-domain restrictions belong to physical evaluation/search.

### What are the references?

Reference 0 is the generating set. Multi-source scenes also include its reversed ordering. Four further witnesses split one source's strength between two sources at the **same position**, preserving total intensity. Consequently some witnesses have **K+1 sources**, up to five despite the generation cap of four.

These references demonstrate permutation and intensity-splitting ambiguity. They do **not** offer nine new spatial positions like the single-source release, cover the inverse set, or define teacher probabilities. Never append them to a teacher sampling support. Floating-point summation can differ by a few ulps; physical sanity checks use absolute tolerance `1e-12`. This tolerance is not a training hyperparameter or search acceptance threshold.

### Restore the local-view tree

```python
root = ds.get_view_tree(s)
for leaf in ds.iter_view_leaves(s):
    print(leaf["x"], leaf["y"], leaf["size"], leaf["view_id"])
    local = ds.get_observation(s, leaf["view_id"])
```

Nodes have `kind="split"`, `"view"`, or `"occupied"`. View leaves link to stored observations; occupied terminal leaves have no observation. Children are top-left, top-right, bottom-left, bottom-right. `include_occupied=True` includes occupied leaves. Tree access is teacher/diagnostic-only because it contains information from source-driven partitioning.

## 6. Generate and read a multi-source teacher

Each candidate is a **complete source set**: all positions and strengths together. Its cost is the mean absolute error of the summed response. Sources can be anywhere in the world outside the window and obstacles; there is no quadtree-parent restriction. The baseline samples configurations reproducibly on a uniform position grid, with independent uniform strengths. It supports all six recorded window sizes.

If K is unknown, explicitly declare candidate counts and their prior. This example considers K=1–4 with equal prior mass. The spacing, sample count and temperature below are **demonstration settings**, not validated research defaults:

```bash
uv sync --locked --extra diagnostics
uv run --extra diagnostics rind-multi-teacher \
  --scene-id 1 --view-id 13 \
  --spacing 64 --samples-per-count 1024 --temperature 0.05 \
  --counts 1 2 3 4 --count-prior 0.25 0.25 0.25 0.25 \
  --seed 20260923 --diagnostics --plot \
  --output outputs/teachers/example-scene1-view13.npz
```

It writes a compressed NPZ, a diagnostic JSON, and a PNG comparing observed/rendered responses, local obstacles, spatial projections and count probabilities. For an externally known K, use `--counts 2` and omit `--count-prior`. The teacher never reads the true K to select its candidates.

```python
from rind_phase1_multi.teacher import load_teacher_record
from rind_phase1_multi.diagnostics import source_space_projections

t = load_teacher_record("outputs/teachers/example-scene1-view13.npz")
q = t["teacher_prob"]        # float64 [N], probability of each complete set
C = t["physical_cost"]       # float64 [N], raw mean absolute intensity error
i = int(q.argmax())         # inspect one set; keep all N for a distribution target
K = int(t["source_counts"][i])
parameters = t["sources"][i, :K]  # [K,3] (x,y,strength); inactive rows are padding
maps = source_space_projections(t)
presence = maps["presence"]  # [Ny,Nx], P(at least one source at this grid point)
```

The **joint target** is `q` over complete sets. The spatial presence map is a marginal, can sum to more than one, and loses correlations between sources. It must not be substituted for the joint target. Complete means every declared sample was evaluated; sampling does not exhaust the continuous inverse set. Budget exhaustion fails explicitly and produces no final teacher target. Reference witnesses are checked separately and never enter proposals or normalization.

For direct Python generation, caller-provided candidates, cache/budget details, and a worked example, see the [teacher guide](docs/teacher.md). Formal spacing, sample count, temperature, source-count support and prior remain explicit choices in `configs/phase1.json`.

## 7. Checks and remaining research work

```bash
uv run python -m unittest discover -s tests -v
uv run python -m rind_phase1_multi.checks --data-only
uv run python -m rind_phase1_multi.checks --data-only --full-audit --report outputs/reports/data-check.json
uv run --extra train python -m rind_phase1_multi.checks --data-only --torch --workers 2
uv run python -m rind_phase1_multi.checks --teacher \
  --spacing 128 --samples-per-count 64 --temperature 0.05 \
  --counts 1 2 3 4 --count-prior 0.25 0.25 0.25 0.25 \
  --report outputs/reports/joint-teacher-check.json
```

The full audit checks metadata, bitmaps and analytic reference equivalence for all scenes/views. Physical rerendering samples the first and last scene of each source count and one view of each available size. It does not rerender every observation in the full release.

The teacher check constructs 24 real targets across K=1–4 and L=16–512, checks reference witnesses separately, and compares cached costs with direct full-set rerenders. Its coarse sampling settings exercise the pipeline; they do not establish inverse-space coverage or training quality. Before a formal run, choose scene splits and sampling/temperature settings, measure convergence with larger supports and multiple seeds, and agree the student/training protocol. Adaptive search remains future work.
