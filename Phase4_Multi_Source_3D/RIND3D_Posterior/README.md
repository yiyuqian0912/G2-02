# rind3d-posterior: certified single-source global posterior labels for RIND3D

> **2026-11-13.** Production FastLabel v3.4 lives in this package, `rind3d_posterior.fastlabel` (statistical certificates on
> top of this engine's outer bounds; scene-format adapters for the 1K and the 100K; Gate A in `fastlabel/gate_a.py`; see
> `docs/FASTLABEL_CORE.md`). `../RIND3D_Dataset1K/rind3d_1k/fastlabel/` is its 1K compatibility layer.
> `experiments/fastlabel/` keeps its research record (`REPORT.md`, `CERTIFICATE_AUDIT.md`,
> `GPUVERIFY.md`) and the research Gate A harness (`validate_verify.py`); benchmark outputs and the earlier
> `RIND3D_Research/posterior_labels` studies (METHOD.md, Phase-1/2 docs) are in the archive tag `archive/rind3d-research-2026-11-13`.

This package is the maintained form of the tau-pencil engine (TPE), Phases 1–2.

## What a label is

**Input.** A local window W of the N³ receiver grid, with these parts:
* the receiver classes `cls[x, y, z]`: 0 = response-inactive FREE (legacy "dark"), 1 = response-active FREE ("lit"), 2 = SOLID;
* the window origin;
* the scene's convex bodies as halfspace rows `P[c, i] = (a, b, c, d)`, with body = {a·x + d ≤ 0}.

**Target.** The source-excluded single-source posterior Uniform(F_W):

    F_W = {s ∈ D(G) \ B̄(W) : the segment s→x is unobstructed exactly for the response-active FREE receivers x of W}

* Visibility is closed-solid, closed-segment.
* D(G) = (0,1)³ minus the solid, with clearance > 2⁻²⁶.
* B̄(W) is the closed window box.

**Output.**
* Records (cell, lo, hi, kind) that bound the volume of F_W inside each 128³ source cell.
* The label q = L/ΣL.
* A deterministic TV bound `tv_bound(L, U)` on TV(q, Uniform(F_W)). This is sparse: there is no dense 128³ tensor.

## Deterministic Local-first research solver

The isolated CBC research entry point reads one v3 channel and writes a sparse 128³ posterior with a deterministic TV bound on its decoded bytes when the bound is established. It has an exact SOLID consistency check, a rational integration backend and an optional projected receiver-patch experiment. [Mathematics and usage](docs/ASTRA_XLF_CERTIFIED_INTERFACE.md) · [Matched results and limitations](docs/ASTRA_XLF_CERTIFIED_RESULTS.md)

## Engines

| Engine | What it is |
|---|---|
| `engine="cpu"` | The reference: Phase-1 engine b4 with counters (`tpl/engine_p2.py`). |
| `engine="gpu"` | The exact GPU evaluation pass: `tpl/engine_g2.py` + `tpl/gpu_pass3.py`. FIFO-preserving device compaction, exact on-device capacity guard, value-aware FP32 filter with FP64 values where they are used. |

**Evidence for the GPU engine.** It reproduces the reference's records and decision statistics bit for bit on every tested window:
* the 12 val windows at 1024³;
* 5 smoke windows;
* 160-task streams;
* `tests/test_regression.py`.

This is software equivalence with the FP64 reference. It is not a proof of real-arithmetic correctness. That rests on the decision margins of the Phase-1 predicates (`docs/` in `../RIND3D_Research/posterior_labels/phase1_tau_pencil`).

## Usage

```python
import rind3d_posterior as RP
rec = RP.label_window(P, K, cls, origin, N, engine="gpu")     # or "cpu"
cells, q, tv = RP.point_estimate(rec)
```

**Environment.** numba ≥ 0.60 and numpy ≥ 2, plus `cupy-cuda12x` for the GPU engine. Tests: `PYTHONPATH=src python -m pytest tests`.

## Measured cost (RTX 3060 Laptop + Ryzen 7 5800H; details in `RIND3D_Research/posterior_labels/phase2_gpu` of the archive tag `archive/rind3d-research-2026-11-13`)

* **128³ windows.** 0.16–1 CPU-s per informative label.
* **1024³ windows.** 1–265 CPU-s per label; L = 256 windows dominate.
* **GPU latency.** The GPU engine is 7.1× faster per window.
* **Machine throughput.** 6 GPU-assisted + 10 CPU workers reach 2794 windows/h, which is 1.55× a fully used 16-thread CPU.

## Independent continuous-geometry experiments

The rational inverse-visibility prototype reconstructs global posterior labels from the observed channel and canonical geometry. The report includes exact multicell examples, bounded approximations, unresolved windows, independent physical rechecks and the cost of falling back to Tau-pencil. [Research report](docs/ASTRA_GLOBAL_INVERSE.md) · [Reproduce the experiments](experiments/global_inverse/README.md)
