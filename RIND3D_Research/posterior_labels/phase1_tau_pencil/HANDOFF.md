# Handoff — RIND3D posterior labels, phase 1 (frozen 2026-11-11)

**Read first:** `RIND3D_POSTERIOR_LABELING_REPORT.md` (conclusion first), then `docs/METHOD.md` (lemmas and
proofs) and `docs/RESULTS_TABLES.md` (all numbers, generated).

## State

* **Status.** Phase 1 is complete and frozen: `EVIDENCE_MANIFEST.json` lists the SHA256 of every file. Phase 2
  (GPU-assisted, scene-amortised production for `sb-global-v0.3`) continues in a **new directory**. Nothing here
  is modified after the freeze.
* **Engines.**

  | Engine | Role |
  |---|---|
  | `engine_b3` (+ `run_b3`) | Frozen; it produced every declared-sample number |
  | `engine_b4` | b3 + deferred expansion on a full work queue + exact in-engine clearance-shell subtraction; used only for the reruns of b3's queue-overflow windows |
  | `engine_v1`, `engine_b1`, `engine_b2`, `engine_v3` | Historical; the report §B says why each was superseded |
  | `engine_v2` | Development head (= b4) |
* **Verification.** All verification is brute force over every FREE receiver (`tpl/verify.py`). Points inside
  emitted polytopes must be feasible, and feasible points must lie inside the support. There were 0
  contradictions anywhere.

## Run

* **Python.** The first study's venv: `../RIND3D_PosteriorLabels/.venv/bin/python`.
* **Environment.** `PYTHONDONTWRITEBYTECODE=1` and `NUMBA_CACHE_DIR=/tmp/...`.
* **Inputs.** The canonical RIND3D package and campaign are read-only (`code/rpl/rindenv.py`).
* **Commands.**
  - One window: `code/try_real.py <scene-prefix> <window>`. `TPL_ENGINE=2` selects the dev head.
  - A batch: `code/batch_real2.py` (b3), `code/batch_real_b4.py` (b4).
  - The tables: `code/make_tables.py`.

## Known limitations and open problems (Phase 2 inputs)

1. **Cost at 1024³.** The mean is tens of CPU-seconds per window, with a long tail (report §D). Profiling shows
   the cost is component evaluations (~10⁸–10⁹ per hard window, at 180–660 ns each). These are driven by:
   * re-evaluation of surviving receivers after every split;
   * face-plane splits of components whose planes cross the window (~40% of regions on hard windows).
2. **Target.** Phase-1 labels exclude sources inside the window, the v0.1 rule. `sb-global-v0.3` requires
   including them, and also labelling all-lit, all-dark and no-FREE natural windows (label = prior for
   no-FREE).
3. **v0.1 selection factor.** It is non-local and modelled, not certified: TV 0.02–0.11 on extended
   posteriors. It is irrelevant for v0.3 natural windows (proposals independent of the source). For v0.3
   selected subsets, the documented weights are part of the training law.
4. **Verification cost.** Brute-force verification of side-256 windows can take more than an hour. Its
   receiver order is random since 18:29; earlier tasks used raster order. This is a cost of the study, not of
   labelling.
5. **Queue capacity.** b3 overflowed its work queue (2²⁴ entries) on some side-256 windows. b4 fixes this
   without more memory.

## Incidents

* **Test split.** One unit test read test-split geometry: `outputs/INCIDENT_TEST_SPLIT.md`.
* **Invalid run.** The selection-study rasteriser overflowed on unbounded intervals. That run is kept as
  `outputs/selection_study_BUGGY_unclamped/` (INVALID) and superseded.
* **Stopped batches** (each with a STOPPED.txt):
  - `real_val64_prev24` (v1; verification too slow);
  - `real_val_random90` (b2; queue capacity);
  - `stageA_results_partial_b2`.
* **Driver changes mid-run.** The random-90 b3 batch ran with two drivers (forward and reverse) and with
  reduced side-256 verification from 17:12. Both are recorded in `outputs/real_val_random90_b3/NOTES.txt`;
  per-window sample sizes are in each JSON.

## Machine

The local laptop has 16 logical CPUs, 31 GiB RAM and an RTX 3060 Laptop GPU (6 GiB; unused in phase 1). It was
shared with an unrelated workload during part of the runs, so the reported cost is CPU time, not wall time.
