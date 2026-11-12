# Phase 2 plan — amortized, GPU-assisted posterior label production (sb-global-v0.3)

**Frozen objective.** Cached global source-posterior labels with TV ≤ 0.05 guarantees (primary; ≤ 0.10
secondary) for `sb-global-v0.3` windows, at practical production cost. Phase 1
(`RIND3D_PosteriorLabels2`) is frozen and read-only.

**Compute.** Colab is the primary research compute: GPU PoC, first batch ≤ ~2 GPU-hours, no paid upgrades.
Drive holds the persistent artefacts. The local CPU is used only for light orchestration and small
correctness checks.

## Step 0 — target audit (sb-global-v0.3)

* **Population.** Natural windows: origin uniform, independent of (G, S). The source may lie inside the
  window (closed box), and source-containing windows are included. No-FREE windows have label = prior
  (Uniform D(G)) cell masses.
* **Fresh scenes.** The v0.3 physical generator is frozen v0.2 (RIND3D_Production). Verify in the repository.
  Generate a small number of fresh single-source scenes with the actual v0.3 contract. Historical v0.1
  scenes and the v0.2 canary are used for benchmarking only, clearly labelled.
* **Engine changes (correctness first).**
  - Remove the s ∉ B(W) exclusion.
  - Handle sources inside the window: receivers on both sides of the source, so the component aspect is fixed
    per source region and receiver-side classes are crossed by the source.
  - Check against brute force.

## Step 1 — profile the b4 engine

Profile per window: stage 1, stage 2 evaluations, face splits, expansions and survivor re-evaluation, on
Stage-A and 1024³ windows. Identify the dominant terms.

## Step 2 — scene-level amortization (CPU first, measured)

* Per-scene geometry preprocessing.
* Per-scene stage-1 sharing across windows.
* A receiver τ cache for overlapping windows.
* Batched windows.

Measure cold-start vs steady-state cost per scene and per certified label.

## Step 3 — GPU candidates (Colab)

* Batched entry evaluation on GPU as a *filter* (COVER/NOCOVER/PARTIAL with margins), with the CPU confirming
  every certifying decision; or provably conservative GPU predicates.
* GPU batched point-membership (brute-force verification, Monte Carlo statistical certification).
* GPU full-field rendering, for scene-level structures.

Record the GPU model and CPU type, GPU-seconds and CPU-seconds separately.

## Step 4 — P64 profile study

Compare P64, P102 and P256 on:
* coverage (spatial, shadow and solid boundary);
* category diversity (informative, extreme, rare, nonlocal);
* intra-scene correlation;
* labelling cost per scene;
* storage.

Propose a clean selection law (natural plus weighted selected windows).

## Step 5 — early stopping with certified TV

* Best-first region order.
* A per-cell interval bound.
* Statistical completion of unresolved regions (99%, adaptive-safe).
* The total time to TV ≤ 0.05 / 0.10 vs to exactness.

## Step 6 — promote to 1024³

Promote only what improves at 128³.

## Step 7 — production cost model

100K × {256, 102, 64}: CPU-core-hours and GPU-hours separately, cold vs steady state.
