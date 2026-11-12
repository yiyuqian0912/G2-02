# Dataset ↔ Teacher interface proposal (rind3d-production-contract-v0.2)

**Status.** This is a proposal from the dataset branch. It is implemented as `rind3d_production.teacher.TeacherPhysics` and `rind3d_production.loader.ProductionDataset`, and tested in `tests/test_production.py`.

**What it does not do.**
- It embeds no Teacher architecture, loss or sampler.
- It does not touch the Teacher branch's repository (`G2-02-3d/Phase3_Single_Source_3D`) or its experiments.

**Compatibility.** The interface wraps the same certified adapter the Phase 3 oracle already uses:
- `rind3d.rendering.context.GeometryContext`;
- `NativeSource.native.points` for arbitrary points;
- certified CPU/CUDA rasters for windows.

An existing `PhysicalOracle` therefore stays valid on v0.2 scenes. Every scene ships canonical `public/geometry.npz` exactly as v0.1 does.

## 1. Identities and versions (stable)

| Field | Where | Meaning |
|---|---|---|
| `contract_id` | `contract.json` | `rind3d-production-contract-v0.2` |
| `profile` | `public/scene.json` | `rind3d-csg-single-binary-v0.2` or `rind3d-csg-multi-additive-v0.2`. Derived variants carry their own names. |
| `scene_id` | `public/scene.json` | SHA256(profile ‖ geometry_id ‖ source float64 bytes ‖ strength float64 bytes) |
| `geometry_id` | `public/scene.json` | Canonical content hash of the CanonicalBundle; the split group |
| `window_id` | `public/windows.npy` | Index within the scene; the window box is `[origin, origin+L)` in 1024³ voxels |
| oracle identity | `TeacherPhysics.ref()` | Producer scientific identity, numerical-policy hash and the certification file SHA256 (recorded in the campaign's `FROZEN.json`) |

## 2. Legal source domain and prior

- **Domain.** The open world (0,1)³, strictly outside the closed solid, with a certified clearance greater than 2⁻²⁶. This is the v0.1 driver rule. `TeacherPhysics.legal(sources)` returns certified booleans.
- **Dataset prior.** Independent uniform proposals in (0,1)³, conditioned on legality.
- **Multi-source sets.** Additionally pairwise separation of at least 1/32. Strengths are i.i.d. U[0.1, 1] in binary64, and the source count is K ∈ {2, 3, 4}.
- **Uncertainty.** Hypothetical sources that fail legality are refused with an error, never silently moved.

## 3. Batched physical queries (certified unless stated)

| Call | Returns | Cost (measured, RTX 3060 laptop, 16 threads; medians) |
|---|---|---|
| `visibility(S, X)` | int8 [S, N]: 1 lit, 0 shadow, −1 SOLID | Certified native clipping, about 1.2–1.5 M pairs/s per thread and 8–9 M pairs/s on 16 threads. Per-source bind 0.26 ms. |
| `response(S, a, X)` | combined Σ a_k lit_k (binary64, channel order) plus channels | as above |
| `window(S, origin, L, backend)` | dense per-source lit [S, L, L, L] plus SOLID | CUDA certified raster: 0.4–2 ms per 64³ window after an 8 ms source bind. CPU certified raster: 0.8–36 ms per 64³. |
| `free_mask(origin, L)` | FREE receivers of a window | stored SOLID mask, 1–40 ms per window read |
| `receiver_bank(origin, L, design, k, seed)` | indices, world coordinates, inclusion probabilities, Horvitz–Thompson weights | negligible |

**Hybrid screening (opt-in, never a label).** A float32 triangulated mesh plus Embree or Warp BVH can pre-screen candidate sources. For example, Warp runs at about 2·10⁸ rays/s, roughly 25× certified 16-thread CPU.

The screen must be labelled `screened`:
- On 26,735 silhouette-grazing rays the screen disagreed with canonical truth 16% of the time.
- Every disagreement vanished after certified arbitration of probe-flagged rays.
- The arbitrated hybrid costs about 5 M rays/s, slower than certified CSG on 16 threads.

Use it only to rank or prune candidates, and evaluate the finalists with `visibility()`.

## 4. Observations: dense versus sampled, weights and mismatch

**Canonical observation.** The complete local response of the window: `observed_response` plus `observed_solid`.
- Single-source: the response is binary.
- Multi-source: the response is binary64, decoded from the window's local palette.

The Teacher can always build any receiver bank from it.

**The single-source mismatch is unchanged:**

  E(s) = (1/|F|) Σ_{x ∈ F} 1[lit_s(x) ≠ y(x)],   with F the FREE receivers of the window.

**Sampled receivers estimate E(s).** They do not redefine it. `receiver_bank` returns first-order inclusion probabilities π_i:
- **uniform** (π = k/|F|): Σ_i w_i c_i / |F| with w_i = 1/π_i is an unbiased estimate of E(s) for any fixed s.
- **stratified** (response classes; Teacher-only because the design depends on y): proportional allocation, at least one receiver per class, with π_i = k_h/N_h.

**Annealed quantities.** For quantities like Z(τ) = Σ exp(−E/τ), the unbiasedness of the estimate of E does not carry over to exp(−Ê/τ). This is the low-temperature Jensen bias the Teacher branch measured (+22% in Z at τ = 0.01 with 256 receivers). The dataset therefore supports:
- full-window E(s) on demand: the CUDA raster at 8 ms per candidate per 64³ window, or the stored field for the true source;
- independent banks with distinct seeds, for split-sample or debiasing estimators.

The dataset does not choose an estimator.

**Multi-source mismatch.** The dataset declares the observable, R(x) = Σ a_k lit_k(x). It does not choose between L1 on R, per-channel binary or set-matching costs. Per-channel targets are privileged.

## 5. Public Student versus private Teacher fields

| Data | `student_local` | `student_geometry` | `teacher` (research_access=True) |
|---|---|---|---|
| window origin, size, profile, scene/window id | yes | yes | yes |
| local response (binary or combined) and local SOLID mask | yes | yes | yes |
| declared uniform sparse sample, with coordinates, π and w | optional | optional | optional |
| canonical geometry arrays | no | yes | yes |
| sources, strengths, K, per-source window channels | no | no | yes |
| window subset tags (natural / informative / extreme / rare / nonlocal), exact counts, nonlocal estimate | no (used internally for sampling only) | no | yes |
| `physics()` → `TeacherPhysics` | no | no | yes |

**Measured from the previous Gaussian study** (it applies to any additive profile): a scene-level response palette reveals the strengths, and response-dependent index sets leak unobserved structure. The v0.2 layout therefore:
- stores multi-source public observations with a **local** palette per window (validated: no value absent from the window);
- has no public global multi-source response;
- allows only response-independent (uniform) sparse designs in student views;
- keeps window subset tags private.

The single-source public `response.rindb` is the complete field, as in v0.1 (full-field training policy). A strictly local release must export windows only.

## 6. Observation distribution the Teacher can rely on

Per scene the recorded windows are, per side L ∈ {64, 128, 256}:

| Subset | Windows per side | Rule |
|---|---|---|
| natural | 16 | uniform, unfiltered: includes all-lit, all-shadow, SOLID-only and source-containing windows (flagged) |
| informative | 8 | mixed, source outside the window |
| extreme | 4 | minority ≤ 1% |
| rare | 2 | minority ≤ 64 receivers |
| nonlocal | 4 | ≥ 90% of sampled shadow receivers blocked outside the window |

- Every record carries its proposal index, so inclusion designs are reproducible.
- Shortfalls are recorded, never filled with non-qualifying windows.
- Difficult windows are therefore an explicit, re-weightable subset, not a distortion of the base distribution.

## 7. What the Teacher branch is asked to confirm

1. `visibility()` / `window()` semantics equal the current `PhysicalOracle` on v0.2 scenes (closed segments, FREE receivers only). The dataset-side test is `test_teacher_physics_matches_stored_fields`.
2. Whether a GPU point-query kernel (arbitrary source–receiver pairs, certified) is worth adding. Today the GPU path is window rasters only. Dense Teacher costs per candidate are already 8 ms on the GPU against about 0.2 s on one CPU thread.
3. The multi-source cost family to declare when the multi-source Teacher starts (it is not chosen here).
4. Whether rare-mode studies need an additional subset, for example windows with widely separated near-equivalent sources. That needs a bounded equivalence search, which is not part of this canary.
