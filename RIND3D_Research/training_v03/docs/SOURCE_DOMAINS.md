# Source domains per task profile (v0.3)

Notation: geometry G; closed solid Σ(G); world (0,1)³; window W with closed box B(W) = [o/1024, (o+L)/1024]³; observation O_W = (local response, local SOLID mask, origin, side, modality).

## Global legal domain

D(G) = { p ∈ (0,1)³ : p ∉ Σ(G), certified clearance(p) > 2⁻²⁶ }.

This is the v0.1/v0.2 driver rule. It is certified with the native convex predicates (`TeacherPhysics.legality`, which uses the same code as `rind3d_production.scenes.legal_source`).

## Dataset prior

| Profile | Prior |
|---|---|
| SB | S ~ Uniform(D(G)), drawn as uniform proposals in (0,1)³ conditioned on certified legality. Strength 1. |
| MA | K follows the campaign schedule (canary: 2, 3, 4 cycled, i.e. uniform). S₁…S_K are i.i.d. Uniform(D(G)) conditioned on ‖S_i − S_j‖ ≥ 1/32. a_k are i.i.d. U[0.1, 1]. |
| G | nine certified convex-union families, uniform by slot (family = slot mod 9) |

## Per task profile

| | sb-global-v0.3 | sb-source-excluded-v0.3 | ma-additive-v0.3 |
|---|---|---|---|
| Prediction domain | (0,1)³ (support D(G)) | (0,1)³ \ B(W) | (0,1)³ per element |
| Source may lie in window | **yes** | **no** (declared condition) | yes |
| Population | W ~ Uniform origins ⟂ (G, S) | W ~ Uniform \| S ∉ B(W) | as sb-global |
| Target | p(S \| O_W, W) | p(S \| O_W, W, S ∉ B(W)) | p({(S_k, a_k)} \| O_W, W) |
| Selection conditioning | subsets condition on S ∉ B(W), so `w_global_mixture` is required | subsets select on O_W only (plus the declared condition), so weights are optional (covariate shift only) | as sb-global |
| Source-containing windows | included; flag privileged | excluded (not samples of this task) | included |
| How the Student knows the task | task id / head (the global task) | task id / head (the excluded task) | modality = 1 |

## Telling source-containing and source-excluding windows apart

The flag `contains_source` is computed exactly on the closed box. It is privileged: it is a function of the hidden source.

The two tasks are separate conditional problems, and a model must be told which one it is solving. If a model is trained on a silent mixture of both, its posterior mass near the window is biased. The source-excluded task cannot be emulated by filtering the global task at inference time, because the Student does not know whether its window contains a source.

## Legality outside the window

The Student sees only the local SOLID mask, so legality outside the window is not an input. A candidate that is legal locally may be illegal globally. That error is measured with the privileged legality oracle, either `TeacherPhysics.legality` or Phase 3 `source_valid`. It is never encoded as a label.

**Box convention.** Phase 3 `Window.contains` uses the half-open box [o, o+L), while the dataset uses the closed box. The two differ only on the box boundary, which has measure zero under the continuous prior. `test_08_candidate_legality` checks that they agree off the boundary.

## No-FREE windows

A no-FREE window (FREE = 0) carries no response. It is part of the natural population, and the Student can see that it is all SOLID.

The mismatch is undefined for such a window. TeacherPhysics returns `status="no_free"` with NaN, or raises `NoFreeReceivers` in strict mode. Measured in the interop audit:
- v0.2 `TeacherPhysics.receiver_bank` raises `ZeroDivisionError`;
- Phase 3 `sample_probes` raises `ValueError` ("Observation has no FREE receiver");
- Phase 3 `full_window_cost` silently returns NaN, with only a NumPy RuntimeWarning ("invalid value encountered in scalar divide") and no status.

The adapter returns an explicit `no_free` status. A no-FREE window is never given a silent zero error.
