# RIND3D 1K release contract: `rind3d-1k-contract-v1`

**Dataset name:** RIND3D. Release `1k-v1`; profile and resolution are recorded in metadata.

The machine-readable copy is `contract/RIND3D_1K_CONTRACT_v1.json`. Any deviation must be a new contract version; there are no silent changes.

## Physical model

| Item | Value |
|---|---|
| World | (0,1)³ |
| Production receiver grid | N = 1024 (voxel centres (i + ½)/N) |
| Development canaries | N = 128, with windows scaled by N/1024 |
| Geometry | Finite union of closed convex polyhedra. Families are listed in `families` of the JSON contract (see GEOMETRY_DIVERSITY.md), assigned as `families[slot mod len]`. v0.1 generator recipes are unchanged; new families come from `rind3d_1k/families.py`. |
| Geometry acceptance | Certified v0.1 gate: geometry proof, world gaps, membership probes, segment reversal, two independent physical predicates |
| Visibility | `closed-solid-closed-segment-v1`; SOLID receivers unlit |
| Source count k | 1..4, chosen per scene by the balancing controller (`rind3d_1k/controller.py`) before sources are drawn. 15% of slots each are forced to k = 1, 2, 3, 4 (keyed uniform); the other 40% use k minimising expected imbalance. |
| Sources | i.i.d. Uniform(D(G)), D(G) = FREE with certified clearance > 2⁻²⁶. Anti-duplicate rule only: \|s_i − s_j\|_∞ ≥ 2⁻²⁰. No MIN_SEPARATION (SELECTION_LAW_1K §4). |
| Responses | One binary channel R_j per source, stored separately. Intensities are 1. No summed field is stored as canonical data. |

## Windows

| Item | Value |
|---|---|
| Per-scene quotas | **exactly** 16 / 24 / 24 windows of physical side 64 / 128 / 256 |
| Proposals | Uniform origins on {0..N−L}³, stream keyed (seed, slot, L, attempt). Repeated origins are skipped. |
| Acceptance A | Every source outside the closed box **and** FREE > 0 **and** m ≥ 1 |
| Thinning | Accept iff U < c(m, L, k). The table is fixed by the controller before the sources are drawn; c ≥ 0.01. |
| Caps | 6000 proposals per side. A scene with any unfilled quota is rejected and retried (≤ 3 attempts, new geometry and sources). Every attempt is recorded. |
| Rate stream | 1024 extra proposals per side. Gives Λ̂_L (training weight ω) and the natural m statistics. |
| Effective layer | I_j = channel j has ≥ 1 lit and ≥ 1 dark FREE receiver in W. m = Σ I_j. |
| Balancing target | Dataset-wide histogram of m ≈ (¼, ¼, ¼, ¼) over 64 000 windows. The weighted histogram is also reported. |

## Posterior labels (one slot per window × source, source axis preserved)

| Item | Value |
|---|---|
| Target of slot (W, j) | Uniform(F_W^j), F_W^j = {s ∈ D(G) \ B̄(W) : s reproduces R_j on FREE(W)}. This is the exact posterior of the weighted population (SELECTION_LAW_1K, Theorem). |
| Engine | `rind3d-posterior` 0.2.x (`tpe-b4/p2-cpu+g2-gpu-pass-v3`). Tier 1 queue capacity 2^25; tier 2 retry at 2^26 when the TV bound > 0.05. |
| Representation | Sparse records (cell int32 index in the 128³ grid, lo, hi float64 volume, kind int8). Label q = L/ΣL. Certified bound `tv_bound` stored per slot. |
| Status | `certified` (TV bound ≤ 0.05); `certified_loose` (≤ 0.10); `uncertified` (> 0.10, never presented as qualified); `unavailable` (informative, not yet computed); `not_computed` (uninformative channel, deferred in this release) |
| Uninformative channels | Response stored; posterior deferred (`not_computed`) because of measured cost (SELECTION_LAW_1K / OPTIMIZATION_REPORT) |

## Storage (per scene directory `scenes/<slot>/`, published atomically)

* `scene.json`: contract, ids, family, k, c-table, per-side selection statistics, Λ̂, timing. No source coordinates.
* `SEAL.json`: SHA-256 of every physical-stage file.
* `public/`: Student-visible window data.
  * `windows.npy`: window, x, y, z, L, Lp, m, informative mask, k.
  * `observations.bin` + `observations_index.npy`: per window, the SOLID crop once and every source's response crop. Packed bits in ZYX order, zstd, indexed by (window, source; −1 = SOLID).
* `privileged/`: global and privileged data.
  * `geometry.npz`: canonical bundle.
  * `geometry_rows.npz`: binary64 rows used by the labeller.
  * `sources.json`: hex floats.
  * `fields/src<j>/{response,solid}.rindb`: certified full fields.
  * `selection_log.npz`: every quota proposal: origin, A, m, mask, U, accepted.
  * `windows_full.json`: counts, overlap histogram, lit fractions, channel diversity.
  * `geometry_acceptance.json`.
* `labels/tasks/w<window>_s<source>.npz`: one file per posterior slot (atomic). `labels/LABELS.json` holds coverage and per-slot metadata.
* Campaign root:
  * `manifest.jsonl`: every attempt (accepted / rejected / failed) with the reason;
  * `labels_log.jsonl`;
  * `controller_state.json`;
  * `RELEASE_MANIFEST.json`: checksums and statistics.

## Views

| View | Contents |
|---|---|
| Student | Local SOLID crop, response channel(s), window side (origin optional) |
| Target | Posterior slots with status and TV bound |
| Privileged | Sources, global geometry, full fields, selection log, ω weights |

Flattening to single-source examples is a loader adapter (`rind3d_1k.loader.flatten`). The canonical loader returns multi-channel window records.

## Splits

By scene: sha256(scene_id) mod 100 < 80 is train, < 90 is val, otherwise test. The test scenes are never used for development; all canaries use separate seeds.

## Seeds

* Release campaign seed **20261120**, slots 0, 1, 2, ….
* Canaries use seeds 20261112 and 20261113.
* The geometry study uses seed 20261114.
