# RIND3D-100K v3 dataset contract (production)

**Normative file:** `RIND3D_Dataset100K/contract/RIND3D_100K_CONTRACT_v3.json` (`rind3d-100k-contract-v3`). This document
explains it. If they disagree, the JSON and the code it names are authoritative. Physical semantics: `docs/RIND3D/SEMANTICS.md`.
Vocabulary: `docs/RIND3D/GLOSSARY.md`.

**Status.** v3 is the production contract and the default of `scripts/run_node.py`.

* **Frozen:** the semantics (physics, eligibility, geometry rule, posterior target, label statuses, storage).
* **As declared, pending the H20 canary review** (`RIND3D_Dataset100K/docs/H20_CANARY_RUNBOOK.md` §7): the population
  parameters κ, quotas, caps and the coverage design. These are v2's, and `docs/SAMPLING_IMPACT_V3.md` measured their
  effect under v3.
* **Kept for regression and reproducibility:** contracts v2 (window rule "mixed") and v1 (thinned, tau-pencil
  labels), with v0.1 geometry (thin-slab T_min = 0).

## 1. World and physics

* **World:** the unit cube [0,1]³ at resolution N = 1024 (receiver voxels).
* **Geometry G:** a union of closed convex polyhedral components, decoded canonically to exact rational halfspaces
  (`rind3d.rendering.polyhedra.decode_convex`).
* **Families:** nine convex-polyhedral families (round robin `slot % 9`): rotated_boxes, thin_slabs, prisms, wedges,
  convex_polyhedra, multiscale_convex, corridors, barriers, occluder_clusters.
* **Curved families** (closed balls) are physically implemented, but **not part of the certified profile**: there is
  no certified labeller for them. `rind3d_100k/profile.py` refuses them in `run_node`; see `docs/CURVED_ASSESSMENT.md`.
* **Thin slabs** (all of thin_slabs, 25% of occluder_clusters):
  * log₂(half-thickness) ~ U(log₂(T_min/(2N)), −5) with T_min = 2 voxels;
  * the exact check requires the voxel-centre sampling of every thin slab to be nonempty and 6-connected;
  * an invalid geometry rejects the scene attempt (`thin_slab_invalid`) and retries up to 3 attempts;
  * the same canonical body defines SOLID and visibility (`docs/THIN_SLABS_V3.md`).
* **Driver sources:**
  * K ∈ {1,2,3,4} per scene with κ = (0.15, 0.15, 0.15, 0.55), keyed draw;
  * each source is uniform on D(G) = the open cube outside the closed solid with certified clearance > 2⁻²⁶;
  * L∞ anti-duplication 2⁻²⁰;
  * every source lies outside every accepted closed window box [o/N, (o+L)/N]³.
* **Binary response.** Receiver r is response-inactive for source s iff the closed segment [s, r] meets the closed
  solid; otherwise it is response-active (legacy key "lit"). Receivers inside the solid are SOLID. The response is
  computed with exact and certified predicates (backend `cuda_certified`). Global response fields are never stored.

## 2. Windows and eligibility (Task 1)

Each scene has exactly 12 windows: 4 each of sides L = 64, 128, 256. Window origins are uniform on {0..N−L}³ (natural
stream). There is also an optional coverage stream (L64, K = 4 scenes, one slot, c(m) = (1/8, 1/4, 1/2, 1), with a
natural fallback). The exact-m rate stream records P_A and Λ for balance-heuristic training weights
(`docs/SELECTION_LAW_100K.md`).

**Direct response interface.** Channel j of window W is informative iff W contains two 6-adjacent receivers u, v
(|u − v|₁ = 1) that are both FREE and have R_j(u) ≠ R_j(v). m(W) is the number of informative channels. A window is
admissible iff no source lies in its closed box, it has a FREE receiver and m ≥ 1.

`rind3d_100k.interface.channel_class` makes the classes explicit:

| class | meaning | v3 |
|---|---|---|
| `interface` | a direct FREE–FREE active/inactive contact | informative, labelled |
| `mixed_no_interface` | active and inactive FREE receivers, every contact through SOLID | excluded (responses kept, `not_computed`) |
| `all_active` / `all_inactive` | uniform response | excluded |
| `no_free` | every receiver SOLID | excluded |

Decisions are exact (`local_eval.evaluate_proposal_interface`: constructive bisection proofs, the exact all-active hull
certificate, 64³ tile scans including edges across tile faces). Screening affects cost only, never a decision. The v2
"mixed" mask of each stored window is kept as `windows.mixed_mask`. Tests: `tests/test_interface_v3.py`,
`tests/test_thin_slabs_v3.py`, `tests/test_local_geometry_v3.py`.

## 3. Posterior target and labels (Task 2)

For channel j of window W, the feasible-source region is F = {s ∈ D(G) \ closed window box : R_G(s)|_W = X_j}, where
X_j is the observed class grid. The **target posterior** is the uniform probability measure on F, projected onto the
global 128³ grid (cell masses). It is defined by the observation and the complete geometry, never by the generated
source. The labeller may use the complete continuous geometry.

Production labeller: **FastLabel v3.4** (`rind3d_posterior.fastlabel`). Label statuses:

| status | guarantee |
|---|---|
| `certified` | deterministic TV ≤ 0.05 (single-cell support or an exact path) |
| `certified_stat` | statistical: TV(q̂, q) ≤ 0.05 except with probability ≤ δ_label = 0.01 / 4.8·10⁶, over the labeller's keyed random streams |
| `certified_loose` / `uncertified` | 0.05 < TV bound ≤ 0.10 / TV bound > 0.10: not accepted by default |
| `not_computed` | uninformative channel (no direct interface): responses kept, no label |

* **Familywise statement.** n_tasks = 4.8·10⁶ = 100,000 scenes × 12 windows × 4 channels, an upper bound on every
  realizable label unit. By the union bound, P(some accepted statistical label has TV > 0.05) ≤ 0.01 for any realized
  campaign.
* The argument, its assumptions and its validation are in `RIND3D_Posterior/docs/FASTLABEL_V34_CERTIFICATION.md`. It
  is a statistical guarantee, not a deterministic proof.
* Timeouts: 1800 / 3600 / 7200 s for L = 64 / 128 / 256. A timed-out, failed or crashed task never produces a
  certified record (quarantine, `validate.py`).
* The exact tau-pencil engine (`--label-engine tpe`) is the deterministic reference.

## 4. Storage and splits

* **Scene containers:** `scenes/<slot//1000:04d>/<slot:06d>.npz`, written atomically (tmp, fsync, rename, directory
  fsync). Each holds the geometry bytes, sources, windows (with streams, m, masks, `mixed_mask`), packed response bits
  per (window, source) as Z, Y, X zstd, SOLID once per window, and meta (provenance, sampling, thin-slab reports).
* **Labels:** append-only checksummed journals `labels/journal/*.r3lj`, consolidated into
  `labels/scenes/<shard>/<slot:06d>.labels.npz` with a sha256 record in `labels/consolidated.jsonl`.
* **Splits:** sha256(scene_id) mod 100 < 80 is train, < 90 is val, else test.
* **Loader:** `rind3d_100k.loader`. `flatten(population=...)` selects Tstar / Tstar_natural / stored. It accepts
  `certified` and `certified_stat` by default, and fills a one-hot fallback only when asked explicitly (labelled as
  such).
* **Serialized keys** keep their legacy names (`lit`, `dark`, `light`). The documented vocabulary is driver source,
  response-active/inactive and response interface (`docs/RIND3D/GLOSSARY.md`).
