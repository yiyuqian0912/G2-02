# Contract v3: sampling impact (laptop canary)

**What changed from v2.** Two things, both declared in `contract/RIND3D_100K_CONTRACT_v3.json`:
1. **Eligibility.** A channel is informative iff the window has a direct FREE–FREE response interface: two 6-adjacent
   FREE receivers with different responses (`rind3d_100k/interface.py`). v2's "mixed" rule (an active and an inactive
   FREE receiver anywhere) also admitted SOLID-separated windows.
2. **Geometry.** Thin slabs are at least 2 voxels thick, with 6-connected samplings (docs/THIN_SLABS_V3.md).

**What did not change.** κ = (0.15, 0.15, 0.15, 0.55), the 4/4/4 quotas, caps, n_probe, the L64/K = 4 coverage slot and
its c(m), the rate-stream design, the T* balance-heuristic weights and the storage format.
* v3 windows additionally store `mixed_mask` (the v2 rule) and per-channel interface edge counts.
* v3 has a new contract id and seed, so scene ids cannot collide with v2.

**Population.** T*_v3: natural multi-source worlds and natural admissible windows. A window is admissible iff no
source is in the closed box, FREE > 0, and m ≥ 1 under the direct-interface rule. Every channel with a direct
interface is one example; other channels of an accepted window are stored with status `not_computed`.
* The rate stream evaluates the v3 m exactly for every proposal, so P_A and the coverage acceptance probability are
  estimated exactly as in v2. The loader's weights are unchanged in form: they consume the v3 m and the v3 rate
  statistics.
* There is no fallback to an ineligible window. A side that cannot fill its quota rejects the scene attempt
  (`quota_unfilled`), exactly as in v2.

**Exactness.** Decisions under the v3 rule are exact (`local_eval.evaluate_proposal_interface`), built from four
facts:
* constructive bisection proofs along lattice paths between an active and an inactive FREE probe;
* the all-active hull certificate;
* exact 64³ tile scans including the edges across tile faces;
* exact fast paths for uniform tiles (an all-FREE tile showing both responses must contain an interface).

The GPU test `test_v3_direct_interface_decisions_are_exact`, and a 148-proposal check on five scenes during
development (lazy and exact decisions), agree with full certified renderings.

## Canary

**Setup.**
* 72 scenes per arm (8 per family), sequential, on a laptop: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB, driver 620.70.06),
  AMD Ryzen 7 5800H, Python 3.12.14 (recorded in each arm's `run.json`).
* Arms:
  - `v2`: contract v2;
  - `v3rule_v2seed`: v2's seed, κ and streams with only the v3 rule, so worlds, sources and proposals are identical to
    `v2` (paired);
  - `v3full_v2seed`: the rule and the thin-slab geometry on v2's seed;
  - `v3`: the v3 contract itself, with its own seed.
* Data are outside Git, under `RIND3D_100K_v3_work/sampling_impact/`. Tables come from `scripts/sampling_impact_report.py`.

**Consistency check.** In the v3-rule arm, the rate stream also records the v2 rule on the same proposals. It
reproduces the v2 arm's own count exactly in 72/72 scenes at all three sides.

| | L64 | L128 | L256 |
|---|---|---|---|
| natural P_A, v2 | 0.3608 | 0.5805 | 0.7891 |
| natural P_A, v3 rule (same proposals) | 0.3578 | 0.5773 | 0.7865 |
| v2-admissible proposals excluded by the v3 rule | 0.84% | 0.56% | 0.33% |
| natural proposals per accepted window, v2 → v3 rule | 3.99 → 4.10 | 2.19 → 2.20 | 1.40 → 1.41 |
| natural P_A, v3 contract (own seed) | 0.3411 | 0.5531 | 0.7574 |
| excluded share inside the v3 contract arm | 1.09% | 0.86% | 0.20% |

**Exclusion by family** (all sides, v3-rule arm). The rule acts where occluders separate the two responses:
* corridors 2.1% and barriers 1.4% (walls of 4–28 voxels);
* multiscale_convex 0.6%, occluder_clusters 0.5%, rotated_boxes 0.4%;
* thin_slabs 0.2%, convex_polyhedra 0.1%, wedges 0.07%, prisms 0%.

With thick slabs, thin_slabs exclusion rises on identical streams: 0.16% in `v3rule_v2seed`, 1.19% in
`v3full_v2seed`. Once a slab is SOLID, windows across it are SOLID-separated rather than false direct interfaces.

**Accepted windows** (864 per arm), by m:

| | m = 1 | m = 2 | m = 3 | m = 4 | K = 4, m = 4 (L64 / L128 / L256) |
|---|---|---|---|---|---|
| v2 | 458 | 184 | 112 | 110 | 11 / 19 / 80 |
| v3 rule, v2 seed | 460 | 185 | 110 | 109 | 10 / 19 / 80 |
| v3 full, v2 seed | 462 | 181 | 115 | 106 | 9 / 17 / 80 |
| v3 contract (own seed) | 486 | 153 | 106 | 119 | 7 / 28 / 84 |

**Readings.**
* **Rare strata.** The v3 rule moves a few windows from high to lower m, because a SOLID-separated channel no longer
  counts. The rare K = 4, m = 4 stratum is preserved; the L64 coverage slot still fills with no fallback.
* **Scene acceptance.** Every scene of every arm was accepted. The only rejected attempts were thin-slab rim cases:
  1 per 72 scenes in each arm with thin slabs, retried successfully.
* **Bias.** The rule conditions on at least one direct interface edge and adds no further weighting: the natural stream
  accepts every admissible proposal. It does not favour larger interfaces beyond excluding windows with none. v3 windows
  record per-channel interface edge counts, so the size distribution can be studied directly.
* **Throughput.** The median scene wall rose from 4.2 to 6.8 s with the first v3 implementation, and fell back to 4.8 s
  (+15% over v2) after the exact uniform-tile fast paths (arm `v3full_v2seed`, run with the final code).
  - The extra cost is the rate stream. Exact v3 m needs an interface proof or a full scan per channel, where v2 stops
    as soon as both responses are seen.
  - The v3 contract arm (6.7 s) was run before the fast paths. The per-family benchmark
    (docs/GENERATOR_BENCHMARK_V3.md) uses the final code.
* **Effective training examples.** About 1% fewer admissible windows and essentially the same examples per window. The
  number of labelled examples per scene is fixed by the 12 windows and m, so it is unchanged at the 1% level.

**What is not established.**
* κ and the coverage table were chosen for v2. Their validity for T*_v3, the m-balance of the coverage slot and the
  label-cost tails still need the H20 canary (H20_CANARY_RUNBOOK.md) before any production.
* These are laptop measurements of 72-scene canaries. They are not H20 throughput.
