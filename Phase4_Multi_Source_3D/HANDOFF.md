# HANDOFF: RIND3D-100K v3 (for the next researcher or agent)

**Branch** `yushu/3d`: the authoritative production branch. Release tag: `release/rind3d-100k-v3-2026-11-18`.
**Root of the RIND3D work:** `Phase4_Multi_Source_3D/`. Every path below is relative to it.

## What RIND3D-100K is

* 100,000 scenes. Each scene is a 1024³ world of closed convex polyhedra with 1–4 point driver sources.
* 12 local windows per scene, at L = 64 / 128 / 256.
* For every (window, source) channel with a direct response interface, the dataset stores the binary response crop
  and a **certified posterior label**: the uniform measure on the feasible-source region F, binned on the global 128³
  grid.
* A Student model learns to infer the global source posterior from the local window. Specification:
  `RIND3D_100K_V3_CONTRACT.md`, `docs/RIND3D/SEMANTICS.md`, `docs/RIND3D/GLOSSARY.md`.

## What is complete

| item | status | where |
|---|---|---|
| Task 1: direct response-interface eligibility | done | `RIND3D_Dataset100K/rind3d_100k/interface.py` (`channel_class`), `local_eval.py`; tests `test_interface_v3.py` |
| Task 1: thin slabs | done | T_min = 2, exact nonempty and 6-connected voxel sampling: `geometry_v3.py`, `docs/THIN_SLABS_V3.md`, `test_thin_slabs_v3.py`. v0.1 (T_min = 0) stays in contracts v1/v2 |
| Task 2: FastLabel v3.4 production labels | done | `RIND3D_Posterior/src/rind3d_posterior/fastlabel/`; certificate `RIND3D_Posterior/docs/FASTLABEL_V34_CERTIFICATION.md` (statistical, Gate B passed with conditions) |
| Task 3: curved obstacles | assessed; excluded from the certified profile | `docs/CURVED_ASSESSMENT.md`, `docs/CURVED_OCCLUDERS.md`, `RIND3D_Generator/curved/`, `rind3d_100k/physical_curved.py`; refused by `rind3d_100k/profile.py` and FastLabel's `inputs.py` |
| Task 4: driver-source / binary-response terminology | done | `docs/RIND3D/GLOSSARY.md`, READMEs, specs; v3 meta key `active_frac`; loader `response(s)`, `informative`, `driver_sources()` |
| Production default | contract v3, FastLabel | `scripts/run_node.py` |
| Durability | atomic containers, checksummed journals, torn-tail repair, STOP/resume, code freeze | `rind3d_100k/storage.py`, `coordinator.py` |
| Validation | tests, smoke test (11 steps), laptop canaries with crash/resume | `PRODUCTION_RUNBOOK.md` §2, `RELEASE_CANARY_2026-11-18.md` |

## What is tested (laptop: RTX 3060 6 GB, Ryzen 7 5800H)

* **pytest:**
  * `RIND3D_Dataset100K/tests`: physics, eligibility, thin slabs, curved spheres, sampling law, storage, loader,
    FastLabel certificate rules, profile guard, code freeze, GPU pipeline;
  * `RIND3D_Dataset1K/tests`;
  * `RIND3D_Posterior/tests`;
  * `RIND3D_Generator/tests`, run from that directory.
* **Smoke test:**
  * physical fixtures for v1, v2 and v3 (bit-exact crop digests);
  * FastLabel fixtures for v2 and v3 (bit-exact label digests);
  * Gate A;
  * a node run with validation;
  * journal recovery.
* **Canary** under contract v3, all families: `verify_root --labels` clean, physics spot checks 0 mismatches, 0 label
  failures, plus `kill -9` resume, STOP drain, torn manifest line and corrupted label detection. FastLabel was
  compared with the exact tau-pencil engine on a sample.
* **FastLabel vs independent references** (deterministic certified tau-pencil labels, Astra's exact rational results):
  99 comparisons with 0 violations of TV ≤ 0.05; worst case 0.0437 against a loose reference, 0.0141 against a tight
  one. See the certification document §7.

## What is NOT done

* The 16 × H20 production gate (`RIND3D_Dataset100K/docs/H20_CANARY_RUNBOOK.md`).
* The κ / coverage review.
* The 100K campaign itself: not launched, and it needs explicit authorization.

See `NEXT.md`.

## Guarantee of a released label (read before using the data)

* **`certified`:** deterministic TV ≤ 0.05.
* **`certified_stat`:** TV(q̂, q) ≤ 0.05 except with probability ≤ δ = 0.01 / 4.8·10⁶ per label, over the labeller's
  keyed random streams. The union bound gives P(some accepted statistical label is wrong) ≤ 0.01 for the whole
  campaign. This is a statistical guarantee, not a deterministic proof.
* **Conditions:**
  * the code is frozen per campaign (`CODE.json`);
  * the contract's seeds and δ are not overridden (enforced);
  * no re-rolls;
  * `verify_root --labels` was run on the root.
* **Numerical premises:** float margins of the outer region and the binary64 cell volumes of B. Stated in the
  certification document §6.

## Important paths and commands

* **Environment:** `bash RIND3D_Dataset100K/handoff/env/setup_env.sh && source r3d_env.sh`
* **Validation:** `PRODUCTION_RUNBOOK.md` §2
* **Canary / production:** `PRODUCTION_RUNBOOK.md` §3
* **Loader:** `rind3d_100k.loader.RIND3D100K`, `flatten(ds, population="Tstar")`
* **Process rule:** one rind3d CUDA runtime per process (physics and FastLabel in separate workers)

## Remaining limitations

* Statistical, not deterministic, labels.
* Curved obstacles unsupported for labels.
* Population parameters provisional until the H20 review.
* Laptop-only performance measurements.

## How to resume or extend

* **Production:** `NEXT.md` items 1–5.
* **Deterministic labelling research** (XLF, the tau-pencil verifier), local-first theory, Astra's results:
  `RESEARCH_ARCHIVE.md`. Every branch is pushed and listed there with its commit.
