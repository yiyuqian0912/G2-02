# D — 1K FastLabel posterior backfill: completed and verified

**Run:**
* approved in the 2026-11-13 brief;
* launched 04:13:46, finished 08:33:56 EST, 2026-11-13;
* `RIND3D_Dataset1K/backfill/run_backfill.sh`, i.e. `scripts/run_labels.py release_1k_v1 15 0 --engine fastlabel --scene-order --retry-failed`;
* FastLabel v3.4: frozen in checkpoint 6 and audited (`RIND3D_Audit/07_TV_CERTIFICATE.md`), with the M3 δ fix.

## 1. Outcome

| item | value |
|---|---|
| informative (mixed) label tasks in the release | 159,802 |
| labelled before the backfill (tau-pencil, kept unchanged) | 333 |
| labelled by the backfill | **159,469** (including the 5 tasks quarantined by the first release pass, retried with FastLabel) |
| failures, timeouts, crashes, quarantines | **0** |
| `certified`: deterministic TV ≤ 0.05 (FastLabel single-cell support 52,190 + tau-pencil 333) | 52,523 |
| `certified_stat`: statistical TV ≤ 0.05, per-label δ = 0.01 / 159,802, so P(any false statistical certificate) ≤ 0.01 | 107,279 |
| `certified_loose`, `uncertified`, `unavailable` | **0** |
| wall clock | **4.33 h** (15,600 s), 10.2 labels/s, 15 workers, one GPU semaphore |
| projection before the run (checkpoint 6) | 4.27 h, 90% interval [3.53, 5.03] h. The realised time is inside it |
| label wall per task (worker) | mean 1.44 s, median 0.82 s, p99 17.1 s, max 71.7 s |
| uninformative channels (all-lit / all-dark) | 39,109, not computed by design (status `not_computed`, stored) |

**Per side:**

| side | certified | certified_stat | n_cells median | n_cells mean | n_cells max |
|---|---|---|---|---|---|
| L64 | 2,171 | 36,460 | 2,997 | 61,779 | 1,842,088 |
| L128 | 14,993 | 45,239 | 121 | 19,412 | 1,704,455 |
| L256 | 35,359 | 25,580 | 1 | 3,642 | 1,408,499 |

**Statistical certificates by round and stage:**

| | stage 1 | stage 2 |
|---|---|---|
| round 1 | 5,495 | 1,286 |
| round 2 | 74,819 | 13,947 |
| round 3 | 10,671 | 909 |
| round 4 | 83 | 49 |
| rounds 5–11 | — | 20 |

## 2. Verification (`backfill/verify_backfill.py`, `backfill/integrity_snapshot.py`; full per-task output `RIND3D_1K_data/backfill_verification_v1/verify_backfill.json` (+ post-backfill sha256 manifest of every label file, `post_backfill_integrity.json`; SHA256SUMS there); summary `chain/verify_backfill.out`)

| check | result |
|---|---|
| every informative task has a task file | 159,802 / 159,802 |
| sha256 of each new file equals the sha256 logged at write time (`labels_log.jsonl`) | 159,469 / 159,469 |
| the 333 pre-existing labels are byte-identical to the pre-backfill snapshot | 333 / 333 |
| arrays finite and non-negative, cells in range, non-empty | 0 problems |
| `certified` ⇒ tv_bound ≤ 0.05 | 0 problems |
| `certified_stat` ⇒ statistical certificate with ε = 0.05, δ_label = 0.01/159,802, n_tasks = 159,802, keyed seed = (base, scene, window, source), p_lower ≥ 0.95, δ_round = δ_label·2⁻ʳ, and thresholds (a1, a2, α) recomputed from `design2` with K within the stage's threshold | 107,279 / 107,279 consistent |
| **soundness (audit S4, privileged): the true source's 128³ cell lies in the label support** | **159,802 / 159,802** |
| SEAL of every scene (25,550 sealed physical files) | 0 mismatches: the physical 1K dataset is unchanged |
| temporary files left by writers | 0 |

**Driver change during the run.**
* At 04:55 a system update replaced the NVIDIA userspace driver (615.78 → 620.70) while the 615 kernel module stayed loaded. NVML (`nvidia-smi`) fails; CUDA works.
* Running workers kept their loaded library. Eight labels recomputed in fresh processes under the new library are **bit-identical** to the files written before the update.
* A reboot is needed eventually. It should not happen during runs.

## 3. Cost by response type (worker wall per label; the release's pilot scenes)

| type | side | n | certified | statistical share | wall mean / median / p99 (s) | posterior cells (median) |
|---|---|---|---|---|---|---|
| boundary-mixed | L64 | 32,557 | 100% | 94% | 1.35 / 0.86 / 14.5 | 2,080 |
| boundary-mixed | L128 | 52,981 | 100% | 75% | 0.97 / 0.68 / 3.9 | 117 |
| boundary-mixed | L256 | 54,506 | 100% | 42% | 1.09 / 0.87 / 3.7 | 1 |
| solid-separated | L64 | 2,348 | 100% | 100% | 13.5 / 11.5 / 38.8 | 637,228 |
| solid-separated | L128 | 1,281 | 100% | 100% | 13.4 / 10.6 / 37.4 | 512,945 |
| solid-separated | L256 | 295 | 100% | 100% | 13.3 / 11.1 / 45.4 | 317,319 |

**Solid-separated mixed channels:**
* They are certified by the existing solver without exception.
* Each costs about 10–13× a boundary-mixed label, because their posteriors are broad: ~5·10⁵ cells, often a half-space-like region behind a slab.
* They are 2.5% of the labels but about 23% of the backfill's compute.

## 4. Loader use

The default loader accepts both statuses (`C_LOADER.md`). With this backfill complete:
* no informative 1K example needs the one-hot fallback;
* the missing-label selection issue (audit M2) does not arise for the 1K.

## 5. NAT-1K supplement labels (separate δ family)

**Run:**
* engine: the same, FastLabel v3.4;
* δ_label = 0.01 / 30,349, base seed 20261114 (`natural/run_nat_labels.sh`);
* 12 workers, sharing the machine with Student training: 08:42 → 09:47, 64.5 min.

| item | value |
|---|---|
| mixed-channel tasks | 30,349 (planned count fixed before labelling, `natural/N_SUP.json`) |
| failures | 0 |
| certified (single-cell support) | 12,050 |
| certified_stat | 18,299 |
| verification (`verify_backfill.py … 30349`; `chain/verify_nat1k.out`) | 0 problems; certificates consistent with the declared family; true source in support 30,349 / 30,349 |

The natural windows of NAT-1K have more single-cell labels at L256 and more broad L64 labels than the release.
* L64: 250 certified + 4,150 statistical.
* L256: 9,745 + 7,100.
* This follows directly from natural window sampling: M = 1 windows dominate at small sides (`B_SAMPLING_WEIGHTING.md` §1).
