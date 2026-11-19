# RIND3D-100K v3 release canary (laptop, 2026-11-18)

* **Machine:** NVIDIA RTX 3060 Laptop (6 GB), AMD Ryzen 7 5800H (16 threads), 31 GB RAM.
* **Contract:** `rind3d-100k-contract-v3` (production default).
* **Labeller:** FastLabel v3.4.
* **Workers:** 1 physical worker, 10 GPU label workers.
* **Release tag:** `release/rind3d-100k-v3-2026-11-18` on `yushu/3d`.
* **Artifacts** (not in git): `/home/yushu/Projects/RIND/RIND3D_100K_v3_work/release/`. This holds `canary/` (VERIFY,
  COVERAGE, REPORT, ESTIMATE, costs), `canary_resume/`, `canary_corrupt/`, `canary_logs/` and `final/`.

These are laptop measurements. They are not H20 numbers and must not be extrapolated as such.

## Environment and tests

* **Clean environment:** `setup_env.sh` built from source in 1 m 38 s. This needed one fix: the rolling Arch mirror
  no longer serves Boost 1.92.0-1, so it is now fetched from the Arch Linux Archive with the same pinned SHA-256.
* **pytest on the release code:**

  | suite | result |
  |---|---|
  | `RIND3D_Dataset100K/tests` | 155 passed |
  | `RIND3D_Dataset1K/tests` | 9 passed |
  | `RIND3D_Posterior/tests` | 26 passed |
  | `RIND3D_Generator/tests` | 306 passed, 17 skipped |

* **Smoke test:** SMOKE PASS, all 11 steps; see `final/smoke.log`. The steps are:
  * environment;
  * engine regression;
  * 1K label fixtures;
  * v1 physical fixtures;
  * v2 physical and FastLabel fixtures;
  * v3 physical and FastLabel fixtures (bit-exact);
  * Gate A on v3 scenes;
  * a node run with validation;
  * journal recovery.

## Canary: 36 scenes, slots 0–35, four per family

* **Scenes:** 36/36 accepted, 0 rejected attempts.
* **K histogram:** {1: 9, 2: 2, 3: 4, 4: 21}, consistent with κ = (0.15, 0.15, 0.15, 0.55).
* **Windows:** 432 (12 per scene). Streams: L64 123 natural + 21 coverage; L128 144; L256 144.
* **Channels:** 1,308 driver-source channels.
  * 807 informative (direct interface), i.e. 22.4 labels per scene: 512 `certified_stat` and 295 `certified`
    (deterministic).
  * 501 `not_computed` (no direct interface: SOLID-separated, uniform or no FREE receiver).
* **Failures:** 0 label failures, timeouts or quarantines.
* **verify_root --labels**, re-run with the release code's strict certificate checks:
  * ok;
  * physics spot checks on 9 scenes, 0 mismatches;
  * every label valid: sha256, certificate rules (`check_stat_certificate`), true source in support, n_tasks.
* **T\* weights:**

  | L | Kish ESS fraction | max/median weight |
  |---|---|---|
  | 64 | 0.78 | 2.6 (coverage weight share 0.22) |
  | 128 | 0.87 | 1.5 |
  | 256 | 0.94 | 1.2 |

  Input to the κ / coverage review: H20 runbook §7, which targets ESS ≥ 0.8 and max ≤ 1.4× at L64.

### Cost (canary means)

* **Physical stage:** 5.5 s per scene (p50 5.7, p95 8.2, max 12.9 s); about 560–620 scenes/h with one physical
  worker.
* **FastLabel CPU-seconds per label:**

  | L | mean | p99 | max |
  |---|---|---|---|
  | 64 | 1.19 | 5.6 | 6.1 |
  | 128 | 0.94 | 4.7 | 5.3 |
  | 256 | 0.72 | — | — |

  Overall mean 0.91 s (90% CI 0.86–0.96). Throughput was 3.8 labels/s, supply-bound by the single physical worker.
* **Storage:** 347 KB per scene and 37 KB per label (journal).

## Robustness

| test | result |
|---|---|
| `kill -9` of the coordinator with 11 scenes and 226 labels in flight, then a rerun | completed 36/36; verify clean; no orphaned workers. Identical re-consolidations of slots 0–13 are expected (last record wins) |
| torn line appended to `manifest/physical.jsonl`, run, STOP, resume (slots 0–39) | **found a bug**: the append after the torn tail created a corrupt middle line and the next resume failed. Fixed in `storage.append_line` (drop the torn tail first, with a test). Rerun: STOP drained in under 5 s, resume completed 40/40, verify clean |
| one byte flipped in a consolidated label file | `verify_root` fails with `slot 7: consolidated labels sha256 mismatch` |
| resume under different code | refused unless `--allow-code-change` (test `test_code_freeze.py`) |

**Coverage-report bug, fixed:** statuses were double counted after a resume. It now counts the last consolidation per
slot.

## 100K estimate (contract v3; `scripts/estimate.py`, laptop rates)

* **Labels:** 100,000 × 22.4 ≈ 2.24·10⁶. The δ budget assumes the upper bound of 4.8·10⁶ label units.
* **Label cost:** 565 core-hours of FastLabel worker time. On this laptop's single GPU with 10 workers: about 66 h
  (90% CI 63–70 h).
* **Physical stage:** 100,000 × 5.5 s ≈ 153 GPU-hours on this laptop GPU.
* **Storage:** about 35 GB of scenes plus about 83 GB of label journals.
* **16 × H20:** must be measured (H20 runbook §9). If each H20 matched this laptop GPU, physical work would take about
  10 h and labels about 4 h on 16 GPUs. That is an assumption, not a measurement.

## FastLabel vs references

* The certification review (`RIND3D_Posterior/docs/FASTLABEL_V34_CERTIFICATION.md` §7) made 99 comparisons against
  independent references: the deterministic certified tau-pencil labels and Astra's exact rational fixtures.
* Result: 0 violations; worst-case TV 0.0437 against a loose reference, 0.0141 against a tight one; single-cell labels
  exact.
* An additional on-canary comparison with the exact tau-pencil engine (`compare_engines.py`) was stopped to
  conclude the release. It is not required, because the references above are independent.
