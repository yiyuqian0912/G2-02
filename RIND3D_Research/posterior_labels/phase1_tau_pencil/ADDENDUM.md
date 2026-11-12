# Phase 1 addendum: runs that were still in progress at the freeze

Phase 1 was frozen on 2026-11-11 at about 20:25 EST:
* the frozen snapshot is `PHASE1_FREEZE_MANIFEST.json` (SHA256 of every file at freeze time);
* the review ZIP is `~/Downloads/RIND3D_POSTERIOR_LABELS2_PHASE1_REVIEW.zip` (see its `.sha256.json`).

The runs listed in the report's "Freeze status" section were not interrupted. Their results are written as new
files in the same output directories. This file records them once they finish; no frozen file is modified.

(To be completed when the pending runs finish.)

Correction: the report's 'Freeze status' heading says ~20:30 EST; the actual freeze, by file timestamps, was about 20:25 EST.


## Final state (recorded 2026-11-11 21:55 EST)

**The pending runs ended with a machine reboot at about 21:52 EST.** They were not restarted: the user declared
Phase 1 done and moved on to Phase 2. Final counts:

| Run | Final |
|---|---|
| Random sample, engine b3 | 89 / 90 windows; 09ade133/82 never finished verification |
| b4 reruns of b3 work-queue overflows | 7 done, **all exact (TV = 0)**: 1c79650a/234, 82b22342/195, b4d7f4f6/217, 380d9ae0/199, 86b09755/229, c3621fd9/177, a636b0aa/254; c6f3aded/180 and c94e14ab/231 not rerun |
| Paired 128³ ↔ 1024³ | 28 / 30, all exact at both resolutions |
| Stage A baseline comparison | 22 / 30 |

**Combined certified count for the random sample.** 80 windows were exact under b3 and 7 more under the b4
reruns, so 87 of 89 are exact. The other 2 are b3 queue overflows that were never rerun under b4 (outcome unknown).

**Frozen-file integrity.**
* The running jobs appended to five logs listed in the freeze manifest:
  - real_b4_reruns.log;
  - real_paired30_b3.log;
  - real_val_random90_b3_part2.log;
  - real_val_random90_b3_rev.log;
  - stageA_compare30.log.

  The frozen versions are in the ZIP.
* The reverse-order second driver recomputed outputs/real_val_random90_b3/c94e14ab_231.json after the freeze. The
  results were identical and only the timings differed. The frozen version was restored from the ZIP; the
  duplicate is kept as `c94e14ab_231.duplicate_run_after_freeze.json`.
* All other 834 frozen files match the manifest.
