# XLF CBC: next steps (prioritized, grounded in the measurements)

Production stays FastLabel v3.4 on `yushu/3d`. Nothing here is a production change. Any production use needs a
separate readiness decision.

## State (v2.8, `70522ab`)

* **Benchmark:** 84/90 tasks certified at 8 s and 88/90 at 20 s.
* **Release canary:** 739/807 channels at 10 s (v2.7), all consistent with the production labels.
* **References:** 0 inconsistencies against 86 exact-certified references.
* **Speed:** certified median 1.6 s, on par with FastLabel's median where certified.
* **Remaining gaps:**
  * uncertified tails (8% of canary channels at 10 s);
  * no machine-checked proof of the transitions;
  * Python polytope code in the trusted base.

## 1. Close the certification gap on hard channels

1. **Residual bitmask for huge pieces.** A piece whose residual set exceeds the 2M-entry list cap rescans the whole
   window; one L = 256 task ran 73 full scans. A per-piece bitmask (2 MB at L = 256) removes this. The scan already
   supports a skip list; a masked variant is straightforward.
2. **Native exact cuts and splits.** Exact enforcement and pair conditioning (about 15% of the time) still run
   Python big-integer polytope code. `native/cells.cpp` already implements `split` in GMP; exposing it for
   enforcement would follow the same differential-test and byte-identity protocol.
3. **Fragmented frontiers next to obstacles** (all remaining failures). Make contact-layer refinement quantitative:
   stop once the charged contact volume is below the tolerance slack.

## 2. Assurance

1. Finish the exact differential audit on all 90 tasks (L64/L128 running). L = 256 needs a faster oracle.
2. Mechanize the frontier accounting theorem and the integer wire bound in a proof assistant, or keep the
   300-instance exact property test. The screen's integer arithmetic is already mechanized (`smt_screen.py`, z3,
   6/6).
3. Pin the builds: Numba, g++, GMP.

## 3. Deployment study (needs a separate decision)

A deterministic-first hybrid gives most channels a deterministic TV certificate at roughly FastLabel's median cost:
CBC with a 10 s budget, then FastLabel for the rest. On the canary that is 91.6% deterministic labels. The open
measurements are H20 throughput and the per-campaign cost of the remaining FastLabel fallbacks.
