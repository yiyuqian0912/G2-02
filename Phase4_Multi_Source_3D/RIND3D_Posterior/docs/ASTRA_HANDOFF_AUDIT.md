# Astra handoff audit (correct-by-construction XLF)

Audit date 2026-11-18. Auditor: Claude, taking over Astra's research. Branch `claude/xlf-correct-by-construction`.
Status words: **PROVEN** (a written proof in this repository), **IMPLEMENTED**, **EXACT-TESTED** (exact rational
references), **EMPIRICALLY TESTED** (real tasks against independent labels), **CONDITIONAL**, **OPEN**.

## 1. Provenance

| item | value |
|---|---|
| latest Astra commit | `c8feef2` "Prove anytime XLF posterior bounds and test online certification on exact fixtures" (2026-11-18 20:24). It is the tip of both local branches `astra/3d-v3-receiver-certificates` and `astra/xlf-correct-by-construction-solver`. Neither was pushed; `origin/astra/3d-v3-receiver-certificates` is still `94b4820` |
| history | `28195e1` blind global posterior prototype → `94b4820` benchmarks and exact fixtures → `c8feef2` anytime bound theorem, accounting reference, closed-room example |
| uncommitted work | `G2-02-astra-xlf-kernel/…/experiments/credit_solver/{inputs,kernel,run}.py` (mtime 20:34–20:35, after `c8feef2`). Imported **byte-identically** as `8d4e3cf` on this branch. sha256: `3953681f…` inputs.py, `37ddd745…` kernel.py, `cd6420d6…` run.py. `kernel.py` cites a `SPEC.md` that does not exist |
| other uncommitted work | `G2-02-astra-inverse/…/experiments/proof_labels/` (unchanged since 12:53; read-only snapshot already in `RIND3D_100K_v3_work/research_archive/astra_uncommitted_snapshot_2026-11-19/`) |
| this branch | `claude/xlf-correct-by-construction` from `c8feef2`, pushed. Astra's worktrees and branches were not modified |

## 2. What exists and runs

| component | files (under `experiments/`) | status |
|---|---|---|
| abstract accounting reference | `xlf_assurance/ledger.py` (rational boxes, single-owner ledger, exact quota wire encoding, bound on decoded bytes) | IMPLEMENTED, EXACT-TESTED (13 unit tests pass) |
| restricted physical example | `xlf_assurance/room.py` (64³ closed room, 262,144 receivers) | reproduced: CERTIFIED, decoded `{1056832: 1}`, TV 0, 0.20 s, same observation sha256 as `room_evidence.json` |
| 214-cell wire check | `xlf_assurance/experiment.py` | reproduced: exact wire sum 1, encoding TV 5.85e-15, same wire sha256 `6fa36134…` |
| missing-subtree regression | `test_assurance.py` (physical corner on a 2³ window) | reproduced: ABSTAIN at 1/20 with upper 17179860992/17179860993 |
| exact geometry | `global_inverse/poly.py` (integer homogeneous polytopes, exact split, rational volume, lazy exact grid integration), `solver.py` (inverse cones, pair conditioning), `envelopes.py` (guaranteed/possible receiver shadows, clearance partition), `raster.py` (Numba reciprocal scanline screen with directed fixed point; exact receiver enforcement) | IMPLEMENTED; 25 tests in `test_global.py`, `mixed_patch/test_*.py` pass |
| **credit solver** (the actual CBC labeller) | `credit_solver/inputs.py` (v3 container decoder), `kernel.py` (single-owner state, frontier accounting, wire bound), `run.py` (untrusted search, atomic envelope writer) | IMPLEMENTED (uncommitted by Astra); **runs end to end on real v3 tasks** (§3) |

## 3. Does the solver run end to end on real RIND3D tasks? Yes.

Baseline measurement, unmodified Astra code, on the 90 frozen tasks of `global_inverse/tasks.json` (27 holdout + 54
archive) and `fresh.json` (9). One process per task, 8 s search budget, 8 parallel jobs, Ryzen 7 5800H.
Raw records: `RIND3D_100K_v3_work/xlf_cbc/baseline_astra/`.

| L | certified TV ≤ 0.05 | median wall |
|---|---|---|
| 64 | 5/30 | 1.7 s |
| 128 | 8/30 | 2.2 s |
| 256 | 21/30 | 4.6 s |
| all | **34/90** | |

The example `000004_w00_s0` gives the 214-cell posterior, CERTIFIED with a TV bound of 5.9e-15 in 1.9 s.

**Reference dispute, resolved in favour of the credit solver.** On `000038_w06_s2` the certified output (bound 0.0189)
was 0.0290 away from the archived Tau-pencil label `cmp_v3/labels/*_tp.npz`, which claimed bound 0. The independent
exact-kernel verifier (`yushu/3d-v3-staging`, `certify/`) re-certified the task and gave TV ≤ 1.05e-6 for its own
label. The credit solver is 0.0097 from that label, inside its bound. The kernel state was consistent with the
verifier's per-cell enclosures: no lower credit above a certified upper, and total upper 1.5003e-11 ≥ certified lower
1.4863e-11. The **archived legacy Tau-pencil, local-first and random-subset labels are wrong by TV 0.0193** on this
task, while their engine reported 0. The FastLabel v3.4 label for the task is 1.7e-7 from the exact label.

So legacy Tau-pencil outputs (`cmp_v3/labels/*_{tp,lf,rand}.npz` and Astra's `global_inverse/records/reference*/`,
which come from the same engine) are not trustworthy references. All comparisons now use freshly verified
exact-kernel labels (`xlf_cbc/ref_cert/`).

## 4. Mathematical guarantees and their implementation

| guarantee | proof | implementation |
|---|---|---|
| global TV bound TV(p,q) ≤ 1 − Σ min(pᵢ, aᵢ) from per-cell lower credits and an upper on outside mass | PROVEN (`XLF_PROOF_FOUNDATIONS.md`) | `kernel.wire_bound` uses the **frontier** form: upper = Σ IN uppers + Σ live-piece volumes; integer units of 2⁻¹⁶⁰ with floor/ceil; evaluated on the decoded payload |
| safety of interruption and omitted siblings | PROVEN for the abstract ledger; the frontier variant needs the invariant *F ⊆ ⋃ IN ∪ ⋃ live (a.e.)* | each kernel transition replaces a live piece by children covering parent ∩ F. Argued per transition in `XLF_CBC_DESIGN.md` §3. No mechanized proof |
| exact inverse shadow C_K(r) | PROVEN (all-pairs elimination) | `solver.cone`, exact integers |
| guaranteed shadow ∩_{s∈P} S_K(s) = ∩ over vertices, with the tightest pair per facet pair | PROVEN by convexity of C_K(r) in s. I re-derived it here: valid for vertices outside K, over-strict but sound for vertices on or inside K | `envelopes.Envelope.inner` |
| possible shadow ⊇ ⋃_{s∈P} S_K(s) | PROVEN as a sound superset (re-derived); exact only when P is strictly on one side of every facet plane involved | `Envelope.outer`; optional exact support planes `penumbra.py` (float proposals, exact validation) |
| scanline screen | directed rounding of normalized rows to 2⁴⁵ is sound because receiver homogeneous coordinates are positive. int64 range: \|coef\| ≤ 2⁴⁵, coordinates ≤ 2¹¹, so \|value\| < 2⁵⁸ | `raster.scan` (Numba, integer only). The range argument is not asserted at run time (only `denom ≤ 2048`) |
| clearance > 2⁻²⁶ | PROVEN (outward offset cover, inward translates) | `Observation.cleared`, `envelopes.clearance_partition`, exact integers |
| exactly normalized output | PROVEN (largest-remainder quotas of 2⁵² ticks) | `ledger.encode/decode`; the run writer re-reads and checks the bytes |

## 5. Trusted-kernel operations without a proof of the code

* `Poly.split`, `cap_hull` and `Poly.volume` rely on a face-cycle invariant: convex faces with vertices in cyclic order
  and no face lying in the cut plane. The fan volume uses `abs`, which is correct only under that invariant. Tested by
  conservation checks, not proved.
* `grid_pieces`: correctness of exact cell clipping. A conservation check guards it at run time.
* Envelope and scan code versus the written lemmas: a manual correspondence, plus exact unit tests on small
  fixtures.
* The frontier invariant across `condition_pair`, `exclude_window` and `advance`: argued manually, no mechanized
  check.
* Input decoding (`inputs.load`): bit order (big-endian packbits of [Z,Y,X]), axes and origin. Agrees with Astra's
  229M-voxel native rerender.
* No isolation: Python private attributes do not stop a hostile search from editing `_state`. This is a research
  semantic model.

## 6. Numerical calculations without rigorous bounds

All decisions use exact integers or rationals. The remaining items:

* the int64 range argument of the Numba scan (fixed below as a run-time guard);
* Numba build identity (`cache=True` keyed on source only);
* float hull proposals in `convex_generators` and `penumbra` are untrusted, and they are validated exactly.

## 7. Are the probability bytes covered by the certificate? Yes, for the research record format.

`wire_bound` decodes the payload and bounds TV against the decoded probabilities. `write_result` re-reads the
envelope, checks the wire sha256, cell count and bound, then publishes atomically. The format (`<Id` records in a
JSON envelope) is a research format, not the production label schema.

## 8. The most important obstacle to scaling

37 of the 56 failures are `unresolved_frontier`: usually **one** live piece with no IN credit after ~192 receiver
cuts, i.e. the hard-coded lineage cap in `Kernel.advance`.

**Cause.** These pieces are long thin cones whose apex lies **on an obstacle vertex or edge** (shadow planes all pass
through occluder edges). At such a vertex the guaranteed-shadow formula over-constrains and the possible shadow
loses its pair rows. Soundness is unaffected, but almost every receiver becomes "ambiguous". The search then proves
receivers one at a time with exact cones: 189 of 197 enforced receivers did not change the piece, and lifting the
cap to 4,000 still did not finish.

**Secondary obstacles.**

* Every scan rescans the whole window, because resolved receivers are not inherited. This is the 19 `time_cap`
  failures, mostly at L = 256.
* Exact Python grid integration of large IN pieces: 7.7 s on `000049_w03_s0`.

## 9. Previous Claude XLF ideas that help without weakening soundness

* Plain geometric splits as an untrusted proposal: bisection away from an obstacle contact, and long-axis
  bisection. The kernel only needs a trusted split transition.
* Inheritance of resolved receivers by containment: children rescan only the parent's ambiguous set.
* Hierarchical pure-class receiver blocks (the XLF multires scan). Not needed for a scanline screen, but useful for
  the residual sets.
* Angular-extreme receiver selection, which finds the pencil extremes that define F's boundary.
* From the XLF failures, keep:
  * never trust a replay of an incomplete tree;
  * charge every unresolved region;
  * one process per task in batches.
* The exact-kernel verifier (`yushu/3d-v3-staging`) is the independent research reference. It already resolved one
  dispute.
