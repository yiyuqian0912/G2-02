# XLF CBC labeller: implementation and assurance

Code in `experiments/credit_solver/` (plus the exact geometry it reuses from `experiments/global_inverse/` and
`experiments/xlf_assurance/`). Run environment:

```sh
cd Phase4_Multi_Source_3D/RIND3D_Posterior/experiments/credit_solver
export PYTHONPATH=".:../../src:../../../RIND3D_Generator/src:../../../RIND3D_Dataset1K" OMP_NUM_THREADS=1
python cbc_run.py CONTAINER.npz WINDOW CHANNEL OUT.json --seconds 8      # one label
python batch.py OUTDIR ../global_inverse/tasks.json:holdout ... --runner cbc_run.py   # benchmark
python compare.py OUTDIR                                                  # reference comparison
python -m pytest -q test_cbc.py test_wire.py ../global_inverse/test_global.py
```

## 1. Modules and trust

| module | role | trusted? |
|---|---|---|
| `inputs.py` (Astra) | decode one v3 container channel: canonical geometry rows, [x,y,z] class array, origin, problem identity | **yes** (input) |
| `cbc_kernel.py` | state, transitions, exact enforcement (mask-restricted), body-subset screens, settlement, octree cell credits, integer encoder, snapshot | **yes** |
| `screen2.py` | Numba integer screens (full window / residual list), candidate-body masks, range guard | **yes** |
| `global_inverse/poly.py` (Astra, + float sign filter) | exact homogeneous-integer polytopes: split, volume, `side` | **yes** |
| `global_inverse/solver.py` (Astra): `cone`, `apply_pair`, `subtract_many` | exact inverse cones, pair conditioning | **yes** |
| `global_inverse/envelopes.py` (Astra): `Envelope`, `clearance_partition` | guaranteed and possible receiver shadows; exact clearance partition | **yes** |
| `global_inverse/raster.py` (Astra): `encoded` | directed fixed-point encoding of envelope rows | **yes** |
| `global_inverse/penumbra.py` (Astra, + axis fallbacks) | support-plane **proposals** (binary64 hull); exact validation inside | validation only |
| `xlf_assurance/ledger.py` (Astra): `encode`, `decode` | exact quota encoding to binary64 probabilities, strict decoding | **yes** |
| `kernel.py` (Astra): `wire_bound`, `ceil_mass` | integer TV bound on the decoded payload | **yes** |
| `cbc_search.py`, `run.py` writer, `cbc_run.py` | search policy; atomic envelope writer that re-reads the bytes | no (the writer re-checks) |
| `exactcheck.py`, `audit_exact.py`, `compare.py`, `diag.py`, `hunt.py` | independent research instruments | no (not in the certificate path) |

The trusted base is about 1,400 lines of Python and Numba. Search heuristics (about 200 lines) are outside it.

## 2. Numerical model

| computation | representation | why it is exact or outward |
|---|---|---|
| obstacles | integer rows from the canonical binary64 parameters (`integer_row`, exact rationals) | exact |
| receivers | homogeneous integers (2(o+i)+1, …, 2048) | exact |
| polytopes | primitive homogeneous integer vertices (about 500 bits typical); exact plane intersection | exact; a split must conserve volume exactly (run-time guard); a cut cannot create volume (guard) |
| sign tests | binary64 filter certain only if \|v\| > 10⁻¹²·Σ\|terms\| (rounding error ≤ 6u·Σ\|terms\|); exact integers otherwise, or when non-finite or tiny | exact decision. Certified payloads are byte-identical with and without the filter on 5 regression tasks |
| receiver screens | rows normalized to 2⁴⁵, rounded up (guaranteed) or down (possible); int64 with \|value\| < 2⁵⁸ | directed, sound because coordinates are positive; range guard on every screen |
| volumes, cell credits | Python Fractions, then floor/ceiling at 2⁻¹⁶⁰; interior cells exactly 2¹³⁹ units | exact, with an outward integer conversion; conservation check per IN piece |
| clearance | outward facet offsets with a directed integer square root; inward rational translates | exact |
| TV bound | integer arithmetic at 2⁻⁹⁶ (floors of the common mass) | outward: ≥ exact bound, ≤ exact + cells·2⁻⁹⁶ (property-tested) |
| output | 2⁵² exact ticks (largest remainder), binary64 values summing exactly to 1 | exact; decoding rejects non-normalized, unsorted or invalid bytes |

The only floating-point computations are two filters: the sign filter (proved margin, exact fallback) and the
support-plane proposals (validated exactly).

## 3. Tests and checks (current status)

| check | scope | result |
|---|---|---|
| Astra's assurance tests (`xlf_assurance/test_assurance.py`) | abstract ledger, missing sibling, wire | 13 pass |
| Astra's geometry tests (`global_inverse/test_global.py`, `mixed_patch/test_*.py`) | cones, polytopes, envelopes, clearance, encoding | 25 pass |
| `test_cbc.py` | `scan_ids` ≡ `scan_collect` (ids and masks); **full-window screen ≡ pointwise int64 reference** (contradictions, ambiguous ids, masks; 16 pieces including cut pieces); corrupt lists rejected; split exact and stale handles rejected; enforced children satisfy their receivers (exact segment tests); masks sound (excluded cones miss the piece, exact); body-subset residual screen never weaker than the all-body screen (11-body scene); window and obstacles excluded; end-to-end certificate on decoded bytes; unvisited mass charged | 11 pass |
| `test_wire.py` | 300 random exact instances: exact encoding, wire_bound within [exact, exact + m·2⁻⁹⁶], true TV ≤ bound; tampered payload rejected; 200 instances: `encode_int` byte-identical to `ledger.encode` (ties included) | 501 pass |
| byte identity | payload sha256 and transition count unchanged across optimizations (float filter, masks, octree, per-edge cache) on 5 tasks | identical |
| run-time guards | split conserves exact volume; an exact cut cannot create volume; octree credit conserves exact volume; facet rows validated; screen ranges checked | abort on violation (never observed) |
| exact differential audit (`audit_exact.py`) | real tasks: sampled exact points in IN pieces must be feasible; sampled points discarded by each transition kind must not be feasible | see `XLF_CBC_RESULTS.md` |
| independent references | exact-kernel certified labels (`yushu/3d-v3-staging` verifier) | see `XLF_CBC_RESULTS.md` |

## 4. Remaining gaps (OPEN)

1. **No machine-checked proof.** The transition lemmas (`XLF_CBC_DESIGN.md`) are manual. The Python code is
   tested against them, not proved equal to them.
2. **Face-structure invariant of `Poly`.** The fan volume and the facet rows assume convex, cyclically ordered faces.
   They are guarded at run time:
   * split volume conservation;
   * facet-row validation;
   * cell-credit conservation.

   They are not proved.
3. **Numba build identity.** `cache=True` keys on the source only. A production build would pin the compiler, flags
   and cache.
4. **Isolation.** Python cannot prevent a hostile search from mutating kernel state. The model assumes the search
   is merely untrusted, not adversarial at the interpreter level.
5. **Input trust.** The container decoder and the canonical geometry reader are trusted inputs. Agreement with the
   native renderer is tested (Astra's rerender: 229M voxels), not proved.
6. **The reference engine itself.** Legacy Tau-pencil labels are unreliable (TV 0.019 error observed). Only the exact
   verifier's labels are used as references.
