# Dataset ↔ Teacher interoperability (v0.3)

**Components.**
- `rind3d_training/teacher.py` (TeacherPhysics v0.3, privileged);
- `rind3d_training/phase3_adapter.py` (Phase 3 Teacher, imported read-only);
- `tests/test_training_ready.py` (integration tests on the actual exported files);
- `rind3d_training/interop_audit.py` (measured audit; output `outputs/tests/interop_audit.json`).

The v0.2 `rind3d_production.teacher` is left unchanged.

## 1. Two supervision routes, one data format

| Route | What it uses | When |
|---|---|---|
| **A: direct** conditional probability training | Student shards + privileged `join.npy` targets (sources, strengths, K) and optional population weights | always available; needs no Teacher call |
| **B: selective privileged Teacher** (physical / Gibbs) | `TeacherPhysics(scene)` or the Phase 3 `PhysicalOracle` through the adapter, called **on demand** for chosen windows; receiver banks with exact inclusion probabilities | for the windows a Teacher experiment selects; no per-window Gibbs precomputation is required or stored |

Source truth travels only in the privileged target channel. The Student feature path (`StudentShards.read/batch`) cannot return it (audit L10, test 10).

## 2. The 10-step integration test

Tests are run on real canary samples (SB and MA) read from the exported student shards and joined to the privileged view by opaque sample id. **All pass:** 21/21 tests, `outputs/tests/pytest.txt`.

| # | Requirement | Test | What is checked |
|---|---|---|---|
| 1 | same scene through both views | `test_01_same_scene_both_views[SB, MA]` | student sample → sample_id → privileged row → SceneRef → `TeacherPhysics.observation` equals the student response and SOLID bit-for-bit; origin and side agree |
| 2 | retrieve the true label | `test_02_true_label[SB, MA]` | privileged sources equal the producer's private sources as an unordered set; NaN padding beyond K; the true sources are certified legal |
| 3 | evaluate source visibility | `test_03_source_visibility[SB, MA]` | certified point visibility of the true source(s) at 2048 sampled FREE receivers reproduces the observed response (MA: Σ a_k lit_k within 1e-12) |
| 4 | reconstruct the local observation | `test_04_reconstruct_observation[SB cpu, MA cpu, SB cuda]` | certified dense re-render (CPU and **CUDA**) equals the student response and returns the certified SOLID mask |
| 5 | uniform and stratified banks | `test_05_uniform_and_stratified_banks` | uniform: k distinct FREE receivers, π = k/N, Σw = N; stratified: **exactly k** for k ∈ {2, 7, 64, 255, 256, 1001}, every stratum represented, Σw = N; census when k ≥ N |
| 6 | inclusion probabilities and weights | `test_06_inclusion_probabilities_and_weights` | Horvitz–Thompson estimates are unbiased for a full-window FREE total (300 banks per design, within 4 SE); population selection weights reproduce the natural rate-stream frequencies (mixed, informative, extreme, source-containing) for every side |
| 7 | full and sampled FREE mismatch | `test_07_full_and_sampled_free_mismatch` | true source: full mismatch 0; perturbed legal candidate: HT estimate (k = 4096) within 5 SE of the exact count; Phase 3 `full_window_cost` equals the v0.3 fraction; Phase 3 sparse cost is 0 for the truth |
| 8 | candidate legality | `test_08_candidate_legality` | outside-world / SOLID / clearance reasons; the source-excluded domain rejects closed-box candidates; agrees with Phase 3 `source_valid` with and without a window |
| 9 | source-containing windows | `test_09_source_containing_windows` | a natural window that contains the true source: weight 0 in the source-excluded task and > 0 in the global task; legality says `inside_window_box`; full mismatch 0; no subset window ever contains a source |
| 10 | Student inputs carry no privileged data | `test_10_student_inputs_contain_no_privileged_data` | file-level leakage audit L1–L10 on the exported files passes; feature keys are exactly the declared five; no id or group in a batch |

Additional audits, also in the test file:
- `test_no_free_windows_flagged_never_zero`;
- `test_ma_permutation_invariance`;
- `test_error_messages`;
- `test_sampler_deterministic_balanced_bucketed`;
- `test_v02_prefix_and_preserved_policy`;
- `test_space_to_depth_and_blocksparse_exact`.

## 3. Audit findings and their fixes [M]

| Item | v0.2 `rind3d_production.teacher` | Phase 3 Teacher (`rind_phase3`) | v0.3 (`rind3d_training.teacher` + adapter) |
|---|---|---|---|
| **CUDA window interface** | without an external factory: `AttributeError: 'NoneType' object has no attribute 'source'`; with one, returns **SOLID = None** | `rerender_window` owns its factory and returns SOLID; single source only | owns a lazy `BackendFactory` (closed by `close()`); returns lit + certified SOLID; **SOLID equal to the stored mask**; L128 in 0.22 s including CUDA initialisation |
| **No-FREE windows** (69 in the canary export: 53 L64, 16 L128) | `receiver_bank` raises `ZeroDivisionError` | `sample_probes` raises `ValueError("Observation has no FREE receiver")` (clear); `full_window_cost` **silently returns NaN** (only a NumPy RuntimeWarning, no status) | bank `status="no_free"`, k = 0; full and sampled mismatch `status="no_free"`, NaN, count None; `strict=True` raises `NoFreeReceivers("window origin=(…) L=64 has no FREE receiver: mismatch is undefined")`; the adapter returns `no_free` for Phase 3 costs |
| **Stratified sample size** | 68 of 120 real cases (k ∈ {7, 64, 256}) returned k ≠ requested (max error 4) | n/a (Phase 3 probes are uniform) | 0 of 120: exact k by largest-remainder allocation, ≥ 1 per stratum, π_h = k_h/N_h; census if k ≥ N; clear error if k < number of strata |
| **Probe inclusion design** | uniform SRS | uniform rejection sampling, but when too few FREE voxels are hit it falls back to a **deterministic packed-byte scan**: non-uniform, so inclusion probabilities are unknown and the plain mean is biased. Expected on ≈ 0.1% of natural SB windows (FREE fraction below ≈ 2.3%); measured on an L256 window with FREE = 36,756 (0.22%) | always SRS over the FREE list (exact π); `phase3_sampled_cost` reports `uniform_design: False` when the fallback fired |
| **MA semantics** | additive response available | single-source only (`rerender_window` accepts exactly one source) | `response` / `combined` compute Σ a_k lit_k in binary64; `additive_response` builds MA on the Phase 3 oracle; sets are unordered (permutation test passes, tolerance 1e-12) |
| **Sample identity** | student view returns the real `scene_id` | `RIND3D.__getitem__` returns the real `scene_id`. In v0.2 this is SHA256(profile, geometry_id, sources, strengths), a commitment to the label | the adapter puts `opaque-<sample_id>` in the Phase 3 dict; v0.3 student shards carry only opaque HMAC ids (audit L8) |
| **Window-box convention** | closed box for source containment | half-open `Window.contains` [o, o+L) | the closed box is declared in the contract; the two agree off the boundary (measure zero); test 8 |
| **Numerical policy** | production `POLICY` | `GeometryContext` default policy (policy hash `3ac569dd…` vs production `942a58d4…`) | certified answers agree: legality, point visibility and a full L128 window are identical (A6) |
| **Error messages** | generic `ValueError("illegal source in query")` | typed `OracleFailure` / `QueryBudgetExceeded` | `IllegalSource` lists indices and reasons; `ValueError` names the length mismatch / unknown backend / stratum count |

## 4. Phase 3 adapter (lightweight, read-only)

```python
from rind3d_training import phase3_adapter as P3
obs = P3.observation_for_phase3(scene_ref, origin, L, sample_id)   # Phase 3 dict; scene_id := opaque id
with P3.oracle_for_phase3(scene_ref) as oracle:                    # rind_phase3.oracle.PhysicalOracle
    P3.phase3_full_cost(oracle, obs, candidates)                   # guarded: status "no_free" instead of 1/0
    P3.phase3_sampled_cost(oracle, obs, candidates, count=256)     # reports whether the probe design stayed uniform
    P3.additive_response(oracle, sources, strengths, receivers)    # MA semantics on the single-source oracle
```

The adapter never writes into the Phase 3 tree. It runs under `PYTHONDONTWRITEBYTECODE=1`, and the preservation trees confirm this.

The Teacher branch can keep its own data access unchanged. To use v0.3 data it needs three things:
1. the `SceneRef` from the privileged `scenes.json`;
2. the origin and side from the privileged join;
3. `observation_for_phase3`.

## 5. Recommended Teacher interface (v0.3)

- `TeacherPhysics(scene_ref)`:
  - `.ref()`;
  - `.legality(points, window=None)` → (valid, reason);
  - `.visibility(sources, receivers)` → int8 1 / 0 / −1(SOLID);
  - `.response(sources, strengths, receivers)` (MA additive, order given);
  - `.window(sources, origin, L, backend)` → (lit [S, L, L, L], SOLID [L, L, L]), cpu or CUDA (CUDA uses an owned factory);
  - `.observation(origin, L)` (stored fields, MA combined);
  - `.receiver_bank(origin, L, design, k, seed, response, strict)`;
  - `.full_mismatch(...)`;
  - `.sampled_mismatch(sources, strengths, bank, observed)` (HT);
  - `.close()`.
- **Errors.**
  - `IllegalSource` lists the offending indices and reasons.
  - `NoFreeReceivers` is raised in strict mode only; the default returns status `no_free`.
  - A `ValueError` names the problem: strength/source length mismatch, unknown backend, `k` below the number of strata, or a stratified design without a response.
- **Budget for route B** [M, this laptop]:

| Teacher call | Cost |
|---|---|
| legality | 0.075 ms per candidate |
| certified point visibility | 3.1 M source–receiver pairs/s (16 threads) |
| sampled HT mismatch (k = 256) | 0.66 / 0.75 / 0.79 ms per candidate (L64 / L128 / L256) |
| full-window mismatch, CPU | 1.1 ms / 136 ms / 1.35 s (L64 / L128 / L256; window dependent) |
| full-window mismatch, CUDA | 13 / 18 / 82 ms (L64 / L128 / L256) |
| Phase 3 full cost, L128 CPU | 74 ms |
| Phase 3 sparse cost (k = 256) | 0.88 ms per candidate |

**Consequence.** Route B with sampled banks costs under 1 ms per candidate per window. For example, a 256-candidate Gibbs sweep on one window costs about 0.2 s, so selective Teacher supervision is affordable on demand. Precomputing full-window costs for every window would not be: at L256 CPU, one candidate on one window takes 1.35 s.
