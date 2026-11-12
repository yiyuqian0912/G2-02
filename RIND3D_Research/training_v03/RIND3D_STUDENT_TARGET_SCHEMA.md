# RIND3D Student and Target Schema (v0.3 candidate)

Schemas: `rind3d-student-window-v0.3` (the Student view) and `rind3d-privileged-window-v0.3` (the privileged view). The normative contract is `RIND3D_TRAINING_CONTRACT.json`. The code is in:
- `rind3d_training/export.py`: the writer, a trusted process;
- `rind3d_training/student.py`: the Student reader, which needs only NumPy and zstd;
- `rind3d_training/leakage.py`: the file-level audit.

## 1. Two views, one join key

| | Student view | Privileged view |
|---|---|---|
| Location | `outputs/student/NAME/<split>/` | `outputs/privileged/export/NAME/` (mode 0700) |
| Who reads it | model input pipeline (DataLoader workers) | trusted training / evaluation process; the Teacher |
| Contents | local response, local SOLID mask, origin, side, modality | scene, geometry and oracle reference; true sources and strengths; selection law; weights; flags |
| Join | `sample_id` (opaque, 64-bit) | `sample_id` → row of `join.npy` |

Sample ids are HMAC-SHA256 values of a private 256-bit export key over `(export, scene_id, window)`. They cannot be inverted to scene ids, and a different export has different ids. They are join keys only. `StudentShards.read()` returns them separately from the feature dictionary, so they cannot reach a model by accident.

The per-scene `group` token is also opaque (a 32-bit HMAC). It exists only so the scene-balanced sampler can balance scenes, and it is never part of a returned batch.

## 2. Student files

```
STUDENT_MANIFEST.json                schema, feature keys, encoding, per (modality, L) counts, shard SHA256s
shards/<SB|MA>-L<064|128|256>-<n>.bin    concatenated zstd payloads
shards/<SB|MA>-L<064|128|256>-<n>.idx.npy  one row per sample
```

There is one shard per side and modality, and each shard holds 16 scenes. Their samples are interleaved by a private permutation, so position in a shard is independent of the hidden subset tags (audit L9).

`idx.npy` dtype (exact; enforced by audit L2):

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | u8 | opaque join key (not a feature) |
| `group` | u4 | opaque per-scene sampling token (not a feature) |
| `modality` | u1 | 0 = SB single-source binary, 1 = MA multi-source additive (**feature**) |
| `L` | u2 | window side 64 / 128 / 256 (**feature**) |
| `x, y, z` | u2 | window origin in voxels (**feature**) |
| `resp_offset, resp_bytes` | u8, u4 | response payload span in the `.bin` |
| `npal, nplanes` | u1 | MA local palette size and number of code bit planes (SB: 2, 1) |
| `solid_offset, solid_bytes` | u8, u4 | SOLID payload span |
| `resp_crc, solid_crc` | u4 | CRC32 of the decompressed payloads |

**Payloads (decompressed).**
- *SB response:* L³/8 bytes of packed lit bits.
- *MA response:* `npal` float64 palette values, ascending and only those present in the window, followed by `nplanes` × L³/8 bytes of packed code planes. The code is Σ_b plane_b·2^b, and the value is `palette[code]`.
- *SOLID:* L³/8 bytes of packed bits.
- *Layout:* all arrays are ZYX, with bits packed little-endian along X.

**Local coordinates.** Receiver (i, j, k) (XYZ) of a window with origin o has world centre (o + (i, j, k) + ½)/1024.

**Features returned by `StudentShards.read(i)`.** The features are `observed_response`, `observed_solid`, `window_origin`, `window_size` and `modality`. The response is bool for SB and float64 for MA.

**`batch(indices, packed=True)`.** It returns packed uint8 tensors (for MA, palette plus code planes) for unpacking on the GPU, and it accepts a single window size only.

**Forbidden in the Student view.** Audits L1–L10 enforce this on the exported files:
- scene and geometry ids; geometry; full fields;
- sources, strengths and K;
- subset, rank, proposal counts, weights, the containment flag and the nonlocal estimate;
- seeds.

Two physical observables are **allowed**:
- MA palettes contain sums of strengths, because the observation is the additive response.
- A window that is entirely SOLID reveals that it has no FREE receiver.

## 3. Privileged join (`join.npy`, one row per sample)

| Field | Meaning |
|---|---|
| `sample_id, scene, record` | join key, scene ordinal (`scenes.json`), index record |
| `split, modality, L, x, y, z` | split 0/1/2 = train/val/test; window |
| `subset, rank` | obs-index-v0.3 subset code and acceptance rank in its stream |
| `in_P102, in_P256` | profile membership (P102 = the exact obs-policy-v0.2 prefix) |
| `contains_source` | a source lies in the closed window box (any source for MA) |
| `no_free, category, free` | 0 no-FREE, 1 all-lit, 2 all-shadow, 3 mixed, 4 constant-lit (MA); FREE count |
| `A_informative, A_extreme, A_rare` | rule indicators (exact, from counts) |
| `nonlocal_flag` | nonlocal estimate ≥ 0.9 for mixed, source-excluded windows (evaluation stratum) |
| `w_global_mixture` | population weight, sb-global / ma-additive mixture profile (0 = not in profile) |
| `w_srcexcl_mixture` | population weight, sb-source-excluded profile (0 for source-containing windows) |
| `K, sources[4,3], strengths[4]` | target set: NaN padded, lexicographic order (order carries no meaning) |

`scenes.json` maps the scene ordinal to:
- scene_id and geometry_id;
- release, family, profile, K and split;
- the field directory, for `TeacherPhysics` and the Phase 3 adapter.

`selection_law` quantities (P(A_s) and P(contains source) per scene and side) are recomputed by `selection.rule_probabilities` from the privileged index reports.

## 4. Targets per task profile

| Task | Target | Domain | Weight column | Notes |
|---|---|---|---|---|
| `sb-global-v0.3` natural | S | (0,1)³ (prior support D(G)) | 1 for `subset==0` | clean population |
| `sb-global-v0.3` mixture | S | (0,1)³ | `w_global_mixture` | weights are required |
| `sb-source-excluded-v0.3` | S | (0,1)³ minus the closed box | `w_srcexcl_mixture` | declared conditional task; source-containing windows excluded |
| `ma-additive-v0.3` | {(S_k, a_k)}, with K | (0,1)³ per element | `w_global_mixture` | permutation-invariant loss |
| evaluation strata | — | — | per stratum | includes nonlocal, no-FREE and contains-source |

No-FREE windows stay in the natural population, because a Student can still learn the prior from them. Mismatch-based supervision excludes them, and they are never scored as zero error.

## 5. Minimal training-side usage

```python
from rind3d_training.student import StudentShards, SceneBalancedSampler, gpu_unpack
sh = StudentShards("outputs/student/canary-v03/train")
join = np.load("outputs/privileged/export/canary-v03/join.npy")      # trusted trainer only
row_of = {int(s): i for i, s in enumerate(join["sample_id"])}
plan = SceneBalancedSampler(sh.index, {64: 32, 128: 8, 256: 2}, seed=0).epoch(e)
for idx in plan:                                                     # one window size per batch
    b = sh.batch(idx, packed=True)                                   # features + sample_id
    rows = [row_of[int(s)] for s in b["sample_id"]]
    y, w = join["sources"][rows], join["w_global_mixture"][rows]     # targets / weights: privileged
```

The working implementation, including on-GPU unpacking and lossless space-to-depth, is `bench_student.py` (`to_input`, `b4_training`).
