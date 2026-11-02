# Certified teacher queries

`PhysicalOracle` reuses the canonical decoder, exact convex geometry context and native parametric clipping/GMP implementation in `RIND3D_Generator`. It does not voxelize geometry or implement another visibility engine. The original canonical CGAL/Arb service remains the independent point-reference backend. Scope is the current nine world-contained convex-union families; unsupported geometry is rejected explicitly, and existing generator READY gates remain unchanged.

```python
from rind_phase3 import RIND3D
from rind_phase3.oracle import PhysicalOracle
from rind_phase3.physics import sample_probes, SparsePhysicalCost, full_window_cost

with RIND3D("data/RIND3D-v0.1", split=None) as data:
    observation = data[0]
    scene = data.teacher_scene(observation["scene_id"])
    probes = sample_probes(observation, count=64, seed=0)
    with PhysicalOracle(scene) as oracle:
        candidates = [[0.1, 0.2, 0.3], [0.10001, 0.2, 0.3]]
        validity = oracle.source_valid(candidates, window=probes.window)
        response = oracle.evaluate_points(candidates, probes.world_xyz, window=probes.window)
        costs, evidence = SparsePhysicalCost(oracle, probes).evaluate(candidates)
```

Sources and receivers are finite float64 XYZ arrays. `evaluate_points` computes the **Cartesian product** `(N_sources,N_receivers)` in bounded receiver chunks. Default maximum product is 1,048,576 and chunk size 4,096. Returned int8 response is 1 for lit, 0 for blocked/solid, and **-1 for a certified-invalid source**. A separate receiver solid mask distinguishes solid from FREE shadow. Inspect `validity.valid`, `status` and `reasons`; invalid rows are not physical labels. `certified=True` means returned binary labels have passed the certified predicate, not that every proposed source is valid.

Source validity enforces the open world, closed-solid exclusion, exact clearance > `2^-26` and, when supplied, half-open observation exclusion. There is no teacher 32/1024 separation restriction. Close candidate sources can be queried. Geometry and a bounded source-model cache are reused. Statistics count attempted source validations, valid candidate evaluations and actual source/receiver pair requests; native work counters separately record clipping/exact fallbacks. Invalid sources consume candidate attempts but no visibility pairs. Budget overflow raises `QueryBudgetExceeded(status='ABORTED')` before evaluating the oversized product.

Native and canonical failures raise `OracleFailure(status='UNRESOLVED')`; UNKNOWN/ABORTED is never converted to zero/one, a high cost or a successful match. Malformed nonfinite/out-of-world receivers raise validation errors. Count/memory limits are enforced; this interface does not promise immediate interruption of an active GMP/CGAL call. Expensive hard watchdogs need process isolation, not an assumption that a Python timer safely interrupts native code.

Arbitrary-point queries support `backend='cpu_native_certified'` and `'cpu_reference'`. The latter calls the original canonical closed-segment service directly; its first call can pay a substantial geometry-initialization cost. No arbitrary-point CUDA kernel is claimed. CUDA is available for `rerender_window` through the separate `raster_backend='cuda_certified'` option, using existing certified 64³ tiles, directed intervals and exact CPU fallback. Missing CUDA is an explicit failure for that option; `'auto'` reports the backend actually selected. CPU native rastering reuses the same immutable context. No complete 1024³ field is generated per candidate.

`oracle.rerender_window(candidate_xyz, Window(...))` returns full local response and solid arrays, actual backend, certified predicate status and sampled voxel count. Sides are the stored 64/128/256 values and arbitrary legal origins are supported. At most a 256³ local array is returned. `full_window_cost` compares all FREE local voxels against the observation; sparse cost compares only the selected FREE probes. Both normalize by their respective FREE receiver counts. Solid voxels are never treated as negative FREE evidence.

The following objects are deliberately distinct:

- Certified visibility: the exact binary response at a specified continuous receiver.
- Sampled physical cost: exact point labels averaged over an incomplete probe set.
- Full-window cost: exact local-grid disagreement under one certified implementation.
- Equivalent-source certification: original clearance/domain plus full-window equivalence and independent recheck/publication requirements.

A zero sparse cost does not establish full-window compatibility. A zero full-window cost from this adapter is still not the producer's independently verified reference certificate. `WindowResponse.equivalence_certification` remains false. This workspace performs no additional exact reference search and reads no private source labels.
