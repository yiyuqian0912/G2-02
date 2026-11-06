# Fixed-source B2 angular research

This isolated CPU experiment accelerates the unchanged continuous Gaussian
sum. It does not create a dataset scene, register a production family, alter a
public reader, or enable READY. See the [mathematics and research mapping](../../../docs/GAUSSIAN_B2_ANGULAR.md).
The [measured results and integration decision](../../../docs/GAUSSIAN_B2_ANGULAR_RESULTS.md)
include negative results and the independent-verification limits.

The Gaussian sum defines **solid geometry**, not light intensity or opacity.
The response is still binary closed-segment point-light visibility. The
angular quadtree is a rendering certificate structure, separate from Phase 3's
teacher source-space octree; no teacher/training code is changed.

The permanent public dataset name is **RIND3D**. All fields here are frozen
synthetic research cases; N is the sampling resolution of the whole unit world,
not the side of a public observation window.

## Build and reproduce

Run from `RIND3D_Generator` using the same scientific Python environment as the
[previous B2 study](../b2_native/README.md): Python 3.12, NumPy, SciPy,
python-flint, gmpy2, pybind11 and pytest. System development dependencies include
CMake, a C++20 compiler, GMP/GMPXX and MPFR. Measured support is Linux x86-64;
no CUDA is required for this experiment. No cross-platform claim follows from
these tests. Compiler flags disable fast-math and floating-point contraction;
the original environment guards reject incompatible rounding/underflow modes.

```bash
python -m experiments.gaussian_geometry.b2_native.build
python -m experiments.gaussian_geometry.b2_regions.build
python -m experiments.gaussian_geometry.b2_angular.build
python -m pytest -q experiments/gaussian_geometry/tests/test_b2_angular.py
python -m experiments.gaussian_geometry.b2_angular.materialize \
  --output outputs/gaussian-b2-angular-v1/recovered-inputs
python -m experiments.gaussian_geometry.b2_angular.benchmark \
  --input outputs/gaussian-b2-angular-v1/recovered-inputs \
  --output outputs/gaussian-b2-angular-v1/reproduced32 \
  --resolutions 32 --methods A_fixed6 A_fixed7 B_spatial C_angular D_union E_hybrid
```

The repository's default pytest path is `tests`, so the full acceptance run
uses both suites explicitly:

```bash
RIND_TEST_CUDA=1 python -m pytest -q tests
python -m pytest -q experiments/gaussian_geometry/tests
```

Omit the CUDA opt-in on a CPU-only host; the new B2 method itself is CPU-only.
Real private/production pilot tests retain their existing opt-in requirements.

`materialize` regenerates only four deterministic **research input parameter
records** and their seeded synthetic sources. It checks the exact frozen
geometry/file/source hashes. It never writes to RIND3D v0.1 or generates its
1,000 dataset scenes. A different numerical environment that changes those
identities fails explicitly. Existing frozen inputs may instead be passed
directly with `--input outputs/gaussian-geometry-v1/distribution-v2`.

The optional old-field baseline is used if present. Every primary/reference
pair is independently rendered and compared even without that baseline.
Checkpoints contain file hashes and full run identity; changed source,
configuration or input requires a new output directory. Field jobs run in
fresh serial child processes for bounded per-job RSS and cold measurements.
Process startup and source-clearance setup are recorded separately from the
physical field cold total, and also as an inclusive end-to-end measurement.

Run all four 64³ cases only after complete 32³ agreement. The bounded 128³
example is:

```bash
python -m experiments.gaussian_geometry.b2_angular.benchmark \
  --input outputs/gaussian-b2-angular-v1/recovered-inputs \
  --output outputs/gaussian-b2-angular-v1/reproduced128 \
  --ordinals 14 7 --resolutions 128 --methods A_fixed7 D_union \
  --cover-depth 7 --cover-nodes 131073 --seconds 480
```

Changing cover depth is a controlled comparison, not the new algorithm. The
new part is a source-specific directional depth envelope and certified union
coverage. A6/A7 invoke the original region module unchanged.

The optional `P_empty_prefix` and `Q_hybrid_prefix` use the complete outer cover
for FREE-prefix reuse. Set `--prefix-builds` to bound their extra construction.
They are experiments, not an assumed optimization.

Two controls isolate simpler explanations: `R_point_cover` applies the existing
outer-cover FREE test again at individual residual receivers; `S_all_inner`
uses the unchanged old renderer with every certified inner box. Neither is
presented as a new angular certificate. `--angular-boxes 256` restricts C/D to
the same original inner subset, separating box-count effects from directional
reuse and multi-blocker fusion.

```bash
python -m experiments.gaussian_geometry.b2_angular.controls \
  --input outputs/gaussian-b2-angular-v1/recovered-inputs \
  --output outputs/gaussian-b2-angular-v1/reproduced-controls
python -m experiments.gaussian_geometry.b2_angular.analyze \
  --inputs outputs/gaussian-b2-angular-v1/reproduced32 \
  --output outputs/gaussian-b2-angular-v1/reproduced-analysis.json
python -m experiments.gaussian_geometry.b2_angular.contact_control \
  --output outputs/gaussian-b2-angular-v1/reproduced-contact.json
```

For independent numerical-source versions, use a new build path and set
`RIND_B2_ANGULAR_BUILD` to its absolute path. Corresponding overrides for the
old modules are `RIND_B2_NATIVE_BUILD` and `RIND_B2_REGIONS_BUILD`. Never replace
a shared library in a running process. Every import verifies source and binary
identities. Historical measured snapshots are preserved in ignored outputs.

Compact `../results/b2-angular-final-to-measured-v1.patch` and
`../results/b2-angular-final-to-measured-v2.patch` source
transitions reconstruct the older measured native and benchmark sources from
this final tree. Reconstruction was tested in fresh temporary copies against
all archived native source hashes. Apply a transition only in a separate copy,
then build there; it is not an in-place update to the working research tree.

## Interpretation

`AngularOracle(scene, independent=False, method=3).field(source, 32)` returns
response, solid and validity arrays, completion/reason, counters and timing.
`tile(source, 1024, [x,y,z], 8)` uses arbitrary legal integer XYZ origins and
allocates only the requested ZYX tile. It is not a full 1024³ streaming writer.
Source clearance must first be proved by the original oracle, as benchmarks do.
`ready_capable` is always false.

With `trace=True`, `evaluation_path` retains the old 1–10 meanings and
`visibility_evidence` additionally distinguishes angular shadow (1), angular
empty prefix (2), and adaptive spatial evidence (3). A zero evidence byte means
the original certificate or scalar path; value 4 marks the point-cover control.
Region and fallback timers include
their invoked angular work; use angular setup/leaf timers to avoid double
counting. `point_cover_seconds` is included in fallback time, not additional.
Trace nanoseconds measure actual scalar calls only. Retained angular
tiles are proved bounds or UNKNOWN, never a sampled reconstruction of the true
first-hit function.

The geometry-only `extension().screen(...)` test accepts **abstract boxes**.
It proves a geometric implication conditional on those boxes being SOLID. It
must not be used as B2 physical truth for arbitrary unverified boxes. Physical
field runs construct their own full-sum certified boxes independently.

No finite map, sparse fixture sample, or benchmark is a complete proof of
all B2 intersections. UNKNOWN is refined or passed to the unchanged canonical
fallback. General unresolved transcendental equality remains unresolved.
