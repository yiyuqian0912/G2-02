# Continuous Gaussian-sum native research

This is B2: `F(x)=sum w_i exp(-||M_i^-1(x-c_i)||²/2)`, solid `F>=tau`.
It is an isolated experimental backend for future **RIND3D** geometry. It does
not register a family, change the production renderer, or publish READY scenes.
The old v1 experiment, A/B1/C implementations, v0.1 dataset and Phase 3 teacher
are unchanged. The research branch is `yushu/gaussian-b2-native`, based on
`bf4caf1` from `yushu/gaussian-geometry`.

Read [numerical design](../../../docs/GAUSSIAN_B2_NATIVE.md) and
[actual results and limits](../../../docs/GAUSSIAN_B2_RESULTS.md).

## Build and test

Tested: Linux x86-64, Python 3.12.14, C++20, system GMP/GMPXX and MPFR 4.2.2.
The x86 rounding/underflow guard is deliberate; ARM/Windows support is not
claimed. A GPU is unnecessary. The public RIND3D reader does not need this
research extension or these dependencies.

First build the normal development environment according to the generator
README. Use a development environment, never the frozen production environment.
Run from `RIND3D_Generator`:

```bash
export RIND_PYTHON="${RIND_PYTHON:-$PWD/.venv/bin/python}"
uv pip install --python "$RIND_PYTHON" -r experiments/gaussian_geometry/requirements.txt
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.build
"$RIND_PYTHON" -m pytest -q experiments/gaussian_geometry/tests/test_b2_native.py
```

The measured run reused a development interpreter and installed research extras
into an isolated `outputs/gaussian-geometry-v1/deps` target via `PYTHONPATH`.
The build goes into ignored `outputs/gaussian-b2-native-v1/build`. For a custom
build path, pass `--output` to the build command and set `RIND_B2_NATIVE_BUILD`
to its absolute directory. Build/source hashes are checked on import. Restart
Python after rebuilding; a loaded binary is never silently relabeled.

## Query API

```python
from experiments.gaussian_geometry.generators import generate
from experiments.gaussian_geometry.b2_native.api import NativeOracle

scene = generate("gaussian_sum", 20261105, 14, 0.05)
primary = NativeOracle(scene)
reference = NativeOracle(scene, independent=True)

print(primary.membership([0.5, 0.5, 0.5]))
print(primary.segment([0.03125]*3, [0.8, 0.7, 0.6]))
print(primary.classify_box([0.4]*3, [0.45]*3))
print(primary.source_clearance([0.03125]*3))
```

Membership is `FREE/SOLID/UNKNOWN/INVALID`. Segment results are
`MISS/HIT/UNKNOWN/INVALID`. Unknown budgets and reasons are separate from physical
labels. Arrays use ZYX, world coordinates use XYZ and binary64. Exact nonbinary
rational endpoints must use the original rational oracle; this API rejects them
rather than silently rounding a window-entry point.

The native object owns an immutable copy of canonical parameters. `geometry_id`
identifies that snapshot; rebuild after changing the scene. Proof helpers reject
scene/engine identity mismatches. Do not mutate options while a call is running.
Counters on a returned field/batch describe that call. Cumulative Python counters
add snapshots; do not interpret a sum of per-call maxima as a global maximum.

`points(Nx3)` and `rays(source, Nx3)` batch queries. Constructor options include
`coefficients`, `filters`, `groups`, `taylor`, `taylor_depth`, `exact_depth`,
`tail_bounds`, `inherited_regions`, `max_nodes`, `max_depth`, `query_seconds`
and `precision`. Groups default off after their negative matched benchmark.
Reference arithmetic remains 256-bit; primary fallback defaults to 192 bits,
with a higher point-precision retry. Its fast exponential enclosures use directed
MPFR at 53 bits, never unchecked `libm.exp`.

```python
field = reference.field([0.03125]*3, resolution=32, seconds=120)
if field["complete"]:
    response = field["response"]  # bool[Z,Y,X], lit FREE only
    solid = field["solid"]
else:
    valid = field["valid"]        # unknown entries are not physical labels
    print(field["reason"])
```

The caller must first prove source acceptance/clearance. Small-field production
is deliberately capped at 8/16/32/64/128. It is a diagnostic API, not a streaming
1024³ producer or a replacement RINDPack format.

## Reproduce the experiments

Use the saved original B2 research inputs when available. A fresh checkout can
recreate B2-only inputs; this does not generate or alter RIND3D v0.1 scenes:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$RIND_PYTHON" -m experiments.gaussian_geometry.study \
  --models gaussian_sum --output outputs/b2-inputs

"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.profile_baseline \
  --input outputs/b2-inputs --output outputs/b2-python-profile
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.benchmark \
  --input outputs/b2-inputs --output outputs/b2-query-benchmark
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.fields \
  --input outputs/b2-inputs --output outputs/b2-fields \
  --resolutions 8 32 64 --seconds 180 --scalar-control
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.controls \
  --output outputs/b2-continuous-proofs
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.generation \
  --output outputs/b2-joint-generation
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.adversarial \
  --output outputs/b2-adversarial
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.field_ablation \
  --input outputs/b2-inputs --output outputs/b2-field-ablation.json
```

Field and matched-query drivers checkpoint cases and reject changed identities
on resume. Use a fresh output directory after modifying code/configuration.
The measured layout preserves failed and incomplete versions separately.
The 128³ attempt is conditional on successful 64³ gates, not a recommended
default sweep. Its actual 480-second reference limit did **not** complete.

Optional diagnostics:

```bash
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.proxy_experiment \
  --input outputs/b2-inputs --queries outputs/b2-query-benchmark \
  --output outputs/b2-proxy-experiment
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.profile_native \
  --input outputs/b2-inputs --queries outputs/b2-query-benchmark \
  --output outputs/b2-native-profile
```

The second command needs Valgrind/Callgrind headers and tools. It measures
instruction costs in a same-source standalone harness, not benchmark wall time.
The proxy experiment uses rigorously contained/containing continuous boxes only
for screening, with Gaussian fallback. It respects the existing native limit of
64 convex components; larger covers are reported unsupported rather than
bypassing that capability. No proxy is used as the Gaussian geometry truth.

Measured artifacts live in `outputs/gaussian-b2-native-v1`. The compact checked-in
summary is `../results/b2-native-measured-v1.json`. Run `b2_native.summarize`
with `--root` and `--output` to regenerate it from the measured directory layout.
Large arrays, native binaries, Callgrind dumps and synthetic source records stay
ignored. No private v0.1 source coordinates are used or committed.
