# B2 whole-receiver-region experiment

This is continuous positive Gaussian-sum **obstacle** geometry. Binary point-light
visibility, closed contact semantics and the public RIND3D dataset are unchanged.
It has no READY authority. Read the [proofs and protocol](../../../docs/GAUSSIAN_B2_REGIONS.md)
and the measured results linked there. The frozen native-v1 implementation is
compiled unchanged into a separate module; its old source/build/artifacts remain.

Tested environment: Linux x86-64, Python 3.12, C++20, GMP/GMPXX and MPFR.
The same nearest-rounding, gradual-underflow, no-fast-math and no-contraction
requirements apply as in native v1. No GPU is needed. This is a research
extension, not a new requirement of the public RIND3D reader.

## Build and tests

From `RIND3D_Generator`, after the normal development environment and the
Gaussian research requirements are installed:

```bash
export RIND_PYTHON="${RIND_PYTHON:-$PWD/.venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.build
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_regions.build
"$RIND_PYTHON" -m pytest -q experiments/gaussian_geometry/tests/test_b2_native.py \
  experiments/gaussian_geometry/tests/test_b2_regions.py
```

The new build is ignored under `outputs/gaussian-b2-regions-v1/build`. Override
it with build `--output` and absolute `RIND_B2_REGIONS_BUILD` if necessary.
Source and binary hashes are checked; restart Python after a rebuild. Never
replace a library while a benchmark using it is running. The old native build
uses its own `RIND_B2_NATIVE_BUILD`, independently.

## API and meanings

```python
from experiments.gaussian_geometry.generators import generate
from experiments.gaussian_geometry.b2_native.api import NativeOracle
from experiments.gaussian_geometry.b2_regions.api import RegionOracle

scene = generate("gaussian_sum", 20261105, 14, 0.05)
source = [0.03125] * 3  # Synthetic example, never a private v0.1 source.
assert NativeOracle(scene).source_clearance(source).status == "FREE"
primary = RegionOracle(scene, mode=5)
reference = RegionOracle(scene, mode=5, independent=True)
field = reference.field(source, 32, seconds=180)
assert field["complete"], field["reason"]
response = field["response"]  # Z,Y,X; lit FREE voxels only
solid = field["solid"]
```

The caller retains the original source-domain, FREE and clearance requirements.
This region/tile implementation supports the closed unit world; it rejects
unsupported receiver/source domains instead of clipping canonical geometry.
An engine owns a parameter snapshot. Reconstruct it after changing the scene;
do not mutate its options/settings while a native call is running.
Finite continuous physics diagnostics do not authorize invalid dataset sources.
Arrays have a separate validity mask; `complete=False` never supplies labels
for invalid/unvisited entries. Unknown proof boxes are retained conservatively.
An exact equality that cannot be proved remains UNKNOWN in scalar fallbacks.
The Lite equivalent-reference separation requirement does not enter this code.

`region(source, lo_xyz, hi_xyz)` returns `((visibility, membership), counters)`:
visibility 1 means ALL-LIT, 0 ALL-SHADOW; membership 0 means all FREE, 1 all
SOLID. Negative values mean UNKNOWN (-1) or INVALID (-2). These are different
predicates. The single-region helper exercises the parameter/cross-section
methods; the reusable proof tree is constructed inside `field`/`tile` calls.
It is not a persistent cross-backend certificate service.

Modes are 0 scalar, 1 original hierarchy, 2 extra whole-ray-volume bounds,
3 extra fused cross-sections, 4 both, and 5 a continuous proof tree plus
independent geometric screening and original fallback. Settings explicitly
bound cover depth/nodes/time, inner-box count and parameter subdivisions.
`volume_after_cache=True` is a separate optional combination. The primary's
`fast_exp=True` enables a rigorously bounded polynomial filter for new parameter
bounds; independent Gaussian predicates retain MPFR. It defaults off until
its complete-cost ablation is assessed.

```python
# Arbitrary integer origin, bounded output; no full 1024³ allocation.
tile = primary.tile(source, 1024, [257, 419, 73], 32, seconds=180)
print(tile["origin_xyz"], tile["world_resolution"], tile["complete"])
```

This diagnostic tile primitive includes cold tree setup per call. It is not
RINDPack streaming/resume and makes no 1024³ complete-field claim. Output side
is bounded to 8/16/32/64/128, with world sampling powers of two up to 1024.
Canonical geometry remains continuous and independent of those voxel centers.

## Reproduce measured runs

Use the frozen previous-study B2 inputs and fields if available. On a fresh
checkout, recreate B2-only experimental inputs and small baseline fields with
the native-v1 guide; do not regenerate RIND3D v0.1.

```bash
"$RIND_PYTHON" -m experiments.gaussian_geometry.study \
  --models gaussian_sum --output outputs/b2-inputs
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_native.fields \
  --input outputs/b2-inputs --output outputs/b2-baseline \
  --resolutions 8 32 64 --seconds 180

"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_regions.benchmark \
  --input outputs/b2-inputs --baseline outputs/b2-baseline \
  --output outputs/b2-region-ablation --ordinals 0 7 14 21 \
  --resolutions 8 32 --seconds 120
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_regions.benchmark \
  --input outputs/b2-inputs --baseline outputs/b2-baseline \
  --output outputs/b2-region-gates64 --ordinals 0 7 14 21 \
  --modes B_old_regions F_proof_reuse --resolutions 8 32 64 --seconds 180
```

Only after complete agreeing 64³ gates should a selected 128³ attempt run:

```bash
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_regions.benchmark \
  --input outputs/b2-inputs --baseline outputs/b2-baseline \
  --output outputs/b2-region-case14-128 --ordinals 14 \
  --modes F_proof_reuse --resolutions 8 32 64 128 --seconds 480
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_regions.exp_benchmark \
  --output outputs/b2-fast-exp
```

`exp_benchmark` is a matched elementary-function numerical workload, not a
physical-ray benchmark. Its equal-status fallback comparison is distinct from
enclosure-only speed and width. A physical filter ablation uses the region
benchmark with `--fast-exp` on the same modes/inputs.

`--trace` optionally records actual scalar-ray and membership wall times and
evaluation paths. It is diagnostic metadata, not model input or a physical
cost label. The `field_diagnostics` module groups this work by explicitly
sampled boundary neighborhoods; those groups are not exact distance or topology
certificates. The `controls` module reuses original native-v1 adversarial
fixtures and performs a complete selected old-Arb cross-check. Its `--input`,
`--previous` and `--field32` select the saved B2 inputs, native-v1 output root
and complete new reference 32³ array.

For the measured artifact layout, regenerate the compact summary and plots:

```bash
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_regions.summarize \
  --root outputs/gaussian-b2-regions-v1 \
  --output experiments/gaussian_geometry/results/b2-regions-measured-v1.json
"$RIND_PYTHON" -m experiments.gaussian_geometry.b2_regions.figures \
  --root outputs/gaussian-b2-regions-v1 \
  --summary experiments/gaussian_geometry/results/b2-regions-measured-v1.json \
  --output docs/assets/gaussian-b2-regions
```

Checkpoint identities include configuration, numerical sources, driver, API,
input manifests and workload hashes. Each completed field is saved once;
changed identities require a new named output. A budget-limited field remains
incomplete and does not advance that sweep. Full comparisons separately check
response and solid against saved baseline arrays and independently produced
primary/reference arrays. Large fields, binaries and synthetic source records
stay ignored. Compact measurements and plots are committed for review.
