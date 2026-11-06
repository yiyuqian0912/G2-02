# Gaussian geometry experiments for RIND3D

This is a geometry-generation study, not a teacher or public dataset format.
It provides clustered canonical convex objects, independent Gaussian ellipsoid
levels, positive Gaussian-sum solids, and a finite continuous tapered Gaussian
random-field baseline. All new families remain `EXPERIMENTAL_ONLY`.

Read [mathematics, scope and certification limits](../../docs/GAUSSIAN_GEOMETRY.md)
before interpreting the [measured results](../../docs/GAUSSIAN_GEOMETRY_RESULTS.md).

## Environment

Use a **development** Python 3.12 environment with this repository's normal
canonical/native build, as documented in the generator README. Never install
research dependencies into the original frozen production environment.
`RIND_PYTHON` below may point to the already-built Phase 3 development environment.

```bash
# Run from RIND3D_Generator.
export RIND_PYTHON="${RIND_PYTHON:-$PWD/.venv/bin/python}"
uv pip install --python "$RIND_PYTHON" -r experiments/gaussian_geometry/requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$RIND_PYTHON" -m pytest -q experiments/gaussian_geometry/tests
```

For the measured run, additions were installed with `uv pip --target` into
`outputs/gaussian-geometry-v1/deps` and that directory added to `PYTHONPATH`.
The existing Phase 3 and frozen producer environments were not modified.
CUDA is not used to label implicit Gaussian fields. Their CPU-only predicates
use exact rationals/Arb and an independent MPFR path. Existing convex CUDA
regressions remain available through `RIND_TEST_CUDA=1` on a capable host.

## Generate, measure and resume

```bash
"$RIND_PYTHON" -m experiments.gaussian_geometry.study \
  --config experiments/gaussian_geometry/configs/study-v1.json \
  --output outputs/my-gaussian-study

# A shorter pilot; saves all failures and does not produce READY scenes.
"$RIND_PYTHON" -m experiments.gaussian_geometry.study \
  --case-ordinals 0 14 --output outputs/my-gaussian-pilot

# Reuse the saved geometry and synthetic sources for small-field gates.
"$RIND_PYTHON" -m experiments.gaussian_geometry.scripts.fields \
  --input outputs/my-gaussian-study --output outputs/my-gaussian-fields --seconds 45

# Existing-family controls, generated from public synthetic seeds.
"$RIND_PYTHON" -m experiments.gaussian_geometry.controls \
  --output outputs/my-convex-controls

# Isolate the world-fitting strategy while keeping center/shape proposals.
"$RIND_PYTHON" -m experiments.gaussian_geometry.scripts.placement_ablation \
  --input outputs/my-gaussian-study --output outputs/my-capped-convex

# A continuous connectivity proof, not merely sampled midpoint agreement.
"$RIND_PYTHON" -m experiments.gaussian_geometry.scripts.bridge_proof \
  --output outputs/my-bridge-proof

# Exact-witness/nonlocal diagnostics, with independent caps for each size.
"$RIND_PYTHON" -m experiments.gaussian_geometry.scripts.nonlocal_audit \
  --input outputs/my-gaussian-study --capped-input outputs/my-capped-convex \
  --output outputs/my-nonlocal-audit

# Reproduce unresolved rays; does not rewrite their incomplete fields.
"$RIND_PYTHON" -m experiments.gaussian_geometry.scripts.hard_cases \
  --input outputs/my-gaussian-study --fields outputs/my-gaussian-fields \
  --output outputs/my-hard-cases.json

"$RIND_PYTHON" -m experiments.gaussian_geometry.scripts.acceleration_ablation \
  --input outputs/my-gaussian-study --output outputs/my-acceleration.json

# Separate covariance/scale/threshold/kernel-count effects on one fixed case.
"$RIND_PYTHON" -m experiments.gaussian_geometry.scripts.parameter_ablation \
  --input outputs/my-gaussian-study --output outputs/my-parameter-ablation.json

# Two explicitly larger-budget controls; no automatic escalation to 1024³.
"$RIND_PYTHON" -m experiments.gaussian_geometry.scripts.extended_gate \
  --input outputs/my-gaussian-study --output outputs/my-extended-gates

"$RIND_PYTHON" -m experiments.gaussian_geometry.plots \
  --input outputs/my-gaussian-study --output outputs/my-gaussian-figures
```

`scripts.comparison_plots` and `scripts.summarize` reproduce the checked-in
figures/aggregate JSON from the **measured run layout**:

```bash
"$RIND_PYTHON" -m experiments.gaussian_geometry.scripts.comparison_plots \
  --root outputs/gaussian-geometry-v1 --output docs/assets/gaussian-geometry
"$RIND_PYTHON" -m experiments.gaussian_geometry.scripts.summarize \
  --root outputs/gaussian-geometry-v1 \
  --output experiments/gaussian_geometry/results/measured-v1.json
```

The aggregation expects `distribution-v1`, `distribution-v2`, `capped-convex-v1`,
`fields-v3`, `convex-controls`, `nonlocal-v2`, `bridge-control`,
`hard-cases.json`, `acceleration-ablation.json`, `extended-gates` and
`parameter-ablation.json`. These are local artifact
directory names, not public dataset versions. Run names preserve historical
failed adapters and corrected audit versions rather than replacing them.

Rerun the same command to skip committed case results. Each case has its own
atomic JSON checkpoint. Config, seeds and source hashes bind a run; changing
code/config requires a new output directory. Source snapshots preserve old
experiments. Failed, incomplete and unresolved cases stay in the record rather
than being silently replaced. A stopped process can rerun its unfinished case;
already committed cases are not recomputed. These are research checkpoints,
not the production campaign journal or a claim of READY crash certification.

The field gate starts with an additional 8³ correctness control, then attempts
32³, 64³ and 128³ only after completed independent agreement at the preceding
resolution. Native fields are assembled from at-most-64³ bricks. Incomplete
fields preserve a separate validity mask. No 1024³ or mass-production entry
point is enabled here. A time/node limit is not proof of physical rejection.

## Use the geometry/query API

```python
from experiments.gaussian_geometry.generators import generate
from experiments.gaussian_geometry.models import save, load
from experiments.gaussian_geometry.queries import make_oracle

scene = generate("gaussian_sum", seed=20261105, ordinal=14, target=0.05)
save("outputs/example-parameters.json", scene)
scene = load("outputs/example-parameters.json")
oracle = make_oracle(scene)
print(oracle.containment(margin=0.025))
print(oracle.membership([0.5, 0.5, 0.5]))
print(oracle.segment([0.03125]*3, [0.8, 0.7, 0.6]))
```

Membership returns `FREE`, `SOLID` or `UNKNOWN`. Segment results are `HIT`,
`MISS`, `UNKNOWN` or `BUDGET_EXHAUSTED`. Only resolved values are physical
labels. `source_clearance()` is a sufficient strict GT proof, not a complete
distance oracle. Rebuild an oracle after changing any scene parameters; its
support/proxy structures assume a fixed geometry.

All binary64 inputs are serialized as hexadecimal strings; no display mesh is
stored as geometry. The experimental `geometry_id` is a checksum of the whole
research parameter/recipe record, not the original canonical production ID.
A exports through `ConvexScene.bundle()`. B1 can additionally export losslessly
through `scripts.canonical_ellipsoids.to_canonical()` using existing
`affine_ball_v1` codecs. B2/C cannot be exported as a convex/mesh replacement.
New implicit codecs and complete production acceptance are still required.

`approx_values()` is for density proposals, estimates and display only. Arrays
are ZYX, coordinates XYZ in the unit world, and receiver centers are
`(integer_index+.5)/resolution`. Window diagnostics instead use the original
1024 sampling lattice and its 64/128/256 sizes. Model/seed/source recipes in
these outputs are newly synthesized research data, not private v0.1 labels.

Outputs, dependency wheels, native binaries and full field arrays stay outside
Git under `outputs/`. Only code, configuration, small explanatory figures and
reports belong in the research branch. No existing scientific file, profile,
reader, generation campaign or Google Drive artifact is overwritten.
