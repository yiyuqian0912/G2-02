# B2 independent verifier study v1

This bounded research experiment leaves the original independent renderer
available as method A. Methods B, C and D still independently search the
closed ray. Only the original independent residual ray call is selectable.
The primary and region cover, membership and visibility certificate code
come from the frozen baseline. The build saves the generated source patch
and exact source and binary fingerprints. Large arrays and trial records
belong under ignored `outputs/`.

See the [derivation](../../../docs/GAUSSIAN_B2_INDEPENDENT.md) and
[measured report](../../../docs/GAUSSIAN_B2_INDEPENDENT_RESULTS.md).

## Reproduce

Run from `RIND3D_Generator` in a Python environment with the installed
generator, existing Gaussian research dependencies, pybind11, GMP and MPFR.
The original `b2_native` and `b2_regions` builds remain required for the
baseline and source acceptance. Recover missing frozen inputs using the
existing `b2_angular.materialize` recipe. That recreates four synthetic
Gaussian inputs from recorded seeds. To materialize the synthetic inputs
in a fresh folder, run the command below and use that folder as
`--input` in the frozen-input commands.

```bash
python -m experiments.gaussian_geometry.b2_angular.materialize \
  --output outputs/gaussian-b2-independent-reproduction/frozen
```

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=outputs/gaussian-b2-independent-reproduction
study=experiments.gaussian_geometry.b2_independent
frozen=outputs/gaussian-geometry-v1/distribution-v2
fixtures=experiments/gaussian_geometry/fixtures/b2-max-cert-v1

python -m "$study.freeze" --output "$out/heldout"
python -m "$study.build" --output "$out/native"
python -m "$study.build" --profile --output "$out/profile"
RIND_B2_INDEPENDENT_BUILD="$out/native" python -m pytest -q \
  experiments/gaussian_geometry/tests/test_b2_independent.py
python -m "$study.adversarial" --native "$out/native" \
  --output "$out/adversarial.json"

for entry in residual:difficult followup:unselected ordinary:ordinary; do
  python -m "$study.benchmark" rays --native "$out/native" \
    --input "$frozen" --fixture "$fixtures/${entry%:*}/frozen-rays.json" \
    --methods A B C D B128 D128 C_Newton D_Newton --arb-per-case 2 \
    --output "$out/rays-${entry#*:}"
done

python -m "$study.benchmark" fields --input "$frozen" \
  --methods A P --output "$out/baseline"
taskset -c 2 python -m "$study.benchmark" fields --native "$out/profile" \
  --input "$frozen" --resolutions 64 --methods A_profile \
  --baseline "$out/baseline" --output "$out/profile-fields"
python -m "$study.benchmark" fields --native "$out/native" \
  --input "$frozen" --methods B C D --baseline "$out/baseline" \
  --output "$out/frozen-fields"
python -m "$study.benchmark" fields --native "$out/native" \
  --input "$frozen" --resolutions 16 --methods A D Arb \
  --output "$out/arb-fields"
python -m "$study.benchmark" fields --native "$out/native" \
  --input "$out/heldout" --ordinals 60 62 63 64 --methods A P D \
  --output "$out/heldout-fields"
python -m "$study.summarize" --root "$out" --output "$out/measured.json"
```

Choose an empty output directory for a new build. A run can resume with the
same input, build and configuration identities. Failed or partial fields
retain their valid mask and reason. A partial field stops the run.
Comparison baselines require every voxel to be valid. Comparisons record the
matched voxel count, response and solid mismatches, and trace support agreement when both
paths use the same region traversal. Budget and precision settings are in
`configs/b2-independent-v1.json` and the corresponding run identities.

The native field clock covers traversal and physical evaluation. Cold compute
also includes setup and array materialization. Cold and fresh-process totals
capture source clearance, writing, imports, integrity checks and process startup.
Ray trials use three balanced method orders on identical binary64 coordinates.
Use a single allowed logical CPU for the diagnostic profile. The measured
run used CPU 2 on this host. Pinning keeps cycle-counter intervals within
one clock domain. Use the uninstrumented A for speed comparisons.

After the 32³/64³ frozen, third-oracle and held-out checks pass, the bounded
128³ follow-up uses case 14 before case 7. Check each completed support
before proceeding. The budget below applies separately to each renderer.

```bash
python -m "$study.benchmark" fields --native "$out/native" \
  --input "$frozen" --ordinals 14 --resolutions 128 --methods A P D \
  --seconds 480 --output "$out/fields128-case14"
python -m "$study.summarize" --root "$out" --output "$out/measured.json"
python -m "$study.benchmark" fields --native "$out/native" \
  --input "$frozen" --ordinals 7 --resolutions 128 --methods A P D \
  --seconds 480 --output "$out/fields128-case7"
python -m "$study.summarize" --root "$out" --output "$out/measured.json"
```
