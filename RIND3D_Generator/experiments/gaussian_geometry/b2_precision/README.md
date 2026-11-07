# B2 interval-local precision study v1

Run from `RIND3D_Generator` with the existing Gaussian research dependencies,
native A/D builds, and frozen inputs. The configuration fingerprints every
generated source file in the previous independent-verifier build. Recreate
that build with its [documented command](../b2_independent/README.md) if
needed, and pass the resulting directory as `--baseline`.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
study=experiments.gaussian_geometry.b2_precision
out=outputs/gaussian-b2-precision-v1
old=outputs/gaussian-b2-independent-v1
holdout="$old/heldout"

python - <<'PY'
from experiments.gaussian_geometry.b2_independent.freeze import freeze
from experiments.gaussian_geometry.b2_precision.build import CONFIG, OUT
freeze(OUT / "fresh-inputs", CONFIG)
PY
python -m "$study.build"
taskset -c 2 python -m "$study.audit" --output "$out/audit-timing.json"
taskset -c 2 python -m "$study.audit" --probe --output "$out/audit-probe64.json"
python -m "$study.local" --output "$out/native-final"
python -m "$study.bench" rays --output "$out/ray-stage"

python -m "$study.bench" fields --ordinals 0 7 14 21 --resolutions 32 \
  --methods E F --output "$out/frozen32"
python -m "$study.bench" fields --input "$holdout" --ordinals 60 62 63 64 \
  --resolutions 32 --methods E F --output "$out/holdout32"
python -m "$study.bench" fields --ordinals 0 7 14 21 --resolutions 64 \
  --methods D E F --output "$out/frozen64"
python -m "$study.bench" fields --input "$holdout" --ordinals 60 62 63 64 \
  --resolutions 64 --methods D E F --output "$out/holdout64"
for n in 32 64; do
  python -m "$study.bench" fields --input "$out/fresh-inputs" --ordinals 72 75 \
    --resolutions "$n" --methods A P D E F --output "$out/fresh$n"
done
```

Pin the audit to one allowed logical CPU. This machine used CPU 2. The
separate counterfactual probe run supplies decision counts. Use the normal
audit for timing. A/D records on identical historical supports are available
through the preserved baseline directory. Fresh D runs on the 64³ workloads
control for run-to-run timing differences. Every current field retains its
valid mask and is checked against the same geometry and source. The measured
report uses the uncontended scene-21 repeat below because compilation
overlapped its first timing.

```bash
python -m "$study.bench" fields --ordinals 21 --resolutions 64 --methods D E F \
  --output "$out/repeats/frozen64-case21"
```

After the complete 64³ gates, run the bounded 128³ checks. Finish case 14
before case 7. The matched field summary rejects incomplete fields and
disagreeing physical bits.

```bash
python -m "$study.bench" fields --ordinals 14 --resolutions 128 \
  --methods D E F --seconds 480 --output "$out/fields128-case14"
python -m "$study.bench" fields --ordinals 7 --resolutions 128 \
  --methods E F --seconds 480 --output "$out/fields128-case7"
python -m "$study.summarize" --output "$out/measured.json"
```

After the numerical experiment supports the reuse study, build the separate
source-bound session extension and compare its tile policies. The complete
256³ attempt starts with sixteen predeclared 64³ tiles in eight octants.
The driver uses their timing and peak RSS for admission. The budget gate and
exact coverage are saved even when the run remains partial. Use a fresh
output directory for a new scale attempt.

```bash
python -m "$study.tile_build"
python -m pytest -q experiments/gaussian_geometry/tests/test_b2_precision.py
python -m "$study.tiles" compare --output "$out/tile-reuse"
python -m "$study.tiles" scale256 --output "$out/scaling256"
python -m "$study.summarize" --output "$out/measured.json"
```

The [numerical design and initial audit](../../../docs/GAUSSIAN_B2_PRECISION.md)
explain the proof invariant. The [report](../../../docs/GAUSSIAN_B2_PRECISION_RESULTS.md)
records the actual gate decisions, resource bounds and matched timings.
