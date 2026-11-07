# Continuous B2 maximum proofs

This CPU research prototype proves statements about the **continuous obstacle
field**, `F(y)=sum w_i exp(-||M_i^-1(y-c_i)||²/2)`. Gaussian values define solid
geometry; they are not light intensity or opacity. A closed source–receiver
segment is blocked when its maximum F reaches the canonical threshold.

Read the [derivation, audit and certificate format](../../../docs/GAUSSIAN_B2_MAX_CERT.md)
and [measured results and go/no-go decision](../../../docs/GAUSSIAN_B2_MAX_CERT_RESULTS.md).
The original optimized native Taylor engine and independently programmed
renderer remain unchanged. `ready_capable` is always false. This experiment
does not generate RIND3D scenes, alter the public dataset, or change its reader.

## Build

Run commands from `RIND3D_Generator` in the existing [B2 development environment](../b2_native/README.md).
The experiment uses NumPy, SciPy, python-flint, gmpy2, pybind11, pytest and the
existing GMP/GMPXX/MPFR development libraries, CMake and a C++20 compiler.
Figures additionally use Matplotlib. Actual measurements are Linux x86-64,
Python 3.12.14 and GCC 16.2.1. No other platform is claimed from this test.
The new methods require no CUDA; existing canonical regression has its own
CUDA opt-in.

```bash
python -m experiments.gaussian_geometry.b2_native.build
python -m experiments.gaussian_geometry.b2_regions.build
python -m experiments.gaussian_geometry.b2_max_cert.build
python -m experiments.gaussian_geometry.b2_max_cert.field_bridge
python -m pytest -q experiments/gaussian_geometry/tests/test_b2_max_cert.py
```

Builds disable fast-math and floating-point contraction. Environment guards
retain nearest rounding and gradual underflow. Imports verify exact source and
binary identities. Use `--output` and `RIND_B2_MAX_CERT_BUILD` for a separate
versioned build; never replace a library in a running Python process.

## Reproduce physical queries without old timing archives

The three [frozen query corpora](../fixtures/b2-max-cert-v1/) are small and
tracked. They contain only seeded **synthetic research sources**, not hidden
source labels from RIND3D v0.1. Coordinates round-trip losslessly as binary64;
certificates additionally use canonical hexadecimal strings.

Recover the four original parameter records using the already checked frozen
generator recipe. This writes research inputs, not a RIND3D dataset:

```bash
python -m experiments.gaussian_geometry.b2_angular.materialize \
  --output outputs/gaussian-b2-max-cert-v1/recovered-inputs
python -m experiments.gaussian_geometry.b2_max_cert.benchmark \
  --input outputs/gaussian-b2-max-cert-v1/recovered-inputs \
  --root experiments/gaussian_geometry/fixtures/b2-max-cert-v1/residual \
  --split development --repeats 3 \
  --output outputs/gaussian-b2-max-cert-v1/my-development
python -m experiments.gaussian_geometry.b2_max_cert.benchmark \
  --input outputs/gaussian-b2-max-cert-v1/recovered-inputs \
  --root experiments/gaussian_geometry/fixtures/b2-max-cert-v1/residual \
  --split evaluation --repeats 3 \
  --output outputs/gaussian-b2-max-cert-v1/my-evaluation
```

Replace `residual` with `ordinary` or `followup` and use `--split all` for the
two 512-query controls. The former samples FREE receivers uniformly. The latter
samples existing independent-renderer residual positions without cost
preference and excludes both previous corpora. All reuse four known geometries;
this is not unseen-geometry generalization.

Methods are A_primary/A_independent (unchanged optimized baselines),
B_derivative, C_stationary, D_hybrid/D_proof, and the post-hoc
E_lean/E_lean_proof endpoint-proposal ablation. D/E have identical proof
requirements. E disables two initial HIT probes, **not closed endpoint coverage**.
Budgets are explicit in [the study configuration](../configs/b2-max-cert-v1.json).
Checker precision is 256 in the matched study; `--precision 128` is a separate
ablation, not a weaker production policy.

Each ray/method/repetition commits a checkpoint. Repeating an identical
command resumes completed cases. Changed source/config/fixtures require a new
output directory. Failed checks and changed proof hashes are not resumed as
successful cases. Proof files are namespaced by method. Older measured runs
shared a query/repetition blob between D/E; their complete contents were
identical and every stored method hash was checked. Source transitions below
preserve exact reproduction of that historical driver.

## Call the prototype

```python
import json
from pathlib import Path
from experiments.gaussian_geometry.models import load
from experiments.gaussian_geometry.b2_max_cert.api import (
    MaximumOracle, CertificateChecker, dumps,
)

folder = Path("outputs/gaussian-b2-max-cert-v1/recovered-inputs/gaussian_sum-014")
scene = load(folder / "geometry.json")
source = json.loads((folder / "result.json").read_text())["source_xyz_synthetic"]
receiver = [0.8, 0.8, 0.8]
solver = MaximumOracle(scene, method=3, endpoint_probes=False)
answer = solver.segment(source, receiver, proof=True)
print(answer["status"], answer["reason"], answer["work"])
if answer["certificate"] is not None:
    checked = CertificateChecker(scene).check(
        dumps(answer["certificate"]), source, receiver, precision=256,
    )
    assert checked["accepted"]
```

This is a pure closed-segment predicate on finite binary64 inputs. Actual scene
experiments first prove the existing source-clearance condition separately.
`HIT`, `MISS`, `UNKNOWN` and `INVALID` remain distinct. A failed proof check is
not a HIT label. Zero unresolved sparse samples are not a full maximum proof.
Only a complete, strictly bounded interval cover establishes MISS.

`fallback=True` optionally retains the old primary oracle on unresolved work;
its extra work is counted. A fallback MISS may have no new partition. Known
analytic contacts can use this old exact reduction, while general unresolved
transcendental equality remains UNKNOWN. The pure optimizer's numerical
completeness is not assumed equal to the old oracle on every adversary.

## Complete fields and independent checks

```bash
python -m experiments.gaussian_geometry.b2_max_cert.fields \
  --input outputs/gaussian-b2-max-cert-v1/recovered-inputs \
  --output outputs/gaussian-b2-max-cert-v1/my-fields \
  --ordinals 14 7 --resolutions 32 64 --seconds 480
python -m experiments.gaussian_geometry.b2_max_cert.adversarial \
  --output outputs/gaussian-b2-max-cert-v1/my-adversarial
```

The generated field hook changes only the final primary scalar ray call. It
retains the old region traversal, geometry/membership predicates and independent
renderer. The default method order renders the original independent field
before experimental variants; an existing complete reference can be supplied
with `--previous`. Inputs and source/geometry identities are checked. No
128³ expansion was run in this iteration because full-path gains did not
justify it. No complete 1024³ field or production campaign is launched.

`D_checked`/`E_checked` add native MISS-proof checking with one temporary proof
at a time. They do not serialize per-voxel JSON, verify HITs by a MISS proof, or
replace the original independent field. The ray benchmark separately includes
serialization, parsing and optional file persistence. The checker is separately
implemented, but is **not the current independent-rendering acceptance contract**.

Timing-only baseline copies are generated by `profile_baseline`; `characterize`
and `derivative_diagnostic` use the preserved old trace archives. They are not
needed to rerun the tracked query corpora. Actual raw traces, all certificates,
binaries and logs stay in ignored `outputs/gaussian-b2-max-cert-v1`.

The compact source transitions [v2](../results/b2-max-final-to-measured-v2.patch)
and [v3](../results/b2-max-final-to-measured-v3.patch) reconstruct measured
native/field/driver sources from the final tree. Use a separate checkout/copy
containing the sibling baselines. From that copy's G2-02 root, for example:

```bash
git apply --directory=RIND3D_Generator/experiments/gaussian_geometry/b2_max_cert \
  RIND3D_Generator/experiments/gaussian_geometry/results/b2-max-final-to-measured-v2.patch
```

Then build from its generator directory. Both reconstructions were checked
against archived source hashes.
Final numerical predicates match v3; later driver guards and method-specific
proof filenames do not change physical decisions.

The full acceptance suite is explicit because default pytest collects `tests`
only:

```bash
RIND_TEST_CUDA=1 python -m pytest -q tests
python -m pytest -q experiments/gaussian_geometry/tests
```

On a CPU-only machine omit the CUDA opt-in. Existing private/real-production
test opt-ins remain unchanged. This isolated research adds no new READY authority.
