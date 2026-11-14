# RIND3D Generator

This is the self-contained generation and certified physics source for **RIND3D — 3D Physical Reasoning Dataset**. RIND3D v0.1 is the completed initial release: 1,000 certified scenes and 256,000 observation windows. `Lite-Core-1K` describes its internal production profile, not a separate dataset brand.

The completed data is downloaded separately. Do not regenerate it for Phase 3 development; follow [the Phase 3 guide](../Phase3_Single_Source_3D/README.md). No response fields, private source labels, credentials, native binaries or old campaign outputs are committed here.

## Architecture and scope

`src/rind3d` contains canonical geometry adapters, certified CPU/CUDA rendering, RINDPack streaming, exact informative-window selection, independent A/B physical verification, scientific acceptance and resumable campaigns. `native/` contains the production C++ predicates and compact storage implementation. `reader/` is the independent reader-only SDK. `vendor/canonical_geometry/` and `vendor/RINDRay/` contain the required project-owned source, with local build paths. Related comparison/quality source under `experiments/` is retained where acceptance tests require it; historical outputs and binaries are excluded.

Truth is continuous binary64 geometry, interpreted exactly. Closed-segment contact and tangency block visibility. The current production scope is strictly world-contained top-level unions of certified convex components across nine families. Functional/random-field geometry and unsupported CSG retain their publication restrictions. Native A parametric clipping/GMP and independent B rational projection do not share visibility decisions. UNKNOWN/ABORTED never become physical labels.

## Build on Linux x86_64 / Python 3.12

System prerequisites: C++20 compiler, CMake, GMP/GMPXX and MPFR development libraries, Zstandard development library, curl, tar, bsdtar and a working Python development environment. CGAL 6.2.1 and Boost 1.92 headers are fetched by the canonical build script with pinned SHA256 checks; overrides are `RIND_CGAL_DIR` and `RIND_BOOST_PREFIX`. See [dependency notices](THIRD_PARTY.md) before distributing compiled applications.

```bash
uv sync --extra dev --extra train
./scripts/build.sh
uv run --no-sync python scripts/verify_source_copy.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m pytest -q
```

The optional `train` extra supplies CPU PyTorch needed by campaign build-identity/loader acceptance; it is not necessary for sparse native physics queries. The preserved legacy comparison tests skip if their separate experimental extensions have not been built. Production extensions are always built by the main package. Fresh builds need fresh scientific certification; the original frozen authority is never copied or relabeled.

To include those two legacy native comparison implementations in the complete regression run, build their retained sources on the tested Linux platform:

```bash
for comparison in shadow_volumes cpu_cuda; do
  cmake -S "experiments/$comparison" -B "build/acceptance-$comparison" \
    -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
    -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)" -DCMAKE_BUILD_TYPE=Release
  cmake --build "build/acceptance-$comparison" --parallel 2
  cp "build/acceptance-$comparison"/_*.so "experiments/$comparison/"
done
RIND_TEST_CUDA=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --no-sync python -m pytest -q
```

Omit the CUDA opt-in on CPU-only hosts. All generated shared libraries/build files stay ignored; they must not be committed.

An NVIDIA GPU is optional. CUDA requires a compatible NVIDIA driver **and NVRTC**, for example from an installed CUDA Toolkit; the CPU PyTorch extra does not install that compiler. Certified CUDA uses the NVRTC/driver API, rigorous directed binary64 intervals and exact CPU fallback for unknowns. Set `RIND_NVRTC_LIBRARY` to your installed `libnvrtc.so` if NVRTC is not discoverable. `cpu_reference` and `cpu_native_certified` remain available; `auto` checks capabilities. `cuda_certified` fails explicitly when unavailable.

## Generation and recovery

These are producer commands for **new**, separately authorized runs, not prerequisites for reading v0.1:

```bash
uv run --no-sync rind3d certify-lite --output outputs/lite-v01-production/scientific-certification.json
uv run --no-sync python scripts/prepare_quality_fixtures.py
uv run --no-sync python -m experiments.lite_quality.certify
uv run --no-sync rind3d core-pilot --config configs/lite-core-research-v1.yaml --scenes 10 --verification-workers 4 --output outputs/new-smoke-10
```

Quality fixtures are newly generated from a public synthetic seed; no hidden labels from the completed dataset are copied. Certification includes expensive complete-field controls. Use `certify-lite --cpu-only` on CPU hosts. New READY authority requires every existing scientific/quality/recovery gate; a relocated build is not certified merely because it imports or answers queries.

The durable campaign CLI is `python -m rind3d.campaign_cli`: `create`, `run`, `status`, `pause`, `verify`, `extend`, `export`. Creation requires a matching complete campaign certificate. `run --resume --max-new-scenes 100 --max-runtime 3600 --verification-workers 4` uses a cumulative target with stable HMAC-derived identities, bounded workers and atomic journals. Signals stop admission and safely checkpoint/drain; incomplete work is not physical rejection. New campaign authority must be established for the new build before production. Do not reuse the v0.1 frozen certificate or change scientific code midway through a campaign.

## Provenance

[ORIGINAL_SOURCE.json](provenance/ORIGINAL_SOURCE.json) records exact copied-file hashes, original release identity and original build `b038ba8842f5ea5e059f42173dbbb840cdef4544e52f02dab3cec82f0d8e19a2`. The original checkout had no Git revision. Relocation changes build paths and compiled identities; this Git tree does **not** claim that frozen identity. The downloaded archive remains byte-identical, including historical internal manifest names. Source and dataset licenses/citation authors remain pending; no permissive license is invented.
