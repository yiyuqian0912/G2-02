#!/usr/bin/env bash
# RIND3D-100K: build the single production environment from source on the target server.
# Run from anywhere; creates <repo>/.venv and <repo>/r3d_env.sh (source it before every command).
#
#   bash RIND3D_Dataset100K/handoff/env/setup_env.sh            # online
#   R3D_DEPS=/path/to/handoff/deps bash .../setup_env.sh         # offline CGAL/Boost tarballs (see README_H20.md)
#   R3D_WHEELS=/path/to/wheelhouse bash .../setup_env.sh         # offline Python wheels (pip download -r requirements-lock.txt)
#
# System prerequisites (Debian/Ubuntu names): build-essential (g++ >= 11), cmake >= 3.24, libgmp-dev, libmpfr-dev,
# libzstd-dev, libboost-dev (>= 1.74; headers only), curl, xz-utils, libarchive-tools (bsdtar), git, nvidia driver.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
PY_VER="${R3D_PYTHON:-3.12}"
cd "${REPO}"
echo "[setup] repo ${REPO}"

for t in g++ cmake git; do command -v "$t" >/dev/null || { echo "missing $t"; exit 1; }; done
if ! command -v uv >/dev/null; then
  if [[ -n "${R3D_UV:-}" ]]; then export PATH="$(dirname "${R3D_UV}"):${PATH}";
  else curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; fi
fi
uv --version

uv venv "${REPO}/.venv" --python "${PY_VER}" --allow-existing
PYBIN="${REPO}/.venv/bin/python"
PIPX=(uv pip install --python "${PYBIN}")
if [[ -n "${R3D_WHEELS:-}" ]]; then PIPX+=(--no-index --find-links "${R3D_WHEELS}"); fi

# 1. Python dependencies (pinned), without the CUDA NVRTC wheel (pinned to the driver below)
grep -v -E '^nvidia-cuda-nvrtc' "${HERE}/requirements-lock.txt" > /tmp/r3d_req_$$.txt
"${PIPX[@]}" -r /tmp/r3d_req_$$.txt
rm -f /tmp/r3d_req_$$.txt

# 2. NVRTC matching the driver: the certified renderer JIT-compiles PTX through the driver, which rejects PTX newer
#    than its own CUDA version. cupy itself loads cubins and works with any 12.x driver.
# driver CUDA version: from the driver API (cuDriverGetVersion via cupy), else parsed from nvidia-smi
# ("CUDA Version: X.Y" on older drivers, "CUDA UMD Version: X.Y" on newer ones), else R3D_DRIVER_CUDA
DRV_CUDA="${R3D_DRIVER_CUDA:-}"
if [[ -z "${DRV_CUDA}" ]]; then
  DRV_CUDA="$("${PYBIN}" -c 'import cupy; v = cupy.cuda.runtime.driverGetVersion(); print(f"{v // 1000}.{(v % 1000) // 10}")' 2>/dev/null || true)"
fi
if [[ -z "${DRV_CUDA}" ]]; then
  DRV_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA[A-Z ]*Version: \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)"
fi
if [[ -z "${DRV_CUDA}" ]]; then echo "[setup] cannot determine the driver CUDA version (set R3D_DRIVER_CUDA=12.4)"; exit 1; fi
DRV_MAJ="${DRV_CUDA%%.*}"; DRV_MIN="${DRV_CUDA##*.}"
if [[ "${DRV_MAJ}" -ge 13 ]]; then NVRTC_SPEC="nvidia-cuda-nvrtc-cu12==12.9.*"; else NVRTC_SPEC="nvidia-cuda-nvrtc-cu12==12.${DRV_MIN}.*"; fi
echo "[setup] driver CUDA ${DRV_CUDA}; installing ${NVRTC_SPEC}"
"${PIPX[@]}" "${NVRTC_SPEC}"

# 3. canonical-geometry natives (CGAL 6.2.1, Boost headers); offline tarballs if R3D_DEPS is given
"${PIPX[@]}" "pybind11>=3,<4"
CG="${REPO}/RIND3D_Generator/vendor/canonical_geometry"
if [[ -n "${R3D_DEPS:-}" ]]; then
  mkdir -p "${CG}/.build/deps"; cp -n "${R3D_DEPS}"/CGAL-6.2.1.tar.xz "${CG}/.build/deps/" 2>/dev/null || true
  cp -n "${R3D_DEPS}"/boost-1.92.0-1-x86_64.pkg.tar.zst "${CG}/.build/deps/" 2>/dev/null || true
fi
if [[ -z "${RIND_BOOST_PREFIX:-}" && -f /usr/include/boost/version.hpp ]]; then export RIND_BOOST_PREFIX=/usr; fi
RIND_PYTHON="${PYBIN}" bash "${CG}/scripts/build_native.sh"

# 4. rind3d (scikit-build-core: pybind11 modules with GMP / zstd)
"${PIPX[@]}" -e "${REPO}/RIND3D_Generator"

# 5. environment file
NVRTC_LIB="$(ls "${REPO}"/.venv/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.* | grep -v alt | head -1)"
cat > "${REPO}/r3d_env.sh" <<EOF
# source this file before running RIND3D-100K tools
export R3D_REPO="${REPO}"
export PATH="${REPO}/.venv/bin:\${PATH}"
export RIND_NVRTC_LIBRARY="${NVRTC_LIB}"
export PYTHONPATH="${REPO}/RIND3D_Dataset100K:${REPO}/RIND3D_Dataset1K:${REPO}/RIND3D_Posterior/src:${REPO}/RIND3D_Generator/curved\${PYTHONPATH:+:\${PYTHONPATH}}"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
EOF
source "${REPO}/r3d_env.sh"
python -c 'import rind._exact2d, rind._exact3d, rind3d._native, rind3d._convex, rind3d._research, numba, cupy; print("[setup] imports ok", numba.__version__, cupy.__version__)'
echo "[setup] done. Next: source ${REPO}/r3d_env.sh && python RIND3D_Dataset100K/scripts/smoke_test.py --work /scratch/r3d_smoke"
