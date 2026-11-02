#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps_dir="${RIND_DEPS_DIR:-${repo_dir}/.build/deps}"
boost_archive="${deps_dir}/boost-1.92.0-1-x86_64.pkg.tar.zst"
boost_prefix="${deps_dir}/boost-package/usr"
boost_sha="0d795c6401c8bfa16012ada7e2e7f34934fb268f9174470edac1b389056f79bb"

if [[ -n "${RIND_CGAL_DIR:-}" ]]; then
  cgal_dir="${RIND_CGAL_DIR}"
else
  cgal_dir="$("${repo_dir}/scripts/fetch_cgal.sh")"
fi

if [[ -n "${RIND_BOOST_PREFIX:-}" ]]; then
  boost_prefix="${RIND_BOOST_PREFIX}"
elif [[ ! -f "${boost_prefix}/lib/cmake/Boost-1.92.0/BoostConfig.cmake" ]]; then
  if [[ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]]; then
    echo "Set RIND_BOOST_PREFIX to an installed Boost 1.92 prefix." >&2
    exit 1
  fi
  mkdir -p "${deps_dir}" "${deps_dir}/boost-package"
  if [[ ! -f "${boost_archive}" ]]; then
    curl -L --fail --show-error --silent \
      "https://geo.mirror.pkgbuild.com/extra/os/x86_64/boost-1.92.0-1-x86_64.pkg.tar.zst" \
      -o "${boost_archive}.download"
    mv "${boost_archive}.download" "${boost_archive}"
  fi
  actual_sha="$(sha256sum "${boost_archive}" | cut -d' ' -f1)"
  if [[ "${actual_sha}" != "${boost_sha}" ]]; then
    echo "Boost package SHA-256 mismatch" >&2
    exit 1
  fi
  bsdtar -xf "${boost_archive}" -C "${deps_dir}/boost-package"
fi

python_executable="${RIND_PYTHON:-${repo_dir}/.venv/bin/python}"
pybind11_dir="$("${python_executable}" -m pybind11 --cmakedir)"

cmake -S "${repo_dir}/native" -B "${repo_dir}/.build/native" \
  -DCGAL_DIR="${cgal_dir}" -DCMAKE_PREFIX_PATH="${boost_prefix}" \
  -Dpybind11_DIR="${pybind11_dir}" -DPython3_EXECUTABLE="${python_executable}" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${repo_dir}/.build/native" --parallel 2
ctest --test-dir "${repo_dir}/.build/native" --output-on-failure
