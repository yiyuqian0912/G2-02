#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps_dir="${RIND_DEPS_DIR:-${repo_dir}/.build/deps}"
archive="${deps_dir}/CGAL-6.2.1.tar.xz"
source_dir="${deps_dir}/CGAL-6.2.1"
expected_sha="b6be77c60765a8456335de991eeaf6ffec55256984e4a9ecc6a97c37bbfe85bf"

mkdir -p "${deps_dir}"
if [[ ! -f "${archive}" ]]; then
  curl -L --fail --show-error --silent \
    "https://github.com/CGAL/cgal/releases/download/v6.2.1/CGAL-6.2.1.tar.xz" \
    -o "${archive}.download"
  mv "${archive}.download" "${archive}"
fi

actual_sha="$(sha256sum "${archive}" | cut -d' ' -f1)"
if [[ "${actual_sha}" != "${expected_sha}" ]]; then
  echo "CGAL archive SHA-256 mismatch" >&2
  exit 1
fi

if [[ ! -f "${source_dir}/CGALConfig.cmake" ]]; then
  tar -xf "${archive}" -C "${deps_dir}"
fi

printf '%s\n' "${source_dir}"

