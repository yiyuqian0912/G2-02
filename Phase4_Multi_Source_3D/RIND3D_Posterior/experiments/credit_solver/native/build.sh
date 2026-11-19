#!/usr/bin/env bash
# Build the exact cell-credit helper (GMP + pybind11). Usage: native/build.sh [python]
set -euo pipefail
PY=${1:-python3}
cd "$(dirname "$0")"
SUF=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O2 -std=c++17 -fPIC -shared -Wall $("$PY" -m pybind11 --includes) cells.cpp -o "cbc_cells$SUF" -lgmpxx -lgmp
sha256sum cells.cpp "cbc_cells$SUF"
