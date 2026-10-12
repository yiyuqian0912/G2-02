#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${RIND_PYTHON:-}" ]]; then
  visualization_python="$RIND_PYTHON"
elif [[ -x "../Phase1_Single_Source_Single_Intensity/.venv/bin/python" ]]; then
  visualization_python="../Phase1_Single_Source_Single_Intensity/.venv/bin/python"
else
  visualization_python=".venv/bin/python"
fi
if [[ ! -x "$visualization_python" ]]; then
  echo "Missing visualization Python environment. Set RIND_PYTHON or create .venv." >&2
  exit 1
fi
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PWD/outputs/.numba_cache}"
mkdir -p "$NUMBA_CACHE_DIR"
exec "$visualization_python" research/layered_residual/visualize_combined.py "$@"
