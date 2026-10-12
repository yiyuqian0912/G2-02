#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${RIND_PYTHON:-}" ]]; then
  training_python="$RIND_PYTHON"
elif [[ -x "../Phase1_Single_Source_Single_Intensity/.venv/bin/python" ]]; then
  training_python="../Phase1_Single_Source_Single_Intensity/.venv/bin/python"
else
  training_python=".venv/bin/python"
fi
if [[ ! -x "$training_python" ]]; then
  echo "Missing training Python environment: $training_python" >&2
  echo "Set RIND_PYTHON=/absolute/path/to/python or create .venv." >&2
  exit 1
fi
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PWD/outputs/.numba_cache}"
mkdir -p "$NUMBA_CACHE_DIR"
exec "$training_python" research/layered_residual/train_combined.py "$@"
