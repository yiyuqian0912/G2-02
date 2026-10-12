#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ -n "${RIND_PYTHON:-}" ]]; then
  RESULTS_PYTHON="$RIND_PYTHON"
elif [[ -x "../Phase1_Single_Source_Single_Intensity/.venv/bin/python" ]]; then
  RESULTS_PYTHON="../Phase1_Single_Source_Single_Intensity/.venv/bin/python"
else
  RESULTS_PYTHON=".venv/bin/python"
fi
if [[ ! -x "$RESULTS_PYTHON" ]]; then
  printf '%s\n' 'Missing results Python environment. Set RIND_PYTHON or create .venv.' >&2
  exit 1
fi
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/rind-phase2-numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/rind-phase2-matplotlib}"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src:$PROJECT_DIR/vendor/rind-dataset:$PROJECT_DIR/research/layered_residual:../Phase1_Single_Source_Single_Intensity/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$RESULTS_PYTHON" -m rind_phase2.results_browser "$@"
