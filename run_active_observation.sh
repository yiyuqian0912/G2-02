#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${RIND_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in "$ROOT/.venv/bin/python" "$ROOT/Phase1_Single_Source_Single_Intensity/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then PYTHON="$candidate"; break; fi
  done
fi
if [[ -z "$PYTHON" ]] || ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo 'Set RIND_PYTHON to your existing Python environment (with NumPy and Matplotlib).' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/Phase2_Multi_Source${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m research.active_observation.demo "$@"
