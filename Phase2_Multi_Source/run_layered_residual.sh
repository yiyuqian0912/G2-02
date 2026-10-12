#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
phase1_python="../Phase1_Single_Source_Single_Intensity/.venv/bin/python"
if [[ -x "$phase1_python" ]]; then
  exec "$phase1_python" research/layered_residual/layered_residual.py "$@"
fi
exec uv run --with matplotlib python research/layered_residual/layered_residual.py "$@"
