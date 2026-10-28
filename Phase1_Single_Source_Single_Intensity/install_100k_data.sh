#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
archive="${1:-$ROOT/RIND-adaptive-v3-100000scenes-single.zip}"
python_bin="${RIND_PYTHON:-}"
if [[ -z "$python_bin" && -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  python_bin="$VIRTUAL_ENV/bin/python"
elif [[ -z "$python_bin" && -x "$ROOT/.venv/bin/python" ]]; then
  python_bin="$ROOT/.venv/bin/python"
elif [[ -z "$python_bin" && -x "$ROOT/../.venv/bin/python" ]]; then
  python_bin="$ROOT/../.venv/bin/python"
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
  echo 'Python environment is missing. From RIND_Phase2_Training_Minimal run: ./setup.sh' >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src:$ROOT/vendor/rind-dataset${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m rind_phase1.install_data "$archive" \
  --data-root "$ROOT/data/raw/phase1-single-source-100k" --keep-archive
