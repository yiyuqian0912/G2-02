#!/usr/bin/env bash
# Example: ./run_training.sh --run-name trial2 --epochs 50 --batch-size 8 --lr 0.0003
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
python_bin=""
if [[ -n "${RIND_PYTHON:-}" ]]; then
  python_bin="$RIND_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  python_bin="$VIRTUAL_ENV/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  python_bin="$PROJECT_DIR/.venv/bin/python"
elif [[ -x "$PROJECT_DIR/../.venv/bin/python" ]]; then
  python_bin="$PROJECT_DIR/../.venv/bin/python"
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
  printf '%s\n' 'Python environment is missing. From RIND_Phase2_Training_Minimal run: ./setup.sh' >&2
  exit 1
fi
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src:$PROJECT_DIR/vendor/rind-dataset${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" scripts/launch_training.py "$@"
