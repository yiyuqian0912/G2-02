#!/usr/bin/env bash
# Example: ./run_training.sh --run-name trial2 --epochs 50 --batch-size 8 --lr 0.0003
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' 'Missing .venv. First run: uv sync --locked --extra train --extra browser' >&2
  exit 1
fi
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src:$PROJECT_DIR/vendor/rind-dataset${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python scripts/launch_training.py "$@"
