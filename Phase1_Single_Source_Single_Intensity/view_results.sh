#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' 'First run: uv sync --locked --extra train --extra browser' >&2
  exit 1
fi
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src:$PROJECT_DIR/vendor/rind-dataset${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m rind_phase1.results_browser "$@"
