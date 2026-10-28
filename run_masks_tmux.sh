#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-status}"
shift || true
task="${1:-}"
if [[ "$action" == start ]]; then
  case "$task" in
    install|bootstrap|prepare|index|train|evaluate|all) shift ;;
    *) task=all ;;
  esac
else
  task=train
fi
export RIND_TMUX_SESSION="${RIND_TMUX_SESSION:-rind_masks_only}"
export RIND_MASKS_CONFIG="${RIND_MASKS_CONFIG:-$ROOT/Phase2_Multi_Source/configs/masks_only.json}"
python_bin="${RIND_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  for candidate in "$ROOT/Phase1_Single_Source_Single_Intensity/.venv/bin/python" "$ROOT/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then python_bin="$candidate"; break; fi
  done
fi
[[ -n "$python_bin" ]] || { echo 'Python environment missing; set RIND_PYTHON=/absolute/path/to/python' >&2; exit 1; }
RIND_MASKS_CONFIG="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$RIND_MASKS_CONFIG")"
RIND_TMUX_OUTPUT_ROOT="$("$python_bin" -c 'import json,sys; from pathlib import Path; p=Path(json.loads(Path(sys.argv[1]).read_text())["output_root"]); print(p if p.is_absolute() else Path(sys.argv[2])/p)' "$RIND_MASKS_CONFIG" "$ROOT/Phase2_Multi_Source")"
export RIND_TMUX_OUTPUT_ROOT
export RIND_TMUX_RUNNER="$ROOT/run_masks.sh"
export RIND_TMUX_TASK="$task"
export RIND_TMUX_FRONTEND="${RIND_TMUX_FRONTEND:-./run_masks_tmux.sh}"
exec "$ROOT/run_implicit_slots_tmux.sh" "$action" "$@"
