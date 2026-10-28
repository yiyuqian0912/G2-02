#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-status}"
shift || true
task="${1:-}"
if [[ "$action" == "start" ]]; then
  case "$task" in
    install|bootstrap|prepare|index|train|evaluate|all) shift ;;
    *) task=train ;;
  esac
else
  task=train
fi
export RIND_TMUX_SESSION="${RIND_TMUX_SESSION:-rind_decomposition_union}"
export RIND_TMUX_OUTPUT_ROOT="$ROOT/Phase2_Multi_Source/outputs/response_decomposition_union_v2"
export RIND_TMUX_RUNNER="$ROOT/run_decomposition_union.sh"
export RIND_TMUX_TASK="$task"
export RIND_TMUX_FRONTEND=./run_decomposition_union_tmux.sh
exec "$ROOT/run_implicit_slots_tmux.sh" "$action" "$@"
