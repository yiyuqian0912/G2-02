#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-status}";shift || true
task=all
if [[ "$action" == start && $# -gt 0 ]];then
  case "$1" in prepare|train|evaluate|all) task="$1";shift ;; esac
fi
run_name=single_source_rl_v1;arguments=()
while [[ $# -gt 0 ]];do
  if [[ "$1" == --run-name ]];then [[ $# -ge 2 ]] || { echo "Missing --run-name";exit 2; };run_name="$2";arguments+=(--run-name "$2");shift 2
  else arguments+=("$1");shift;fi
done
[[ "$run_name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid run name';exit 2; }
export RIND_TMUX_SESSION="rind_rl_$run_name"
export RIND_TMUX_OUTPUT_ROOT="$ROOT/Phase1_Single_Source_Single_Intensity/outputs/active_rl/$run_name"
export RIND_TMUX_RUNNER="$ROOT/run_single_source_rl.sh"
export RIND_TMUX_TASK="$task"
export RIND_TMUX_FRONTEND=./run_single_source_rl_tmux.sh
# Status/logs locate saved output; they must not forward training-only flags.
if [[ "$action" == start ]];then
  command=("$ROOT/run_implicit_slots_tmux.sh" "$action")
  if [[ ${#arguments[@]} -gt 0 ]];then command+=("${arguments[@]}");fi
  exec "${command[@]}"
fi
exec "$ROOT/run_implicit_slots_tmux.sh" "$action"
