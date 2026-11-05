#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-status}";shift || true
source_session=rind_masks_disentangled_more
session="${RIND_TMUX_SESSION:-rind_failure_diagnostics}"
arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-session) [[ $# -ge 2 ]] || exit 2;source_session="$2";shift 2 ;;
    --session) [[ $# -ge 2 ]] || exit 2;session="$2";shift 2 ;;
    *) arguments+=("$1");shift ;;
  esac
done
for name in "$session" "$source_session"; do
  [[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid session name' >&2;exit 2; }
done
python_bin="${RIND_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  for candidate in "$ROOT/Phase1_Single_Source_Single_Intensity/.venv/bin/python" "$ROOT/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then python_bin="$candidate";break;fi
  done
fi
[[ -n "$python_bin" ]] || { echo 'Python environment missing; set RIND_PYTHON' >&2;exit 1; }
export RIND_PYTHON="$python_bin"
base="$ROOT/Phase2_Multi_Source/outputs/failure_diagnostics"
job="$base/jobs/$session"
if [[ "$action" == start ]]; then
  if [[ -z "${RIND_MASKS_CONFIG:-}" ]]; then
    pointer="$ROOT/Phase2_Multi_Source/outputs/disentangled_v5_launcher/$source_session.txt"
    [[ -f "$pointer" ]] || { echo "Source session config missing: $pointer; pass --source-session with your training session name." >&2;exit 2; }
    export RIND_MASKS_CONFIG="$(cat "$pointer")"
  fi
else
  if [[ -f "$job/source_config.txt" ]]; then export RIND_MASKS_CONFIG="$(cat "$job/source_config.txt")";fi
fi
export RIND_TMUX_SESSION="$session"
export RIND_TMUX_OUTPUT_ROOT="$job"
export RIND_TMUX_RUNNER="$ROOT/run_masks.sh"
export RIND_TMUX_TASK=evaluate
export RIND_TMUX_JOB_LABEL=diagnostics
export RIND_TMUX_FRONTEND=./run_failure_diagnostics_tmux.sh
command=("$ROOT/run_implicit_slots_tmux.sh" "$action")
if [[ "$action" == start ]]; then
  # Defaults read a saved validation snapshot; never evaluate the locked test
  # while deciding how to improve the model. Explicit later overrides allowed.
  command+=(--gpus 1 --batch-size 64 --checkpoint last --split validation --diagnostics-only --diagnostics-root "$base")
fi
if [[ "${#arguments[@]}" -gt 0 ]]; then command+=("${arguments[@]}");fi
if [[ "$action" == start ]]; then
  [[ -f "$RIND_MASKS_CONFIG" ]] || { echo 'Source experiment config missing' >&2;exit 2; }
  "${command[@]}"
  mkdir -p "$job";printf '%s\n' "$RIND_MASKS_CONFIG" > "$job/source_config.txt"
else exec "${command[@]}";fi
