#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:-status}";shift || true
python_bin="${RIND_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  for candidate in "$ROOT/Phase1_Single_Source_Single_Intensity/.venv/bin/python" "$ROOT/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then python_bin="$candidate";break;fi
  done
fi
[[ -n "$python_bin" ]] || { echo 'Python environment missing; set RIND_PYTHON' >&2;exit 1; }
export RIND_PYTHON="$python_bin"
source_session=rind_masks_disentangled_more
source_root="";run_name="";source_set=0;arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-session|--source-root|--run-name)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2;exit 2; }
      case "$1" in
        --source-session) source_session="$2" ;;
        --source-root) source_root="$2" ;;
        --run-name) run_name="$2" ;;
      esac
      source_set=1;shift 2 ;;
    *) arguments+=("$1");shift ;;
  esac
done
session="${RIND_TMUX_SESSION:-rind_masks_sequential_v6}"
for ((i=0;i<${#arguments[@]};i++)); do
  if [[ "${arguments[$i]}" == --session ]]; then
    j=$((i+1));[[ "$j" -lt "${#arguments[@]}" ]] || exit 2;session="${arguments[$j]}"
  fi
done
[[ "$session" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid session name' >&2;exit 2; }
pointer="$ROOT/Phase2_Multi_Source/outputs/sequential_v6_launcher/$session.txt"
if [[ "$action" == start ]]; then
  if [[ -z "${RIND_MASKS_CONFIG:-}" ]]; then
    if [[ "$source_set" == 0 && -f "$pointer" ]]; then
      export RIND_MASKS_CONFIG="$(cat "$pointer")"
    else
      options=(--source-session "$source_session")
      if [[ -n "$source_root" ]]; then options+=(--source-root "$source_root");fi
      if [[ -n "$run_name" ]]; then options+=(--run-name "$run_name");fi
      export RIND_MASKS_CONFIG="$("$python_bin" "$ROOT/Phase2_Multi_Source/research/implicit_slots/configure_sequential.py" "${options[@]}")"
    fi
  fi
else
  export RIND_MASKS_CONFIG="${RIND_MASKS_CONFIG:-$(if [[ -f "$pointer" ]]; then cat "$pointer";else printf '%s' "$ROOT/Phase2_Multi_Source/configs/masks_sequential.json";fi)}"
fi
[[ -f "$RIND_MASKS_CONFIG" ]] || { echo "Configuration missing: $RIND_MASKS_CONFIG" >&2;exit 2; }
export RIND_TMUX_SESSION="$session"
export RIND_TMUX_FRONTEND="${RIND_TMUX_FRONTEND:-./run_sequential_masks_tmux.sh}"
command=("$ROOT/run_masks_tmux.sh" "$action")
if [[ "${#arguments[@]}" -gt 0 ]]; then command+=("${arguments[@]}");fi
if [[ "$action" == start ]]; then
  "${command[@]}"
  mkdir -p "$(dirname "$pointer")";printf '%s\n' "$RIND_MASKS_CONFIG" > "$pointer"
else exec "${command[@]}";fi
