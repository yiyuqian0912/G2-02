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
[[ -n "$python_bin" ]] || { echo 'Python environment missing; set RIND_PYTHON=/absolute/path/to/python' >&2;exit 1; }
export RIND_PYTHON="$python_bin"
extra=0;extra_set=0;arguments=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --add-complex ]]; then
    [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo '--add-complex needs a nonnegative integer' >&2;exit 2; }
    extra="$2";extra_set=1;shift 2
  else arguments+=("$1");shift;fi
done
session="${RIND_TMUX_SESSION:-rind_masks_disentangled_v5}"
for ((i=0;i<${#arguments[@]};i++)); do
  if [[ "${arguments[$i]}" == --session ]]; then
    j=$((i+1));[[ "$j" -lt "${#arguments[@]}" ]] || { echo 'Missing --session value' >&2;exit 2; }
    session="${arguments[$j]}"
  fi
done
[[ "$session" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid tmux session name' >&2;exit 2; }
pointer="$ROOT/Phase2_Multi_Source/outputs/disentangled_v5_launcher/$session.txt"
if [[ "$action" == start ]]; then
  if [[ -z "${RIND_MASKS_CONFIG:-}" ]]; then
    if [[ "$extra_set" == 0 && -f "$pointer" ]]; then
      export RIND_MASKS_CONFIG="$(cat "$pointer")"
      [[ -f "$RIND_MASKS_CONFIG" ]] || { echo "Previous run config is missing: $RIND_MASKS_CONFIG" >&2;exit 2; }
      echo "Reusing previous session config: $RIND_MASKS_CONFIG" >&2
    else
      export RIND_MASKS_CONFIG="$("$python_bin" "$ROOT/Phase2_Multi_Source/research/implicit_slots/configure_disentangled.py" --add-complex "$extra")"
    fi
  fi
else
  export RIND_MASKS_CONFIG="${RIND_MASKS_CONFIG:-$(if [[ -f "$pointer" ]]; then cat "$pointer"; else printf '%s' "$ROOT/Phase2_Multi_Source/configs/masks_disentangled.json";fi)}"
fi
export RIND_TMUX_SESSION="$session"
export RIND_TMUX_FRONTEND=./run_disentangled_masks_tmux.sh
command=("$ROOT/run_masks_tmux.sh" "$action")
if [[ "${#arguments[@]}" -gt 0 ]]; then command+=("${arguments[@]}");fi
if [[ "$action" == start ]]; then
  "${command[@]}"
  mkdir -p "$(dirname "$pointer")";printf '%s\n' "$RIND_MASKS_CONFIG" > "$pointer"
else exec "${command[@]}";fi
