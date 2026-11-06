#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task="${1:-all}";shift || true
case "$task" in prepare|train|evaluate|all) ;; *) echo 'Expected prepare, train, evaluate, or all';exit 2 ;; esac
gpus=auto;device=cuda;arguments=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) [[ $# -ge 2 ]] || { echo "Missing --gpus value";exit 2; };gpus="$2";shift 2 ;;
    --device) [[ $# -ge 2 ]] || { echo "Missing --device value";exit 2; };device="$2";arguments+=(--device "$2");shift 2 ;;
    *) arguments+=("$1");shift ;;
  esac
done
PY="${RIND_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for candidate in "$ROOT/.venv/bin/python" "$ROOT/Phase1_Single_Source_Single_Intensity/.venv/bin/python";do
    if [[ -x "$candidate" ]];then PY="$candidate";break;fi
  done
fi
[[ -n "$PY" ]] || { echo 'Set RIND_PYTHON or run ./setup.sh';exit 1; }
export PYTHONPATH="$ROOT/Phase1_Single_Source_Single_Intensity/src:$ROOT/Phase1_Single_Source_Single_Intensity/vendor/rind-dataset${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONFAULTHANDLER=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/runtime_cache/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$ROOT/runtime_cache/numba}"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
command=("$PY" -u -m rind_phase1.active_rl.experiment "$task")
if [[ ${#arguments[@]} -gt 0 ]];then command+=("${arguments[@]}");fi
if [[ "$task" == prepare || "$device" == cpu ]];then exec "${command[@]}";fi
if [[ "$task" == all ]];then
  pre=("$PY" -u -m rind_phase1.active_rl.experiment prepare)
  if [[ ${#arguments[@]} -gt 0 ]];then pre+=("${arguments[@]}");fi
  "${pre[@]}"
fi
ids="$("$PY" "$ROOT/Phase2_Multi_Source/research/implicit_slots/gpu_runtime.py" --gpus "$gpus" --policy adaptive --samples 3 --max-utilization 20 --min-free-memory-mib 8192)"
export CUDA_VISIBLE_DEVICES="$ids"
IFS=, read -r -a selected <<< "$ids"
echo "Single-source RL | GPUs=$ids | process/GPU; frozen Phase1, GPU ROI decode and deduplicated fusion"
if [[ "${#selected[@]}" -gt 1 ]];then
  command=("$PY" -u -m torch.distributed.run --standalone --nproc-per-node="${#selected[@]}" -m rind_phase1.active_rl.experiment "$task")
  if [[ ${#arguments[@]} -gt 0 ]];then command+=("${arguments[@]}");fi
  exec "${command[@]}"
fi
exec "${command[@]}"
