#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
echo 'RIND build: phase2-decomposition-2026-10-28-v9-expanded'
task="${1:-}"
shift || true
case "$task" in install|bootstrap|prepare|index|train|evaluate|all) ;; *) echo 'Usage: run_decomposition.sh {install|bootstrap|prepare|index|train|evaluate|all} [--gpus auto|N] [--gpu-ids 2,3]' >&2; exit 2 ;; esac
gpus=auto
gpu_ids=""
seed=0
bootstrap=1000
max_used=2048
max_utilization=10
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
  case "$1" in
    --gpus) gpus="$2" ;;
    --gpu-ids) gpu_ids="$2" ;;
    --seed) seed="$2" ;;
    --bootstrap-repetitions) bootstrap="$2" ;;
    --max-used-memory-mib) max_used="$2" ;;
    --max-utilization) max_utilization="$2" ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift 2
done
if [[ "$gpus" != auto ]] && ! [[ "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo '--gpus must be auto or a positive integer' >&2; exit 2
fi
for value in "$max_used" "$max_utilization" "$seed" "$bootstrap"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Thresholds, seed and bootstrap must be nonnegative integers' >&2; exit 2; }
done
PACKAGE="$(cd "$ROOT/.." && pwd)"
config="${RIND_DECOMPOSITION_CONFIG:-$ROOT/configs/decomposition_student.json}"
if [[ -n "${RIND_PYTHON:-}" ]]; then
  python_bin="$RIND_PYTHON"
elif [[ -x "$PACKAGE/Phase1_Single_Source_Single_Intensity/.venv/bin/python" ]]; then
  python_bin="$PACKAGE/Phase1_Single_Source_Single_Intensity/.venv/bin/python"
elif [[ -x "$PACKAGE/.venv/bin/python" ]]; then
  python_bin="$PACKAGE/.venv/bin/python"
else
  echo 'Python environment missing; set RIND_PYTHON=/absolute/path/to/python' >&2; exit 1
fi
export PYTHONPATH="$ROOT:$ROOT/research/implicit_slots:$ROOT/src:$ROOT/vendor/rind-dataset:$PACKAGE/Phase1_Single_Source_Single_Intensity/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PACKAGE/runtime_cache/numba-regions-v2}"
mkdir -p "$NUMBA_CACHE_DIR"
if [[ "$task" == install || "$task" == bootstrap || "$task" == prepare || "$task" == all ]]; then
  "$python_bin" -u -c 'from rind_phase2.regions import check_geometry_backend; print("Teacher geometry preflight:", check_geometry_backend(), flush=True)'
fi
if [[ "$task" == install || "$task" == bootstrap ]]; then
  "$ROOT/dataset_update/install_new_dataset.sh" "$PACKAGE/RIND-random-v3-100000scenes-multi-unit-256views.zip"
  [[ "$task" != install ]] || exit 0
  task=all
fi
if [[ "$task" == index ]]; then
  exec "$python_bin" -u research/implicit_slots/train_decomposition.py index --config "$config"
fi
# A malformed config must fail here, rather than falling through to full
# teacher preparation because a Python validation error looked like "false".
masks_only="$("$python_bin" -c 'import sys; from train_decomposition import load_config; print(int(load_config(sys.argv[1]).get("task") == "decomposition_only"))' "$config")"
if [[ "$masks_only" == 1 && ( "$task" == prepare || "$task" == all ) ]]; then
  echo 'Decomposition only: preparing local mask/strength labels; candidate renders=0; Phase1 locator disabled.'
  "$python_bin" -u research/implicit_slots/train_decomposition.py index --config "$config"
  [[ "$task" != prepare ]] || exit 0
  task=train
fi
selected=()
if [[ -n "$gpu_ids" ]]; then
  IFS=, read -r -a selected <<< "$gpu_ids"
  seen=,
  for i in "${!selected[@]}"; do
    value="${selected[$i]//[[:space:]]/}"
    [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Invalid --gpu-ids' >&2; exit 2; }
    [[ "$seen" != *",$value,"* ]] || { echo 'Duplicate --gpu-ids' >&2; exit 2; }
    seen+="$value,"
    selected[$i]="$value"
  done
else
  command -v nvidia-smi >/dev/null || { echo 'nvidia-smi unavailable' >&2; exit 1; }
  inventory="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  while IFS=, read -r id memory utilization; do
    id="${id//[[:space:]]/}"; memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
    if [[ "$id" =~ ^[0-9]+$ && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] && (( memory <= max_used && utilization <= max_utilization )); then
      selected+=("$id")
    fi
  done <<< "$inventory"
  if [[ "$gpus" != auto && "${#selected[@]}" -gt "$gpus" ]]; then selected=("${selected[@]:0:$gpus}"); fi
fi
[[ "${#selected[@]}" -gt 0 ]] || { echo 'No idle GPUs; task stopped without occupying another job GPU' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${selected[*]}")"
export RIND_PHYSICAL_GPU_IDS="$CUDA_VISIBLE_DEVICES"
gpu_count="${#selected[@]}"
echo "Selected physical GPUs: $CUDA_VISIBLE_DEVICES | workers=$gpu_count | config=$config"
if [[ "$task" == prepare || "$task" == all ]]; then
  # Existing-book mode does not regenerate teachers. Union mode only appends
  # newly installed scene configurations, preserving all original teacher files.
  if "$python_bin" -c 'import sys; from train_decomposition import load_config; sys.exit(0 if "new_teacher" in load_config(sys.argv[1]) else 1)' "$config"; then
    "$python_bin" -u research/implicit_slots/prepare_decomposition_teachers.py --config "$config"
  else
    echo 'Existing complete teacher book selected; physical rendering is reused.'
  fi
  "$python_bin" -u research/implicit_slots/train_decomposition.py index --config "$config"
  [[ "$task" != prepare ]] || exit 0
  task=train
fi
if [[ "$task" == evaluate ]]; then
  evaluator=research/implicit_slots/evaluate_decomposition.py
  if [[ "$masks_only" == 1 ]]; then evaluator=research/implicit_slots/evaluate_masks.py; fi
  arguments=("$evaluator" --config "$config" --seed "$seed" --bootstrap-repetitions "$bootstrap")
else
  arguments=(research/implicit_slots/train_decomposition.py train --config "$config")
fi
if [[ "$gpu_count" -gt 1 ]]; then
  exec "$python_bin" -u -m torch.distributed.run --standalone --nproc-per-node="$gpu_count" "${arguments[@]}"
fi
exec "$python_bin" -u "${arguments[@]}"
