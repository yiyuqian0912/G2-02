#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
echo 'RIND build: phase2-2026-11-04-best16-bounded-pairwise-oom-fix'
task="${1:-}"
shift || true
case "$task" in install|bootstrap|prepare|index|train|evaluate|all) ;; *) echo 'Usage: run_decomposition.sh {install|bootstrap|prepare|index|train|evaluate|all} [--gpus auto|N] [--gpu-ids 2,3]' >&2; exit 2 ;; esac
gpus=auto
gpu_ids=""
seed=0
bootstrap=1000
max_used=2048
max_utilization=20
gpu_policy=""
min_free=24576
gpu_samples=3
auto_batch=""
batch_size=""
reserve_gib=6
checkpoint=best
eval_split=""
diagnostics_only=0
diagnostics_root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto-batch) auto_batch=1;shift;continue ;;
    --no-auto-batch) auto_batch=0;shift;continue ;;
    --diagnostics-only) diagnostics_only=1;shift;continue ;;
  esac
  [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
  case "$1" in
    --gpus) gpus="$2" ;;
    --gpu-ids) gpu_ids="$2" ;;
    --gpu-policy) gpu_policy="$2" ;;
    --min-free-memory-mib) min_free="$2" ;;
    --gpu-samples) gpu_samples="$2" ;;
    --batch-size) batch_size="$2" ;;
    --gpu-reserve-gib) reserve_gib="$2" ;;
    --seed) seed="$2" ;;
    --bootstrap-repetitions) bootstrap="$2" ;;
    --checkpoint) checkpoint="$2" ;;
    --split) eval_split="$2" ;;
    --diagnostics-root) diagnostics_root="$2" ;;
    --max-used-memory-mib) max_used="$2" ;;
    --max-utilization) max_utilization="$2" ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift 2
done
if [[ "$gpus" != auto ]] && ! [[ "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo '--gpus must be auto or a positive integer' >&2; exit 2
fi
for value in "$max_used" "$max_utilization" "$seed" "$bootstrap" "$min_free" "$gpu_samples"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Thresholds, seed and bootstrap must be nonnegative integers' >&2; exit 2; }
done
[[ "$gpu_samples" -ge 1 && "$max_utilization" -le 100 ]] || { echo 'Invalid GPU samples/utilization' >&2;exit 2; }
if [[ -n "$gpu_policy" && "$gpu_policy" != adaptive && "$gpu_policy" != idle ]]; then
  echo '--gpu-policy must be adaptive or idle' >&2;exit 2
fi
if [[ -n "$batch_size" ]]; then
  [[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo '--batch-size must be positive' >&2;exit 2; }
  export RIND_BATCH_SIZE="$batch_size"
fi
export RIND_GPU_RESERVE_GIB="$reserve_gib"
if [[ "$checkpoint" != best && "$checkpoint" != last ]]; then echo '--checkpoint must be best or last' >&2;exit 2;fi
if [[ -n "$eval_split" && "$eval_split" != validation && "$eval_split" != test ]]; then echo '--split must be validation or test' >&2;exit 2;fi
if [[ "$task" != evaluate && ( "$diagnostics_only" == 1 || "$checkpoint" != best || -n "$eval_split" || -n "$diagnostics_root" ) ]]; then
  echo 'Diagnostic/evaluation flags require evaluate task' >&2;exit 2
fi
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
inventory_only="$("$python_bin" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("training_expansion",{}).get("mode")=="inventory"))' "$config")"
if [[ "$inventory_only" == 1 && "$task" != prepare && "$task" != index && "$task" != all ]]; then
  echo 'Inventory-only configuration: use prepare/index; training/evaluation is disabled until a size is chosen.' >&2
  exit 2
fi
cached_labels="$("$python_bin" -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1])).get("cache_source_root"))))' "$config")"
if [[ "$cached_labels" != 1 && ( "$task" == install || "$task" == bootstrap || "$task" == prepare || "$task" == all ) ]]; then
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
  extend_eval="$("$python_bin" -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1])).get("evaluation_expansion"))))' "$config")"
  complex_append="$("$python_bin" -c 'import json,sys; print(int(bool(json.load(open(sys.argv[1])).get("training_expansion"))))' "$config")"
  if [[ "$inventory_only" == 1 ]]; then
    echo 'Inventory only: metadata catalogue and counts; no model training or train-cache assembly.'
  elif [[ "$complex_append" == 1 ]]; then
    echo 'Appending unique complex training windows; validation/test frozen; local labels only; candidate renders=0.'
  elif [[ "$extend_eval" == 1 ]]; then
    echo 'Reusing training and legacy evaluation labels; adding natural held-out windows from RIND; candidate renders=0.'
  elif [[ "$cached_labels" == 1 ]]; then
    echo 'Reusing complete response/layer label cache; no raw data, no rendering; Phase1 locator disabled.'
  else
    echo 'Decomposition only: preparing local mask/strength labels; candidate renders=0; Phase1 locator disabled.'
  fi
  "$python_bin" -u research/implicit_slots/train_decomposition.py index --config "$config"
  [[ "$task" != prepare && "$inventory_only" != 1 ]] || exit 0
  task=train
fi
architecture="$("$python_bin" -c 'from train_decomposition import load_config;import sys;print(load_config(sys.argv[1])["model"].get("decomposition_architecture","legacy"))' "$config")"
if [[ -z "$gpu_policy" ]]; then
  if [[ "$architecture" == disentangled_slots_v5 || "$architecture" == sequential_count_shape_v6 ]]; then gpu_policy=adaptive;else gpu_policy=legacy;fi
fi
if [[ -z "$auto_batch" ]]; then
  if [[ ( "$architecture" == disentangled_slots_v5 || "$architecture" == sequential_count_shape_v6 ) && -z "$batch_size" ]]; then auto_batch=1;else auto_batch=0;fi
fi
export RIND_AUTO_BATCH="$auto_batch"
selected=()
if [[ "$gpu_policy" == adaptive || "$gpu_policy" == idle ]]; then
  inventory_args=(research/implicit_slots/gpu_runtime.py --gpus "$gpus" --policy "$gpu_policy"
    --min-free-memory-mib "$min_free" --max-used-memory-mib "$max_used"
    --max-utilization "$max_utilization" --samples "$gpu_samples")
  if [[ -n "$gpu_ids" ]]; then inventory_args+=(--gpu-ids "$gpu_ids");fi
  selected_ids="$("$python_bin" "${inventory_args[@]}")"
  IFS=, read -r -a selected <<< "$selected_ids"
elif [[ -n "$gpu_ids" ]]; then
  IFS=, read -r -a selected <<< "$gpu_ids"
  seen=,
  for i in "${!selected[@]}"; do
    value="${selected[$i]//[[:space:]]/}"
    [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Invalid --gpu-ids' >&2; exit 2; }
    [[ "$seen" != *",$value,"* ]] || { echo 'Duplicate --gpu-ids' >&2; exit 2; }
    seen+="$value,"
    selected[$i]="$value"
  done
elif [[ "$("$python_bin" -c 'from train_decomposition import load_config; import sys; print(int(load_config(sys.argv[1])["training"].get("require_idle_compute", False)))' "$config")" == 1 ]]; then
  selected_ids="$("$python_bin" -c 'from architecture_search import select_gpus; from argparse import Namespace; import sys; print(",".join(select_gpus(Namespace(device="cuda", gpu_ids=None, gpus=sys.argv[1], max_used_memory_mib=int(sys.argv[2]), max_utilization=int(sys.argv[3])))))' "$gpus" "$max_used" "$max_utilization")"
  IFS=, read -r -a selected <<< "$selected_ids"
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
echo "Selected physical GPUs: $CUDA_VISIBLE_DEVICES | workers=$gpu_count | policy=$gpu_policy | auto-batch=$auto_batch | config=$config"
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
  if [[ "$masks_only" == 1 ]]; then
    arguments+=(--checkpoint "$checkpoint")
    if [[ -n "$eval_split" ]]; then arguments+=(--split "$eval_split");fi
    if [[ "$diagnostics_only" == 1 ]]; then arguments+=(--diagnostics-only);fi
    if [[ -n "$diagnostics_root" ]]; then arguments+=(--diagnostics-root "$diagnostics_root");fi
  elif [[ "$diagnostics_only" == 1 || "$checkpoint" != best || -n "$eval_split" ]]; then
    echo 'Failure diagnostics require mask-only decomposition' >&2;exit 2
  fi
else
  arguments=(research/implicit_slots/train_decomposition.py train --config "$config")
fi
if [[ "$gpu_count" -gt 1 ]]; then
  exec "$python_bin" -u -m torch.distributed.run --standalone --nproc-per-node="$gpu_count" "${arguments[@]}"
fi
exec "$python_bin" -u "${arguments[@]}"
