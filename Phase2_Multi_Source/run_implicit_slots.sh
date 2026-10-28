#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
BUILD_ID="phase2-implicit-slots-2026-10-27-metrics-v4"
echo "RIND build: $BUILD_ID"

command_name="${1:-}"
if [[ "$command_name" != "index" && "$command_name" != "train" && "$command_name" != "all" && "$command_name" != "evaluate" ]]; then
  echo "Usage: ./run_implicit_slots.sh {index|train|evaluate|all} [--gpus auto|N] [--gpu-ids 2,3]" >&2
  exit 2
fi
shift
gpus=auto
gpu_ids=""
max_used_memory_mib=2048
max_utilization=10
evaluation_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) gpus="$2"; shift 2 ;;
    --gpu-ids) gpu_ids="$2"; shift 2 ;;
    --max-used-memory-mib) max_used_memory_mib="$2"; shift 2 ;;
    --max-utilization) max_utilization="$2"; shift 2 ;;
    --seed) evaluation_args+=(--seed "$2"); shift 2 ;;
    --bootstrap-repetitions) evaluation_args+=(--bootstrap-repetitions "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$command_name" != "evaluate" && "${#evaluation_args[@]}" -gt 0 ]]; then
  echo "--seed and --bootstrap-repetitions are evaluation-only options." >&2
  exit 2
fi
if [[ "$gpus" != "auto" ]] && ! [[ "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "--gpus must be auto or a positive integer." >&2
  exit 2
fi
if ! [[ "$max_used_memory_mib" =~ ^[0-9]+$ ]] || ! [[ "$max_utilization" =~ ^[0-9]+$ ]]; then
  echo "GPU thresholds must be nonnegative integers." >&2
  exit 2
fi

if [[ -n "${RIND_PYTHON:-}" ]]; then
  python_bin="$RIND_PYTHON"
elif [[ -x "../Phase1_Single_Source_Single_Intensity/.venv/bin/python" ]]; then
  python_bin="../Phase1_Single_Source_Single_Intensity/.venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
else
  echo "Python environment missing. Activate .venv or set RIND_PYTHON=/absolute/path/to/python." >&2
  exit 1
fi

export PYTHONPATH="$ROOT/research/implicit_slots:$ROOT/src:$ROOT/vendor/rind-dataset:$ROOT/../Phase1_Single_Source_Single_Intensity/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

common=(research/implicit_slots/train_slots.py --config configs/implicit_slots.json)
if [[ "$command_name" == "index" ]]; then
  exec "$python_bin" -u "${common[@]}" index --gpus 1
fi
if [[ "$command_name" == "all" ]]; then
  "$python_bin" -u "${common[@]}" index --gpus 1
  command_name="train"
fi

selected=()
if [[ -n "$gpu_ids" ]]; then
  IFS=',' read -r -a selected <<< "$gpu_ids"
  seen_gpu_ids=","
  for index in "${!selected[@]}"; do
    id="${selected[$index]//[[:space:]]/}"
    if ! [[ "$id" =~ ^[0-9]+$ ]]; then
      echo "--gpu-ids must be a comma-separated list of nonnegative integers." >&2
      exit 2
    fi
    if [[ "$seen_gpu_ids" == *",$id,"* ]]; then
      echo "--gpu-ids contains duplicate physical GPU $id." >&2
      exit 2
    fi
    seen_gpu_ids+="$id,"
    selected[$index]="$id"
  done
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for automatic GPU selection; use --gpu-ids explicitly." >&2
    exit 1
  fi
  while IFS=',' read -r id used utilization; do
    id="${id//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if [[ "$id" =~ ^[0-9]+$ && "$used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] && \
       (( used <= max_used_memory_mib && utilization <= max_utilization )); then
      selected+=("$id")
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  if [[ "$gpus" != "auto" && "${#selected[@]}" -gt "$gpus" ]]; then
    selected=("${selected[@]:0:$gpus}")
  fi
fi
if [[ "${#selected[@]}" -eq 0 ]]; then
  echo "No idle GPU satisfies memory.used<=${max_used_memory_mib} MiB and utilization<=${max_utilization}%." >&2
  exit 1
fi
if [[ "$gpus" != "auto" && "${#selected[@]}" -lt "$gpus" ]]; then
  echo "Requested $gpus GPUs, but only ${#selected[@]} pass the idle thresholds; using the available set." >&2
fi
selected_csv="$(IFS=,; echo "${selected[*]}")"
export CUDA_VISIBLE_DEVICES="$selected_csv"
export RIND_PHYSICAL_GPU_IDS="$selected_csv"
gpu_count="${#selected[@]}"
if [[ "$command_name" == "evaluate" ]]; then
  echo "Selected physical GPUs: $selected_csv | evaluation ranks: 0-$((gpu_count-1)) | best.pt inference"
else
  echo "Selected physical GPUs: $selected_csv | visible DDP ranks: 0-$((gpu_count-1)) | per-GPU micro-batch=8; effective global batch is held near 128 by gradient accumulation"
fi

if [[ "$command_name" == "evaluate" ]]; then
  evaluator=(research/implicit_slots/evaluate_slots.py --config configs/implicit_slots.json "${evaluation_args[@]}")
  if [[ "$gpu_count" -gt 1 ]]; then
    exec "$python_bin" -u -m torch.distributed.run --standalone --nproc-per-node="$gpu_count" \
      "${evaluator[@]}"
  fi
  exec "$python_bin" -u "${evaluator[@]}"
fi

if [[ "$gpu_count" -gt 1 ]]; then
  exec "$python_bin" -u -m torch.distributed.run --standalone --nproc-per-node="$gpu_count" \
    "${common[@]}" "$command_name" --gpus "$gpu_count"
fi
exec "$python_bin" -u "${common[@]}" "$command_name" --gpus 1
