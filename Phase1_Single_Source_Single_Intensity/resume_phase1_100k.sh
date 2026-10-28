#!/usr/bin/env bash
# Resume already-rendered teachers after a training-runtime update.
# Example: ./resume_phase1_100k.sh train --gpus 8
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/Phase1_Single_Source_Single_Intensity/src/rind_phase1" ]]; then
  ROOT="$SCRIPT_DIR/Phase1_Single_Source_Single_Intensity"
else
  ROOT="$SCRIPT_DIR"
fi
stage="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in train|evaluate|summarize) ;; *) echo 'First argument must be train, evaluate, or summarize.' >&2; exit 2;; esac
gpus=8
global_batch=128
accumulation=6
run_label=""
objective_args=()
output="$ROOT/outputs/experiments/phase1_100k_multiscale_h20_memmap"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) gpus="$2"; shift 2;;
    --global-batch) global_batch="$2"; shift 2;;
    --accumulation) accumulation="$2"; shift 2;;
    --output-root) output="$2"; shift 2;;
    --run-label) run_label="$2"; shift 2;;
    --entropy-match-weight|--ranking-loss-weight|--ranking-positive-mass|--ranking-negative-fraction|--ranking-margin|--ranking-softmin-temperature)
      objective_args+=("$1" "$2"); shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
if ! [[ "$gpus" =~ ^[1-9][0-9]*$ ]]; then echo '--gpus must be positive.' >&2; exit 2; fi
if ! [[ "$global_batch" =~ ^[1-9][0-9]*$ ]] || [[ "$global_batch" -lt "$gpus" ]]; then
  echo '--global-batch must be a positive integer at least as large as --gpus.' >&2; exit 2
fi
if ! [[ "$accumulation" =~ ^[1-9][0-9]*$ ]]; then echo '--accumulation must be positive.' >&2; exit 2; fi
python_bin="${RIND_PYTHON:-}"
if [[ -z "$python_bin" && -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then python_bin="$VIRTUAL_ENV/bin/python"; fi
if [[ -z "$python_bin" && -x "$ROOT/.venv/bin/python" ]]; then python_bin="$ROOT/.venv/bin/python"; fi
if [[ -z "$python_bin" && -x "$ROOT/../.venv/bin/python" ]]; then python_bin="$ROOT/../.venv/bin/python"; fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then echo 'Python environment is missing. Run ./setup.sh first.' >&2; exit 1; fi
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/vendor/rind-dataset${PYTHONPATH:+:$PYTHONPATH}"
export RIND_RUNTIME_BATCH_SIZE="$global_batch"
export RIND_GRADIENT_ACCUMULATION_STEPS="$accumulation"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT"
common=(-m rind_phase1.resume_prepared "$stage" --output-root "$output" --variant response)
if [[ -n "$run_label" ]]; then common+=(--run-label "$run_label"); fi
common+=("${objective_args[@]}")
mkdir -p "$output"
log="$output/training.log"
echo "Launching prepared-cache stage=$stage | GPUs=$gpus | micro batch=$global_batch | accumulation=$accumulation | effective batch=$((global_batch*accumulation)) | log=$log"
if [[ "$stage" == train && "$gpus" -gt 1 ]]; then
  "$python_bin" -u -m torch.distributed.run --standalone --nproc-per-node="$gpus" "${common[@]}" 2>&1 | tee -a "$log"
  exit "${PIPESTATUS[0]}"
fi
"$python_bin" -u "${common[@]}" 2>&1 | tee -a "$log"
exit "${PIPESTATUS[0]}"
