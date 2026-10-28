#!/usr/bin/env bash
# Evaluation-only core analysis using existing teachers/checkpoints/predictions.
# The default always reads the original seed_0 result. It never trains,
# rerenders teachers, or reads a *_sharpened experiment unless explicitly
# requested with --run-label.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/Phase1_Single_Source_Single_Intensity/src/rind_phase1" ]]; then
  ROOT="$SCRIPT_DIR/Phase1_Single_Source_Single_Intensity"
else
  ROOT="$SCRIPT_DIR"
fi

mode="${1:-report}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  report|comprehensive|full) ;;
  *) echo 'First argument must be report, comprehensive, or full.' >&2; exit 2 ;;
esac

output="$ROOT/outputs/experiments/phase1_100k_multiscale_h20_memmap"
seed=0
gpu=0
run_label=""
bootstrap_repetitions=1000
max_observations=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) output="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run-label) run_label="$2"; shift 2 ;;
    --bootstrap-repetitions) bootstrap_repetitions="$2"; shift 2 ;;
    --max-observations) max_observations="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

python_bin="${RIND_PYTHON:-}"
if [[ -z "$python_bin" && -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  python_bin="$VIRTUAL_ENV/bin/python"
fi
if [[ -z "$python_bin" && -x "$ROOT/.venv/bin/python" ]]; then python_bin="$ROOT/.venv/bin/python"; fi
if [[ -z "$python_bin" && -x "$ROOT/../.venv/bin/python" ]]; then python_bin="$ROOT/../.venv/bin/python"; fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
  echo 'Python environment is missing. Activate .venv or set RIND_PYTHON=/path/to/python.' >&2
  exit 1
fi

export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/vendor/rind-dataset${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

if [[ "$mode" == "full" ]]; then
  echo "Running locked test evaluation on GPU $gpu. Teacher rendering will not run."
  evaluation=(evaluate --output-root "$output" --variant response --seed "$seed")
  if [[ -n "$run_label" ]]; then evaluation+=(--run-label "$run_label"); fi
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m rind_phase1.resume_prepared "${evaluation[@]}"
fi

if [[ "$mode" == "comprehensive" || "$mode" == "full" ]]; then
  if [[ "$mode" == "comprehensive" ]]; then
    echo "Evaluating saved seed_${seed}${run_label:+_$run_label} predictions; no training or inference will run."
  fi
  comprehensive=("$ROOT/scripts/comprehensive_phase1_evaluation.py"
    --output-root "$output" --variant response --seed "$seed"
    --bootstrap-repetitions "$bootstrap_repetitions")
  if [[ -n "$run_label" ]]; then comprehensive+=(--run-label "$run_label"); fi
  if [[ -n "$max_observations" ]]; then comprehensive+=(--max-observations "$max_observations"); fi
  "$python_bin" -u "${comprehensive[@]}"
fi

args=("$ROOT/scripts/evaluate_phase1_100k.py" --output-root "$output" --variant response --seed "$seed")
if [[ -n "$run_label" ]]; then args+=(--run-label "$run_label"); fi
if [[ "$mode" == "full" ]]; then args+=(--require-test); fi
"$python_bin" -u "${args[@]}"
