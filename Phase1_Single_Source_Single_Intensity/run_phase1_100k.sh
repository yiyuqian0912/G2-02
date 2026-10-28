#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The same launcher is installed both inside the Phase1 checkout and at the
# minimal-package root.  Always resolve the actual Phase1 project explicitly;
# otherwise a package-root Phase2 run_training.sh can receive Phase1 arguments.
if [[ -d "$SCRIPT_DIR/Phase1_Single_Source_Single_Intensity/src/rind_phase1" ]]; then
  ROOT="$SCRIPT_DIR/Phase1_Single_Source_Single_Intensity"
else
  ROOT="$SCRIPT_DIR"
fi
stage="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
gpus=8
extra=()
append_caches=()
append_layer_caches=()
scratch_root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      [[ $# -ge 2 ]] || { echo '--gpus needs a positive integer' >&2; exit 2; }
      gpus="$2"; shift 2 ;;
    --append-cache)
      [[ $# -ge 2 ]] || { echo '--append-cache needs an old Phase-I output/response path' >&2; exit 2; }
      append_caches+=("$2"); shift 2 ;;
    --append-layer-cache)
      [[ $# -ge 2 ]] || { echo '--append-layer-cache needs an old Phase-II teacher output path' >&2; exit 2; }
      append_layer_caches+=("$2"); shift 2 ;;
    --scratch-root)
      [[ $# -ge 2 ]] || { echo '--scratch-root needs a local NVMe directory' >&2; exit 2; }
      scratch_root="$2"; shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done
case "$stage" in all|plan|prepare|train|evaluate|summarize|smoke) ;;
  *) echo 'Usage: ./run_phase1_100k.sh {smoke|all|plan|prepare|train|evaluate|summarize} [--gpus N] [--scratch-root LOCAL_NVME] [--append-cache PATH] [--append-layer-cache PATH]' >&2; exit 2 ;;
esac
if [[ ${#append_caches[@]} -gt 0 ]]; then
  joined="$(IFS=:; printf '%s' "${append_caches[*]}")"
  export RIND_PHASE1_TEACHER_CACHE_ROOTS="${RIND_PHASE1_TEACHER_CACHE_ROOTS:+$RIND_PHASE1_TEACHER_CACHE_ROOTS:}$joined"
fi
if [[ ${#append_layer_caches[@]} -gt 0 ]]; then
  joined="$(IFS=:; printf '%s' "${append_layer_caches[*]}")"
  export RIND_PHASE2_LAYER_CACHE_ROOTS="${RIND_PHASE2_LAYER_CACHE_ROOTS:+$RIND_PHASE2_LAYER_CACHE_ROOTS:}$joined"
fi
path_args=()
if [[ -n "$scratch_root" ]]; then
  mkdir -p "$scratch_root/data" "$scratch_root/outputs"
  source_data="$ROOT/data/raw/phase1-single-source-100k"
  fast_data="$scratch_root/data/phase1-single-source-100k"
  if [[ ! -f "$fast_data/manifest.json" ]]; then
    [[ -f "$source_data/manifest.json" ]] || { echo "Dataset missing: $source_data" >&2; exit 1; }
    partial="$scratch_root/data/.phase1-single-source-100k.partial.$$"
    mkdir -p "$partial"
    echo "One-time copy of the 100k dataset to local storage: $fast_data"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --info=progress2 "$source_data/" "$partial/"
    else
      cp -a "$source_data/." "$partial/"
    fi
    mv "$partial" "$fast_data"
  fi
  path_args+=(--data-root "$fast_data" --output-root "$scratch_root/outputs/phase1_100k_multiscale_h20_fast")
  echo "Fast local dataset: $fast_data"
  echo "Fast output/cache: $scratch_root/outputs/phase1_100k_multiscale_h20_fast"
fi
if [[ "$stage" == smoke ]]; then
  launch_args=(--run-name phase1_100k_multiscale_h20_memmap_smoke
    --config configs/experiment_100k_multiscale_h20.json
    --stage all --gpus "$gpus" --scene-counts 100 20 20
    --epochs 2 --patience 2)
  if [[ ${#path_args[@]} -gt 0 ]]; then launch_args+=("${path_args[@]}"); fi
  if [[ ${#extra[@]} -gt 0 ]]; then launch_args+=("${extra[@]}"); fi
  exec "$ROOT/run_training.sh" "${launch_args[@]}"
fi
launch_args=(--run-name phase1_100k_multiscale_h20_memmap
  --config configs/experiment_100k_multiscale_h20.json
  --stage "$stage" --gpus "$gpus")
if [[ ${#path_args[@]} -gt 0 ]]; then launch_args+=("${path_args[@]}"); fi
if [[ ${#extra[@]} -gt 0 ]]; then launch_args+=("${extra[@]}"); fi
exec "$ROOT/run_training.sh" "${launch_args[@]}"
