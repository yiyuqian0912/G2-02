#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/Phase1_Single_Source_Single_Intensity/evaluate_phase1_100k.sh" "$@"
