#!/usr/bin/env bash
# Train a separate entropy-matched/ranking student from existing teachers.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/resume_phase1_100k.sh" train \
  --run-label sharpened \
  --entropy-match-weight 0.05 \
  --ranking-loss-weight 0.10 \
  --ranking-positive-mass 0.90 \
  --ranking-negative-fraction 0.50 \
  --ranking-margin 0.50 \
  --ranking-softmin-temperature 1.0 \
  "$@"
