#!/bin/bash
# Approved 1K FastLabel v3.4 posterior backfill (2026-11-13). Resumable: rerunning this script skips finished task files.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/RIND3D_HANDOFF/env.sh"   # RIND3D_REPO, RIND3D_1K_DATA, interpreters, NVRTC
cd $RIND3D_REPO/RIND3D_Dataset1K
exec $RIND3D_LABEL_PY scripts/run_labels.py $RIND3D_1K_DATA/release_1k_v1 15 0 --engine fastlabel --scene-order --retry-failed
