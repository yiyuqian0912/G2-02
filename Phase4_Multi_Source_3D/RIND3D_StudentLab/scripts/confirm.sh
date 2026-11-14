#!/bin/bash
# Confirmatory evaluation (docs/A_RESEARCH_PLAN.md §7 with amendments A1, A2): the 1K val split, opened once, for the
# pre-declared runs only.
# usage: scripts/confirm.sh <best_projective_arm>
#   A1: the arm among F3, F4, F5 with the largest dev T* IG if it beats F1 s0 by > 2x the F1 seed spread (0.06), else F1_s0
#   A2: if the best arm is F5_s0, F5 with M = 4 test-time quadrature (F5q4) is evaluated as a secondary, descriptive comparison
# Every evaluation is appended to $RIND3D_LAB_OUT/CONFIRM_LOG.jsonl by slab.evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=${RIND3D_LAB_OUT:-${RIND3D_1K_DATA:-$HOME/Projects/RIND/RIND3D_1K_data}/studentlab_v1}
R=$OUT/runs
BEST=${1:?best projective arm}
for a in G0_s0 G0_s1 F1_s0 F1_s1 P0; do
  [ -f $R/$a/eval_confirm.npz ] || $PY -m slab.evaluate $R/$a --split confirm --confirm --sets tstar,b1k
done
if [ "$BEST" != F1_s0 ]; then
  [ -f $R/$BEST/eval_confirm.npz ] || $PY -m slab.evaluate $R/$BEST --split confirm --confirm --sets tstar,b1k
fi
if [ "$BEST" == F5_s0 ]; then
  [ -f $R/F5_s0/eval_confirm_q4.npz ] || $PY -m slab.evaluate $R/F5_s0 --split confirm --confirm --sets tstar,b1k --tag q4 --override '{"quad": 4}'
fi
# C3: OOD strata for the best projective arm and G0 s0 (the confirm split has the 360 certified OOD references)
[ -f $R/G0_s0/eval_confirm_ood.npz ] || $PY -m slab.evaluate $R/G0_s0 --split confirm --confirm --sets sep,alllit,alldark --tag ood
[ -f $R/$BEST/eval_confirm_ood.npz ] || $PY -m slab.evaluate $R/$BEST --split confirm --confirm --sets sep,alllit,alldark --tag ood
P=$OUT/pilot_evals/FL_MIS_s0/eval_confirm.npz
[ -f $P ] || $PY -m slab.evaluate pilot:$HOME/Projects/RIND/RIND3D_1K_data/pilot_runs_v1/FL_MIS_s0/best.pt --split confirm --confirm --sets tstar
mkdir -p docs/confirm
EXTRA=""; PAIRS=""
if [ "$BEST" != F1_s0 ]; then EXTRA="$BEST=$R/$BEST/eval_confirm.npz"; PAIRS=",$BEST:G0_s0,$BEST:F1_s0"; fi
if [ "$BEST" == F5_s0 ]; then EXTRA="$EXTRA F5q4=$R/F5_s0/eval_confirm_q4.npz"; PAIRS="$PAIRS,F5q4:F5_s0,F5q4:F1_s0"; fi
$PY -m slab.analyze G0_s0=$R/G0_s0/eval_confirm.npz G0_s1=$R/G0_s1/eval_confirm.npz F1_s0=$R/F1_s0/eval_confirm.npz \
  F1_s1=$R/F1_s1/eval_confirm.npz P0=$R/P0/eval_confirm.npz $EXTRA --sets tstar,b1k \
  --pairs F1_s0:G0_s0,F1_s1:G0_s1,P0:G0_s0$PAIRS --json docs/confirm/C1_C2.json > docs/confirm/C1_C2.md
$PY -m slab.analyze G0_s0=$R/G0_s0/eval_confirm_ood.npz $BEST=$R/$BEST/eval_confirm_ood.npz --sets sep,alllit,alldark \
  --pairs $BEST:G0_s0 --json docs/confirm/C3.json > docs/confirm/C3.md
$PY -m slab.analyze G0_s0=$R/G0_s0/eval_confirm.npz pilotFL_MIS_s0=$P --sets tstar --pairs G0_s0:pilotFL_MIS_s0 \
  --json docs/confirm/C4.json > docs/confirm/C4.md
cat docs/confirm/C1_C2.md docs/confirm/C3.md docs/confirm/C4.md
