#!/bin/bash
# Final pilot analysis over all 13 arms with the declared paired comparisons; writes experiments/out/pilot_analysis.json
# and the markdown tables experiments/out/pilot_tables.md; plus seed-averaged effects (pilot_seed_effects.json).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/RIND3D_HANDOFF/env.sh"   # RIND3D_REPO, RIND3D_1K_DATA, interpreters, NVRTC
R=$RIND3D_1K_DATA/pilot_runs_v1
PY=$RIND3D_TORCH_PY
cd $RIND3D_REPO/RIND3D_StudentResearch/pilot
O=../experiments/out
ARMS=""
for a in FL_B1K_s0 FL_B1K_s1 OH_B1K_s0 OH_B1K_s1 FL_MIS_s0 FL_MIS_s1 OH_MIS_s0 OH_MIS_s1 FL_TREL_s0 FL_B1K_sc25_s0 OH_B1K_sc25_s0 FL_B1K_ex25_s0 OH_B1K_ex25_s0; do
  ARMS="$ARMS $a=$R/$a/eval.npz"
done
PAIRS="OH_B1K_s0:FL_B1K_s0,OH_B1K_s1:FL_B1K_s1,OH_MIS_s0:FL_MIS_s0,OH_MIS_s1:FL_MIS_s1"
PAIRS="$PAIRS,FL_MIS_s0:FL_B1K_s0,FL_MIS_s1:FL_B1K_s1,FL_TREL_s0:FL_B1K_s0,OH_MIS_s0:OH_B1K_s0,OH_MIS_s1:OH_B1K_s1"
PAIRS="$PAIRS,FL_B1K_sc25_s0:FL_B1K_ex25_s0,OH_B1K_sc25_s0:OH_B1K_ex25_s0,OH_B1K_sc25_s0:FL_B1K_sc25_s0,OH_B1K_ex25_s0:FL_B1K_ex25_s0"
PAIRS="$PAIRS,FL_B1K_s1:FL_B1K_s0,FL_MIS_s1:FL_MIS_s0,OH_B1K_s1:OH_B1K_s0,OH_MIS_s1:OH_MIS_s0"
$PY analyze.py $O/pilot_analysis.json $ARMS --pairs $PAIRS > $O/pilot_analysis.log 2>&1
$PY tables.py $O/pilot_analysis.json > $O/pilot_tables.md
$PY - <<'PYEOF' > $O/pilot_seed_effects.json
import json
R = json.load(open("../experiments/out/pilot_analysis.json"))["arms"]
def g(a, s): return R[a][s]["info_gain_vs_prior"]["mean"]
out = {}
for s in ("tstar", "b1k", "ood_nat_sep", "ood_rel_sep", "ood_nat_alllit", "ood_nat_alldark"):
    m = {k: [g(f"{k}_s0", s), g(f"{k}_s1", s)] for k in ("FL_B1K", "OH_B1K", "FL_MIS", "OH_MIS")}
    mean = {k: sum(v) / 2 for k, v in m.items()}
    out[s] = dict(per_seed=m, seed_mean=mean, seed_half_range={k: abs(v[0] - v[1]) / 2 for k, v in m.items()},
                  FL_minus_OH_B1K=mean["FL_B1K"] - mean["OH_B1K"], FL_minus_OH_MIS=mean["FL_MIS"] - mean["OH_MIS"],
                  MIS_minus_B1K_FL=mean["FL_MIS"] - mean["FL_B1K"], MIS_minus_B1K_OH=mean["OH_MIS"] - mean["OH_B1K"],
                  MIS_minus_B1K_FL_per_seed=[m["FL_MIS"][i] - m["FL_B1K"][i] for i in (0, 1)])
print(json.dumps(out, indent=1))
PYEOF
echo done
