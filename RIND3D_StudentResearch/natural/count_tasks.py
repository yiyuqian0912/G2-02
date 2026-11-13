"""Count NAT-1K label tasks (mixed channels) and extreme channels per role and side; write natural/N_SUP.json.
The task count fixes the delta family of the NAT-1K labels (delta_label = 0.01 / n_tasks) BEFORE labelling."""
import json, sys
from pathlib import Path
import numpy as np

NAT = Path("/home/yushu/Projects/RIND/RIND3D_1K_data/natural_supplement_v1")
tot = 0; by = {}
scenes = 0
for d in sorted((NAT / "scenes").iterdir()):
    nj = d / "natural.json"
    if not nj.exists():
        continue
    scenes += 1
    nat = json.loads(nj.read_text())
    for wm in nat["windows"]:
        for t in wm["types"]:
            key = f"{nat['role']}:L{wm['Lp']}:{t}"
            by[key] = by.get(key, 0) + 1
            tot += t == "mixed"
out = dict(n_tasks=tot, scenes=scenes, by_role_side_type=dict(sorted(by.items())))
Path(__file__).with_name("N_SUP.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
