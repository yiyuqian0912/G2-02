import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
for p in (HERE, REPO / "RIND3D_Dataset1K", REPO / "RIND3D_Posterior" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
