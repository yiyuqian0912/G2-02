"""Read-only access to the canonical RIND3D package (exact oracle, geometry, stored fields).
RIND3D and the canonical `rind` package are imported from their checkouts without installation;
always run with PYTHONDONTWRITEBYTECODE=1 so nothing is written into those trees."""
import os, sys
RIND3D = os.environ.get("RIND3D_ROOT", "/home/yushu/Projects/RIND/RIND3D")
for p in (os.path.join(RIND3D, "src"), "/home/yushu/Projects/RIND/dataset/src",
          os.path.join(RIND3D, ".venv/lib/python3.12/site-packages")):
    if p not in sys.path:
        sys.path.append(p) if p.endswith("site-packages") else sys.path.insert(0, p)
SCENES = os.path.join(RIND3D, "outputs/lite-core-1k-v01/campaign/scenes")
