"""Data and output locations (overridable by environment variables; no machine-specific path is required in code)."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent                                   # RIND3D_StudentLab
PHASE4 = LAB.parent                                 # Phase4_Multi_Source_3D
PILOT = PHASE4 / "RIND3D_StudentResearch" / "pilot"  # frozen pilot code (Table, Student, evaluation helpers)
DATA = Path(os.environ.get("RIND3D_1K_DATA", Path.home() / "Projects/RIND/RIND3D_1K_data"))
REL = DATA / "release_1k_v1"
NAT = DATA / "natural_supplement_v1"
FEATS = Path(os.environ.get("RIND3D_LAB_FEATS", DATA / "pilot_features_v1"))
OUT = Path(os.environ.get("RIND3D_LAB_OUT", DATA / "studentlab_v1"))
FACETS = OUT / "facets_v1.npz"


def facets_file(version="v1"):
    return OUT / f"facets_{version}.npz"
