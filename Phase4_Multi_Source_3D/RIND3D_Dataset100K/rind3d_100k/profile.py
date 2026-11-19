"""Certified production profile guard (RIND3D-100K v3).

A contract may enter the certified production pipeline (scripts/run_node.py) only if every geometry family is one of
the validated convex-polyhedral families: FastLabel v3.4 and the exact tau-pencil reference decode canonical convex
halfspace components only (rind3d.rendering.polyhedra.decode_convex). Curved families (balls, contract
rind3d-100k-curved-research-v0) are physically implemented and tested but have NO certified labeller; they stay in
research configurations (rind3d_100k.physical_curved) and are refused here, so they cannot enter certified output."""

CERTIFIED_FAMILIES = ("rotated_boxes", "thin_slabs", "prisms", "wedges", "convex_polyhedra", "multiscale_convex",
                      "corridors", "barriers", "occluder_clusters")
LABEL_ENGINES = ("fastlabel", "tpe")


class UnsupportedProfile(ValueError):
    pass


def check_certified_profile(contract, label_engine=None):
    """Raise UnsupportedProfile unless the contract is a certified-label profile (polyhedral families, a certified
    labeller). Returns the label engine in effect."""
    fams = list(contract.get("families") or [])
    bad = [f for f in fams if f not in CERTIFIED_FAMILIES]
    if not fams or bad:
        raise UnsupportedProfile(f"contract {contract.get('contract')}: families {bad or fams} have no certified labeller; "
                                 f"certified profiles use only {CERTIFIED_FAMILIES}")
    eng = label_engine or (contract.get("labels") or {}).get("engine")
    if isinstance(eng, str) and eng not in LABEL_ENGINES:        # v1 stores the engine version string
        eng = "tpe" if "tpe" in eng else ("fastlabel" if eng.startswith("fastlabel") else eng)
    if eng not in LABEL_ENGINES:
        raise UnsupportedProfile(f"contract {contract.get('contract')}: label engine {eng!r} is not a certified labeller")
    return eng
