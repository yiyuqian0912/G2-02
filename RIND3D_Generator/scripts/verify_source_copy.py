"""Check copied scientific source against the frozen relocation inventory."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "provenance/ORIGINAL_SOURCE.json").read_text())
    errors = []
    checked = set()
    for original, expected in inventory["copied_source"].items():
        prefix, tail = original.split("/", 1)
        relative = inventory.get("relocated_paths", {}).get(original)
        if relative is None:
            relative = {
                "RIND3D": "",
                "canonical_geometry": "vendor/canonical_geometry",
                "dataset": "vendor/canonical_geometry",
                "RINDRay": "vendor/RINDRay",
            }[prefix]
            relative = str(Path(relative) / tail)
        path = root / relative
        digest = inventory.get("adapted_files_sha256", {}).get(relative, expected["sha256"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(relative)
        checked.add(relative)
    print(json.dumps({"copied_files_checked": len(checked), "mismatches": errors}))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
