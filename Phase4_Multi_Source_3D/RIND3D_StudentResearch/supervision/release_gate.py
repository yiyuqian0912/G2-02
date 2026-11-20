"""Qualify the frozen executable and export a complete journal's digest inventory."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

BUILD_DIGEST = "0f4b04a10e674964a9b8713a44cd2a992aa31c2be2e5420de7af6b457bf19ed6"
EXTRA_FILES = {
    "rind.identity": "f340a82681ddb989c71adb569bdb2b5dbca09d3a1b00ddd2428522fe4c6b2d2a",
    "campaign": "1fe13c6013a390fdb4f6a241ab0cc6b084b60b4f1f65b18419002068ccf9a37c",
}
NOTES = Path(__file__).resolve().parents[2] / "RIND3D_Posterior/experiments/credit_solver/assurance"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def preflight(kernel):
    sys.path[:0] = [str(kernel.resolve()), str(kernel.resolve() / "assurance")]
    import campaign
    import evidence
    from rind import identity
    expected = (NOTES / "results/build.json").read_bytes()
    if sha(expected) != BUILD_DIGEST:
        raise ValueError("unapproved build manifest")
    build = json.loads(evidence._build_bytes())
    if json.loads(expected) != build:
        raise ValueError("runtime differs from the qualified numerical build")
    for mod in (identity, campaign):
        if sha(Path(mod.__file__).read_bytes()) != EXTRA_FILES[mod.__name__]:
            raise ValueError(f"unapproved implementation of {mod.__name__}")
    return build, campaign


def collect(root, manifest_sha, build, campaign):
    raw = (root / "manifest.json").read_bytes()
    if sha(raw) != manifest_sha:
        raise ValueError("manifest identity mismatch")
    plan = json.loads(raw)
    if (plan["schema"] != "cbc-campaign/1" or plan["build"] != build or
            plan["backend"] != build["backend"] or plan["tolerance"] != "1/20" or not plan["tasks"]):
        raise ValueError("unqualified or empty production shard")
    folders = {d.name for d in (root / "tasks").iterdir() if d.is_dir()}
    if folders != set(plan["tasks"]):
        raise ValueError("attempt inventory differs from the frozen task set")
    issued = {}
    for key, ident in plan["tasks"].items():
        if key != f"{ident['slot']:06d}_w{ident['window']:02d}_s{ident['channel']}":
            raise ValueError("task identity mismatch")
        for start in sorted((root / "tasks" / key).glob("*/start.json")):
            request = json.loads(start.read_bytes())
            if key in issued or request["key"] != key or request["manifest_sha"] != manifest_sha:
                raise ValueError(f"conflicting attempt for {key}")
            receipt = json.loads((start.parent / "receipt.json").read_bytes())
            status = receipt["status"]
            if status in ("CERTIFIED", "INCOMPLETE"):
                campaign.artifact_status(start.parent / "label.json", receipt, ident, build)
                if status == "CERTIFIED":
                    issued[key] = receipt["sha256"]
            elif status not in ("RESOURCE_INCOMPLETE", "INTERRUPTED"):
                raise ValueError(f"unrecognized attempt status for {key}")
        if key not in issued:
            raise ValueError(f"unissued channel {key}")
    return (json.dumps(issued, indent=2, sort_keys=True) + "\n").encode()


def export(root, destination, manifest_sha, build, campaign):
    # Lock the same inode as the frozen coordinator. Never recover or publish a worker label here.
    with (root / "owner.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        data = collect(root, manifest_sha, build, campaign)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
            try:
                tmp.write(data)
                tmp.flush()
                os.fsync(tmp.fileno())
                os.link(tmp.name, destination)
                campaign.sync_dir(destination.parent)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
    return sha(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path, help="credit_solver directory in the frozen checkout")
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("check")
    ex = actions.add_parser("export")
    ex.add_argument("root", type=Path)
    ex.add_argument("manifest_sha")
    ex.add_argument("destination", type=Path, help="new file in an existing directory")
    args = parser.parse_args()
    build, runtime = preflight(args.kernel)
    if args.action == "check":
        print(json.dumps(dict(commit=build["commit"], build_sha256=BUILD_DIGEST, additional_sha256=EXTRA_FILES)))
    else:
        print(export(args.root, args.destination, args.manifest_sha, build, runtime))
