"""Freeze and replay exact scalar-residual tile workloads."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy import ndimage

from ..b2_angular.preservation import sha
from ..b2_independent.benchmark import INPUT, config, identity_file, read_input, trace_summary
from ..b2_precision.bench import settings
from ..b2_precision.build import extension
from ..b2_precision.tiles import digest
from ..study import atomic_json

OUT = Path("outputs/gaussian-b2-miss-v1")
PREVIOUS = Path("outputs/gaussian-b2-precision-v1")
F_BUILD = PREVIOUS / "tile-native"
GRID = 256


def read(path):
    return json.loads(path.read_text())


def load_inputs(row):
    scene, source, identity = read_input(Path(row["input"]) / row["case"])
    if row["geometry"] != identity:
        raise ValueError("Frozen input identity differs")
    return scene, source


def freeze(path):
    if path.exists():
        raise ValueError("Keep the original frozen plan")
    frozen = read(PREVIOUS / "scaling256/F/probe.json")
    rows = []
    dev = {(64, 192, 0), (0, 128, 64), (64, 0, 0)}
    for i, tile in enumerate(frozen["rows"]):
        origin = tile["origin_xyz"]
        reference = PREVIOUS / "scaling256/F" / ("-".join(map(str, origin)) + ".npz")
        if sha(reference) != tile["archive_SHA256"] or not tile["complete"]:
            raise ValueError("Frozen probe archive differs")
        rows.append({"id": f"14-{i:02}", "case": "gaussian_sum-014", "input": str(INPUT),
                     "geometry": frozen["input"], "grid": GRID, "side": 64, "origin": origin,
                     "split": "development" if tuple(origin) in dev else "evaluation",
                     "prior_archive": str(reference), "prior_archive_SHA256": tile["archive_SHA256"],
                     "prior_seconds": tile["tile_seconds"]})
    # Freeze two receiver neighborhoods per case using a fixed draw from the
    # lower half and upper quartile of prior residual MISS costs. The previous
    # traces select test inputs only; the candidate renderer never sees them.
    rng = np.random.default_rng(202611071529)
    specs = (
        (7, 128, INPUT, PREVIOUS / "fields128-case7"),
        (60, 64, Path("outputs/gaussian-b2-independent-v1/heldout"), PREVIOUS / "holdout64"),
        (63, 64, Path("outputs/gaussian-b2-independent-v1/heldout"), PREVIOUS / "holdout64"),
        (72, 64, PREVIOUS / "fresh-inputs", PREVIOUS / "fresh64"),
    )
    for ordinal, n, root, folder in specs:
        case = f"gaussian_sum-{ordinal:03}"
        _, _, identity = read_input(root / case)
        ref = folder / f"{case}-{n}-F.npz"
        rr = read(ref.with_suffix(".json"))
        if (sha(ref) != rr["archive_SHA256"] or not rr["complete"]
                or any(rr[k] != identity[k] for k in ("source_SHA256", "geometry_id"))):
            raise ValueError("Selection trace identity differs")
        with np.load(ref) as arrays:
            hits = np.flatnonzero(arrays["evaluation_path"].ravel() == 6)
            times = arrays["ray_nanoseconds"].ravel()[hits]
        ranked = hits[np.argsort(times, kind="stable")]
        choices = (ranked[:len(ranked)//2], ranked[3*len(ranked)//4:])
        for k, ids in enumerate(choices):
            idx = int(rng.choice(ids))
            z, y, x = np.unravel_index(idx, (n,) * 3)
            center = (np.array([x, y, z]) * 2 + 1) * GRID // (2 * n)
            origin = np.clip(center - [13, 17, 11], 0, GRID-32).tolist()
            rows.append({"id": f"{ordinal}-{'ordinary' if k==0 else 'hot'}", "case": case,
                         "input": str(root), "geometry": identity, "grid": GRID, "side": 32,
                         "origin": origin, "split": "evaluation",
                         "selection_archive": str(ref), "selection_archive_SHA256": sha(ref),
                         "selection_index_zyx": idx, "selection_population": len(ids)})
    atomic_json(path, {"schema": "b2-miss-tile-plan-v1",
                       "historical_probe_SHA256": sha(PREVIOUS / "scaling256/F/probe.json"),
                       "F_build": read(F_BUILD / "identity.json"), "tiles": rows})


def clustering(path, nanoseconds):
    """Face connectivity and adjacency among the top quartile of scalar MISS."""
    miss = path == 6
    times = nanoseconds[miss]
    if not len(times):
        return None
    cutoff = float(np.quantile(times, .75))
    hot = miss & (nanoseconds >= cutoff)
    labels, n = ndimage.label(hot)
    sizes = np.bincount(labels[hot])
    sizes[0] = 0
    hot_contacts = miss_contacts = touch = 0
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        hot_contacts += int(np.count_nonzero(hot[a] & hot[b]))
        miss_contacts += int(np.count_nonzero(miss[a] & miss[b]))
        touch += int(np.count_nonzero(hot[a] & miss[b]) + np.count_nonzero(miss[a] & hot[b]))
    hot_n = int(hot.sum())
    ratio = (hot_n * (hot_n - 1) / (len(times) * (len(times) - 1))
             if len(times) > 1 else 0.)
    return {
        "threshold_nanoseconds": cutoff, "hot_count": hot_n, "components": n,
        "largest_component": int(sizes.max()), "largest_hot_cost_fraction":
            float(nanoseconds[labels == sizes.argmax()].sum() / nanoseconds[hot].sum()),
        "hot_face_pairs": hot_contacts, "MISS_face_pairs": miss_contacts,
        "randomized_expected_hot_pairs": miss_contacts * ratio,
        "neighbor_hot_fraction_given_hot_to_MISS_face": 2 * hot_contacts / touch if touch else None,
        "hot_fraction": hot_n / len(times),
        "hot_fraction_of_original_MISS_time": float(nanoseconds[hot].sum()/times.sum()),
    }


def make_session(module, row, mode, independent=True, trace=True):
    scene, source = load_inputs(row)
    start = time.perf_counter()
    engine = module.Engine([k.center.tolist() for k in scene.kernels],
                           [k.matrix.ravel().tolist() for k in scene.kernels],
                           [k.weight for k in scene.kernels], scene.threshold.value,
                           scene.threshold.kind == "exp", independent)
    opts, cfg = settings(module, "F" if independent else "A", config())
    cfg.trace = trace
    if mode:
        cfg.study.miss_mode = mode
    session = engine.session(source, row["grid"], opts, cfg, True)
    return session, time.perf_counter() - start


def compare(bits, archive, expected_sha=None):
    if expected_sha is None:
        expected_sha = read(archive.with_suffix(".json"))["archive_SHA256"]
    if sha(archive) != expected_sha:
        raise ValueError(f"Reference archive checksum differs: {archive}")
    with np.load(archive) as base:
        if not base["valid"].all():
            raise ValueError("Incomplete comparator")
        common = bits["valid"]
        if not all(np.array_equal(bits[k][common], base[k][common]) for k in ("response", "solid")):
            raise AssertionError(f"Physical tile bits differ from {archive}")


def run(args):
    if args.primary and args.mode:
        raise ValueError("The primary control cannot use MISS propagation")
    module = extension(args.native)
    plan = read(args.plan)
    selected = set(args.ids or [r["id"] for r in plan["tiles"]])
    rows = [r for r in plan["tiles"] if r["id"] in selected]
    if {r["id"] for r in rows} != selected:
        raise ValueError("Unknown frozen tile id")
    identity_file(args.output / "identity.json", {
        "build": module.study_identity, "plan_SHA256": sha(args.plan),
        "method": "P" if args.primary else args.mode,
        "baseline": str(args.reference) if args.reference else None,
        "ids": [r["id"] for r in rows], "driver_SHA256": sha(Path(__file__)),
        "options": config(),
    })
    results, setups = [], []
    session = None
    last = None
    wall, cpu = time.perf_counter(), time.process_time()
    for row in rows:
        if row["case"] != last:
            session, elapsed = make_session(module, row, args.mode, independent=not args.primary)
            setups.append({"case": row["case"], "seconds": elapsed, "info": session.info(),
                           "work": session.preparation_work()})
            last = row["case"]
        started, started_cpu = time.perf_counter(), time.process_time()
        tile = session.tile(row["origin"], row["side"], args.seconds)
        elapsed = time.perf_counter() - started
        cpu_s = time.process_time() - started_cpu
        bits = {k: np.asarray(tile.pop(k), dtype=bool) for k in ("response", "solid", "valid")}
        for k in ("evaluation_path", "ray_nanoseconds", "point_nanoseconds"):
            bits[k] = np.asarray(tile.pop(k))
        if "propagated_seed" in tile:
            bits["propagated_seed"] = np.asarray(tile.pop("propagated_seed"))
        complete = bool(bits["valid"].all())
        if complete != tile["complete"]:
            raise AssertionError("Tile completeness disagrees with mask")
        if row.get("prior_archive"):
            compare(bits, Path(row["prior_archive"]), row["prior_archive_SHA256"])
        if args.reference:
            compare(bits, args.reference / (row["id"] + ".npz"))
        archive = args.output / (row["id"] + ".npz")
        np.savez_compressed(archive, **bits)
        record = {**row, **tile, "tile_call_seconds": elapsed, "tile_CPU_seconds": cpu_s,
                  "valid_voxels": int(bits["valid"].sum()), "archive_SHA256": sha(archive),
                  "physical_bits_SHA256": digest(bits) if complete else None,
                  "trace": trace_summary(bits),
                  "clustering": clustering(bits["evaluation_path"], bits["ray_nanoseconds"])}
        if "propagated_seed" in bits and args.reference:
            with np.load(args.reference / (row["id"] + ".npz")) as old:
                used = bits["propagated_seed"] >= 0
                t = old["ray_nanoseconds"][old["evaluation_path"] == 6]
                covered = old["ray_nanoseconds"][used]
                if not np.all(old["evaluation_path"][used] == 6):
                    raise AssertionError("Propagated receiver was not an original MISS")
                record["saved_scalar"] = {
                    "original_MISS_nanoseconds": int(covered.sum()),
                    "original_MISS_fraction": float(covered.sum()/t.sum()) if t.sum() else 0.,
                    "count": int(used.sum()),
                    "original_top_quartile_count": int((covered >= np.quantile(t, .75)).sum()) if len(t) else 0,
                    "original_above_median_count": int((covered >= np.median(t)).sum()) if len(t) else 0,
                }
            remaining = (bits["evaluation_path"] == 6) & ~used
            record["new_scalar"] = {
                "MISS_nanoseconds_including_proof_work": int(bits["ray_nanoseconds"][remaining].sum()),
                "MISS_count": int(remaining.sum()),
                "propagated_lookup_nanoseconds": int(bits["ray_nanoseconds"][used].sum()),
            }
        atomic_json(args.output / (row["id"] + ".json"), record)
        results.append(record)
        atomic_json(args.output / "summary.json", {
            "records": results, "setups": setups, "elapsed_wall_seconds": time.perf_counter()-wall,
            "CPU_seconds": time.process_time()-cpu, "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        })
        print(row["id"], row["origin"], record["valid_voxels"], round(elapsed, 3),
              record["trace"]["MISS"], record["trace"]["HIT"], flush=True)
        if not complete:
            raise RuntimeError((row["id"], tile["reason"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("freeze", "run"))
    p.add_argument("--plan", type=Path, default=OUT / "tile-plan.json")
    p.add_argument("--native", type=Path, default=F_BUILD)
    p.add_argument("--output", type=Path, default=OUT / "baseline")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--mode", type=int, default=0)
    p.add_argument("--primary", action="store_true")
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--reference", type=Path)
    args = p.parse_args()
    if args.task == "freeze":
        freeze(args.plan)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "summary.json").exists():
            raise ValueError("Use a new measured run directory")
        run(args)
