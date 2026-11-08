"""Check sampled actual cache hits and axial interior points with Arb."""
import argparse
from pathlib import Path
import time

import numpy as np

from ..b2_angular.preservation import sha
from ..b2_miss.bench import load_inputs, read
from ..queries import FieldOracle, Limits
from ..study import atomic_json
from .freeze import OUT


def run(args):
    rng = np.random.default_rng(202611072248)
    records = []
    for tile_id in ("14-05","14-13","14-14","new-0-boundary","new-62-boundary","new-75-boundary"):
        folder = args.output.parent / (args.variant + ("-heldout" if tile_id.startswith("new-") else ""))
        archive = folder/(tile_id + ".npz")
        tile = read(archive.with_suffix(".json"))
        if sha(archive) != tile["archive_SHA256"] or not tile["complete"]:
            raise ValueError("Invalid measured tile")
        scene, source = load_inputs(tile)
        n, grid = tile["side"], tile["grid"]
        with np.load(archive) as data:
            seed = data["hit_propagated_seed"].ravel()
        used = np.flatnonzero(seed >= 0)
        oracle = FieldOracle(scene, Limits(max_nodes=8192, max_depth=40, seconds=.5))

        def point(i):
            xyz = np.array(np.unravel_index(i, (n,)*3))[::-1]
            return (xyz + tile["origin"] + .5)/grid

        for i in np.sort(rng.choice(used, min(len(used),4), replace=False)):
            receiver, origin = point(i), point(seed[i])
            for location, p in (("center",receiver),("axial_interior",(receiver+origin)/2)):
                t = time.perf_counter()
                answer = oracle.segment(source,p)
                row = {"tile":tile_id, "receiver_index":int(i), "seed_index":int(seed[i]),
                       "kind":location, "receiver_hex":[float(x).hex() for x in p],
                       "status":answer.status, "reason":answer.reason, "nodes":answer.nodes,
                       "seconds":time.perf_counter()-t}
                if answer.status not in ("HIT","UNKNOWN","BUDGET_EXHAUSTED"):
                    raise AssertionError(("Arb contradicts a certified HIT interval",row))
                records.append(row)
    atomic_json(args.output,{"variant":args.variant,"checks":records})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant",default="P_JHA")
    p.add_argument("--output",type=Path,default=OUT/"arb-controls.json")
    a=p.parse_args()
    if a.output.exists():
        raise ValueError("Keep the measured checks")
    run(a)
