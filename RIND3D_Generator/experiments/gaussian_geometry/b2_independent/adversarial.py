"""Numerical cases for independently discovered labels, including exact contact."""
import argparse
from pathlib import Path
import time

import numpy as np

from ..b2_max_cert.adversarial import fixtures as earlier_fixtures
from ..models import Kernel, KernelScene, Threshold, record
from ..queries import FieldOracle, Limits
from ..study import atomic_json
from .benchmark import LABELS, METHODS, config, native_engine, native_settings
from .build import DEFAULT, extension


def fixtures():
    rows = earlier_fixtures()
    thin = np.array([[1., 1., 0.], [1., 1. + 2**-48, 0.], [0., 0., 2**-5]])
    scene = KernelScene("gaussian_sum", [
        Kernel([.5, .5, .5], thin, .7),
        Kernel([.5, .5, .5 + 2**-20], np.eye(3) * 2**-20, .2),
    ], Threshold("binary64", .6))
    rows += [
        ("ill_conditioned_mixed_scales_HIT", scene, [.5, .5, .0625], [.5, .5, .9375], "HIT"),
        ("ill_conditioned_mixed_scales_MISS", scene, [.625, .375, .0625], [.625, .375, .9375], "MISS"),
    ]
    tiny = np.nextafter(0., 1.)
    scene = KernelScene("gaussian_sum", [
        Kernel([0, 0, 0], np.eye(3), tiny),
        Kernel([.125, 0, 0], np.eye(3) * 2**-8, tiny),
    ], Threshold("binary64", tiny))
    rows += [
        ("subnormal_weight_exact_closed_end", scene, [-1, 0, 0], [0, 0, 0], "HIT"),
        ("subnormal_weights_strict_MISS", scene, [-2, 1, 0], [-1, 1, 0], "MISS"),
    ]
    return rows


def run(native, output):
    module = extension(native)
    settings = {m: native_settings(module, m, config()) for m in METHODS}
    for opts, _ in settings.values():
        opts.max_nodes = 2048
        opts.query_seconds = .4
    rows = []
    for name, scene, source, receiver, expected in fixtures():
        engine = native_engine(module, scene)
        variants = {}
        for method, options in settings.items():
            result = engine.ray(source, receiver, *options)
            result["status"] = LABELS[result["status"]]
            if result["status"] not in (expected, "UNKNOWN"):
                raise AssertionError((name, method, result))
            variants[method] = result
        start = time.perf_counter()
        arb = FieldOracle(scene, Limits(max_nodes=8192, max_depth=40, seconds=.5))
        d = arb.segment(source, receiver)
        if d.status in ("HIT", "MISS") and d.status != expected:
            raise AssertionError((name, "Arb", d))
        variants["Arb"] = {"status": d.status, "reason": d.reason, "nodes": d.nodes,
                           "wall_seconds": time.perf_counter() - start}
        rows.append({
            "name": name, "expected": expected, "canonical_scene": record(scene),
            "source_hex": [float(x).hex() for x in source],
            "receiver_hex": [float(x).hex() for x in receiver], "variants": variants,
        })
    atomic_json(output, {"build": module.study_identity, "fixtures": rows,
                         "no_opposite_labels": True, "ray_node_budget": 2048,
                         "ray_seconds": .4, "Arb_seconds": .5})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=DEFAULT / "native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.native, args.output)
