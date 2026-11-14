"""Small CLI: inspect, generate descriptors, export a query batch, or benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np

from . import Dataset, capabilities


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rindray")
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info", help="show implemented capabilities")
    info.add_argument("--probe-devices", action="store_true", help="explicitly probe optional GPU hardware")
    generate = commands.add_parser("generate", help="write reproducible scene descriptors")
    generate.add_argument("path", type=Path)
    generate.add_argument("--train", type=int, default=8)
    generate.add_argument("--val", type=int, default=2)
    generate.add_argument("--test", type=int, default=2)
    generate.add_argument("--seed", type=int, default=12345)
    generate.add_argument("--boxes", type=int, default=64)
    generate.add_argument("--free-space", action="store_true", help="sample receivers outside solid geometry")
    from .sampling import SAMPLERS
    generate.add_argument("--query-sampler", choices=SAMPLERS, default="uniform_v1")
    generate.add_argument("--query-params", type=json.loads, default=None, help="versioned sampler parameters as a JSON object")
    sample = commands.add_parser("sample", help="export one explicit query batch as NPY arrays")
    sample.add_argument("dataset", type=Path)
    sample.add_argument("scene_id")
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--count", type=int, default=4096)
    sample.add_argument("--realization", type=int, default=0)
    sample.add_argument("--query-id", type=int, default=0)
    sample.add_argument("--model", choices=("binary_any", "count_visible", "sum_weighted_visibility"), default="sum_weighted_visibility")
    bench = commands.add_parser("benchmark", help="run fixed workloads and compare to reference")
    bench.add_argument("--quick", action="store_true")
    bench.add_argument("--large", action="store_true", help="include one-million-point workloads")
    bench.add_argument("--vulkan", action="store_true", help="include Vulkan; use identical sphere-free geometry for every candidate")
    bench.add_argument("--json", type=Path)
    bench.add_argument("--consumer-sleep-ms", type=float, default=0)
    profile = commands.add_parser("profile", help="compare bounded prefetch depths on a fixed dataset stream")
    profile.add_argument("dataset", type=Path)
    profile.add_argument("scene_id")
    profile.add_argument("--count", type=int, default=16384)
    profile.add_argument("--chunk-size", type=int, default=4096)
    profile.add_argument("--consumer-sleep-ms", type=float, default=0)
    profile.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.command == "info":
        print(json.dumps(capabilities(probe_devices=args.probe_devices), indent=2))
    elif args.command == "generate":
        if args.free_space and args.query_sampler != "uniform_v1":
            parser.error("use --free-space or --query-sampler, not both")
        data = Dataset.create(args.path, splits={"train": args.train, "val": args.val, "test": args.test},
                              master_seed=args.seed, box_count=args.boxes,
                              query_sampler="uniform_free_v1" if args.free_space else args.query_sampler,
                              query_params=args.query_params)
        print(data.path / "manifest.json")
    elif args.command == "sample":
        data = Dataset(args.dataset)
        if args.out.exists():
            raise FileExistsError(f"sample output already exists: {args.out}")
        sample = data.sample(args.scene_id, count=args.count, realization_id=args.realization,
                             query_id=args.query_id, model=args.model)
        args.out.mkdir(parents=True)
        np.save(args.out / "points.npy", sample.points, allow_pickle=False)
        np.save(args.out / "responses.npy", sample.labels, allow_pickle=False)
        metadata = {**sample.metadata, "dtype": str(sample.labels.dtype), "shape": list(sample.labels.shape)}
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        print(args.out)
    elif args.command == "profile":
        from .benchmark import benchmark_stream
        report = benchmark_stream(Dataset(args.dataset), args.scene_id, count=args.count,
                                   chunk_size=args.chunk_size, consumer_sleep_ms=args.consumer_sleep_ms)
        print("prefetch total_ms wait_ewma_ms peak_payload_bytes")
        for row in report["records"]:
            stats = row["statistics"]
            print(row["prefetch"], round(row["total_ms"], 3), round(stats["ewma_wait_ms"], 3), stats.get("queue_peak_bytes", 0))
        print("selected_prefetch:", report["selected_prefetch"])
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    else:
        from .benchmark import run_benchmark, table
        sizes = [256, 4096] if args.quick else [256, 4096, 65536]
        if args.large:
            sizes.append(1_000_000)
        report = run_benchmark(sizes=sizes, thread_counts=(1, 2) if args.quick else None,
                               repeats=2 if args.quick else 3, consumer_sleep_ms=args.consumer_sleep_ms,
                               include_vulkan=args.vulkan)
        print(table(report))
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if any(row["unexpected_mismatches"] for row in report["records"]):
            raise SystemExit("benchmark candidates disagree with the reference outside the diagnosed float32 boundary envelope")


if __name__ == "__main__":
    main()
