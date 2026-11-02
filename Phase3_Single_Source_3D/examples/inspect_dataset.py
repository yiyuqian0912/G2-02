import argparse, json, os
from rind_phase3 import RIND3D


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", default=os.environ.get("RIND3D_DATA_ROOT", "data/RIND3D-v0.1")
    )
    a = p.parse_args()
    with RIND3D(a.root, split=None) as data:
        print(json.dumps(data.metadata(), indent=2))
        for i in (0, 85, 170):
            sample = data[i]
            print(
                "Window",
                sample["window_id"],
                "XYZ",
                sample["origin"],
                "side",
                sample["side"],
                "counts",
                data.statistics(i),
            )


if __name__ == "__main__":
    main()
