import argparse, os
from rind_phase3 import RIND3D


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", default=os.environ.get("RIND3D_DATA_ROOT", "data/RIND3D-v0.1")
    )
    p.add_argument("--index", type=int, default=0)
    a = p.parse_args()
    with RIND3D(a.root, split=None) as data:
        sample = data[a.index]
        print("Student keys:", sorted(sample))
        for key in ("observed_response", "local_solid_mask"):
            print(
                key,
                "packed ZYX:",
                sample[key].data.shape,
                "dense ZYX:",
                sample[key].unpack().shape,
            )
        print("World XYZ origin:", data.window(a.index).world_origin)


if __name__ == "__main__":
    main()
