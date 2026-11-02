import argparse, os
from rind_phase3 import RIND3D, make_dataloader


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", default=os.environ.get("RIND3D_DATA_ROOT", "data/RIND3D-v0.1")
    )
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    with RIND3D(a.root, split="train") as data:
        it = iter(make_dataloader(data, batch_size=2, num_workers=a.workers, seed=0))
        try:
            for _ in range(3):
                b = next(it)
                print(
                    "side",
                    b["side"],
                    "response",
                    tuple(b["observed_response"].shape),
                    "solid",
                    tuple(b["local_solid_mask"].shape),
                )
                assert "structure" not in b
        finally:
            it.close()


if __name__ == "__main__":
    main()
