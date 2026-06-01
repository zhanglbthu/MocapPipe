from argparse import ArgumentParser
from pathlib import Path

import torch


def main():
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/translation_viz/huawei_single/input")
    parser.add_argument("--flip-xz", action="store_true")
    parser.add_argument("--tran-scale", type=float, default=1.0)
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(source, map_location="cpu")
    tran = data["tran_gt"].float().view(-1, 3)
    if args.flip_xz:
        tran = tran.clone()
        tran[:, 0].neg_()
        tran[:, 2].neg_()
    tran = tran * float(args.tran_scale)

    out = {
        "pose_t": data["pose_gt"].float().view(-1, 24, 3, 3),
        "pose_p": data["pose_gt"].float().view(-1, 24, 3, 3),
        "tran_t": tran,
        "tran_p": tran,
    }
    torch.save(out, output_dir / "1.pt")
    print(output_dir / "1.pt")


if __name__ == "__main__":
    main()
