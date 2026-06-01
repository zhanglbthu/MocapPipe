import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="data/toy/imuposer_toy_multimodal/reference_poses.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/toy/imuposer_toy_multimodal/visualize_input",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(source, map_location="cpu")
    out = {
        "pose_t": data["pose_down"].float(),
        "pose_p": data["pose_up"].float(),
        "tran_t": data["tran"].float(),
        "tran_p": data["tran"].float(),
    }

    torch.save(out, output_dir / "1.pt")
    print(output_dir / "1.pt")


if __name__ == "__main__":
    main()
