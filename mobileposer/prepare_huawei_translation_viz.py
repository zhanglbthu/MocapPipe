import json
from argparse import ArgumentParser
from pathlib import Path

import torch


def sequence_stats(seq_path: Path):
    data = torch.load(seq_path, map_location="cpu")
    tran = data["tran_gt"].float().view(-1, 3)
    displacement = (tran[-1] - tran[0]).norm().item()
    path_length = (tran[1:] - tran[:-1]).norm(dim=1).sum().item() if len(tran) > 1 else 0.0
    span = (tran.max(dim=0).values - tran.min(dim=0).values).tolist()
    return {
        "path": seq_path,
        "displacement": displacement,
        "path_length": path_length,
        "span": span,
        "num_frames": int(tran.shape[0]),
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--source-root", type=str, default="/root/autodl-tmp/dataset/raw/Huawei_new")
    parser.add_argument("--output-dir", type=str, default="data/translation_viz/huawei_new/input")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sort-by", type=str, default="displacement", choices=["displacement", "path_length"])
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for subject_dir in sorted(source_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for seq_path in sorted(subject_dir.glob("*.pt"), key=lambda p: int(p.stem)):
            stats = sequence_stats(seq_path)
            rows.append(stats)

    rows.sort(key=lambda item: item[args.sort_by], reverse=True)
    selected = rows[: args.top_k]

    summary = []
    for out_idx, item in enumerate(selected, start=1):
        data = torch.load(item["path"], map_location="cpu")
        viz = {
            "pose_t": data["pose_gt"].float().view(-1, 24, 3, 3),
            "pose_p": data["pose_gt"].float().view(-1, 24, 3, 3),
            "tran_t": data["tran_gt"].float().view(-1, 3),
            "tran_p": data["tran_gt"].float().view(-1, 3),
        }
        torch.save(viz, output_dir / f"{out_idx}.pt")
        summary.append(
            {
                "output_index": out_idx,
                "source": str(item["path"]),
                "displacement": item["displacement"],
                "path_length": item["path_length"],
                "span": item["span"],
                "num_frames": item["num_frames"],
            }
        )

    (output_dir.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
