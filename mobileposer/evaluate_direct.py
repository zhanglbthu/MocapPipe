from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

import articulate as art
from config import datasets, joint_set, model_config, paths
from data import PoseDataset
from evaluate import PoseEvaluator
from models.directposer import DirectPoserNet


def load_direct_model(model_path: str, backbone: str, transformer_kwargs: dict):
    device = model_config.device
    try:
        model = DirectPoserNet.load_from_checkpoint(model_path)
    except Exception:
        model = DirectPoserNet(backbone=backbone, **transformer_kwargs)
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


@torch.no_grad()
def evaluate_pose(model, dataset, save_dir=None):
    device = model_config.device
    xs, ys = zip(*[(imu.to(device), pose.to(device)) for imu, pose, _, _ in dataset])
    evaluator = PoseEvaluator()
    pose_errs = []

    model.eval()
    for idx, (x, pose_t) in enumerate(zip(xs, ys)):
        print(f"Evaluating sample {idx + 1}/{len(xs)}...")
        model.reset()
        pose_t = art.math.r6d_to_rotation_matrix(pose_t).view(-1, 24, 3, 3)

        pose_p = []
        for i in tqdm(range(x.shape[0])):
            pose_p.append(model.forward_frame(x[i]))
        pose_p = torch.stack(pose_p)

        pose_errs.append(evaluator.eval(pose_p, pose_t))

        if save_dir is not None:
            out_path = save_dir / f"{idx + 1}.pt"
            torch.save({"pose_p": pose_p.cpu(), "pose_t": pose_t.cpu()}, out_path)

    errors = torch.stack(pose_errs)
    mean = errors.mean(dim=0)
    std = errors.std(dim=0)
    stats = torch.stack([mean, std], dim=1)
    PoseEvaluator.print(stats)
    return stats


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="imuposer")
    parser.add_argument("--combo", type=str, default="lw_rp_h")
    parser.add_argument("--backbone", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--transformer-d-model", type=int, default=192)
    parser.add_argument("--transformer-nhead", type=int, default=8)
    parser.add_argument("--transformer-num-layers", type=int, default=6)
    parser.add_argument("--transformer-dim-feedforward", type=int, default=768)
    parser.add_argument("--transformer-dropout", type=float, default=0.4)
    args = parser.parse_args()

    if args.dataset not in datasets.test_datasets:
        raise ValueError(f"Test dataset: {args.dataset} not found.")

    transformer_kwargs = {
        "transformer_d_model": args.transformer_d_model,
        "transformer_nhead": args.transformer_nhead,
        "transformer_num_layers": args.transformer_num_layers,
        "transformer_dim_feedforward": args.transformer_dim_feedforward,
        "transformer_dropout": args.transformer_dropout,
    }
    model = load_direct_model(args.model, args.backbone, transformer_kwargs)
    dataset = PoseDataset(fold="test", evaluate=args.dataset)

    save_dir = Path("data") / "eval" / args.dataset / args.combo / "directposer"
    save_dir.mkdir(parents=True, exist_ok=True)
    evaluate_pose(model, dataset, save_dir=save_dir)


if __name__ == "__main__":
    main()
