import torch
from argparse import ArgumentParser
from contextlib import redirect_stdout
from pathlib import Path
from tqdm import tqdm

from config import datasets, model_config
from data import DiffusionPoseDataset
from diffusionposer import DiffusionPoser, DiffusionPoserConfig, DiffusionPoserInference
from evaluate import PoseEvaluator
from utils.file_utils import get_best_checkpoint


def resolve_checkpoint(args):
    if args.model:
        return Path(args.model)

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        checkpoint_dir = run_dir if run_dir.name == "diffusionposer" else run_dir / "diffusionposer"
    else:
        raise ValueError("Provide either --model, --checkpoint-dir, or --run-dir.")

    best_checkpoint = get_best_checkpoint(str(checkpoint_dir))
    if best_checkpoint is None:
        raise FileNotFoundError(f"No validation checkpoints found in {checkpoint_dir}")
    checkpoint_path = checkpoint_dir / best_checkpoint
    print(f"Using best checkpoint: {checkpoint_path}")
    return checkpoint_path


def build_save_dir(args, checkpoint_path):
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        checkpoint_dir = checkpoint_path.parent
        run_name = checkpoint_dir.parent.name if checkpoint_dir.name == "diffusionposer" else checkpoint_dir.name
        save_dir = (
            Path("data")
            / "eval"
            / "diffusionposer"
            / args.dataset
            / args.combo
            / run_name
            / f"steps_{args.num_steps}"
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def load_diffusion_model(checkpoint_path, args):
    config = DiffusionPoserConfig(
        state_dim=args.state_dim,
        window_length=args.window_length,
        diffusion_steps=args.diffusion_steps,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    model = DiffusionPoser(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model


@torch.no_grad()
def evaluate_diffusion(model, dataset, combo, save_dir, num_steps=10, max_samples=None):
    device = model_config.device
    model = model.to(device)
    model.eval()

    inference = DiffusionPoserInference(model, num_steps=num_steps)
    evaluator = PoseEvaluator()
    pose_errs = []

    sample_count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in tqdm(range(sample_count), desc=f"Evaluating {combo}"):
        sample = dataset[idx]
        x0 = sample['x0'].to(device)
        pose_t = sample['pose'].to(device)
        tran_t = sample['tran'].to(device)

        pred_state = inference.autoregressive(x0, combo=combo, num_steps=num_steps)
        pose_p = inference.state_to_pose(pred_state)
        tran_p = inference.state_to_tran(pred_state)
        pose_errs.append(evaluator.eval(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t))

        torch.save(
            {
                'pred_state': pred_state.cpu(),
                'pose_p': pose_p.cpu(),
                'pose_t': pose_t.cpu(),
                'tran_p': tran_p.cpu(),
                'tran_t': tran_t.cpu(),
            },
            save_dir / f"{idx + 1}.pt",
        )

    errors = torch.stack(pose_errs)
    summary = errors.mean(dim=0)
    PoseEvaluator.print(summary)

    with open(save_dir / "log.txt", "w") as f:
        with redirect_stdout(f):
            PoseEvaluator.print(summary)
        for i, err in enumerate(errors):
            print(f"Sample {i + 1}", file=f)
            PoseEvaluator.print_single(err, file=f)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="huawei")
    parser.add_argument("--combo", type=str, default="lw_rp")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)

    parser.add_argument("--state-dim", type=int, default=171)
    parser.add_argument("--window-length", type=int, default=125)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    args = parser.parse_args()

    if args.dataset not in datasets.test_datasets:
        raise ValueError(f"Test dataset {args.dataset} not found.")

    checkpoint_path = resolve_checkpoint(args)
    save_dir = build_save_dir(args, checkpoint_path)
    print(f"Saving evaluation outputs to: {save_dir}")

    model = load_diffusion_model(checkpoint_path, args)
    dataset = DiffusionPoseDataset(fold='test', evaluate=args.dataset, window_length=args.window_length)
    evaluate_diffusion(model, dataset, args.combo, save_dir, num_steps=args.num_steps, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
