import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

from config import datasets, model_config, paths
from data import DiffusionPoseDataset
from diffusionposer import DiffusionPoserInference
from evaluate import PoseEvaluator
from evaluate_diffusion import load_diffusion_model, resolve_checkpoint


def build_save_dir(args, checkpoint_path):
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        checkpoint_dir = checkpoint_path.parent
        run_name = checkpoint_dir.parent.name if checkpoint_dir.name == "diffusionposer" else checkpoint_dir.name
        save_dir = (
            paths.eval_output_dir
            / "diffusion_prior"
            / run_name
            / args.mode
            / f"steps_{args.num_steps}"
        )
        if args.mode == "denoise_gt":
            save_dir = save_dir / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def parse_indices(value):
    text = str(value).strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def unconditional_sample(model, num_steps, window_length, seed):
    generator_device = model.device if hasattr(model, "device") else next(model.parameters()).device
    if generator_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    inference = DiffusionPoserInference(model, num_steps=num_steps)
    x_input = torch.zeros(window_length, inference.layout.state_dim, device=generator_device)
    observed_mask = torch.zeros_like(x_input)
    pred_state = inference.inpaint(x_input, observed_mask, num_steps=num_steps)
    pose = inference.state_to_pose(pred_state)
    tran = inference.state_to_tran(pred_state)
    return pred_state, pose, tran


def denoise_gt_sample(model, x0, num_steps, start_timestep, seed):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    x0 = x0.to(device)
    x0_norm = model.normalize_state(x0).unsqueeze(0)
    t0 = torch.tensor([start_timestep], device=device, dtype=torch.long)
    noise = torch.randn_like(x0_norm)
    x_t, _ = model.q_sample(x0_norm, t0, noise=noise)

    inference = DiffusionPoserInference(model, num_steps=num_steps)
    timesteps = torch.linspace(start_timestep, 0, num_steps, device=device).long()

    current = x_t
    for i, timestep in enumerate(timesteps):
        t = torch.full((1,), int(timestep.item()), device=device, dtype=torch.long)
        pred_x0 = model(current, t)
        if i == len(timesteps) - 1:
            current = pred_x0
        else:
            next_t = timesteps[i + 1]
            current = inference._ddim_step(current, pred_x0, timestep, next_t)

    pred_state = model.denormalize_state(current.squeeze(0))
    pose = inference.state_to_pose(pred_state)
    tran = inference.state_to_tran(pred_state)
    return pred_state, pose, tran


def denoise_gt_sequence(model, x0, num_steps, start_timestep, seed, window_length):
    state_chunks = []
    pose_chunks = []
    tran_chunks = []

    for start in range(0, x0.shape[0], window_length):
        end = min(start + window_length, x0.shape[0])
        pred_state, pose, tran = denoise_gt_sample(
            model=model,
            x0=x0[start:end],
            num_steps=num_steps,
            start_timestep=start_timestep,
            seed=seed + start,
        )
        state_chunks.append(pred_state)
        pose_chunks.append(pose)
        tran_chunks.append(tran)

    return (
        torch.cat(state_chunks, dim=0),
        torch.cat(pose_chunks, dim=0),
        torch.cat(tran_chunks, dim=0),
    )


def unconditional_motion_stats(pose, tran):
    pose_delta = pose[1:] - pose[:-1] if pose.shape[0] > 1 else torch.zeros_like(pose[:0])
    root_delta = tran[1:] - tran[:-1] if tran.shape[0] > 1 else torch.zeros_like(tran[:0])
    return {
        "num_frames": int(pose.shape[0]),
        "mean_pose_delta": float(pose_delta.abs().mean().item()) if pose_delta.numel() else 0.0,
        "mean_root_speed": float(root_delta.norm(dim=1).mean().item()) if root_delta.numel() else 0.0,
        "root_path_length": float(root_delta.norm(dim=1).sum().item()) if root_delta.numel() else 0.0,
    }


@torch.no_grad()
def run_unconditional(args, model, save_dir):
    summary = []
    for sample_idx in range(1, args.num_samples + 1):
        seed = args.seed + sample_idx - 1
        pred_state, pose, tran = unconditional_sample(
            model=model,
            num_steps=args.num_steps,
            window_length=args.window_length,
            seed=seed,
        )
        torch.save(
            {
                "pred_state": pred_state.cpu(),
                "pose_p": pose.cpu(),
                "pose_t": pose.cpu(),
                "tran_p": tran.cpu(),
                "tran_t": tran.cpu(),
            },
            save_dir / f"{sample_idx}.pt",
        )
        stats = unconditional_motion_stats(pose.cpu(), tran.cpu())
        stats["sample_id"] = sample_idx
        stats["seed"] = seed
        summary.append(stats)

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


@torch.no_grad()
def run_denoise_gt(args, model, save_dir):
    if args.dataset not in datasets.test_datasets:
        raise ValueError(f"Unknown dataset for denoise_gt: {args.dataset}")

    dataset = DiffusionPoseDataset(fold="test", evaluate=args.dataset, window_length=args.window_length)
    evaluator = PoseEvaluator()
    summary = []
    for seq_idx in parse_indices(args.sequence_indices):
        sample = dataset[seq_idx - 1]
        pred_state, pose_p, tran_p = denoise_gt_sequence(
            model=model,
            x0=sample["x0"],
            num_steps=args.num_steps,
            start_timestep=args.start_timestep,
            seed=args.seed + seq_idx - 1,
            window_length=args.window_length,
        )
        pose_t = sample["pose"]
        tran_t = sample["tran"]
        err = evaluator.eval(pose_p.cpu(), pose_t.cpu(), tran_p=tran_p.cpu(), tran_t=tran_t.cpu())

        torch.save(
            {
                "pred_state": pred_state.cpu(),
                "pose_p": pose_p.cpu(),
                "pose_t": pose_t.cpu(),
                "tran_p": tran_p.cpu(),
                "tran_t": tran_t.cpu(),
            },
            save_dir / f"{seq_idx}.pt",
        )
        summary.append(
            {
                "sequence_id": seq_idx,
                "num_frames": int(pose_t.shape[0]),
                "sip_error_deg": float(err[0, 0].item()),
                "angular_error_deg": float(err[1, 0].item()),
                "masked_angular_error_deg": float(err[2, 0].item()),
                "positional_error_cm": float(err[3, 0].item()),
                "masked_positional_error_cm": float(err[4, 0].item()),
                "mesh_error_cm": float(err[5, 0].item()),
                "jitter_error_100m_s3": float(err[6, 0].item()),
                "distance_error_cm": float(err[7, 0].item()),
            }
        )

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def maybe_render(args, save_dir):
    if not args.render:
        return

    base_cmd = [
        sys.executable,
        str((paths.root_dir / "visualize.py").resolve()),
        "--input-dir",
        str(save_dir),
        "--fps",
        str(args.fps),
        "--stride",
        str(args.stride),
        "--image-width",
        str(args.image_width),
        "--image-height",
        str(args.image_height),
        "--face-stride",
        str(args.face_stride),
        "--subject-spacing",
        str(args.subject_spacing),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.max_frames is not None:
        base_cmd.extend(["--max-frames", str(args.max_frames)])
    if args.visualize_tran:
        base_cmd.append("--visualize-tran")

    subprocess.run(base_cmd, check=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["unconditional", "denoise_gt"], required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)

    parser.add_argument("--dataset", type=str, default="imuposer")
    parser.add_argument("--sequence-indices", type=str, default="1,7,24")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--start-timestep", type=int, default=None)

    parser.add_argument("--state-dim", type=int, default=150)
    parser.add_argument("--window-length", type=int, default=125)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=datasets.fps)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--face-stride", type=int, default=1)
    parser.add_argument("--subject-spacing", type=float, default=1.1)
    parser.add_argument("--visualize-tran", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.start_timestep is None:
        args.start_timestep = args.diffusion_steps - 1

    checkpoint_path = resolve_checkpoint(args)
    save_dir = build_save_dir(args, checkpoint_path)
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Saving prior-eval outputs to: {save_dir}")

    model = load_diffusion_model(checkpoint_path, args).to(model_config.device).eval()

    if args.mode == "unconditional":
        run_unconditional(args, model, save_dir)
    else:
        run_denoise_gt(args, model, save_dir)

    maybe_render(args, save_dir)


if __name__ == "__main__":
    main()
