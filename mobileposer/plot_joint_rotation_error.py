import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import torch


JOINT_NAME_TO_INDEX = {
    "root": 0,
    "pelvis": 0,
    "lhip": 1,
    "rhip": 2,
    "spine1": 3,
    "lknee": 4,
    "rknee": 5,
    "spine2": 6,
    "lankle": 7,
    "rankle": 8,
    "spine3": 9,
    "lfoot": 10,
    "rfoot": 11,
    "neck": 12,
    "lclavicle": 13,
    "rclavicle": 14,
    "head": 15,
    "lshoulder": 16,
    "rshoulder": 17,
    "lelbow": 18,
    "relbow": 19,
    "lwrist": 20,
    "rwrist": 21,
    "lhand": 22,
    "rhand": 23,
}


def load_pose_pair(result_file: str):
    data = torch.load(result_file, map_location="cpu")
    pose_p = data["pose_p"]
    pose_t = data["pose_t"]
    if pose_t.ndim == 3:
        pose_t = pose_t.view(pose_p.shape[0], 24, 3, 3)
    return pose_p, pose_t


def geodesic_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(a.transpose(-1, -2), b)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.rad2deg(torch.acos(cos))


def compute_error_curve(result_file: str, joint_idx: int) -> torch.Tensor:
    pose_p, pose_t = load_pose_pair(result_file)
    return geodesic_deg(pose_p[:, joint_idx], pose_t[:, joint_idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--ours-dir", required=True)
    parser.add_argument("--tic-dir", required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--joint", default="rhip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    joint_idx = JOINT_NAME_TO_INDEX[args.joint.lower()]
    fname = f"{args.sequence}.pt"
    curves: Dict[str, torch.Tensor] = {
        "w/o calibrator": compute_error_curve(os.path.join(args.raw_dir, fname), joint_idx),
        "ours": compute_error_curve(os.path.join(args.ours_dir, fname), joint_idx),
        "tic": compute_error_curve(os.path.join(args.tic_dir, fname), joint_idx),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.figure(figsize=(10, 4.8))
    colors = {
        "w/o calibrator": "#7f7f7f",
        "ours": "#d62728",
        "tic": "#1f77b4",
    }

    for label, curve in curves.items():
        y = curve.numpy()
        x = range(len(y))
        mean = float(curve.mean())
        plt.plot(x, y, label=f"{label} ({mean:.2f}°)", linewidth=1.6, color=colors[label])

    plt.xlabel("Frame")
    plt.ylabel("Rotation Error (deg)")
    plt.title(args.title or f"Sequence {args.sequence} {args.joint.upper()} Rotation Error")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
