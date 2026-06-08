#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.aitviewer_render import render_meshes_side_by_side
from scripts.eval.eval_imu_streaming import load_pose_evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Render GT-only IMUPoser test sequences with sequence/frame labels.")
    parser.add_argument(
        "--eval-pt",
        type=str,
        default="/root/autodl-tmp/dataset/processed/eval/imuposer_test.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/project/GENMO/outputs/imuposer_test_gt_renders",
    )
    parser.add_argument(
        "--seq-ids",
        type=int,
        nargs="*",
        default=None,
        help="1-based sequence ids to render. Default: all sequences.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--subject-spacing", type=float, default=1.0)
    parser.add_argument(
        "--use-gt-translation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use GT translation. Default is pose-only rendering with zero translation.",
    )
    return parser.parse_args()


def overlay_sequence_text(input_video, output_video, seq_id, total_frames):
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open rendered video: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_video}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            label = f"Seq {seq_id:02d}  Frame {frame_idx:04d}/{total_frames:04d}"
            cv2.rectangle(frame, (24, 18), (460, 74), (255, 255, 255), thickness=-1)
            cv2.putText(
                frame,
                label,
                (36, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


@torch.no_grad()
def render_sequence_gt(body_model, pose_t, tran_t, seq_id, output_path, fps, width, height, subject_spacing):
    if pose_t.ndim == 3:
        pose_t = pose_t.view(-1, 24, 3, 3)
    if tran_t is None:
        tran_t = torch.zeros(pose_t.shape[0], 3, dtype=pose_t.dtype, device=pose_t.device)

    _, _, vertices = body_model.forward_kinematics(pose_t, tran=tran_t, calc_mesh=True)
    vertices = vertices.unsqueeze(0).cpu().numpy()
    faces = body_model.face.cpu().numpy() if torch.is_tensor(body_model.face) else body_model.face

    with tempfile.TemporaryDirectory(prefix=f"seq{seq_id:02d}_gt_") as tmpdir:
        tmp_video = Path(tmpdir) / f"seq{seq_id:02d}_raw.mp4"
        render_meshes_side_by_side(
            vertices=vertices,
            faces=faces,
            output_path=tmp_video,
            width=width,
            height=height,
            fps=fps,
            subject_spacing=subject_spacing,
            xvfb_width=width,
            xvfb_height=height,
        )
        overlay_sequence_text(tmp_video, output_path, seq_id, pose_t.shape[0])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.eval_pt, map_location="cpu")
    _, body_model = load_pose_evaluator()

    num_sequences = len(data["pose"])
    seq_ids = args.seq_ids if args.seq_ids is not None else list(range(1, num_sequences + 1))

    for seq_id in tqdm(seq_ids, desc="render_gt_sequences"):
        idx = seq_id - 1
        pose_t = data["pose"][idx].float().view(-1, 24, 3, 3)
        if args.use_gt_translation:
            tran_t = data["tran"][idx].float()
        else:
            tran_t = torch.zeros(pose_t.shape[0], 3, dtype=pose_t.dtype)

        seq_dir = output_dir / f"seq{seq_id:02d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        video_path = seq_dir / f"seq{seq_id:02d}_gt.mp4"
        render_sequence_gt(
            body_model=body_model,
            pose_t=pose_t,
            tran_t=tran_t,
            seq_id=seq_id,
            output_path=video_path,
            fps=args.fps,
            width=args.width,
            height=args.height,
            subject_spacing=args.subject_spacing,
        )

    manifest_path = output_dir / "manifest.txt"
    with manifest_path.open("w") as handle:
        for seq_id in seq_ids:
            handle.write(f"seq{seq_id:02d}: {output_dir / f'seq{seq_id:02d}' / f'seq{seq_id:02d}_gt.mp4'}\n")


if __name__ == "__main__":
    main()
