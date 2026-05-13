import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

import articulate as art
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import Meshes
from config import datasets, paths


def natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]


def list_sequence_files(input_dir, sequence=None):
    if sequence is not None:
        seq_path = input_dir / f"{sequence}.pt"
        if not seq_path.exists():
            raise FileNotFoundError(f"Sequence file not found: {seq_path}")
        return [seq_path]

    return sorted([p for p in input_dir.glob("*.pt") if p.stem.isdigit()], key=natural_key)


@torch.no_grad()
def pose_to_vertices(body_model, pose, batch_size=256):
    pose = pose.float().view(-1, 24, 3, 3)
    tran = torch.zeros((pose.shape[0], 3), dtype=torch.float32)
    vertices = []

    for start in range(0, pose.shape[0], batch_size):
        end = min(start + batch_size, pose.shape[0])
        _, _, vert = body_model.forward_kinematics(
            pose=pose[start:end],
            tran=tran[start:end],
            calc_mesh=True,
        )
        vertices.append(vert.cpu())

    return torch.cat(vertices, dim=0).numpy()


def get_subject_colors(include_gt=False):
    colors = matplotlib.colormaps["tab10"].colors
    base = [
        tuple(float(x) for x in colors[0]) + (1.0,),
        tuple(float(x) for x in colors[1]) + (1.0,),
    ]
    if include_gt:
        base.append(tuple(float(x) for x in colors[2]) + (1.0,))
    return base


def prepare_subject_vertices(vertex_groups, subject_spacing):
    vertex_groups = [vertices.copy() for vertices in vertex_groups]
    flat_vertices = [vertices.reshape(-1, 3) for vertices in vertex_groups]
    all_vertices = np.concatenate(flat_vertices, axis=0)
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2.0

    for vertices in vertex_groups:
        vertices -= center

    offsets = np.linspace(
        -subject_spacing * (len(vertex_groups) - 1) / 2.0,
        subject_spacing * (len(vertex_groups) - 1) / 2.0,
        num=len(vertex_groups),
        dtype=np.float32,
    )
    for vertices, offset in zip(vertex_groups, offsets):
        vertices[:, :, 0] += float(offset)
    return vertex_groups


class AITViewerCompareRenderer:
    def __init__(self, faces, width, height, fps, face_stride=1, subject_spacing=1.4):
        self.faces = faces[::face_stride].astype(np.int32)
        self.width = width
        self.height = height
        self.fps = fps
        self.subject_spacing = subject_spacing

    def _make_meshes(self, vertices_lstm, vertices_transformer, vertices_gt=None):
        vertex_groups = [vertices_lstm, vertices_transformer]
        names = ["LSTM", "Transformer"]
        if vertices_gt is not None:
            vertex_groups.append(vertices_gt)
            names.append("GT")

        vertex_groups = prepare_subject_vertices(vertex_groups, self.subject_spacing)
        colors = get_subject_colors(include_gt=vertices_gt is not None)
        meshes = []
        for vertices, color, name in zip(vertex_groups, colors, names):
            mesh = Meshes(vertices, self.faces, name=name, flat_shading=False, draw_edges=False)
            mesh.color = color
            meshes.append(mesh)
        return meshes

    def render_video(self, vertices_lstm, vertices_transformer, output_path, vertices_gt=None):
        nodes = self._make_meshes(vertices_lstm, vertices_transformer, vertices_gt=vertices_gt)
        viewer = HeadlessRenderer(size=(self.width, self.height), backend="egl")
        try:
            viewer.auto_set_camera_target = False
            for node in nodes:
                viewer.scene.add(node)
            viewer._init_scene()
            viewer.scene.fps = self.fps
            viewer.playback_fps = self.fps
            viewer.export_fps = self.fps
            viewer.scene.camera.target = np.array([0.0, 0.9, 0.0], dtype=np.float32)
            viewer.scene.camera.position = np.array([0.0, 0.2, 5.6], dtype=np.float32)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            viewer.save_video(video_dir=str(output_path), output_fps=self.fps)
        finally:
            if hasattr(viewer, "close"):
                viewer.close()


def visualize_sequence(seq_path_a, seq_path_b, output_path, body_model, args):
    data_a = torch.load(seq_path_a, map_location="cpu")
    data_b = torch.load(seq_path_b, map_location="cpu")
    if "pose_p" not in data_a or "pose_p" not in data_b or "pose_t" not in data_a:
        raise KeyError("Sequence files must contain pose_p, and the first input must contain pose_t")

    vertices_a = pose_to_vertices(body_model, data_a["pose_p"], batch_size=args.batch_size)
    vertices_b = pose_to_vertices(body_model, data_b["pose_p"], batch_size=args.batch_size)
    vertices_gt = pose_to_vertices(body_model, data_a["pose_t"], batch_size=args.batch_size) if args.include_gt else None

    n_frames = min(len(vertices_a), len(vertices_b))
    if vertices_gt is not None:
        n_frames = min(n_frames, len(vertices_gt))
    frame_ids = list(range(0, n_frames, args.stride))
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]

    vertices_a = vertices_a[frame_ids]
    vertices_b = vertices_b[frame_ids]
    if vertices_gt is not None:
        vertices_gt = vertices_gt[frame_ids]

    renderer = AITViewerCompareRenderer(
        body_model.face.astype(np.int32),
        width=args.image_width,
        height=args.image_height,
        fps=args.fps,
        face_stride=args.face_stride,
        subject_spacing=args.subject_spacing,
    )
    renderer.render_video(vertices_a, vertices_b, output_path, vertices_gt=vertices_gt)
    print(f"Saved {output_path}")


def resolve_default_output_dir(dir_a, dir_b, include_gt):
    suffix = f"{dir_a.name}_vs_{dir_b.name}"
    if include_gt:
        suffix += "_vs_gt"
    return paths.video_output_dir / "imuposer" / "lw_rp_h" / suffix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir-a", type=str, required=True, help="First result directory, e.g. LSTM.")
    parser.add_argument("--input-dir-b", type=str, required=True, help="Second result directory, e.g. Transformer.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sequence", type=int, default=None)
    parser.add_argument("--fps", type=int, default=datasets.fps)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--face-stride", type=int, default=1)
    parser.add_argument("--subject-spacing", type=float, default=1.4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--include-gt", action="store_true")
    args = parser.parse_args()

    input_dir_a = Path(args.input_dir_a)
    input_dir_b = Path(args.input_dir_b)
    output_dir = Path(args.output_dir) if args.output_dir else resolve_default_output_dir(input_dir_a, input_dir_b, args.include_gt)

    seq_files_a = list_sequence_files(input_dir_a, args.sequence)
    seq_files_b = list_sequence_files(input_dir_b, args.sequence)
    if len(seq_files_a) != len(seq_files_b):
        raise ValueError("Input directories must contain the same number of numeric .pt files")

    stems_a = [p.stem for p in seq_files_a]
    stems_b = [p.stem for p in seq_files_b]
    if stems_a != stems_b:
        raise ValueError("Input directories must contain matching sequence ids")

    if args.sequence is None and len(seq_files_a) > 1:
        base_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--input-dir-a",
            str(input_dir_a),
            "--input-dir-b",
            str(input_dir_b),
            "--output-dir",
            str(output_dir),
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
        if args.include_gt:
            base_cmd.append("--include-gt")

        for seq_path in seq_files_a:
            seq_cmd = base_cmd + ["--sequence", seq_path.stem]
            subprocess.run(seq_cmd, check=True)
        return

    body_model = art.ParametricModel(paths.smpl_file)
    for seq_path_a, seq_path_b in zip(seq_files_a, seq_files_b):
        output_path = output_dir / f"{seq_path_a.stem}.mp4"
        visualize_sequence(seq_path_a, seq_path_b, output_path, body_model, args)


if __name__ == "__main__":
    main()
