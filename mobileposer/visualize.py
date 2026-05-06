import argparse
import re
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

    return sorted(
        [p for p in input_dir.glob("*.pt") if p.stem.isdigit()],
        key=natural_key,
    )


def to_local_pose(body_model, pose):
    pose = pose.float().view(-1, 24, 3, 3)
    return body_model.inverse_kinematics_R(pose)


@torch.no_grad()
def pose_to_vertices(body_model, pose, tran, batch_size=256):
    pose = to_local_pose(body_model, pose)
    tran = tran.float().view(-1, 3)
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


def resolve_output_dir(input_dir, output_dir):
    if output_dir is not None:
        return Path(output_dir)

    repo_root = paths.root_dir
    eval_root = paths.eval_output_dir.resolve()
    input_abs = input_dir.resolve()
    if input_abs.is_relative_to(eval_root):
        relative = input_abs.relative_to(eval_root)
        return paths.video_output_dir / relative

    return paths.video_output_dir / input_dir.name


def maybe_zero_translation(pose, tran, visualize_tran):
    if visualize_tran:
        return tran
    return torch.zeros((pose.shape[0], 3), dtype=torch.float32)


def get_subject_colors():
    colors = matplotlib.colormaps["tab10"].colors
    return [
        tuple(float(x) for x in colors[0]) + (1.0,),
        tuple(float(x) for x in colors[1]) + (1.0,),
    ]


def prepare_subject_vertices(gt_vertices, pred_vertices, subject_spacing):
    gt_vertices = gt_vertices.copy()
    pred_vertices = pred_vertices.copy()

    all_vertices = np.concatenate(
        [
            gt_vertices.reshape(-1, 3),
            pred_vertices.reshape(-1, 3),
        ],
        axis=0,
    )
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2.0

    gt_vertices -= center
    pred_vertices -= center
    gt_vertices[:, :, 0] -= subject_spacing / 2.0
    pred_vertices[:, :, 0] += subject_spacing / 2.0
    return gt_vertices, pred_vertices


class AITViewerVideoRenderer:
    def __init__(self, faces, width, height, fps, face_stride=1, subject_spacing=1.1):
        self.faces = faces[::face_stride].astype(np.int32)
        self.width = width
        self.height = height
        self.fps = fps
        self.subject_spacing = subject_spacing

    def _make_meshes(self, gt_vertices, pred_vertices):
        gt_vertices, pred_vertices = prepare_subject_vertices(
            gt_vertices,
            pred_vertices,
            self.subject_spacing,
        )

        colors = get_subject_colors()

        gt_mesh = Meshes(
            gt_vertices,
            self.faces,
            name="GT",
            flat_shading=False,
            draw_edges=False,
        )
        pred_mesh = Meshes(
            pred_vertices,
            self.faces,
            name="Pred",
            flat_shading=False,
            draw_edges=False,
        )

        gt_mesh.color = colors[0]
        pred_mesh.color = colors[1]
        return [gt_mesh, pred_mesh]

    def render_video(self, gt_vertices, pred_vertices, output_path):
        nodes = self._make_meshes(gt_vertices, pred_vertices)

        viewer = HeadlessRenderer(size=(self.width, self.height))
        try:
            viewer.auto_set_camera_target = False
            for node in nodes:
                viewer.scene.add(node)
            viewer._init_scene()
            viewer.scene.camera.target = np.array([0.0, 0.9, 0.0], dtype=np.float32)
            viewer.scene.camera.position = np.array([0.0, 0.2, 5.2], dtype=np.float32)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            viewer.save_video(video_dir=str(output_path), output_fps=self.fps)
        finally:
            if hasattr(viewer, "close"):
                viewer.close()


def visualize_sequence(seq_path, output_path, body_model, args):
    data = torch.load(seq_path, map_location="cpu")
    required = ["pose_t", "pose_p"]
    if args.visualize_tran:
        required.extend(["tran_t", "tran_p"])
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{seq_path} is missing required fields: {missing}")

    gt_tran = maybe_zero_translation(data["pose_t"], data.get("tran_t"), args.visualize_tran)
    pred_tran = maybe_zero_translation(data["pose_p"], data.get("tran_p"), args.visualize_tran)

    gt_vertices = pose_to_vertices(body_model, data["pose_t"], gt_tran, args.batch_size)
    pred_vertices = pose_to_vertices(body_model, data["pose_p"], pred_tran, args.batch_size)

    n_frames = min(len(gt_vertices), len(pred_vertices))
    frame_ids = list(range(0, n_frames, args.stride))
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]

    gt_vertices = gt_vertices[frame_ids]
    pred_vertices = pred_vertices[frame_ids]

    renderer = AITViewerVideoRenderer(
        body_model.face.astype(np.int32),
        width=args.image_width,
        height=args.image_height,
        fps=args.fps,
        face_stride=args.face_stride,
        subject_spacing=args.subject_spacing,
    )
    renderer.render_video(gt_vertices, pred_vertices, output_path)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Evaluation result directory containing sequence .pt files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save mp4 files. Defaults to INPUT_DIR/videos.")
    parser.add_argument("--sequence", type=int, default=None, help="Render only one sequence id, e.g. 1 for 1.pt.")
    parser.add_argument("--fps", type=int, default=datasets.fps)
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--face-stride", type=int, default=1, help="Use every Nth SMPL face.")
    parser.add_argument("--subject-spacing", type=float, default=1.1, help="Horizontal spacing between GT and Pred meshes.")
    parser.add_argument("--visualize-tran", action="store_true", help="Include translation in the visualization.")
    parser.add_argument("--batch-size", type=int, default=256, help="SMPL FK batch size.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = resolve_output_dir(input_dir, args.output_dir)

    seq_files = list_sequence_files(input_dir, args.sequence)
    if not seq_files:
        raise FileNotFoundError(f"No numeric sequence .pt files found in {input_dir}")

    body_model = art.ParametricModel(paths.smpl_file)
    for seq_path in seq_files:
        output_path = output_dir / f"{seq_path.stem}.mp4"
        visualize_sequence(seq_path, output_path, body_model, args)


if __name__ == "__main__":
    main()
