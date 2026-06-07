import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

import articulate as art
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import Meshes
from config import datasets, paths
from visualize import maybe_zero_translation, pose_to_vertices


def prepare_vertices_quad(gt_vertices, ours_vertices, tic_vertices, base_vertices, subject_spacing):
    gt_vertices = gt_vertices.copy()
    ours_vertices = ours_vertices.copy()
    tic_vertices = tic_vertices.copy()
    base_vertices = base_vertices.copy()

    all_vertices = np.concatenate(
        [
            gt_vertices.reshape(-1, 3),
            ours_vertices.reshape(-1, 3),
            tic_vertices.reshape(-1, 3),
            base_vertices.reshape(-1, 3),
        ],
        axis=0,
    )
    center = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2.0

    gt_vertices -= center
    ours_vertices -= center
    tic_vertices -= center
    base_vertices -= center

    gt_vertices[:, :, 0] -= 1.5 * subject_spacing
    ours_vertices[:, :, 0] -= 0.5 * subject_spacing
    tic_vertices[:, :, 0] += 0.5 * subject_spacing
    base_vertices[:, :, 0] += 1.5 * subject_spacing
    return gt_vertices, ours_vertices, tic_vertices, base_vertices


class QuadRenderer:
    def __init__(self, faces, width, height, fps, face_stride=1, subject_spacing=1.25):
        self.faces = faces[::face_stride].astype(np.int32)
        self.width = width
        self.height = height
        self.fps = fps
        self.subject_spacing = subject_spacing

    def _make_meshes(self, gt_vertices, ours_vertices, tic_vertices, base_vertices):
        gt_vertices, ours_vertices, tic_vertices, base_vertices = prepare_vertices_quad(
            gt_vertices,
            ours_vertices,
            tic_vertices,
            base_vertices,
            self.subject_spacing,
        )
        colors = matplotlib.colormaps["tab10"].colors
        specs = [
            ("GT", gt_vertices, tuple(float(x) for x in colors[0]) + (1.0,)),
            ("Ours", ours_vertices, tuple(float(x) for x in colors[2]) + (1.0,)),
            ("TIC", tic_vertices, tuple(float(x) for x in colors[3]) + (1.0,)),
            ("Baseline", base_vertices, tuple(float(x) for x in colors[1]) + (1.0,)),
        ]
        meshes = []
        for name, vertices, color in specs:
            mesh = Meshes(vertices, self.faces, name=name, flat_shading=False, draw_edges=False)
            mesh.color = color
            meshes.append(mesh)
        return meshes

    def render_video(self, gt_vertices, ours_vertices, tic_vertices, base_vertices, output_path):
        nodes = self._make_meshes(gt_vertices, ours_vertices, tic_vertices, base_vertices)

        viewer = HeadlessRenderer(size=(self.width, self.height))
        try:
            viewer.auto_set_camera_target = False
            for node in nodes:
                viewer.scene.add(node)
            viewer._init_scene()
            viewer.scene.fps = self.fps
            viewer.playback_fps = self.fps
            viewer.export_fps = self.fps
            viewer.scene.camera.target = np.array([0.0, 0.9, 0.0], dtype=np.float32)
            viewer.scene.camera.position = np.array([0.0, 0.2, 8.2], dtype=np.float32)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            viewer.save_video(video_dir=str(output_path), output_fps=self.fps)
        finally:
            if hasattr(viewer, "close"):
                viewer.close()


def load_vertices(body_model, result_path, batch_size, pose_is_local, visualize_tran, global_scale):
    data = torch.load(result_path, map_location="cpu")
    gt_tran = maybe_zero_translation(data["pose_t"], data.get("tran_t"), visualize_tran)
    pred_tran = maybe_zero_translation(data["pose_p"], data.get("tran_p"), visualize_tran)
    gt_vertices = pose_to_vertices(
        body_model,
        data["pose_t"],
        gt_tran,
        batch_size=batch_size,
        pose_is_local=pose_is_local,
        global_scale=global_scale,
    )
    pred_vertices = pose_to_vertices(
        body_model,
        data["pose_p"],
        pred_tran,
        batch_size=batch_size,
        pose_is_local=pose_is_local,
        global_scale=global_scale,
    )
    return gt_vertices, pred_vertices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-dir", type=str, required=True)
    parser.add_argument("--tic-dir", type=str, required=True)
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--fps", type=int, default=datasets.fps)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--face-stride", type=int, default=1)
    parser.add_argument("--subject-spacing", type=float, default=1.25)
    parser.add_argument("--visualize-tran", action="store_true")
    parser.add_argument("--pose-is-local", action="store_true")
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    ours_path = Path(args.ours_dir) / f"{args.sequence}.pt"
    tic_path = Path(args.tic_dir) / f"{args.sequence}.pt"
    base_path = Path(args.baseline_dir) / f"{args.sequence}.pt"
    output_path = Path(args.output_dir) / f"{args.sequence}.mp4"

    body_model = art.ParametricModel(paths.smpl_file)
    gt_vertices, ours_vertices = load_vertices(
        body_model, ours_path, args.batch_size, args.pose_is_local, args.visualize_tran, args.global_scale
    )
    _, tic_vertices = load_vertices(
        body_model, tic_path, args.batch_size, args.pose_is_local, args.visualize_tran, args.global_scale
    )
    _, base_vertices = load_vertices(
        body_model, base_path, args.batch_size, args.pose_is_local, args.visualize_tran, args.global_scale
    )

    n_frames = min(len(gt_vertices), len(ours_vertices), len(tic_vertices), len(base_vertices))
    frame_ids = list(range(0, n_frames, args.stride))
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]

    gt_vertices = gt_vertices[frame_ids]
    ours_vertices = ours_vertices[frame_ids]
    tic_vertices = tic_vertices[frame_ids]
    base_vertices = base_vertices[frame_ids]

    renderer = QuadRenderer(
        body_model.face.astype(np.int32),
        width=args.image_width,
        height=args.image_height,
        fps=args.fps,
        face_stride=args.face_stride,
        subject_spacing=args.subject_spacing,
    )
    renderer.render_video(gt_vertices, ours_vertices, tic_vertices, base_vertices, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
