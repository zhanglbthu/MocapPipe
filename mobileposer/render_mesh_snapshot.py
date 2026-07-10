import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import articulate as art
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import Meshes
from config import paths
from visualize import maybe_zero_translation, pose_to_vertices


def _load_vertices(
    body_model,
    result_path: Path,
    frame_idx: int,
    batch_size: int,
    pose_key: str,
    tran_key: Optional[str],
    pose_is_local: bool,
    visualize_tran: bool,
    global_scale: float,
):
    data = torch.load(result_path, map_location="cpu")
    if pose_key not in data:
        raise KeyError(f"{result_path} is missing required field: {pose_key}")

    pose = data[pose_key]
    tran = data.get(tran_key) if tran_key is not None else None
    tran = maybe_zero_translation(pose, tran, visualize_tran)
    vertices = pose_to_vertices(
        body_model,
        pose,
        tran,
        batch_size=batch_size,
        pose_is_local=pose_is_local,
        global_scale=global_scale,
    )
    if frame_idx < 0 or frame_idx >= len(vertices):
        raise IndexError(f"frame_idx={frame_idx} out of range for {result_path} with {len(vertices)} frames")
    return vertices[frame_idx : frame_idx + 1]


def _center_and_offset(vertex_groups, subject_spacing: float):
    vertex_groups = [v.copy() for v in vertex_groups]
    flat = np.concatenate([v.reshape(-1, 3) for v in vertex_groups], axis=0)
    center = (flat.min(axis=0) + flat.max(axis=0)) / 2.0
    vertex_groups = [v - center for v in vertex_groups]

    offsets = np.linspace(
        -subject_spacing * (len(vertex_groups) - 1) / 2.0,
        subject_spacing * (len(vertex_groups) - 1) / 2.0,
        num=len(vertex_groups),
        dtype=np.float32,
    )
    for verts, offset in zip(vertex_groups, offsets):
        verts[:, :, 0] += float(offset)
    return vertex_groups


def _default_colors():
    return [
        (0.20, 0.45, 0.90, 1.0),
        (0.13, 0.65, 0.35, 1.0),
        (0.88, 0.36, 0.20, 1.0),
        (0.55, 0.35, 0.85, 1.0),
        (0.15, 0.70, 0.75, 1.0),
        (0.85, 0.70, 0.10, 1.0),
    ]


def render_snapshot(
    vertex_groups,
    output_path: Path,
    faces: np.ndarray,
    width: int,
    height: int,
    subject_spacing: float,
    camera_distance: float,
    transparent_background: bool,
):
    vertex_groups = _center_and_offset(vertex_groups, subject_spacing)
    viewer = HeadlessRenderer(size=(width, height), backend="egl")
    try:
        viewer.auto_set_camera_target = False
        if hasattr(viewer, "auto_set_floor"):
            viewer.auto_set_floor = False
        colors = _default_colors()
        for idx, vertices in enumerate(vertex_groups):
            mesh = Meshes(vertices, faces.astype(np.int32), name=f"mesh_{idx}", flat_shading=False, draw_edges=False)
            mesh.color = colors[idx % len(colors)]
            viewer.scene.add(mesh)

        viewer._init_scene()
        if hasattr(viewer.scene, "floor") and viewer.scene.floor is not None:
            viewer.scene.floor.enabled = False
        viewer.scene.fps = 30
        viewer.playback_fps = 30
        viewer.export_fps = 30
        viewer.scene.camera.target = np.array([0.0, 0.9, 0.0], dtype=np.float32)
        viewer.scene.camera.position = np.array([0.0, 0.2, camera_distance], dtype=np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        viewer.export_frame(str(output_path), transparent_background=transparent_background)
    finally:
        if hasattr(viewer, "close"):
            viewer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Render a single-frame mesh snapshot from one or more result .pt files."
    )
    parser.add_argument("--results", type=str, nargs="+", required=True, help="One or more result .pt files.")
    parser.add_argument(
        "--pose-keys",
        type=str,
        nargs="+",
        default=None,
        help="Pose field for each result file. Defaults to pose_p for all inputs.",
    )
    parser.add_argument(
        "--tran-keys",
        type=str,
        nargs="+",
        default=None,
        help="Translation field for each result file. Defaults to tran_p for pred, tran_t for pose_t.",
    )
    parser.add_argument("--frame-idx", type=int, required=True, help="Frame index to render.")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path.")
    parser.add_argument("--pose-is-local", action="store_true", help="Treat pose tensors as local SMPL rotations.")
    parser.add_argument("--visualize-tran", action="store_true", help="Use translation when rendering.")
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=2400)
    parser.add_argument("--image-height", type=int, default=1200)
    parser.add_argument("--subject-spacing", type=float, default=1.3)
    parser.add_argument("--camera-distance", type=float, default=6.8)
    parser.add_argument("--transparent-background", action="store_true", help="Export PNG with transparent background.")
    args = parser.parse_args()

    result_paths = [Path(p) for p in args.results]
    pose_keys = args.pose_keys or ["pose_p"] * len(result_paths)
    if len(pose_keys) != len(result_paths):
        raise ValueError("--pose-keys must have the same length as --results")

    if args.tran_keys is None:
        tran_keys = ["tran_t" if pk == "pose_t" else "tran_p" for pk in pose_keys]
    else:
        tran_keys = args.tran_keys
    if len(tran_keys) != len(result_paths):
        raise ValueError("--tran-keys must have the same length as --results")

    body_model = art.ParametricModel(paths.smpl_file)
    vertex_groups = []
    for result_path, pose_key, tran_key in zip(result_paths, pose_keys, tran_keys):
        vertex_groups.append(
            _load_vertices(
                body_model=body_model,
                result_path=result_path,
                frame_idx=args.frame_idx,
                batch_size=args.batch_size,
                pose_key=pose_key,
                tran_key=tran_key,
                pose_is_local=args.pose_is_local,
                visualize_tran=args.visualize_tran,
                global_scale=args.global_scale,
            )
        )

    render_snapshot(
        vertex_groups=vertex_groups,
        output_path=Path(args.output),
        faces=body_model.face,
        width=args.image_width,
        height=args.image_height,
        subject_spacing=args.subject_spacing,
        camera_distance=args.camera_distance,
        transparent_background=args.transparent_background,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
