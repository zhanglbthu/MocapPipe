import argparse
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import Meshes


TAB10_COLORS = (
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--subject_spacing", type=float, default=1.25)
    parser.add_argument("--xvfb_display", type=str, default=":0")
    parser.add_argument("--xvfb_width", type=int, default=1920)
    parser.add_argument("--xvfb_height", type=int, default=1080)
    return parser.parse_args()


@contextmanager
def managed_xvfb(display, width, height):
    proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", f"{width}x{height}x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(1.0)
        os.environ["DISPLAY"] = display
        yield
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def prepare_subject_vertices(vertices, subject_spacing):
    vertices = vertices.copy()
    num_subjects = vertices.shape[0]
    center = (vertices.reshape(-1, 3).min(axis=0) + vertices.reshape(-1, 3).max(axis=0)) / 2.0
    vertices -= center

    offsets = (np.arange(num_subjects, dtype=np.float32) - (num_subjects - 1) / 2.0) * subject_spacing
    vertices[:, :, :, 0] += offsets[:, None, None]
    return vertices


def make_meshes(vertices, faces, labels, subject_spacing):
    vertices = prepare_subject_vertices(vertices, subject_spacing)
    meshes = []
    for idx, label in enumerate(labels):
        mesh = Meshes(vertices[idx], faces, name=str(label), flat_shading=False, draw_edges=False)
        color = TAB10_COLORS[idx % len(TAB10_COLORS)]
        mesh.color = tuple(float(x) for x in color) + (1.0,)
        meshes.append(mesh)
    return meshes


class AITViewerMultiRenderer:
    def __init__(self, faces, width, height, fps, subject_spacing):
        self.faces = faces.astype(np.int32)
        self.width = width
        self.height = height
        self.fps = fps
        self.subject_spacing = subject_spacing

    def render_video(self, vertices, labels, output_path):
        meshes = make_meshes(vertices, self.faces, labels, self.subject_spacing)
        num_subjects = vertices.shape[0]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_prefix = output_path.with_suffix("")

        viewer = HeadlessRenderer(size=(self.width, self.height))
        try:
            viewer.auto_set_camera_target = False
            for mesh in meshes:
                viewer.scene.add(mesh)
            viewer._init_scene()
            viewer.scene.fps = self.fps
            viewer.playback_fps = self.fps
            viewer.export_fps = self.fps
            viewer.scene.camera.target = np.array([0.0, 0.9, 0.0], dtype=np.float32)
            camera_z = max(4.8, 3.4 + 0.85 * num_subjects)
            viewer.scene.camera.position = np.array([0.0, 0.35, camera_z], dtype=np.float32)
            viewer.save_video(video_dir=str(tmp_prefix), output_fps=self.fps)
        finally:
            if hasattr(viewer, "close"):
                viewer.close()

        produced = sorted(output_path.parent.glob(f"{tmp_prefix.name}_*.mp4"))
        if not produced:
            raise FileNotFoundError(f"AITViewer did not produce an output mp4 for prefix {tmp_prefix}.")
        shutil.move(str(produced[0]), str(output_path))
        for extra in produced[1:]:
            extra.unlink(missing_ok=True)


def main():
    args = parse_args()
    payload = np.load(args.input_npz)
    vertices = payload["vertices"]
    faces = payload["faces"]
    labels = [str(x) for x in payload["labels"]]

    with managed_xvfb(args.xvfb_display, args.xvfb_width, args.xvfb_height):
        renderer = AITViewerMultiRenderer(
            faces=faces,
            width=args.width,
            height=args.height,
            fps=args.fps,
            subject_spacing=args.subject_spacing,
        )
        renderer.render_video(vertices, labels, args.output_path)


if __name__ == "__main__":
    main()
