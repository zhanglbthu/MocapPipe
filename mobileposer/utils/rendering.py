"""Small shared helpers for paper-result rendering."""

from __future__ import annotations

import torch


def _to_local_pose(body_model, pose):
    pose = pose.float().view(-1, 24, 3, 3)
    return body_model.inverse_kinematics_R(pose)


@torch.no_grad()
def pose_to_vertices(body_model, pose, translation, batch_size=256, pose_is_local=False, global_scale=1.0):
    pose = pose.float().view(-1, 24, 3, 3) if pose_is_local else _to_local_pose(body_model, pose)
    translation = translation.float().view(-1, 3)
    vertices = []
    for start in range(0, pose.shape[0], batch_size):
        end = min(start + batch_size, pose.shape[0])
        _, _, mesh = body_model.forward_kinematics(
            pose=pose[start:end],
            tran=translation[start:end],
            calc_mesh=True,
        )
        vertices.append(mesh.cpu())
    return (torch.cat(vertices, dim=0) * float(global_scale)).numpy()


def maybe_zero_translation(pose, translation, visualize_translation):
    if visualize_translation and translation is not None:
        return translation
    return torch.zeros((pose.shape[0], 3), dtype=torch.float32)
