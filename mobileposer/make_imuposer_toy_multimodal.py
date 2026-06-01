import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import articulate as art
from articulate.armature import SMPLJoint
from config import paths


RIGHT_ARM_JOINTS = [
    SMPLJoint.RCLAVICLE.value,
    SMPLJoint.RSHOULDER.value,
    SMPLJoint.RELBOW.value,
    SMPLJoint.RWRIST.value,
    SMPLJoint.RHAND.value,
]


def geodesic_midpoint(rot_a: torch.Tensor, rot_b: torch.Tensor) -> torch.Tensor:
    rel = rot_a.transpose(-1, -2).matmul(rot_b)
    delta = art.math.rotation_matrix_to_axis_angle(rel.reshape(-1, 3, 3)).view(-1, 3)
    mid = art.math.axis_angle_to_rotation_matrix((0.5 * delta).view(-1, 3)).view_as(rot_a)
    return rot_a.matmul(mid)


def replace_right_arm(sequence_pose: torch.Tensor, template_pose: torch.Tensor) -> torch.Tensor:
    pose = sequence_pose.clone()
    pose[:, RIGHT_ARM_JOINTS] = template_pose[RIGHT_ARM_JOINTS].unsqueeze(0)
    return pose


def wrist_height(body_model: art.ParametricModel, pose: torch.Tensor) -> torch.Tensor:
    _, joints = body_model.forward_kinematics(pose=pose.float())
    joints = joints.view(-1, 24, 3)
    return joints[:, SMPLJoint.RWRIST.value, 1]


def build_dataset_entry(acc, ori, pose, tran):
    return {
        "acc": acc.clone(),
        "ori": ori.clone(),
        "pose": pose.clone(),
        "tran": tran.clone(),
    }


def save_dataset(entries, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "acc": [entry["acc"] for entry in entries],
            "ori": [entry["ori"] for entry in entries],
            "pose": [entry["pose"] for entry in entries],
            "tran": [entry["tran"] for entry in entries],
        },
        output_path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=str(paths.processed_datasets / "eval" / "imuposer_test.pt"))
    parser.add_argument("--sequence-index", type=int, default=1, help="Zero-based sequence index. 1 means the second sequence.")
    parser.add_argument("--repeats", type=int, default=64, help="Copies per mode for the training file.")
    parser.add_argument("--output-dir", type=str, default=str(paths.processed_datasets / "eval"))
    parser.add_argument("--viz-dir", type=str, default="data/toy/imuposer_toy_multimodal")
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    viz_dir = Path(args.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(source, map_location="cpu")
    acc = data["acc"][args.sequence_index].float()
    ori = data["ori"][args.sequence_index].float()
    pose = data["pose"][args.sequence_index].float()
    tran = data["tran"][args.sequence_index].float()

    body_model = art.model.ParametricModel(paths.smpl_file)
    wrist_y = wrist_height(body_model, pose)
    down_index = int(torch.argmin(wrist_y).item())
    up_index = int(torch.argmax(wrist_y).item())

    pose_down = replace_right_arm(pose, pose[down_index])
    pose_up = replace_right_arm(pose, pose[up_index])
    pose_mean = pose_down.clone()
    pose_mean[:, RIGHT_ARM_JOINTS] = geodesic_midpoint(
        pose_down[:, RIGHT_ARM_JOINTS],
        pose_up[:, RIGHT_ARM_JOINTS],
    )

    train_entries = []
    for _ in range(args.repeats):
        train_entries.append(build_dataset_entry(acc, ori, pose_down, tran))
        train_entries.append(build_dataset_entry(acc, ori, pose_up, tran))

    test_entries = [
        build_dataset_entry(acc, ori, pose_down, tran),
        build_dataset_entry(acc, ori, pose_up, tran),
        build_dataset_entry(acc, ori, pose_mean, tran),
    ]

    train_path = output_dir / "imuposer_toy_multimodal_train.pt"
    test_path = output_dir / "imuposer_toy_multimodal_test.pt"
    save_dataset(train_entries, train_path)
    save_dataset(test_entries, test_path)

    down_y = wrist_height(body_model, pose_down)
    up_y = wrist_height(body_model, pose_up)
    mean_y = wrist_height(body_model, pose_mean)

    frames = torch.arange(len(wrist_y))
    plt.figure(figsize=(12, 4))
    plt.plot(frames, wrist_y.numpy(), label="original", linewidth=1.0, alpha=0.7)
    plt.plot(frames, down_y.numpy(), label="right-arm down", linewidth=2.0)
    plt.plot(frames, up_y.numpy(), label="right-arm up", linewidth=2.0)
    plt.plot(frames, mean_y.numpy(), label="geodesic midpoint", linewidth=2.0, linestyle="--")
    plt.xlabel("Frame")
    plt.ylabel("Right wrist height (Y)")
    plt.title("IMUPoser toy multimodal target: same LW input, two right-arm solutions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / "right_wrist_height.png", dpi=180)
    plt.close()

    torch.save(
        {
            "pose_original": pose,
            "pose_down": pose_down,
            "pose_up": pose_up,
            "pose_mean": pose_mean,
            "tran": tran,
        },
        viz_dir / "reference_poses.pt",
    )

    summary = {
        "source": str(source),
        "sequence_index": args.sequence_index,
        "sequence_length": int(acc.shape[0]),
        "down_index": down_index,
        "up_index": up_index,
        "right_arm_joints": RIGHT_ARM_JOINTS,
        "repeats_per_mode": args.repeats,
        "train_sequences": len(train_entries),
        "test_sequences": len(test_entries),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "plot_path": str(viz_dir / "right_wrist_height.png"),
        "reference_pose_path": str(viz_dir / "reference_poses.pt"),
    }
    (viz_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
