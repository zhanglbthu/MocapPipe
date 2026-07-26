"""Shared metric aggregation and reproducible evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

import articulate as art
from config import datasets, joint_set, paths


METRIC_NAMES = (
    "SIP Error (deg)",
    "Angular Error (deg)",
    "Masked Angular Error (deg)",
    "Positional Error (cm)",
    "Masked Positional Error (cm)",
    "Mesh Error (cm)",
    "Jitter Error (100m/s^3)",
    "Distance Error (cm)",
)


class PoseEvaluator:
    """Compute the common sparse-inertial motion-capture metrics."""

    def __init__(self):
        self._evaluator = art.FullMotionEvaluator(
            paths.smpl_file,
            joint_mask=torch.tensor([2, 5, 16, 20]),
            fps=datasets.fps,
        )

    def evaluate(self, predicted_pose, target_pose, predicted_translation=None, target_translation=None):
        predicted_pose = predicted_pose.clone().view(-1, 24, 3, 3)
        target_pose = target_pose.clone().view(-1, 24, 3, 3)
        if predicted_translation is None or target_translation is None:
            predicted_translation = torch.zeros(predicted_pose.shape[0], 3, device=predicted_pose.device)
            target_translation = torch.zeros(target_pose.shape[0], 3, device=target_pose.device)
        else:
            predicted_translation = predicted_translation.clone().view(-1, 3)
            target_translation = target_translation.clone().view(-1, 3)

        ignored = torch.tensor(joint_set.ignored, device=predicted_pose.device, dtype=torch.long)
        identity = torch.eye(3, device=predicted_pose.device).view(1, 1, 3, 3)
        predicted_pose[:, ignored] = identity
        target_pose[:, ignored.to(target_pose.device)] = identity.to(target_pose.device)

        errors = self._evaluator(
            predicted_pose,
            target_pose,
            tran_p=predicted_translation,
            tran_t=target_translation,
        )
        return torch.stack(
            [
                errors[9],
                errors[3],
                errors[9],
                errors[0] * 100,
                errors[7] * 100,
                errors[1] * 100,
                errors[4] / 100,
                errors[6],
            ]
        )

    @staticmethod
    def print(stats: torch.Tensor) -> None:
        for index, name in enumerate(METRIC_NAMES):
            print(f"{name}: {stats[index, 0]:.2f} (+/- {stats[index, 1]:.2f})")


def aggregate_sequence_metrics(sequence_metrics: Iterable[torch.Tensor]) -> torch.Tensor:
    """Average evaluator [metric, (mean, std)] tensors across sequences."""
    values = list(sequence_metrics)
    if not values:
        raise ValueError("Cannot aggregate an empty evaluation set.")
    stats = torch.stack(values).mean(dim=0)
    if stats.shape != (len(METRIC_NAMES), 2):
        raise ValueError(f"Expected metric shape {(len(METRIC_NAMES), 2)}, got {tuple(stats.shape)}")
    return stats


def write_evaluation_report(
    output_dir: Path,
    stats: torch.Tensor,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write one machine-readable and one human-readable result summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        name: {"mean": float(stats[index, 0]), "std": float(stats[index, 1])}
        for index, name in enumerate(METRIC_NAMES)
    }
    report = {"metadata": metadata, "metrics": metrics}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    lines = [f"{key}: {value}" for key, value in metadata.items()]
    lines.append("")
    lines.extend(
        f"{name}: {values['mean']:.2f} (+/- {values['std']:.2f})"
        for name, values in metrics.items()
    )
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n")
    return report
