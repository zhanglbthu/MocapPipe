"""Single command entry point for the research pipeline.

This module intentionally delegates to the existing experiment scripts.  It
gives the project one stable public interface while legacy commands remain
available during the refactor.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Command:
    script: str
    description: str


COMMANDS = {
    "preprocess": Command("process.py", "Build AMASS, IMUPoser, Huawei, or calibration datasets."),
    "train-mocap": Command("train_direct.py", "Train the frozen downstream DirectPoser baseline."),
    "train-calibrator": Command("train_combo_imu_calibrator.py", "Train the proposed plug-in IMU canonicalizer."),
    "train-tic": Command("train_tic_calibrator.py", "Train the TIC calibration baseline."),
    "eval-mocap": Command("evaluate_direct.py", "Evaluate a downstream mocap model without calibration."),
    "eval-calibrator": Command("evaluate_combo_calibrated_direct_online.py", "Evaluate the proposed canonicalizer online."),
    "eval-tic": Command("evaluate_tic_calibrated_direct_online.py", "Evaluate TIC online under the same protocol."),
    "demo": Command("livedemo.py", "Run the Huawei-device real-time interaction demo."),
    "visualize": Command("visualize_quad.py", "Render GT/raw/TIC/ours comparison results."),
    "summarize": Command("summarize_experiments.py", "Index all training histories and evaluation reports."),
}


def _print_help() -> None:
    print("Usage: python -m mobileposer.cli <command> [arguments]\n")
    print("Core research commands:")
    width = max(map(len, COMMANDS))
    for name, command in COMMANDS.items():
        print(f"  {name:<{width}}  {command.description}")
    print("\nArguments after <command> are passed to the underlying script.")
    print("Example: python -m mobileposer.cli preprocess --dataset amass")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    name, *script_args = argv
    command = COMMANDS.get(name)
    if command is None:
        available = ", ".join(COMMANDS)
        print(f"Unknown command: {name}. Available commands: {available}", file=sys.stderr)
        return 2

    script = PACKAGE_DIR / command.script
    completed = subprocess.run([sys.executable, str(script), *script_args], cwd=PACKAGE_DIR)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
