import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from config import paths
from evaluate_diffusion import build_save_dir as build_eval_save_dir
from evaluate_diffusion import resolve_checkpoint as resolve_eval_checkpoint
from train_diffusion import build_checkpoint_path as build_train_checkpoint_path
from utils.file_utils import get_datestring
from visualize import resolve_output_dir as resolve_vis_output_dir


REPO_ROOT = Path(__file__).resolve().parent


DEFAULT_CONFIGS = {
    "train": paths.experiments_dir / "train.default.yaml",
    "eval": paths.experiments_dir / "eval.default.yaml",
    "visualize": paths.experiments_dir / "visualize.default.yaml",
}


SCRIPT_BY_MODE = {
    "train": REPO_ROOT / "train_diffusion.py",
    "eval": REPO_ROOT / "evaluate_diffusion.py",
    "visualize": REPO_ROOT / "visualize.py",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=SCRIPT_BY_MODE.keys())
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--set", nargs="*", default=[], help="Override config values, e.g. --set epochs=5 run_name=exp1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def parse_scalar(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if value.lower() == "null":
            return None
        return value


def apply_overrides(config, overrides):
    result = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}")
        key, value = item.split("=", 1)
        result[key] = parse_scalar(value)
    return result


def normalize_config(mode, config):
    cfg = dict(config)
    cfg.setdefault("conda_env", "mobileposer")
    cfg.setdefault("save_resolved_config", True)
    if mode == "train":
        cfg.setdefault("save_dir", str(paths.checkpoint))
    elif mode == "eval":
        cfg.setdefault("model", None)
        cfg.setdefault("run_dir", None)
        cfg.setdefault("checkpoint_dir", None)
        cfg.setdefault("save_dir", None)
        cfg.setdefault("max_samples", None)
    elif mode == "visualize":
        cfg.setdefault("output_dir", None)
        cfg.setdefault("sequence", None)
        cfg.setdefault("max_frames", None)
    return cfg


def config_to_cli_args(config):
    args = []
    skip_keys = {"conda_env", "save_resolved_config", "experiment_name"}
    for key, value in config.items():
        if key in skip_keys or value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        args.extend([flag, str(value)])
    return args


def as_namespace(config):
    return argparse.Namespace(**config)


def train_output_dir(config):
    checkpoint_path = build_train_checkpoint_path(as_namespace(config))
    return checkpoint_path.parent


def eval_output_dir(config):
    checkpoint_path = resolve_eval_checkpoint(as_namespace(config))
    save_dir = build_eval_save_dir(as_namespace(config), checkpoint_path)
    return save_dir


def visualize_output_dir(config):
    input_dir = Path(config["input_dir"])
    output_dir = resolve_vis_output_dir(input_dir, config.get("output_dir"))
    return output_dir


def resolve_artifact_dir(mode, config):
    if mode == "train":
        return train_output_dir(config)
    if mode == "eval":
        return eval_output_dir(config)
    if mode == "visualize":
        return visualize_output_dir(config)
    raise ValueError(f"Unsupported mode: {mode}")


def write_resolved_config(mode, config, output_dir, source_config):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "timestamp": get_datestring(),
        "source_config": str(source_config) if source_config else None,
        "config": config,
    }
    with open(output_dir / "experiment.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def build_command(mode, config):
    script = SCRIPT_BY_MODE[mode]
    cli_args = config_to_cli_args(config)
    return [sys.executable, str(script), *cli_args]


def main():
    args = parse_args()
    config_path = Path(args.config) if args.config else DEFAULT_CONFIGS[args.mode]
    base_config = load_yaml(config_path)
    config = normalize_config(args.mode, apply_overrides(base_config, args.set))
    command = build_command(args.mode, config)

    print("Command:")
    print(" ".join(shlex.quote(part) for part in command))

    artifact_dir = resolve_artifact_dir(args.mode, config)
    print(f"Artifacts: {artifact_dir}")

    if config.get("save_resolved_config", True):
        write_resolved_config(args.mode, config, artifact_dir, config_path)
        print(f"Saved resolved config to: {artifact_dir / 'experiment.yaml'}")

    if args.print_only or args.dry_run:
        return

    completed = subprocess.run(command, cwd=REPO_ROOT)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
