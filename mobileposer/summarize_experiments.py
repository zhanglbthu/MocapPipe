"""Index training histories and evaluation reports without loading models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
SELECTION_METRICS = (
    "total_loss",
    "validation_step_loss",
    "val_loss",
    "ori_loss",
    "drift_err_deg",
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {"read_error": str(error)}


def _summarize_history(path: Path, root: Path) -> dict[str, Any]:
    history = _read_json(path)
    result: dict[str, Any] = {"path": str(path.relative_to(root))}
    if not isinstance(history, list) or not history:
        result["error"] = "history must be a non-empty list"
        return result

    metric = next(
        (name for name in SELECTION_METRICS if any(name in epoch for epoch in history if isinstance(epoch, dict))),
        None,
    )
    result["epochs"] = len(history)
    result["last_epoch"] = history[-1]
    if metric is not None:
        candidates = [epoch for epoch in history if isinstance(epoch, dict) and isinstance(epoch.get(metric), (int, float))]
        if candidates:
            best = min(candidates, key=lambda epoch: epoch[metric])
            result.update({"selection_metric": metric, "best_value": best[metric], "best_epoch": best.get("epoch")})
    return result


def build_index(data_dir: Path) -> dict[str, Any]:
    histories = sorted(data_dir.rglob("history.json")) if data_dir.exists() else []
    reports = sorted(data_dir.rglob("report.json")) if data_dir.exists() else []
    return {
        "data_dir": str(data_dir.resolve()),
        "training_runs": [_summarize_history(path, data_dir) for path in histories],
        "evaluations": [
            {"path": str(path.relative_to(data_dir)), "report": _read_json(path)}
            for path in reports
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    data_dir = args.data_dir.expanduser().resolve()
    output = args.output or data_dir / "experiment_index.json"
    index = build_index(data_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2, sort_keys=True))
    print(f"Indexed {len(index['training_runs'])} training runs and {len(index['evaluations'])} evaluations.")
    print(f"EXPERIMENT_INDEX={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
