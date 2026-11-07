"""Run every trainer-hyperparameter row against every environment row."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load_trials(path: Path) -> list[dict[str, Any]]:
    """Load one plain JSON list of Hydra override dictionaries."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Every entry in {path} must be a JSON object")
    return data


def _override(name: str, value: Any) -> str:
    """Render a Python/JSON value as one Hydra command-line override."""

    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, (list, tuple)):
        rendered = "[" + ",".join(map(str, value)) + "]"
    else:
        rendered = str(value)
    return f"{name}={rendered}"


def _write(path: Path, value: Any) -> None:
    """Write stable, human-readable JSON and terminate it with a newline."""

    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_sweep(
    trainer_path: Path,
    environment_path: Path,
    output_dir: Path,
    *,
    smoke: bool,
    video: bool,
) -> int:
    """Train the Cartesian product and write a persistent leaderboard."""

    trainer_trials = _load_trials(trainer_path)
    environment_trials = _load_trials(environment_path)
    combinations = list(itertools.product(trainer_trials, environment_trials))
    output_dir.mkdir(parents=True, exist_ok=False)
    _write(output_dir / "training_hyperparameters.json", trainer_trials)
    _write(output_dir / "environment_hyperparameters.json", environment_trials)
    records: list[dict[str, Any]] = []

    for trial_number, (trainer, environment) in enumerate(combinations, start=1):
        trial_dir = output_dir / f"trial_{trial_number:03d}"
        overrides = {**trainer, **environment}
        overlap = set(trainer) & set(environment)
        if overlap:
            raise ValueError(f"Trainer/environment keys overlap: {sorted(overlap)}")
        command = [
            sys.executable,
            str(ROOT / "App" / "Scripts" / "train.py"),
            *( ["--smoke"] if smoke else [] ),
            "--run-dir", str(trial_dir),
            "--video" if video else "--no-video",
            *[_override(key, value) for key, value in overrides.items()],
        ]
        print(f"\n[{trial_number}/{len(combinations)}] {trial_dir.name}")
        print(" ".join(command))
        completed = subprocess.run(command, cwd=ROOT)
        summary_path = trial_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        record = {
            "trial": trial_number,
            "trial_dir": str(trial_dir),
            "trainer_overrides": trainer,
            "environment_overrides": environment,
            "exit_code": completed.returncode,
            "score": summary.get("best_episode_reward"),
            "success": completed.returncode == 0 and summary.get("status") == "completed",
        }
        records.append(record)
        _write(output_dir / "results.json", records)

    ranked = sorted(
        records,
        key=lambda row: (row["score"] is not None, row["score"] if row["score"] is not None else float("-inf")),
        reverse=True,
    )
    _write(output_dir / "leaderboard.json", ranked)
    successful = [row for row in ranked if row["success"]]
    if successful:
        _write(output_dir / "best_trial.json", successful[0])
    return 0 if all(row["success"] for row in records) else 1


def main() -> int:
    """Parse the two obvious JSON inputs and launch the nested sweep."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-hyperparameters",
        type=Path,
        default=ROOT / "App" / "Configs" / "sweeps" / "training_hyperparameters.json",
        help="JSON list of PPO/trainer value bundles to test.",
    )
    parser.add_argument(
        "--environment-hyperparameters",
        type=Path,
        default=ROOT / "App" / "Configs" / "sweeps" / "environment_hyperparameters.json",
        help="JSON list of MJX environment value bundles to test.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Use the tiny integration profile for every trial.")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "Artifacts" / "Sweeps" / time.strftime("sweep_%Y%m%d_%H%M%S")
    return run_sweep(
        args.training_hyperparameters.resolve(),
        args.environment_hyperparameters.resolve(),
        output.resolve(),
        smoke=args.smoke,
        video=args.video,
    )


if __name__ == "__main__":
    raise SystemExit(main())
