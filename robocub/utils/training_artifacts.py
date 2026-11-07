"""Human-facing plots and summaries for every PPO training run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


def _scalar(value: Any) -> float | None:
    """Convert a finite scalar or scalar array to float; reject other values."""

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


class TrainingArtifactMonitor:
    """Collect progress callbacks and write stable final artifacts."""

    def __init__(self, run_dir: Path):
        """Create artifact folders and start an empty in-memory metric history."""

        self.run_dir = run_dir
        self.plots_dir = run_dir / "plots"
        self.videos_dir = run_dir / "videos"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []

    def record(self, step: int, metrics: Mapping[str, Any]) -> None:
        """Store one PPO progress callback using JSON-compatible scalars."""

        row: dict[str, Any] = {"step": int(step)}
        for key, value in metrics.items():
            scalar = _scalar(value)
            if scalar is not None:
                row[str(key)] = scalar
        self.rows.append(row)

    def finalize(
        self,
        *,
        status: str,
        best_checkpoint_step: int,
        best_episode_reward: float,
        wall_time_seconds: float,
    ) -> dict[str, Any]:
        """Write the run summary, raw progress history, and final graph."""

        summary = {
            "status": status,
            "progress_callbacks": len(self.rows),
            "final_step": self.rows[-1]["step"] if self.rows else 0,
            "best_checkpoint_step": int(best_checkpoint_step),
            "best_episode_reward": _scalar(best_episode_reward),
            "wall_time_seconds": float(wall_time_seconds),
            "plot": "plots/final_progress.png",
            "video": (
                "videos/final_walk-eval.mp4"
                if (self.videos_dir / "final_walk-eval.mp4").is_file()
                else None
            ),
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (self.run_dir / "progress.json").write_text(
            json.dumps(self.rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._save_plot()
        return summary

    def _save_plot(self) -> None:
        """Plot reward and available PPO diagnostics into one PNG."""

        from matplotlib import pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(14, 5))
        steps = [row["step"] for row in self.rows]
        reward_key = next(
            (key for key in ("eval/episode_reward", "training/episode_reward") if any(key in row for row in self.rows)),
            None,
        )
        if reward_key:
            axes[0].plot(
                [row["step"] for row in self.rows if reward_key in row],
                [row[reward_key] for row in self.rows if reward_key in row],
            )
        axes[0].set_title("Episode Reward")
        axes[0].set_xlabel("Environment Steps")
        axes[0].grid(alpha=0.3)

        secondary = [
            key
            for key in ("training/policy_loss", "training/value_loss", "training/entropy_loss", "training/kl_mean")
            if any(key in row for row in self.rows)
        ]
        for key in secondary:
            axes[1].plot(
                [row["step"] for row in self.rows if key in row],
                [row[key] for row in self.rows if key in row],
                label=key.removeprefix("training/"),
            )
        axes[1].set_title("Training Diagnostics")
        axes[1].set_xlabel("Environment Steps")
        axes[1].grid(alpha=0.3)
        if secondary:
            axes[1].legend()
        if not steps:
            axes[0].text(0.5, 0.5, "No progress callbacks", ha="center", va="center")
        figure.tight_layout()
        figure.savefig(self.plots_dir / "final_progress.png", dpi=150)
        plt.close(figure)
