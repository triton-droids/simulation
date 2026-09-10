"""Tests for the Gate 4 three-seed result aggregator."""

from pathlib import Path

import pytest

from source.scripts.report_g1_baseline import (
    EXPECTED_COMMANDS,
    EXPECTED_SEEDS,
    _best_checkpoint,
    _stats,
    _validate_episode_grid,
    _validate_summary,
)


def test_best_checkpoint_uses_highest_training_evaluation_return() -> None:
    records = [
        {"step": 0, "metrics": {"eval/episode_reward": -1.0}},
        {"step": 10, "metrics": {"eval/episode_reward": 2.0}},
        {"step": 20, "metrics": {"eval/episode_reward": 1.5}},
    ]

    assert _best_checkpoint(records) == (10, 2.0)


def test_three_seed_statistics_use_sample_interval() -> None:
    result = _stats([1.0, 2.0, 3.0])

    assert result["mean"] == pytest.approx(2.0)
    assert result["sample_std"] == pytest.approx(1.0)
    assert result["ci95_low"] == pytest.approx(-0.4841377117)
    assert result["ci95_high"] == pytest.approx(4.4841377117)


def test_summary_validation_rejects_changed_held_out_protocol() -> None:
    summary = {
        "reset_seeds": EXPECTED_SEEDS,
        "commands": EXPECTED_COMMANDS,
        "steps_per_episode": 500,
        "trained_checkpoint": 10,
        "untrained_checkpoint": 0,
        "reset_randomized": True,
        "observation_noise": False,
        "aggregate": {"trained": {}, "untrained": {}, "standing": {}},
    }
    _validate_summary(summary, 10, Path("matched"))

    changed = {**summary, "reset_seeds": [1000, 1001, 1002]}
    with pytest.raises(ValueError, match="Unexpected held-out seeds"):
        _validate_summary(changed, 10, Path("changed"))


def test_episode_grid_validation_rejects_duplicates() -> None:
    rows = []
    for controller in ("trained", "untrained", "standing"):
        for name, command in EXPECTED_COMMANDS.items():
            for seed in EXPECTED_SEEDS:
                rows.append(
                    {
                        "controller": controller,
                        "command_name": name,
                        "reset_seed": str(seed),
                        "requested_steps": "500",
                        "command_vx": str(command[0]),
                        "command_vy": str(command[1]),
                        "command_yaw_rate": str(command[2]),
                    }
                )

    _validate_episode_grid(rows, Path("matched"))
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        _validate_episode_grid(rows, Path("duplicate"))
