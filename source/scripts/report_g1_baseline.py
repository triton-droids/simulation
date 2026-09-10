"""Aggregate the frozen three-seed Gate 4 G1 baseline evaluation.

This script deliberately consumes completed evaluation artifacts rather than
running policies.  It validates the frozen commands/reset seeds, preserves
per-policy results, and computes the decision rule declared in
``research/EXPERIMENT_PLAN.md`` before the held-out suite was inspected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


EXPECTED_SEEDS = [2000, 2001, 2002]
EXPECTED_COMMANDS = {
    "stand": [0.0, 0.0, 0.0],
    "forward": [0.5, 0.0, 0.0],
    "backward": [-0.3, 0.0, 0.0],
    "left": [0.0, 0.3, 0.0],
    "right": [0.0, -0.3, 0.0],
    "turn_left": [0.0, 0.0, 0.5],
    "turn_right": [0.0, 0.0, -0.5],
    "combined": [0.4, 0.2, 0.35],
}
CONTROLLERS = ("trained", "untrained", "standing")
METRICS = (
    "episode_duration_seconds",
    "linear_velocity_vector_rmse",
    "linear_velocity_mae",
    "yaw_rate_rmse",
    "yaw_rate_mae",
    "tracking_success_fraction",
    "episode_return",
    "mean_torso_tilt_degrees",
    "mean_pelvis_height",
    "minimum_pelvis_height",
    "mean_actuator_effort",
    "mechanical_energy_proxy",
    "mean_action_rate_cost",
    "mean_joint_acceleration_cost",
    "mean_foot_slip_cost",
    "undesired_contact_rate",
    "collision_contact_rate",
    "joint_limit_violation_rate",
    "gait_contact_asymmetry",
    "fall_rate",
    "episode_success_rate",
    "finite_rate",
)
T_CRITICAL_95_DF2 = 4.302652729696142


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate fixed held-out G1 evaluations across policy seeds."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        type=Path,
        help="Completed run directory; pass exactly once for each of seeds 0, 1, 2.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _metric_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    with (run_dir / "logs" / "metrics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("event") == "metrics":
                records.append(record)
    if not records:
        raise ValueError(f"No metric records in {run_dir}")
    return records


def _best_checkpoint(records: list[dict[str, Any]]) -> tuple[int, float]:
    candidates = [
        (int(record["step"]), float(record["metrics"]["eval/episode_reward"]))
        for record in records
        if "eval/episode_reward" in record.get("metrics", {})
    ]
    if not candidates:
        raise ValueError("No evaluation reward was logged")
    return max(candidates, key=lambda item: item[1])


def _validate_summary(summary: dict[str, Any], checkpoint: int, run_dir: Path) -> None:
    if summary.get("reset_seeds") != EXPECTED_SEEDS:
        raise ValueError(f"Unexpected held-out seeds in {run_dir}: {summary.get('reset_seeds')}")
    if summary.get("commands") != EXPECTED_COMMANDS:
        raise ValueError(f"Unexpected held-out commands in {run_dir}")
    if int(summary.get("steps_per_episode", -1)) != 500:
        raise ValueError(f"Expected 500-step episodes in {run_dir}")
    if int(summary.get("trained_checkpoint", -1)) != checkpoint:
        raise ValueError(
            f"Evaluation checkpoint {summary.get('trained_checkpoint')} does not match "
            f"training-side selector {checkpoint} in {run_dir}"
        )
    if int(summary.get("untrained_checkpoint", -1)) != 0:
        raise ValueError(f"Expected checkpoint-zero untrained control in {run_dir}")
    if summary.get("reset_randomized") is not True:
        raise ValueError(f"Expected randomized held-out resets in {run_dir}")
    if summary.get("observation_noise") is not False:
        raise ValueError(f"Expected observation noise disabled in {run_dir}")
    if set(summary.get("aggregate", {})) != set(CONTROLLERS):
        raise ValueError(f"Missing controller aggregate in {run_dir}")


def _validate_episode_grid(rows: list[dict[str, str]], run_dir: Path) -> None:
    expected = {
        (controller, command, str(seed))
        for controller in CONTROLLERS
        for command in EXPECTED_COMMANDS
        for seed in EXPECTED_SEEDS
    }
    observed = {
        (row.get("controller", ""), row.get("command_name", ""), row.get("reset_seed", ""))
        for row in rows
    }
    if len(rows) != len(expected) or observed != expected:
        raise ValueError(f"Incomplete or duplicate held-out episode grid in {run_dir}")
    for row in rows:
        command = EXPECTED_COMMANDS[row["command_name"]]
        recorded = [
            float(row["command_vx"]),
            float(row["command_vy"]),
            float(row["command_yaw_rate"]),
        ]
        if recorded != command or int(row["requested_steps"]) != 500:
            raise ValueError(f"Changed episode protocol in {run_dir}: {row}")


def _stats(values: list[float]) -> dict[str, float]:
    count = len(values)
    avg = mean(values)
    sample_std = stdev(values) if count > 1 else 0.0
    half_width = (
        T_CRITICAL_95_DF2 * sample_std / math.sqrt(count) if count == 3 else math.nan
    )
    return {
        "mean": avg,
        "sample_std": sample_std,
        "ci95_low": avg - half_width,
        "ci95_high": avg + half_width,
        "policy_seed_count": float(count),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_training_curves(rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for seed in sorted({int(row["policy_seed"]) for row in rows}):
        selected = [row for row in rows if int(row["policy_seed"]) == seed]
        steps_m = [float(row["step"]) / 1_000_000 for row in selected]
        axes[0].plot(
            steps_m,
            [row["eval_episode_reward"] for row in selected],
            label=f"seed {seed}",
        )
        axes[1].plot(
            steps_m,
            [row["eval_avg_episode_length"] for row in selected],
            label=f"seed {seed}",
        )
    axes[0].set(
        title="Training-side evaluation return",
        xlabel="environment steps (millions)",
        ylabel="mean return",
    )
    axes[1].set(
        title="Training-side episode length",
        xlabel="environment steps (millions)",
        ylabel="control steps",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_held_out_comparison(
    statistics: dict[str, dict[str, dict[str, float]]], output_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    panels = (
        ("linear_velocity_vector_rmse", "Linear velocity RMSE", "m/s"),
        ("yaw_rate_rmse", "Yaw-rate RMSE", "rad/s"),
        ("episode_duration_seconds", "Episode duration", "seconds"),
        ("fall_rate", "Fall rate", "fraction"),
    )
    labels = ["trained", "untrained", "standing"]
    colors = ["#2878b5", "#9c9c9c", "#d07c2c"]
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 7), constrained_layout=True)
    for axis, (metric, title, unit) in zip(axes.flat, panels):
        values = [statistics[controller][metric]["mean"] for controller in labels]
        errors = [
            statistics[controller][metric]["ci95_high"] - values[index]
            for index, controller in enumerate(labels)
        ]
        axis.bar(labels, values, yerr=errors, capsize=4, color=colors)
        axis.set(title=title, ylabel=unit)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Frozen held-out commands; mean and 95% t interval across 3 policy seeds")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    run_dirs = [path.resolve() for path in args.run_dir]
    if len(run_dirs) != 3:
        raise ValueError(f"Expected exactly three run directories, got {len(run_dirs)}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()

    for run_dir in run_dirs:
        config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
        policy_seed = int(config["agent"]["seed"])
        if policy_seed in seen_seeds:
            raise ValueError(f"Duplicate policy seed {policy_seed}")
        seen_seeds.add(policy_seed)
        records = _metric_records(run_dir)
        best_step, best_reward = _best_checkpoint(records)
        evaluation_dir = run_dir / "evaluation" / f"checkpoint_{best_step}"
        summary = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
        _validate_summary(summary, best_step, run_dir)

        for controller in CONTROLLERS:
            aggregate = summary["aggregate"][controller]
            row: dict[str, Any] = {
                "policy_seed": policy_seed,
                "run_name": run_dir.name,
                "selected_checkpoint": best_step,
                "training_eval_reward_at_selection": best_reward,
                "controller": controller,
            }
            row.update({metric: float(aggregate[metric]) for metric in METRICS})
            per_seed.append(row)

        with (evaluation_dir / "episodes.csv").open(newline="", encoding="utf-8") as stream:
            run_episodes = list(csv.DictReader(stream))
        _validate_episode_grid(run_episodes, run_dir)
        for episode in run_episodes:
            episodes.append(
                {"policy_seed": policy_seed, "run_name": run_dir.name, **episode}
            )

        for record in records:
            metrics = record["metrics"]
            curves.append(
                {
                    "policy_seed": policy_seed,
                    "step": int(record["step"]),
                    "eval_episode_reward": float(metrics["eval/episode_reward"]),
                    "eval_avg_episode_length": float(metrics["eval/avg_episode_length"]),
                    "training_kl_mean": float(metrics.get("training/kl_mean", 0.0)),
                    "training_walltime_seconds": float(metrics.get("training/walltime", 0.0)),
                    "evaluation_walltime_seconds": float(metrics.get("eval/walltime", 0.0)),
                }
            )

    if seen_seeds != {0, 1, 2}:
        raise ValueError(f"Expected policy seeds 0, 1, 2; got {sorted(seen_seeds)}")
    if len(episodes) != 3 * 3 * 8 * 3:
        raise ValueError(f"Expected 216 episode rows, got {len(episodes)}")

    controller_statistics: dict[str, dict[str, dict[str, float]]] = {}
    for controller in CONTROLLERS:
        controller_rows = [row for row in per_seed if row["controller"] == controller]
        controller_statistics[controller] = {
            metric: _stats([float(row[metric]) for row in controller_rows])
            for metric in METRICS
        }

    seed_decisions = []
    for seed in sorted(seen_seeds):
        by_controller = {
            row["controller"]: row for row in per_seed if row["policy_seed"] == seed
        }
        trained = by_controller["trained"]
        controls = [by_controller["untrained"], by_controller["standing"]]
        checks = {
            "linear_rmse_better_than_both": trained["linear_velocity_vector_rmse"]
            < min(control["linear_velocity_vector_rmse"] for control in controls),
            "yaw_rmse_better_than_both": trained["yaw_rate_rmse"]
            < min(control["yaw_rate_rmse"] for control in controls),
            "duration_better_than_both": trained["episode_duration_seconds"]
            > max(control["episode_duration_seconds"] for control in controls),
            "has_full_horizon_rollout": trained["fall_rate"] < 1.0,
            "all_finite": all(row["finite_rate"] == 1.0 for row in by_controller.values()),
        }
        seed_decisions.append(
            {"policy_seed": seed, **checks, "passes_matched_rule": all(checks.values())}
        )

    family = {
        controller: {
            metric: controller_statistics[controller][metric]["mean"]
            for metric in METRICS
        }
        for controller in CONTROLLERS
    }
    trained = family["trained"]
    controls = [family["untrained"], family["standing"]]
    family_checks = {
        "linear_rmse_better_than_both": trained["linear_velocity_vector_rmse"]
        < min(control["linear_velocity_vector_rmse"] for control in controls),
        "yaw_rmse_better_than_both": trained["yaw_rate_rmse"]
        < min(control["yaw_rate_rmse"] for control in controls),
        "duration_better_than_both": trained["episode_duration_seconds"]
        > max(control["episode_duration_seconds"] for control in controls),
        "has_full_horizon_rollout": trained["fall_rate"] < 1.0,
        "all_finite": all(controller["finite_rate"] == 1.0 for controller in family.values()),
        "at_least_two_seeds_pass": sum(
            decision["passes_matched_rule"] for decision in seed_decisions
        )
        >= 2,
    }
    result = {
        "kind": "gate4_three_seed_held_out_aggregate",
        "policy_seeds": sorted(seen_seeds),
        "reset_seeds": EXPECTED_SEEDS,
        "commands": EXPECTED_COMMANDS,
        "controller_statistics_across_policy_seeds": controller_statistics,
        "per_seed_decisions": seed_decisions,
        "family_checks": family_checks,
        "evaluation_rule_passes": all(family_checks.values()),
        "note": "Video review and clean-checkout commands are separate Gate 4 requirements.",
    }

    _write_csv(output_dir / "per_policy_seed.csv", per_seed)
    _write_csv(output_dir / "all_episodes.csv", episodes)
    _write_csv(output_dir / "training_curves.csv", curves)
    _plot_training_curves(curves, output_dir / "training_curves.png")
    _plot_held_out_comparison(
        controller_statistics, output_dir / "heldout_comparison.png"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "family_checks": family_checks,
                "evaluation_rule_passes": result["evaluation_rule_passes"],
            },
            indent=2,
        )
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
