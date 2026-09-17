"""Evaluate G1 PPO checkpoints against matched untrained and standing controls.

The suite uses fixed held-out commands and reset seeds.  It writes per-episode
CSV, aggregate JSON, and optional representative videos without cloud services.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "jax_compilation_cache")
)
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "win32" else "egl")

from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp
import mediapy as media
import numpy as np
from omegaconf import OmegaConf

from source.locomotion import get_env_class
from source.robots import make_robot


COMMANDS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("stand", (0.0, 0.0, 0.0)),
    ("forward", (0.5, 0.0, 0.0)),
    ("backward", (-0.3, 0.0, 0.0)),
    ("left", (0.0, 0.3, 0.0)),
    ("right", (0.0, -0.3, 0.0)),
    ("turn_left", (0.0, 0.0, 0.5)),
    ("turn_right", (0.0, 0.0, -0.5)),
    ("combined", (0.4, 0.2, 0.35)),
)

_MEDIAPY_FFMPEG_SOURCE: str | None = None
ORIGINAL_ARGV = tuple(sys.argv)


def _utc_timestamp() -> str:
    """Return an unambiguous UTC timestamp for evaluation provenance."""

    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Trained checkpoint step; default is best logged evaluation reward.",
    )
    parser.add_argument("--untrained-checkpoint", type=int, default=0)
    parser.add_argument(
        "--reference-label", choices=("untrained", "initial"), default="untrained",
        help="Use initial when checkpoint 0 contains warm-started parameters.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--commands", nargs="+", choices=[name for name, _ in COMMANDS],
        default=None, help="Diagnostic command subset; default is the full frozen grid.",
    )
    parser.add_argument(
        "--seeds",
        default="2000,2001,2002",
        help="Comma-separated reset seeds (default: frozen Gate 4 held-out set).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--video-command", choices=[name for name, _ in COMMANDS], default="combined",
        help="Command to render for each controller on the first reset seed.",
    )
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument(
        "--nominal-reset",
        action="store_true",
        help="Disable initial-state randomization (fixed seeds remain recorded).",
    )
    parser.add_argument(
        "--min-pelvis-height",
        type=float,
        default=None,
        help="Evaluation-only override for diagnosing the configured fall cutoff.",
    )
    args = parser.parse_args()
    if args.video and args.commands is not None and args.video_command not in args.commands:
        parser.error("--video-command must belong to the selected --commands")
    return args


def _best_logged_checkpoint(run_dir: Path) -> int:
    metrics_path = run_dir / "logs" / "metrics.jsonl"
    candidates: list[tuple[float, int]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") == "metrics":
            reward = record["metrics"].get("eval/episode_reward")
            if reward is not None:
                candidates.append((float(reward), int(record["step"])))
    if not candidates:
        raise ValueError(f"No evaluation metrics found in {metrics_path}")
    return max(candidates)[1]


def _git_record() -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "core.autocrlf=true", *args],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    try:
        return {
            "available": True,
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Export archives deliberately omit .git.  Evaluation remains usable
        # there while making the missing provenance explicit in its summary.
        return {
            "available": False,
            "commit": None,
            "branch": None,
            "dirty": None,
        }


def _replace_command(state: Any, command: jax.Array) -> Any:
    """Set both command state and the command slice in existing observations."""

    info = dict(state.info)
    info["command"] = command
    observations = {
        name: value.at[9:12].set(command) for name, value in state.obs.items()
    }
    return state.replace(info=info, obs=observations)


def _step_with_held_command(
    env: Any, state: Any, action: jax.Array, command: jax.Array
) -> Any:
    """Take one transition while preventing environment command resampling.

    Playground samples a replacement command after its 501st transition.  A
    held-out evaluation must not silently become an evaluation of that sampled
    command, so install the scheduled command on both sides of every step.
    Installing it before the step also guarantees that transition rewards use
    the held command.
    """

    state = _replace_command(state, command)
    next_state = env.step(state, action)
    return _replace_command(next_state, command)


def _is_playground_environment(env: Any) -> bool:
    """Return whether ``env`` is the pinned Playground adapter."""

    return "playground_source" in getattr(env, "__dict__", {})


def _state_data(env: Any, state: Any) -> Any:
    """Return the simulator data carried by either environment State type."""

    return state.data if _is_playground_environment(env) else state.pipeline_state


def _qpos(env: Any, data: Any) -> jax.Array:
    return data.qpos if _is_playground_environment(env) else data.q


def _qvel(env: Any, data: Any) -> jax.Array:
    return data.qvel if _is_playground_environment(env) else data.qd


def _control_bounds(env: Any) -> tuple[jax.Array, jax.Array]:
    if _is_playground_environment(env):
        bounds = jp.asarray(env.mj_model.actuator_ctrlrange)
        return bounds[:, 0], bounds[:, 1]
    return env.ctrl_lower, env.ctrl_upper


def _soft_joint_bounds(env: Any) -> tuple[jax.Array, jax.Array]:
    if _is_playground_environment(env):
        return env._env._soft_lowers, env._env._soft_uppers
    return env.soft_joint_lower, env.soft_joint_upper


def _foot_global_linvel(env: Any, data: Any) -> jax.Array:
    if _is_playground_environment(env):
        return data.sensordata[env._env._foot_linvel_sensor_adr]
    return env.get_foot_global_linvel(data)


def _sensor_active(env: Any, data: Any, sensor_id: int) -> jax.Array:
    address = env.mj_model.sensor_adr[sensor_id]
    return data.sensordata[address] > 0


def _contact_diagnostics(
    env: Any, data: Any
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return support, illegal, collision, non-foot-floor and self contacts."""

    if not _is_playground_environment(env):
        contact, undesired, collision = env.contact_state(data)
        nonfoot_ground, self_collision = _contact_breakdown(env, data)
        return contact, undesired, collision, nonfoot_ground, self_collision

    upstream = env._env
    contact = env._contact(data)
    illegal = _sensor_active(
        env, data, upstream._right_foot_left_foot_found_sensor
    )
    illegal |= _sensor_active(
        env, data, upstream._left_foot_right_shin_found_sensor
    )
    illegal |= _sensor_active(
        env, data, upstream._right_foot_left_shin_found_sensor
    )
    # The authoritative feet-only scene exposes curated hand--thigh collision
    # sensors but not an exhaustive non-foot-floor collision classifier.
    self_collision = upstream._cost_collision(data)
    nonfoot_ground = jp.array(False)
    return contact, illegal, self_collision, nonfoot_ground, self_collision


def _instrumentation_metadata(env: Any) -> dict[str, Any]:
    """Describe backend-specific coverage without changing episode columns."""

    if _is_playground_environment(env):
        return {
            "environment_backend": "mujoco_playground_g1_joystick_adapter",
            "contact_support": "left/right foot-floor found sensors",
            "illegal_contact": "cross-foot and cross-foot/shin found sensors",
            "self_collision": "same-side hand/thigh found sensors",
            "nonfoot_ground_contact_available": False,
            "collision_coverage_limitation": (
                "The pinned authoritative feet-only scene provides curated "
                "collision sensors, not exhaustive robot/self/floor collision "
                "coverage; nonfoot_ground_contact_rate is therefore unavailable "
                "and emitted as zero."
            ),
        }
    return {
        "environment_backend": "native_brax_g1",
        "contact_support": "raw MJX geom contacts",
        "illegal_contact": "verified cross-foot and cross-foot/shin geom pairs",
        "self_collision": "verified same-side hand/thigh geom pairs",
        "nonfoot_ground_contact_available": True,
        "collision_coverage_limitation": None,
    }


def _action_diagnostics(
    env: Any, action: jax.Array
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Return the applied action/target and actuator-bound diagnostics.

    Policies can emit values outside the normalized action interval and the
    default-pose offset can reach a physical actuator limit before a normalized
    action reaches +/-1.  Keep those two saturation mechanisms separate.
    """

    applied_action, motor_targets = env.action_to_targets(action)
    ctrl_lower, ctrl_upper = _control_bounds(env)
    diagnostics = {
        "action_clipping_fraction": jp.mean(
            (jp.abs(action - applied_action) > 1.0e-6).astype(jp.float32)
        ),
        "action_saturation_fraction": jp.mean(
            (jp.abs(applied_action) >= 1.0 - 1.0e-6).astype(jp.float32)
        ),
        "target_saturation_fraction": jp.mean(
            (
                jp.isclose(motor_targets, ctrl_lower, rtol=0.0, atol=1.0e-6)
                | jp.isclose(
                    motor_targets, ctrl_upper, rtol=0.0, atol=1.0e-6
                )
            ).astype(jp.float32)
        ),
        "maximum_abs_applied_action": jp.max(jp.abs(applied_action)),
    }
    return applied_action, motor_targets, diagnostics


def _applied_action_rate(
    applied_action: jax.Array, previous_applied_action: jax.Array
) -> jax.Array:
    """Quadratic action delta using actuator-bound normalized actions."""

    return jp.sum(jp.square(applied_action - previous_applied_action))


def _contact_breakdown(env: Any, pipeline_state: Any) -> tuple[jax.Array, jax.Array]:
    """Separate non-foot floor contact from robot self-contact.

    The flat G1 scene has one floor geom.  Every active contact containing that
    geom is a ground contact, and every active contact containing no floor geom
    is a robot--robot contact.  The known support geoms are excluded from the
    non-foot-ground category.  This is intentionally evaluation-side
    instrumentation; it does not alter reward or termination behavior.
    """

    geom_pairs = pipeline_state.contact.geom
    active = pipeline_state.contact.dist < 0.0
    geom1 = geom_pairs[:, 0]
    geom2 = geom_pairs[:, 1]
    floor_first = geom1 == env.floor_geom_id
    floor_second = geom2 == env.floor_geom_id
    floor_contact = floor_first | floor_second
    foot_geom_ids = jp.concatenate(tuple(env.foot_geom_ids))
    foot_ground = (floor_first & jp.isin(geom2, foot_geom_ids)) | (
        floor_second & jp.isin(geom1, foot_geom_ids)
    )
    nonfoot_ground = jp.any(active & floor_contact & ~foot_ground)
    self_collision = jp.any(active & ~floor_contact)
    return nonfoot_ground, self_collision


def _build_rollout(
    env: Any, ppo_network: Any, steps: int, *, capture_pipeline: bool = False
):
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    def rollout(params: Any, use_policy: jax.Array, seed: jax.Array, command: jax.Array):
        reset_key, policy_key = jax.random.split(jax.random.PRNGKey(seed))
        state = _replace_command(env.reset(reset_key), command)
        initial_pipeline_state = _state_data(env, state)
        inference = make_policy(params, deterministic=True)

        def one_step(carry, _):
            (
                state,
                active,
                policy_key,
                previous_applied_action,
                previous_contact,
            ) = carry
            # Reinstate the held/scheduled command before policy inference as
            # well as before the reward-bearing transition.  This makes the
            # invariant explicit even if a backend resampled on the prior step.
            state = _replace_command(state, command)
            policy_key, action_key = jax.random.split(policy_key)
            policy_action, _ = inference(state.obs, action_key)
            action = jp.where(use_policy, policy_action, jp.zeros(env.action_size))
            applied_action, _, action_diagnostics = _action_diagnostics(env, action)

            def active_step(current_state):
                # The environments use mutable Python dictionaries as PyTree
                # containers. Copy them so the inactive branch remains intact.
                current_state = current_state.replace(
                    info=dict(current_state.info), metrics=dict(current_state.metrics)
                )
                return _step_with_held_command(env, current_state, action, command)

            next_state = jax.lax.cond(active, active_step, lambda value: value, state)
            done = next_state.done > 0.0
            next_active = active & ~done

            data = _state_data(env, next_state)
            qpos = _qpos(env, data)
            qvel = _qvel(env, data)
            local_velocity = env.get_local_linvel(data, "pelvis")
            gyro = env.get_gyro(data, "pelvis")
            torso_up = env.get_gravity(data, "torso")
            (
                contact,
                undesired,
                collision,
                nonfoot_ground,
                self_collision,
            ) = _contact_diagnostics(
                env, data
            )
            contact_transition = contact != previous_contact
            foot_velocity = _foot_global_linvel(env, data)[:, :2]
            joint_position = qpos[7:]
            soft_lower, soft_upper = _soft_joint_bounds(env)
            low = qpos[2] < env.cfg.termination.min_pelvis_height
            high = qpos[2] > env.cfg.termination.max_pelvis_height
            inverted = torso_up[2] < env.cfg.termination.min_torso_up_z
            invalid = ~jp.isfinite(qpos).all() | ~jp.isfinite(qvel).all()
            limit_violation = jp.any(
                (joint_position < soft_lower) | (joint_position > soft_upper)
            )
            trace = {
                "valid": active,
                "done": done,
                "reward": next_state.reward,
                "linear_error": local_velocity[:2] - command[:2],
                "yaw_error": gyro[2] - command[2],
                "pelvis_height": qpos[2],
                "torso_up_z": torso_up[2],
                "effort": jp.sum(jp.abs(data.actuator_force)),
                "power": jp.sum(
                    jp.abs(qvel[6:] * data.actuator_force)
                ),
                # Match the environment reward semantics: action rate is based
                # on the normalized action actually applied after clipping.
                "action_rate": _applied_action_rate(
                    applied_action, previous_applied_action
                ),
                **action_diagnostics,
                "joint_acceleration": jp.sum(
                    jp.square(data.qacc[6:])
                ),
                "foot_slip": jp.sum(
                    jp.sum(jp.square(foot_velocity), axis=-1) * contact
                ),
                "left_contact": contact[0],
                "right_contact": contact[1],
                "undesired_contact": undesired,
                "collision_contact": collision,
                "nonfoot_ground_contact": nonfoot_ground,
                "self_collision_contact": self_collision,
                "joint_limit_violation": limit_violation,
                "left_contact_transition": contact_transition[0],
                "right_contact_transition": contact_transition[1],
                "cause_low": low,
                "cause_high": high,
                "cause_inverted": inverted,
                "cause_undesired": undesired,
                "cause_invalid": invalid,
            }
            if capture_pipeline:
                trace["pipeline_state"] = data
            return (
                next_state,
                next_active,
                policy_key,
                applied_action,
                contact,
            ), trace

        initial_carry = (
            state,
            jp.array(True),
            policy_key,
            state.info["last_act"],
            _contact_diagnostics(env, _state_data(env, state))[0],
        )
        _, trace = jax.lax.scan(one_step, initial_carry, xs=None, length=steps)
        return initial_pipeline_state, trace

    return jax.jit(rollout)


def _make_evaluation_network(
    network_factory: Any,
    observation_size: Any,
    action_size: int,
    *,
    normalize_observations: bool,
) -> Any:
    """Reconstruct a PPO network with the training-time preprocessing.

    Brax stores running observation statistics alongside the policy weights,
    but those statistics are only applied when the network is constructed with
    ``running_statistics.normalize``.  Omitting that callback silently loads a
    valid checkpoint whose actions no longer match training-time inference.
    """

    kwargs: dict[str, Any] = {}
    if normalize_observations:
        kwargs["preprocess_observations_fn"] = running_statistics.normalize
    return network_factory(observation_size, action_size, **kwargs)


def _masked_mean(value: np.ndarray, valid: np.ndarray) -> float:
    return float(np.asarray(value)[valid].mean())


def _summarize_trace(
    trace: dict[str, np.ndarray],
    *,
    controller: str,
    command_name: str,
    command: tuple[float, float, float],
    seed: int,
    dt: float,
    requested_steps: int,
) -> dict[str, Any]:
    valid = np.asarray(trace["valid"], dtype=bool)
    completed_steps = int(valid.sum())
    if completed_steps == 0:
        raise ValueError("Evaluation rollout contained no valid control step")
    terminal_index = completed_steps - 1
    linear_error = np.asarray(trace["linear_error"])[valid]
    yaw_error = np.asarray(trace["yaw_error"])[valid]
    linear_norm = np.linalg.norm(linear_error, axis=-1)
    yaw_abs = np.abs(yaw_error)
    left_contact = np.asarray(trace["left_contact"])[valid].astype(bool)
    right_contact = np.asarray(trace["right_contact"])[valid].astype(bool)
    double_support = left_contact & right_contact
    left_only_support = left_contact & ~right_contact
    right_only_support = ~left_contact & right_contact
    single_support = left_only_support | right_only_support
    flight = ~left_contact & ~right_contact
    left_transition_count = int(
        np.count_nonzero(np.asarray(trace["left_contact_transition"])[valid])
    )
    right_transition_count = int(
        np.count_nonzero(np.asarray(trace["right_contact_transition"])[valid])
    )
    episode_duration = completed_steps * dt
    fall = bool(np.asarray(trace["done"])[terminal_index])
    all_numeric = [
        np.asarray(value)[valid]
        for name, value in trace.items()
        if name not in {"pipeline_state", "valid"}
        and np.issubdtype(np.asarray(value).dtype, np.number)
    ]

    return {
        "controller": controller,
        "command_name": command_name,
        "command_vx": command[0],
        "command_vy": command[1],
        "command_yaw_rate": command[2],
        "reset_seed": seed,
        "requested_steps": requested_steps,
        "episode_steps": completed_steps,
        "episode_duration_seconds": completed_steps * dt,
        "fall": fall,
        "survived_full_horizon": completed_steps == requested_steps and not fall,
        "linear_velocity_rmse": float(np.sqrt(np.mean(np.square(linear_error)))),
        "linear_velocity_vector_rmse": float(
            np.sqrt(np.mean(np.sum(np.square(linear_error), axis=-1)))
        ),
        "linear_velocity_mae": float(linear_norm.mean()),
        "yaw_rate_rmse": float(np.sqrt(np.mean(np.square(yaw_error)))),
        "yaw_rate_mae": float(yaw_abs.mean()),
        "tracking_success_fraction": float(
            np.mean((linear_norm <= 0.25) & (yaw_abs <= 0.25))
        ),
        "episode_success": bool(
            completed_steps == requested_steps
            and not fall
            and linear_norm.mean() <= 0.25
            and yaw_abs.mean() <= 0.25
        ),
        "mean_reward": _masked_mean(trace["reward"], valid),
        "episode_return": float(np.asarray(trace["reward"])[valid].sum()),
        "mean_torso_tilt_degrees": float(
            np.degrees(
                np.arccos(np.clip(np.asarray(trace["torso_up_z"])[valid], -1.0, 1.0))
            ).mean()
        ),
        "mean_pelvis_height": _masked_mean(trace["pelvis_height"], valid),
        "minimum_pelvis_height": float(
            np.asarray(trace["pelvis_height"])[valid].min()
        ),
        "mean_actuator_effort": _masked_mean(trace["effort"], valid),
        "mechanical_energy_proxy": float(
            np.asarray(trace["power"])[valid].sum() * dt
        ),
        "mean_action_rate_cost": _masked_mean(trace["action_rate"], valid),
        "mean_action_clipping_fraction": _masked_mean(
            trace["action_clipping_fraction"], valid
        ),
        "mean_action_saturation_fraction": _masked_mean(
            trace["action_saturation_fraction"], valid
        ),
        "mean_target_saturation_fraction": _masked_mean(
            trace["target_saturation_fraction"], valid
        ),
        "maximum_abs_applied_action": float(
            np.asarray(trace["maximum_abs_applied_action"])[valid].max()
        ),
        "mean_joint_acceleration_cost": _masked_mean(
            trace["joint_acceleration"], valid
        ),
        "mean_foot_slip_cost": _masked_mean(trace["foot_slip"], valid),
        "undesired_contact_rate": _masked_mean(trace["undesired_contact"], valid),
        "collision_contact_rate": _masked_mean(trace["collision_contact"], valid),
        "nonfoot_ground_contact_rate": _masked_mean(
            trace["nonfoot_ground_contact"], valid
        ),
        "self_collision_contact_rate": _masked_mean(
            trace["self_collision_contact"], valid
        ),
        "joint_limit_violation_rate": _masked_mean(
            trace["joint_limit_violation"], valid
        ),
        "left_contact_duty": float(left_contact.mean()),
        "right_contact_duty": float(right_contact.mean()),
        "gait_contact_asymmetry": abs(
            float(left_contact.mean()) - float(right_contact.mean())
        ),
        "double_support_fraction": float(double_support.mean()),
        "single_support_fraction": float(single_support.mean()),
        "flight_fraction": float(flight.mean()),
        "left_only_support_fraction": float(left_only_support.mean()),
        "right_only_support_fraction": float(right_only_support.mean()),
        "left_contact_transition_count": left_transition_count,
        "right_contact_transition_count": right_transition_count,
        "contact_transitions_per_second": float(
            (left_transition_count + right_transition_count) / episode_duration
        ),
        "both_feet_transitioned": bool(
            left_transition_count > 0 and right_transition_count > 0
        ),
        "terminal_low_pelvis": bool(np.asarray(trace["cause_low"])[terminal_index]),
        "terminal_high_pelvis": bool(np.asarray(trace["cause_high"])[terminal_index]),
        "terminal_inverted": bool(np.asarray(trace["cause_inverted"])[terminal_index]),
        "terminal_undesired_contact": bool(
            np.asarray(trace["cause_undesired"])[terminal_index]
        ),
        "terminal_invalid_state": bool(
            np.asarray(trace["cause_invalid"])[terminal_index]
        ),
        "all_finite": all(np.isfinite(value).all() for value in all_numeric),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    numeric_keys = (
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
        "mean_action_clipping_fraction",
        "mean_action_saturation_fraction",
        "mean_target_saturation_fraction",
        "maximum_abs_applied_action",
        "mean_joint_acceleration_cost",
        "mean_foot_slip_cost",
        "undesired_contact_rate",
        "collision_contact_rate",
        "nonfoot_ground_contact_rate",
        "self_collision_contact_rate",
        "joint_limit_violation_rate",
        "left_contact_duty",
        "right_contact_duty",
        "gait_contact_asymmetry",
        "double_support_fraction",
        "single_support_fraction",
        "flight_fraction",
        "left_only_support_fraction",
        "right_only_support_fraction",
        "left_contact_transition_count",
        "right_contact_transition_count",
        "contact_transitions_per_second",
    )
    for controller in sorted({row["controller"] for row in rows}):
        selected = [row for row in rows if row["controller"] == controller]
        aggregate = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in numeric_keys
        }
        aggregate["episodes"] = float(len(selected))
        aggregate["fall_rate"] = float(np.mean([row["fall"] for row in selected]))
        aggregate["episode_success_rate"] = float(
            np.mean([row["episode_success"] for row in selected])
        )
        aggregate["finite_rate"] = float(
            np.mean([row["all_finite"] for row in selected])
        )
        aggregate["both_feet_transitioned_rate"] = float(
            np.mean([row["both_feet_transitioned"] for row in selected])
        )
        output[controller] = aggregate
    return output


def _write_video(
    env: Any,
    initial_pipeline_state: Any,
    trace: dict[str, Any],
    episode_steps: int,
    path: Path,
    render_every: int,
) -> str:
    global _MEDIAPY_FFMPEG_SOURCE

    stacked = trace["pipeline_state"]
    frame_data = [initial_pipeline_state]
    for index in range(episode_steps):
        frame_data.append(jax.tree.map(lambda value: value[index], stacked))
    if _is_playground_environment(env):
        # Playground's renderer accepts complete mjx_env.State objects even
        # though it consumes only ``state.data``.  Rebuild that public state
        # type from each captured frame rather than passing raw mjx.Data.
        zero = jp.zeros(())
        states = [
            env._mjx_env_module.State(data, {}, zero, zero, {}, {})
            for data in frame_data
        ]
    else:
        states = frame_data
    frames = env.render(
        states[::render_every], height=480, width=640, camera="track"
    )
    fps = 1.0 / env.dt / render_every
    if not media.video_is_available():
        import imageio_ffmpeg

        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        _MEDIAPY_FFMPEG_SOURCE = "mediapy_imageio_ffmpeg"
    elif _MEDIAPY_FFMPEG_SOURCE is None:
        _MEDIAPY_FFMPEG_SOURCE = "mediapy_system_ffmpeg"
    try:
        media.write_video(path, frames, fps=fps)
        assert _MEDIAPY_FFMPEG_SOURCE is not None
        return _MEDIAPY_FFMPEG_SOURCE
    except RuntimeError as error:
        if "ffmpeg" not in str(error).lower():
            raise

        # MediaPy requires a separately installed ffmpeg executable.  OpenCV's
        # wheels include an MP4 backend, which keeps local/WSL rendering usable
        # without mutating the host with a system package installation.
        import cv2

        if not frames:
            raise RuntimeError("Cannot encode an empty rollout") from error
        height, width = np.asarray(frames[0]).shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open its MP4 writer") from error
        try:
            for frame in frames:
                rgb = np.asarray(frame)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("OpenCV produced no MP4 output") from error
        return "opencv_mp4v"


def main() -> None:
    started_at_utc = _utc_timestamp()
    started_monotonic = time.perf_counter()
    args = parse_args()
    commands = tuple((name, value) for name, value in COMMANDS
                     if args.commands is None or name in args.commands)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")

    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = _best_logged_checkpoint(run_dir)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "evaluation" / f"checkpoint_{checkpoint}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    cfg = OmegaConf.load(run_dir / "resolved_config.json")
    cfg.robot.fetch_model = False
    cfg.sim.noise.add_noise = False
    cfg.sim.push.add_push = False
    cfg.sim.domain_rand.add_domain_rand = False
    cfg.sim.commands.resample_time = (args.steps + 1) * (
        cfg.sim.sim.timestep * cfg.sim.action.n_frames
    )
    if args.nominal_reset:
        cfg.sim.reset.randomize = False
    if args.min_pelvis_height is not None:
        if args.min_pelvis_height <= 0.0:
            raise ValueError("--min-pelvis-height must be positive")
        cfg.sim.termination.min_pelvis_height = args.min_pelvis_height

    robot = make_robot(cfg.robot)
    EnvClass = get_env_class(cfg.env.name)
    env = EnvClass(cfg.robot.name, robot, cfg.env.terrain, cfg.sim)
    factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    ppo_network = _make_evaluation_network(
        factory,
        env.observation_size,
        env.action_size,
        normalize_observations=bool(cfg.agent.normalize_observations),
    )
    rollout = _build_rollout(env, ppo_network, args.steps)
    video_rollout = (
        _build_rollout(env, ppo_network, args.steps, capture_pipeline=True)
        if args.video
        else None
    )

    checkpoint_root = run_dir / "logs" / "checkpoints"
    trained_params = model.load_params(checkpoint_root / str(checkpoint) / "policy")
    untrained_params = model.load_params(
        checkpoint_root / str(args.untrained_checkpoint) / "policy"
    )
    controllers = (
        ("trained", trained_params, True),
        (args.reference_label, untrained_params, True),
        ("standing", untrained_params, False),
    )

    rows: list[dict[str, Any]] = []
    videos: list[str] = []
    video_encoders: dict[str, str] = {}
    for controller, params, use_policy in controllers:
        for command_name, command_values in commands:
            for seed in seeds:
                initial_state, device_trace = rollout(
                    params,
                    jp.asarray(use_policy),
                    jp.asarray(seed, dtype=jp.int32),
                    jp.asarray(command_values),
                )
                jax.block_until_ready(device_trace["reward"])
                host_trace = jax.tree.map(np.asarray, device_trace)
                row = _summarize_trace(
                    host_trace,
                    controller=controller,
                    command_name=command_name,
                    command=command_values,
                    seed=seed,
                    dt=env.dt,
                    requested_steps=args.steps,
                )
                rows.append(row)
                # Preserve completed numeric evidence even if a later render fails.
                with (output_dir / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(
                    f"{controller:9s} {command_name:10s} seed={seed} "
                    f"steps={row['episode_steps']:4d} "
                    f"lin_rmse={row['linear_velocity_vector_rmse']:.3f} "
                    f"yaw_rmse={row['yaw_rate_rmse']:.3f}"
                )
                if args.video and command_name == args.video_command and seed == seeds[0]:
                    assert video_rollout is not None
                    video_initial_state, video_trace = video_rollout(
                        params,
                        jp.asarray(use_policy),
                        jp.asarray(seed, dtype=jp.int32),
                        jp.asarray(command_values),
                    )
                    jax.block_until_ready(video_trace["reward"])
                    video_path = output_dir / f"{controller}_{command_name}_seed{seed}.mp4"
                    encoder = _write_video(
                        env,
                        video_initial_state,
                        video_trace,
                        row["episode_steps"],
                        video_path,
                        args.render_every,
                    )
                    videos.append(video_path.name)
                    video_encoders[video_path.name] = encoder

    csv_path = output_dir / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "kind": ("gate4_fixed_held_out_velocity_evaluation" if args.commands is None
                 else "g1_diagnostic_command_subset"),
        "command": list(ORIGINAL_ARGV),
        "started_at_utc": started_at_utc,
        "run_dir": str(run_dir),
        "trained_checkpoint": checkpoint,
        "untrained_checkpoint": args.untrained_checkpoint,
        "reference_controller_label": args.reference_label,
        "steps_per_episode": args.steps,
        "control_dt_seconds": env.dt,
        "reset_seeds": seeds,
        "reset_randomized": bool(cfg.sim.reset.randomize),
        "minimum_pelvis_height": float(cfg.sim.termination.min_pelvis_height),
        "observation_noise": bool(cfg.sim.noise.add_noise),
        "commands": {name: values for name, values in commands},
        "thresholds": {
            "instantaneous_linear_error_norm": 0.25,
            "instantaneous_yaw_error_abs": 0.25,
            "episode_success_requires_full_horizon": True,
        },
        "instrumentation": _instrumentation_metadata(env),
        "aggregate": _aggregate(rows),
        "videos": videos,
        "video_command": args.video_command if args.video else None,
        "video_encoders": video_encoders,
        "model": robot.source_record,
        "git": _git_record(),
        "platform": platform.platform(),
        "jax_devices": [str(device) for device in jax.devices()],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mujoco", "jax", "jaxlib", "brax")
        },
    }
    result["ended_at_utc"] = _utc_timestamp()
    result["wall_time_seconds"] = time.perf_counter() - started_monotonic
    json_path = output_dir / "summary.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")

    if not all(row["all_finite"] for row in rows):
        raise SystemExit("Evaluation produced non-finite metrics")


if __name__ == "__main__":
    main()
