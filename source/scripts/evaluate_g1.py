"""Evaluate G1 PPO checkpoints against matched untrained and standing controls.

The suite uses fixed held-out commands and reset seeds.  It writes per-episode
CSV, aggregate JSON, and optional representative videos without cloud services.
"""

from __future__ import annotations

import argparse
import csv
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "jax_compilation_cache")
)
os.environ.setdefault("MUJOCO_GL", "egl")

from brax.io import model
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
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--seeds",
        default="2000,2001,2002",
        help="Comma-separated reset seeds (default: frozen Gate 4 held-out set).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--video", action="store_true")
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
    return parser.parse_args()


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
            ["git", *args],
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


def _build_rollout(
    env: Any, ppo_network: Any, steps: int, *, capture_pipeline: bool = False
):
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    def rollout(params: Any, use_policy: jax.Array, seed: jax.Array, command: jax.Array):
        reset_key, policy_key = jax.random.split(jax.random.PRNGKey(seed))
        state = _replace_command(env.reset(reset_key), command)
        initial_pipeline_state = state.pipeline_state
        inference = make_policy(params, deterministic=True)

        def one_step(carry, _):
            state, active, policy_key, previous_action = carry
            policy_key, action_key = jax.random.split(policy_key)
            policy_action, _ = inference(state.obs, action_key)
            action = jp.where(use_policy, policy_action, jp.zeros(env.action_size))

            def active_step(current_state):
                # The environments use mutable Python dictionaries as PyTree
                # containers. Copy them so the inactive branch remains intact.
                current_state = current_state.replace(
                    info=dict(current_state.info), metrics=dict(current_state.metrics)
                )
                return env.step(current_state, action)

            next_state = jax.lax.cond(active, active_step, lambda value: value, state)
            done = next_state.done > 0.0
            next_active = active & ~done

            local_velocity = env.get_local_linvel(next_state.pipeline_state, "pelvis")
            gyro = env.get_gyro(next_state.pipeline_state, "pelvis")
            torso_up = env.get_gravity(next_state.pipeline_state, "torso")
            contact, undesired, collision = env.contact_state(
                next_state.pipeline_state
            )
            foot_velocity = next_state.pipeline_state.xd.vel[env.foot_link_ids, :2]
            joint_position = next_state.pipeline_state.q[7:]
            low = next_state.pipeline_state.q[2] < env.cfg.termination.min_pelvis_height
            high = next_state.pipeline_state.q[2] > env.cfg.termination.max_pelvis_height
            inverted = torso_up[2] < env.cfg.termination.min_torso_up_z
            invalid = (
                jp.isnan(next_state.pipeline_state.q).any()
                | jp.isnan(next_state.pipeline_state.qd).any()
            )
            limit_violation = jp.any(
                (joint_position < env.soft_joint_lower)
                | (joint_position > env.soft_joint_upper)
            )
            trace = {
                "valid": active,
                "done": done,
                "reward": next_state.reward,
                "linear_error": local_velocity[:2] - command[:2],
                "yaw_error": gyro[2] - command[2],
                "pelvis_height": next_state.pipeline_state.q[2],
                "torso_up_z": torso_up[2],
                "effort": jp.sum(jp.abs(next_state.pipeline_state.actuator_force)),
                "power": jp.sum(
                    jp.abs(
                        next_state.pipeline_state.qd[6:]
                        * next_state.pipeline_state.actuator_force
                    )
                ),
                "action_rate": jp.sum(jp.square(action - previous_action)),
                "joint_acceleration": jp.sum(
                    jp.square(next_state.pipeline_state.qacc[6:])
                ),
                "foot_slip": jp.sum(
                    jp.sum(jp.square(foot_velocity), axis=-1) * contact
                ),
                "left_contact": contact[0],
                "right_contact": contact[1],
                "undesired_contact": undesired,
                "collision_contact": collision,
                "joint_limit_violation": limit_violation,
                "cause_low": low,
                "cause_high": high,
                "cause_inverted": inverted,
                "cause_undesired": undesired,
                "cause_invalid": invalid,
            }
            if capture_pipeline:
                trace["pipeline_state"] = next_state.pipeline_state
            return (next_state, next_active, policy_key, action), trace

        initial_carry = (
            state,
            jp.array(True),
            policy_key,
            jp.zeros(env.action_size),
        )
        _, trace = jax.lax.scan(one_step, initial_carry, xs=None, length=steps)
        return initial_pipeline_state, trace

    return jax.jit(rollout)


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
        "mean_joint_acceleration_cost": _masked_mean(
            trace["joint_acceleration"], valid
        ),
        "mean_foot_slip_cost": _masked_mean(trace["foot_slip"], valid),
        "undesired_contact_rate": _masked_mean(trace["undesired_contact"], valid),
        "collision_contact_rate": _masked_mean(trace["collision_contact"], valid),
        "joint_limit_violation_rate": _masked_mean(
            trace["joint_limit_violation"], valid
        ),
        "left_contact_duty": _masked_mean(trace["left_contact"], valid),
        "right_contact_duty": _masked_mean(trace["right_contact"], valid),
        "gait_contact_asymmetry": abs(
            _masked_mean(trace["left_contact"], valid)
            - _masked_mean(trace["right_contact"], valid)
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
        "mean_joint_acceleration_cost",
        "mean_foot_slip_cost",
        "undesired_contact_rate",
        "collision_contact_rate",
        "joint_limit_violation_rate",
        "gait_contact_asymmetry",
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
    states = [initial_pipeline_state]
    for index in range(episode_steps):
        states.append(jax.tree.map(lambda value: value[index], stacked))
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
    args = parse_args()
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
    output_dir.mkdir(parents=True, exist_ok=True)

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
    ppo_network = factory(env.observation_size, env.action_size)
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
        ("untrained", untrained_params, True),
        ("standing", untrained_params, False),
    )

    rows: list[dict[str, Any]] = []
    videos: list[str] = []
    video_encoders: dict[str, str] = {}
    for controller, params, use_policy in controllers:
        for command_name, command_values in COMMANDS:
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
                print(
                    f"{controller:9s} {command_name:10s} seed={seed} "
                    f"steps={row['episode_steps']:4d} "
                    f"lin_rmse={row['linear_velocity_vector_rmse']:.3f} "
                    f"yaw_rmse={row['yaw_rate_rmse']:.3f}"
                )
                if args.video and command_name == "combined" and seed == seeds[0]:
                    assert video_rollout is not None
                    video_initial_state, video_trace = video_rollout(
                        params,
                        jp.asarray(use_policy),
                        jp.asarray(seed, dtype=jp.int32),
                        jp.asarray(command_values),
                    )
                    jax.block_until_ready(video_trace["reward"])
                    video_path = output_dir / f"{controller}_combined_seed{seed}.mp4"
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
        "kind": "gate4_fixed_held_out_velocity_evaluation",
        "run_dir": str(run_dir),
        "trained_checkpoint": checkpoint,
        "untrained_checkpoint": args.untrained_checkpoint,
        "steps_per_episode": args.steps,
        "control_dt_seconds": env.dt,
        "reset_seeds": seeds,
        "reset_randomized": bool(cfg.sim.reset.randomize),
        "minimum_pelvis_height": float(cfg.sim.termination.min_pelvis_height),
        "observation_noise": bool(cfg.sim.noise.add_noise),
        "commands": {name: values for name, values in COMMANDS},
        "thresholds": {
            "instantaneous_linear_error_norm": 0.25,
            "instantaneous_yaw_error_abs": 0.25,
            "episode_success_requires_full_horizon": True,
        },
        "aggregate": _aggregate(rows),
        "videos": videos,
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
