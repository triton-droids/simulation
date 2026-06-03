# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout logger for cross-simulator parity checks.

This script is similar to ``scripts/rl_games/play.py`` but focuses on:
1) scripted/action-replay rollouts with deterministic command profiles, and
2) logging observation/action channels to a file for parity analysis.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Parity logger for Isaac Lab rollouts.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_steps", type=int, default=2000, help="Number of rollout steps to log.")
parser.add_argument("--env_index", type=int, default=0, help="Which env index to log.")

parser.add_argument(
    "--action_source",
    type=str,
    default="scripted",
    choices=["scripted", "replay", "policy"],
    help="Action source for rollout.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (required for policy mode).")

parser.add_argument("--replay_file", type=str, default=None, help="Replay actions file (.npz) for replay mode.")
parser.add_argument("--replay_key", type=str, default="actions", help="Key in replay .npz to read actions from.")
parser.add_argument("--replay_loop", action="store_true", default=False, help="Loop replay actions if shorter.")
parser.add_argument("--replay_env_index", type=int, default=0, help="Env index to use if replay actions are [T,N,A].")

parser.add_argument("--scripted_zero_s", type=float, default=1.0, help="Seconds of zero actions at rollout start.")
parser.add_argument("--scripted_sine_hz", type=float, default=0.7, help="Sine frequency for scripted actions.")
parser.add_argument("--scripted_sine_amp", type=float, default=0.25, help="Sine amplitude for scripted actions.")

parser.add_argument(
    "--command_profile",
    type=str,
    default="stand_forward_yaw",
    choices=["none", "stand_forward_yaw"],
    help="Command profile to apply manually each step.",
)
parser.add_argument("--stand_s", type=float, default=2.0, help="Stand segment duration (seconds).")
parser.add_argument("--forward_s", type=float, default=4.0, help="Forward segment duration (seconds).")
parser.add_argument("--yaw_s", type=float, default=4.0, help="Yaw segment duration (seconds).")
parser.add_argument("--forward_vx", type=float, default=0.6, help="Forward vx command for forward segment.")
parser.add_argument("--yaw_rate", type=float, default=0.6, help="Yaw command for yaw segment.")

parser.add_argument("--output", type=str, default=None, help="Output .npz path.")
parser.add_argument("--metadata_out", type=str, default=None, help="Optional metadata .json path.")
parser.add_argument(
    "--sim2sim-log",
    action="store_true",
    default=False,
    help="Record MuJoCo playback/eval channels in addition to the base parity channels.",
)
parser.add_argument(
    "--compare_with",
    type=str,
    default=None,
    help="Optional other .npz log file. If set, compute parity metrics against it.",
)
parser.add_argument("--compare_max_lag", type=int, default=10, help="Max lag (steps) for best-lag search.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from rl_games.common import env_configurations, vecenv

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import tritonhumanoid.tasks  # noqa: F401


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_output_path(task_name: str) -> str:
    return os.path.abspath(os.path.join("logs", "parity", f"{task_name}_{_timestamp()}.npz"))


def _extract_obs_for_agent(obs: Any) -> Any:
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        if "obs" in obs:
            return obs["obs"]
        if "policy" in obs:
            return obs["policy"]
    return obs


def _parse_step_out(step_out: Any) -> tuple[Any, Any, Any, Any, Any]:
    if not isinstance(step_out, tuple):
        return step_out, None, None, None, None
    if len(step_out) == 5:
        return step_out
    if len(step_out) == 4:
        obs, rew, dones, info = step_out
        return obs, rew, dones, None, info
    raise RuntimeError(f"Unexpected env.step output length: {len(step_out)}")


def _obs_row_for_env(obs_any: Any, env_index: int) -> np.ndarray:
    obs_any = _extract_obs_for_agent(obs_any)
    if torch.is_tensor(obs_any):
        if obs_any.ndim == 1:
            return obs_any.detach().cpu().numpy().astype(np.float32)
        return obs_any[env_index].detach().cpu().numpy().astype(np.float32)
    arr = np.asarray(obs_any, dtype=np.float32)
    if arr.ndim == 1:
        return arr.copy()
    return arr[env_index].copy()


def _row_np(value: Any, env_index: int, dtype=np.float32) -> np.ndarray:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().reshape(1).numpy().astype(dtype)
        if value.ndim == 1:
            return value.detach().cpu().numpy().astype(dtype)
        return value[env_index].detach().cpu().numpy().astype(dtype)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1).astype(dtype)
    if arr.ndim == 1:
        return arr.astype(dtype)
    return arr[env_index].astype(dtype)


def _scalar_row_np(value: Any, env_index: int) -> np.ndarray:
    if value is None:
        return np.asarray([0.0], dtype=np.float32)
    if torch.is_tensor(value):
        if value.ndim == 0:
            return np.asarray([float(value.detach().cpu().item())], dtype=np.float32)
        return np.asarray([float(value[env_index].detach().cpu().item())], dtype=np.float32)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.asarray([float(arr.item())], dtype=np.float32)
    return np.asarray([float(arr[env_index])], dtype=np.float32)


def _compute_clock(env_obj) -> torch.Tensor | None:
    if not bool(getattr(env_obj.cfg, "use_phase_obs", False)):
        return None
    t = env_obj.episode_length_buf.float() * float(env_obj._control_dt)
    phase = 2.0 * torch.pi * (t / float(env_obj.cfg.gait_period_s))
    if hasattr(env_obj, "phase_offset"):
        phase = phase + env_obj.phase_offset
    if bool(getattr(env_obj.cfg, "freeze_phase_when_standing", False)):
        stand_mask = (
            torch.norm(env_obj.commands[:, :2], dim=1) < float(env_obj.cfg.stand_phase_lin_threshold)
        ) & (torch.abs(env_obj.commands[:, 2]) < float(env_obj.cfg.stand_phase_yaw_threshold))
        phase = torch.where(stand_mask, torch.full_like(phase, float(env_obj.cfg.stand_phase_value)), phase)
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)


def _command_profile(step: int, dt: float, args: argparse.Namespace) -> tuple[float, float, float]:
    if args.command_profile == "none":
        return (0.0, 0.0, 0.0)
    t = float(step) * float(dt)
    cycle = float(args.stand_s + args.forward_s + args.yaw_s)
    if cycle <= 0.0:
        return (0.0, 0.0, 0.0)
    t_mod = t % cycle
    if t_mod < args.stand_s:
        return (0.0, 0.0, 0.0)
    if t_mod < args.stand_s + args.forward_s:
        return (float(args.forward_vx), 0.0, 0.0)
    return (0.0, 0.0, float(args.yaw_rate))


def _scripted_actions(step: int, dt: float, num_envs: int, num_actions: int, device: torch.device, args: argparse.Namespace):
    t = float(step) * float(dt)
    actions = torch.zeros((num_envs, num_actions), dtype=torch.float32, device=device)
    if t < float(args.scripted_zero_s):
        return actions
    sine_t = t - float(args.scripted_zero_s)
    base_phase = 2.0 * math.pi * float(args.scripted_sine_hz) * sine_t
    joint_phase = torch.arange(num_actions, device=device, dtype=torch.float32) * (2.0 * math.pi / float(num_actions))
    joint_actions = float(args.scripted_sine_amp) * torch.sin(base_phase + joint_phase)
    actions[:] = joint_actions.unsqueeze(0).expand(num_envs, -1)
    return torch.clamp(actions, -1.0, 1.0)


def _load_replay_actions(path: str, key: str) -> np.ndarray:
    data = np.load(path)
    if key not in data:
        raise KeyError(f"Replay key '{key}' not found in '{path}'. Keys: {list(data.keys())}")
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Replay actions must be rank-2 or rank-3. Got shape {arr.shape}")
    return arr


def _replay_actions_at(
    replay_actions: np.ndarray,
    step: int,
    num_envs: int,
    num_actions: int,
    device: torch.device,
    replay_env_index: int,
    loop: bool,
) -> torch.Tensor:
    T = replay_actions.shape[0]
    if T <= 0:
        raise ValueError("Replay action file is empty.")
    if step >= T and not loop:
        raise IndexError(f"Replay exhausted at step {step} (length={T}). Use --replay_loop to wrap.")
    idx = step % T if loop else step
    row = replay_actions[idx]
    if row.ndim == 1:
        if row.shape[0] != num_actions:
            raise ValueError(f"Replay action dim mismatch: got {row.shape[0]}, expected {num_actions}")
        out = np.repeat(row[None, :], num_envs, axis=0)
    else:
        if row.shape[-1] != num_actions:
            raise ValueError(f"Replay action dim mismatch: got {row.shape[-1]}, expected {num_actions}")
        if row.shape[0] == num_envs:
            out = row
        else:
            env_idx = int(np.clip(replay_env_index, 0, row.shape[0] - 1))
            out = np.repeat(row[env_idx : env_idx + 1, :], num_envs, axis=0)
    return torch.from_numpy(out).to(device=device, dtype=torch.float32).clamp_(-1.0, 1.0)


def _make_agent(task_name: str, checkpoint: str, env, num_actors: int):
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    agent_cfg = load_cfg_from_registry(task_name, "rl_games_cfg_entry_point")
    resume_path = retrieve_file_path(checkpoint)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    rl_env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    try:
        vecenv.register(
            "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
    except Exception:
        pass
    env_entry = {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: rl_env}
    if "rlgpu" in env_configurations.configurations:
        env_configurations.configurations["rlgpu"] = env_entry
    else:
        env_configurations.register("rlgpu", env_entry)

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = int(num_actors)
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    return agent, rl_env


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _align_by_lag(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return x[:-lag], y[lag:]
    if lag < 0:
        return x[-lag:], y[:lag]
    return x, y


def _fit_gain_offset(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def compare_logs(path_a: str, path_b: str, max_lag: int = 10) -> dict[str, Any]:
    a = np.load(path_a)
    b = np.load(path_b)
    channels = [
        "commands",
        "actions_input",
        "act_pos_scaled",
        "act_vel_scaled",
        "actions_applied",
        "prev_actions",
        "q_des",
        "act_pos",
        "act_vel",
        "clock",
        "up_b",
        "base_lin_vel",
        "base_ang_vel",
        "obs_latest",
    ]
    metrics: dict[str, Any] = {}
    for ch in channels:
        if ch not in a or ch not in b:
            continue
        xa = np.asarray(a[ch], dtype=np.float64)
        xb = np.asarray(b[ch], dtype=np.float64)
        T = min(xa.shape[0], xb.shape[0])
        xa = xa[:T]
        xb = xb[:T]
        if xa.ndim == 1:
            xa = xa[:, None]
        if xb.ndim == 1:
            xb = xb[:, None]
        D = min(xa.shape[1], xb.shape[1])
        xa = xa[:, :D]
        xb = xb[:, :D]
        dim_metrics = []
        for d in range(D):
            x = xa[:, d]
            y = xb[:, d]
            best_lag = 0
            best_corr = -1.0
            for lag in range(-max_lag, max_lag + 1):
                xl, yl = _align_by_lag(x, y, lag)
                corr = abs(_safe_corr(xl, yl))
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
            xl, yl = _align_by_lag(x, y, best_lag)
            corr_signed = _safe_corr(xl, yl)
            gain, offset = _fit_gain_offset(xl, yl)
            rmse_raw = float(np.sqrt(np.mean((xl - yl) ** 2)))
            y_hat = gain * xl + offset
            rmse_affine = float(np.sqrt(np.mean((y_hat - yl) ** 2)))
            dim_metrics.append(
                {
                    "dim": d,
                    "corr": corr_signed,
                    "gain": gain,
                    "offset": offset,
                    "best_lag_steps": int(best_lag),
                    "rmse_raw": rmse_raw,
                    "rmse_affine": rmse_affine,
                }
            )
        metrics[ch] = dim_metrics
    return metrics


def _log_state(
    env_obj,
    env_index: int,
    action_input_row: torch.Tensor,
    obs_out: Any,
    reward: Any = None,
    dones: Any = None,
    truncated: Any = None,
    *,
    sim2sim_log: bool = False,
) -> dict[str, np.ndarray]:
    if hasattr(env_obj, "_update_state"):
        env_obj._update_state()
    idx = int(np.clip(env_index, 0, int(env_obj.num_envs) - 1))
    clock = _compute_clock(env_obj)
    out: dict[str, np.ndarray] = {}
    out["lin_vel_cmd"] = env_obj.imu_lin_vel_cmd[idx].detach().cpu().numpy().astype(np.float32)
    out["ang_vel_cmd"] = env_obj.imu_ang_vel_cmd[idx].detach().cpu().numpy().astype(np.float32)
    out["ang_vel_cmd_scaled"] = (
        (env_obj.imu_ang_vel_cmd[idx] * float(env_obj.cfg.ang_vel_scale)).detach().cpu().numpy().astype(np.float32)
    )
    out["up_cmd"] = env_obj.up_cmd[idx].detach().cpu().numpy().astype(np.float32)
    out["commands"] = env_obj.commands[idx].detach().cpu().numpy().astype(np.float32)
    out["act_pos_scaled"] = env_obj.act_pos_scaled[idx].detach().cpu().numpy().astype(np.float32)
    out["act_vel_scaled"] = (env_obj.act_vel[idx] * float(env_obj.cfg.dof_vel_scale)).detach().cpu().numpy().astype(np.float32)
    out["prev_actions"] = env_obj.prev_actions[idx].detach().cpu().numpy().astype(np.float32)
    out["clock"] = (
        np.zeros((2,), dtype=np.float32)
        if clock is None
        else clock[idx].detach().cpu().numpy().astype(np.float32)
    )
    out["actions_input"] = action_input_row.detach().cpu().numpy().astype(np.float32)
    out["actions_applied"] = env_obj.actions[idx].detach().cpu().numpy().astype(np.float32)
    out["q_des"] = env_obj.q_des[idx].detach().cpu().numpy().astype(np.float32)
    out["act_pos"] = env_obj.act_pos[idx].detach().cpu().numpy().astype(np.float32)
    out["act_vel"] = env_obj.act_vel[idx].detach().cpu().numpy().astype(np.float32)
    out["base_lin_vel"] = env_obj.com_lin_vel_cmd[idx].detach().cpu().numpy().astype(np.float32)
    out["base_ang_vel"] = env_obj.com_ang_vel_cmd[idx].detach().cpu().numpy().astype(np.float32)
    out["up_b"] = env_obj.up_b[idx].detach().cpu().numpy().astype(np.float32)
    out["obs_latest"] = _obs_row_for_env(obs_out, idx)
    if not sim2sim_log:
        return out

    env_origin = env_obj.scene.env_origins[idx].detach().cpu().numpy().astype(np.float32)
    root_pos_w = env_obj.root_pos_w[idx].detach().cpu().numpy().astype(np.float32)
    root_pos_local = (env_obj.root_pos_w[idx] - env_obj.scene.env_origins[idx]).detach().cpu().numpy().astype(np.float32)
    root_quat_w = env_obj.root_quat_w[idx].detach().cpu().numpy().astype(np.float32)
    root_lin_vel_w = env_obj.root_lin_vel_w[idx].detach().cpu().numpy().astype(np.float32)
    root_ang_vel_w = env_obj.root_ang_vel_w[idx].detach().cpu().numpy().astype(np.float32)

    out["env_origin_w"] = env_origin
    out["root_pos_w"] = root_pos_w
    out["root_pos_local"] = root_pos_local
    out["root_quat_w"] = root_quat_w
    out["root_lin_vel_w"] = root_lin_vel_w
    out["root_ang_vel_w"] = root_ang_vel_w
    out["joint_pos"] = out["act_pos"]
    out["joint_vel"] = out["act_vel"]
    out["qpos"] = np.concatenate([root_pos_local, root_quat_w, out["act_pos"]]).astype(np.float32)
    out["qvel"] = np.concatenate([root_lin_vel_w, root_ang_vel_w, out["act_vel"]]).astype(np.float32)
    out["reward"] = _scalar_row_np(reward, idx)
    out["done"] = _scalar_row_np(dones, idx)
    out["timeout"] = _scalar_row_np(truncated, idx)

    joint_torque = env_obj.robot.data.applied_torque[:, env_obj._joint_dof_idx]
    out["joint_torque"] = joint_torque[idx].detach().cpu().numpy().astype(np.float32)

    if hasattr(env_obj, "_init_feet"):
        env_obj._init_feet()
        foot_body_ids = env_obj._feet_body_ids
        foot_sensor_ids = env_obj._feet_sensor_ids
        out["foot_pos_w"] = env_obj.robot.data.body_pos_w[idx, foot_body_ids, :].detach().cpu().numpy().astype(np.float32)
        out["foot_quat_w"] = env_obj.robot.data.body_quat_w[idx, foot_body_ids, :].detach().cpu().numpy().astype(np.float32)
        out["foot_lin_vel_w"] = env_obj.robot.data.body_lin_vel_w[idx, foot_body_ids, :].detach().cpu().numpy().astype(np.float32)
        forces_hist = env_obj._contact_sensor.data.net_forces_w_history
        foot_forces = forces_hist[:, 0, foot_sensor_ids, :]
        foot_force_z = torch.clamp(foot_forces[:, :, 2], min=0.0)
        foot_contact = foot_force_z > env_obj.cfg.foot_contact_force_thresh
        out["foot_contact_forces_w"] = foot_forces[idx].detach().cpu().numpy().astype(np.float32)
        out["foot_contact"] = foot_contact[idx].detach().cpu().numpy().astype(np.float32)

    reward_terms = getattr(env_obj, "last_reward_terms", {})
    if reward_terms:
        keys = sorted(reward_terms.keys())
        out["reward_terms"] = np.asarray(
            [float(reward_terms[key][idx].detach().cpu().item()) for key in keys],
            dtype=np.float32,
        )
    return out


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env_obj = env.unwrapped
    step_env = env
    dt = float(env_obj.step_dt)
    num_envs = int(env_obj.num_envs)
    num_actions = int(env_obj.num_actions)
    device = env_obj.device

    reset_out = step_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    # Disable internal command resampling if we are manually applying a command profile each step.
    if args_cli.command_profile != "none" and hasattr(env_obj, "_cmd_resample_interval_steps"):
        env_obj._cmd_resample_interval_steps = 0

    replay_actions = None
    if args_cli.action_source == "replay":
        if not args_cli.replay_file:
            raise ValueError("--replay_file is required when --action_source replay")
        replay_actions = _load_replay_actions(args_cli.replay_file, args_cli.replay_key)

    agent = None
    if args_cli.action_source == "policy":
        if not args_cli.checkpoint:
            raise ValueError("--checkpoint is required when --action_source policy")
        agent, step_env = _make_agent(args_cli.task, args_cli.checkpoint, env, num_envs)
        reset_out = step_env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        agent_obs = _extract_obs_for_agent(obs)
        _ = agent.get_batch_size(agent_obs, 1)
        if agent.is_rnn:
            agent.init_rnn()

    logs: dict[str, list[np.ndarray]] = {}
    step_times = []

    for step in range(int(args_cli.num_steps)):
        if not simulation_app.is_running():
            break

        if args_cli.command_profile != "none":
            cmd = torch.tensor(_command_profile(step, dt, args_cli), dtype=torch.float32, device=device)
            env_obj.commands[:] = cmd.unsqueeze(0).expand(num_envs, -1)

        with torch.inference_mode():
            if args_cli.action_source == "scripted":
                actions = _scripted_actions(step, dt, num_envs, num_actions, device, args_cli)
            elif args_cli.action_source == "replay":
                actions = _replay_actions_at(
                    replay_actions,
                    step,
                    num_envs,
                    num_actions,
                    device,
                    args_cli.replay_env_index,
                    bool(args_cli.replay_loop),
                )
            else:
                agent_obs = _extract_obs_for_agent(obs)
                obs_t = agent.obs_to_torch(agent_obs)
                actions = agent.get_action(obs_t, is_deterministic=True)
                actions = torch.as_tensor(actions, device=device, dtype=torch.float32).clamp_(-1.0, 1.0)
                if actions.ndim == 1:
                    actions = actions.unsqueeze(0).expand(num_envs, -1).contiguous()

            step_out = step_env.step(actions)
            obs, reward, dones, truncated, info = _parse_step_out(step_out)

            if agent is not None and agent.is_rnn and agent.states is not None:
                if dones is not None:
                    dones_t = torch.as_tensor(dones, device=device).bool()
                    if truncated is not None:
                        dones_t = torch.logical_or(dones_t, torch.as_tensor(truncated, device=device).bool())
                    done_ids = torch.nonzero(dones_t, as_tuple=False).squeeze(-1)
                    if done_ids.numel() > 0:
                        for s in agent.states:
                            s[:, done_ids, :] = 0.0

        logged = _log_state(
            env_obj,
            args_cli.env_index,
            actions[int(np.clip(args_cli.env_index, 0, num_envs - 1))],
            obs,
            reward=reward,
            dones=dones,
            truncated=truncated,
            sim2sim_log=bool(args_cli.sim2sim_log),
        )
        for k, v in logged.items():
            logs.setdefault(k, []).append(v)
        step_times.append(np.array([step * dt], dtype=np.float32))

    if len(step_times) == 0:
        raise RuntimeError("No steps were logged. Increase --num_steps or ensure simulator is running.")

    out_path = args_cli.output or _default_output_path(args_cli.task)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data_to_save: dict[str, np.ndarray] = {"time_s": np.concatenate(step_times, axis=0)}
    for k, seq in logs.items():
        data_to_save[k] = np.stack(seq, axis=0)
    np.savez_compressed(out_path, **data_to_save)
    print(f"[INFO] Saved parity log: {out_path}")

    metadata = {
        "task": args_cli.task,
        "num_steps_logged": int(data_to_save["time_s"].shape[0]),
        "dt": dt,
        "decimation": int(getattr(env_obj.cfg, "decimation", 1)),
        "num_envs": num_envs,
        "num_actions": num_actions,
        "env_index": int(args_cli.env_index),
        "action_source": args_cli.action_source,
        "checkpoint": args_cli.checkpoint,
        "sim2sim_log": bool(args_cli.sim2sim_log),
        "quaternion_order": "wxyz",
        "joint_order": [
            str(env_obj.robot.data.joint_names[int(jid)])
            for jid in getattr(env_obj, "_joint_dof_idx", [])
        ] if hasattr(env_obj.robot.data, "joint_names") else [],
        "observation_layout": [
            "base_lin_vel_cmd[3]",
            "base_ang_vel_cmd_scaled[3]",
            "up_cmd[3]",
            "commands[3]",
            "act_pos_scaled[10]",
            "act_vel_scaled[10]",
            "prev_actions[10]",
        ],
        "obs_stack_frames": int(getattr(env_obj.cfg, "obs_stack_frames", 1)),
        "action_scale": float(getattr(env_obj.cfg, "action_scale", 1.0)),
        "action_scale_per_joint": (
            env_obj._action_scale_per_joint.detach().cpu().numpy().astype(float).tolist()
            if hasattr(env_obj, "_action_scale_per_joint")
            else []
        ),
        "joint_velocity_limit": 15.0,
        "armature": 0.01,
        "reward_term_names": sorted(getattr(env_obj, "last_reward_terms", {}).keys()),
        "command_profile": args_cli.command_profile,
        "command_profile_params": {
            "stand_s": float(args_cli.stand_s),
            "forward_s": float(args_cli.forward_s),
            "yaw_s": float(args_cli.yaw_s),
            "forward_vx": float(args_cli.forward_vx),
            "yaw_rate": float(args_cli.yaw_rate),
        },
    }
    meta_path = args_cli.metadata_out or (os.path.splitext(out_path)[0] + ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved metadata: {meta_path}")

    if args_cli.compare_with:
        metrics = compare_logs(out_path, args_cli.compare_with, max_lag=int(args_cli.compare_max_lag))
        cmp_path = os.path.splitext(out_path)[0] + "_compare.json"
        with open(cmp_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved comparison metrics: {cmp_path}")

    step_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
