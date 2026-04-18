"""
Rollout logger for MuJoCo locomotion parity checks.

This mirrors the IsaacLab parity logger flow:
1) scripted / replay / policy action sources,
2) deterministic command profiles, and
3) per-step channel logging to .npz for cross-simulator comparison.
"""

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch

from envs.locomotion_env import HumanoidLocomotionEnv


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_output_path() -> str:
    return os.path.abspath(os.path.join("logs", "parity", f"mujoco_locomotion_{_timestamp()}.npz"))


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


def _scripted_actions(step: int, dt: float, num_actions: int, args: argparse.Namespace) -> np.ndarray:
    t = float(step) * float(dt)
    actions = np.zeros((num_actions,), dtype=np.float32)
    if t < float(args.scripted_zero_s):
        return actions
    sine_t = t - float(args.scripted_zero_s)
    base_phase = 2.0 * math.pi * float(args.scripted_sine_hz) * sine_t
    joint_phase = np.arange(num_actions, dtype=np.float32) * (2.0 * math.pi / float(num_actions))
    actions = float(args.scripted_sine_amp) * np.sin(base_phase + joint_phase)
    return np.clip(actions.astype(np.float32), -1.0, 1.0)


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
    num_actions: int,
    replay_env_index: int,
    loop: bool,
) -> np.ndarray:
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
        out = row
    else:
        if row.shape[-1] != num_actions:
            raise ValueError(f"Replay action dim mismatch: got {row.shape[-1]}, expected {num_actions}")
        env_idx = int(np.clip(replay_env_index, 0, row.shape[0] - 1))
        out = row[env_idx]
    return np.clip(np.asarray(out, dtype=np.float32), -1.0, 1.0)


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
        "actions_applied",
        "act_delay_steps",
        "prev_actions",
        "q_des",
        "act_pos",
        "act_vel",
        "act_pos_scaled",
        "act_vel_scaled",
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


def _latest_frame_obs(obs: np.ndarray, env: HumanoidLocomotionEnv) -> np.ndarray:
    if env._frame_stack <= 1:
        return obs.copy()
    if env._stack_frame_major:
        return obs.reshape(env._frame_stack, env._single_frame_size)[-1].copy()
    return obs.reshape(env._single_frame_size, env._frame_stack)[:, -1].copy()


def _compute_q_des_policy(env: HumanoidLocomotionEnv) -> np.ndarray:
    q_des_mj = np.asarray(env.data.ctrl, dtype=np.float32).copy()
    return q_des_mj[env._policy_to_mj]


def _compute_clock(env: HumanoidLocomotionEnv) -> np.ndarray:
    if not env._use_phase_obs:
        return np.zeros((2,), dtype=np.float32)

    # Isaac-style phase definition for parity:
    # phase = 2*pi*(episode_time/gait_period) + phase_offset,
    # with optional freezing to stand_phase_value under standing command.
    step_count = int(
        getattr(
            env,
            "_episode_step",
            getattr(env, "_steps", getattr(env, "_step_count", 0)),
        )
    )
    dt = float(env.dt)
    gait_period = float(getattr(env, "_gait_period_s", 1.0))
    phase_offset = float(getattr(env, "_phase_offset", 0.0))

    t = step_count * dt
    phase = 2.0 * math.pi * (t / gait_period) + phase_offset

    freeze = bool(getattr(env, "_freeze_phase_when_standing", False))
    if freeze:
        cmd = np.asarray(env._commands, dtype=np.float32)
        lin_thr = float(getattr(env, "_stand_phase_lin_threshold", 1e-3))
        yaw_thr = float(getattr(env, "_stand_phase_yaw_threshold", 1e-3))
        stand_val = float(getattr(env, "_stand_phase_value", 0.0))
        stand = (np.linalg.norm(cmd[:2]) < lin_thr) and (abs(cmd[2]) < yaw_thr)
        if stand:
            phase = stand_val

    return np.array([math.sin(phase), math.cos(phase)], dtype=np.float32)


def _log_state(env: HumanoidLocomotionEnv, action_input_row: np.ndarray, obs_out: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    out["lin_vel_cmd"] = env.torso_lin_vel_cmd.astype(np.float32).copy()
    out["ang_vel_cmd"] = env.torso_ang_vel_cmd.astype(np.float32).copy()
    out["ang_vel_cmd_scaled"] = (env.torso_ang_vel_cmd * float(env._ang_vel_scale)).astype(np.float32).copy()
    out["up_cmd"] = env.up_cmd.astype(np.float32).copy()
    out["commands"] = env._commands.astype(np.float32).copy()
    out["act_pos_scaled"] = env.act_pos_scaled.astype(np.float32).copy()
    out["act_vel_scaled"] = env.act_vel_scaled.astype(np.float32).copy()
    out["prev_actions"] = env._last_act.astype(np.float32).copy()
    out["clock"] = _compute_clock(env)
    out["actions_input"] = np.asarray(action_input_row, dtype=np.float32).copy()
    out["actions_policy"] = env._policy_actions.astype(np.float32).copy()
    out["actions_applied"] = env._actions.astype(np.float32).copy()
    out["act_delay_steps"] = np.asarray(env._act_delay_steps_policy, dtype=np.int32).copy()
    out["q_des"] = _compute_q_des_policy(env)  # policy/interleaved order
    out["q_des_mj"] = np.asarray(env.data.ctrl, dtype=np.float32).copy()  # raw MuJoCo actuator order
    out["act_pos"] = env.act_pos.astype(np.float32).copy()
    out["act_vel"] = env.act_vel.astype(np.float32).copy()
    out["track_pos_w"] = env.track_pos_w.astype(np.float32).copy()
    out["base_lin_vel"] = env.torso_lin_vel_cmd.astype(np.float32).copy()
    out["base_ang_vel"] = env.torso_ang_vel_cmd.astype(np.float32).copy()
    out["up_b"] = env.up_b.astype(np.float32).copy()
    out["obs_latest"] = _latest_frame_obs(np.asarray(obs_out, dtype=np.float32), env)
    return out


def main():
    parser = argparse.ArgumentParser(description="Parity logger for MuJoCo locomotion rollouts.")
    parser.add_argument("--num_steps", type=int, default=2000, help="Number of rollout steps to log.")

    parser.add_argument(
        "--action_source",
        type=str,
        default="scripted",
        choices=["scripted", "replay", "policy"],
        help="Action source for rollout.",
    )
    parser.add_argument("--policy", type=str, default=None, help="TorchScript policy path (.pt) for policy mode.")

    parser.add_argument("--replay_file", type=str, default=None, help="Replay actions file (.npz) for replay mode.")
    parser.add_argument("--replay_key", type=str, default="actions", help="Key in replay .npz to read actions from.")
    parser.add_argument("--replay_loop", action="store_true", default=False, help="Loop replay actions if shorter.")
    parser.add_argument("--replay_env_index", type=int, default=0, help="Env index to use if replay actions are [T,N,A].")
    parser.add_argument(
        "--replay_order",
        type=str,
        default="policy",
        choices=["policy", "mujoco"],
        help="Order of replay actions. 'policy' is interleaved. 'mujoco' is actuator order.",
    )

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

    parser.add_argument("--xml_path", type=str, default="robot_description/scene.xml", help="MuJoCo scene XML path.")
    parser.add_argument("--output", type=str, default=None, help="Output .npz path.")
    parser.add_argument("--metadata_out", type=str, default=None, help="Optional metadata .json path.")
    parser.add_argument(
        "--compare_with",
        type=str,
        default=None,
        help="Optional other .npz log file. If set, compute parity metrics against it.",
    )
    parser.add_argument("--compare_max_lag", type=int, default=10, help="Max lag (steps) for best-lag search.")
    args = parser.parse_args()

    env = HumanoidLocomotionEnv(xml_path=args.xml_path)
    dt = float(env.dt)
    num_actions = int(env._nu)

    obs = env.reset()

    replay_actions = None
    if args.action_source == "replay":
        if not args.replay_file:
            raise ValueError("--replay_file is required when --action_source replay")
        replay_actions = _load_replay_actions(args.replay_file, args.replay_key)

    policy = None
    if args.action_source == "policy":
        if not args.policy:
            raise ValueError("--policy is required when --action_source policy")
        policy = torch.jit.load(args.policy, map_location="cpu")
        policy.eval()

    logs: dict[str, list[np.ndarray]] = {}
    step_times = []

    with torch.inference_mode():
        for step in range(int(args.num_steps)):
            if args.command_profile != "none":
                cmd = _command_profile(step, dt, args)
                env.set_command(*cmd)

            if args.action_source == "scripted":
                action_policy = _scripted_actions(step, dt, num_actions, args)
            elif args.action_source == "replay":
                action_replay = _replay_actions_at(
                    replay_actions,
                    step,
                    num_actions,
                    args.replay_env_index,
                    bool(args.replay_loop),
                )
                if args.replay_order == "mujoco":
                    action_policy = action_replay[env._policy_to_mj]
                else:
                    action_policy = action_replay
            else:
                obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                action_policy = policy(obs_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
                action_policy = np.clip(action_policy, -1.0, 1.0)

            obs = env.step(action_policy)

            logged = _log_state(env, action_policy, obs)
            for k, v in logged.items():
                logs.setdefault(k, []).append(v)
            step_times.append(np.array([step * dt], dtype=np.float32))

    if len(step_times) == 0:
        raise RuntimeError("No steps were logged. Increase --num_steps.")

    out_path = args.output or _default_output_path()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data_to_save: dict[str, np.ndarray] = {"time_s": np.concatenate(step_times, axis=0)}
    for k, seq in logs.items():
        data_to_save[k] = np.stack(seq, axis=0)
    np.savez_compressed(out_path, **data_to_save)
    print(f"[INFO] Saved parity log: {out_path}")

    metadata = {
        "sim": "mujoco",
        "task": "locomotion_env",
        "num_steps_logged": int(data_to_save["time_s"].shape[0]),
        "dt": dt,
        "num_actions": num_actions,
        "action_source": args.action_source,
        "command_profile": args.command_profile,
        "command_profile_params": {
            "stand_s": float(args.stand_s),
            "forward_s": float(args.forward_s),
            "yaw_s": float(args.yaw_s),
            "forward_vx": float(args.forward_vx),
            "yaw_rate": float(args.yaw_rate),
        },
        "replay_order": args.replay_order,
        "xml_path": args.xml_path,
        "act_delay_mode": env._act_delay_mode,
        "act_delay_steps_policy": np.asarray(env._act_delay_steps_policy, dtype=int).tolist(),
        "act_delay_steps_by_name": {k: int(v) for k, v in env._act_delay_steps_by_name.items()},
        "act_delay_range_by_name": {k: [int(v[0]), int(v[1])] for k, v in env._act_delay_range_by_name.items()},
    }
    meta_path = args.metadata_out or (os.path.splitext(out_path)[0] + ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved metadata: {meta_path}")

    if args.compare_with:
        metrics = compare_logs(out_path, args.compare_with, max_lag=int(args.compare_max_lag))
        cmp_path = os.path.splitext(out_path)[0] + "_compare.json"
        with open(cmp_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved comparison metrics: {cmp_path}")


if __name__ == "__main__":
    main()
