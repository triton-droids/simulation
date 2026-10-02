"""Evaluate old or new tracking checkpoints in the hardware task."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hardware_tracking_task import (
  JOINTS,
  PEAK_TORQUE,
  STALLED_TORQUE,
  HardwareTrackingConfig,
  make_hardware_leg_tracking_cfg,
)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

MOTOR_SPEED_LIMIT_RAD_S = (18.0, 19.0, 19.0, 18.0, 40.0) * 2


def fixed_extreme_imu(env, env_ids):
  if not hasattr(env, "hardware_imu_tilt"):
    env.hardware_imu_tilt = torch.zeros(env.num_envs, 3, device=env.device)
    env.hardware_gyro_bias = torch.zeros_like(env.hardware_imu_tilt)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env.hardware_imu_tilt[env_ids, 0] = 0.0349066
  env.hardware_gyro_bias[env_ids] = 0.05


def make_condition_cfg(reference, condition):
  if condition == "nominal":
    hardware = HardwareTrackingConfig(
      randomize=False,
      observation_delay_steps=(0, 0),
      action_delay_physics_steps=(0, 0),
    )
  elif condition == "latency":
    hardware = HardwareTrackingConfig(
      randomize=False,
      observation_delay_steps=(1, 1),
      action_delay_physics_steps=(1, 1),
    )
  else:
    hardware = HardwareTrackingConfig(
      randomize=True,
      observation_delay_steps=(1, 1),
      action_delay_physics_steps=(1, 1),
    )
  cfg = make_hardware_leg_tracking_cfg(reference, hardware)
  cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
  cfg.events["foot_friction"].params["ranges"] = (0.8, 0.8)
  if condition == "extreme":
    from mjlab.utils.noise import UniformNoiseCfg

    cfg.events["encoder_bias"].params["bias_range"] = (0.01, 0.01)
    cfg.events["foot_friction"].params["ranges"] = (0.3, 0.3)
    cfg.events["imu_calibration"].func = fixed_extreme_imu
    cfg.events["pd_gains"].params["kp_range"] = (0.8, 0.8)
    cfg.events["pd_gains"].params["kd_range"] = (1.5, 1.5)
    for name in ("friction_rs04", "friction_rs03", "friction_rs02"):
      hi = cfg.events[name].params["ranges"][1]
      cfg.events[name].params["ranges"] = (hi, hi)
    cfg.events["armature"].params["ranges"] = (2.0, 2.0)
    cfg.events["mass_inertia"].params["alpha_range"] = (0.0477, 0.0477)
    cfg.events["base_com"].params["ranges"] = {axis: (0.02, 0.02) for axis in range(3)}
    cfg.events["push_robot"].params["velocity_range"] = {
      key: (bounds[1], bounds[1])
      for key, bounds in cfg.events["push_robot"].params["velocity_range"].items()
    }
    actor = cfg.observations["actor"].terms
    for name, magnitude in (
      ("base_ang_vel", 0.2),
      ("joint_pos", 0.01),
      ("joint_vel", 0.5),
      ("gravity", 0.05),
    ):
      actor[name].noise = UniformNoiseCfg(n_min=magnitude - 1e-6, n_max=magnitude)
  else:
    # Nominal physical friction, with no randomization.
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    for name, pattern, value in (
      ("rs04", ".*(hip1|knee)_joint", 0.65),
      ("rs03", ".*(hip2|thigh)_joint", 0.5),
      ("rs02", ".*ankle_joint", 0.25),
    ):
      cfg.events[f"friction_{name}"] = EventTermCfg(
        mode="reset",
        func=dr.joint_friction,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=pattern),
          "ranges": (value, value),
          "operation": "abs",
        },
      )
  return cfg


def evaluate(checkpoint, reference, condition, num_envs, steps):
  cfg = make_condition_cfg(reference, condition)
  cfg.scene.num_envs = num_envs
  cfg.seed = 123
  agent = json.loads((checkpoint.parent / "agent.json").read_text())
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    assert tuple(env.scene["robot"].joint_names) == JOINTS
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])
    runner = MotionTrackingOnPolicyRunner(wrapped, agent, None, "cuda:0")
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device="cuda:0")
    obs, _ = wrapped.reset()
    robot = env.scene["robot"]
    completed, failures, timeouts = [], 0, 0
    efforts, velocities, rewards = [], [], []
    metrics = {}
    with torch.inference_mode():
      for _ in range(steps):
        before = env.episode_length_buf.clone()
        obs, reward, done, _ = wrapped.step(policy(obs))
        failures += int(env.reset_terminated.sum().item())
        timeouts += int(env.reset_time_outs.sum().item())
        completed.extend(((before[done.bool()] + 1) * env.step_dt).cpu().tolist())
        efforts.append(robot.data.actuator_force.abs().cpu().numpy())
        velocities.append(robot.data.joint_vel.abs().cpu().numpy())
        rewards.append(float(reward.mean().item()))
        cmd = env.command_manager.get_term("motion")
        for name, value in cmd.metrics.items():
          if name.startswith("error_"):
            metrics.setdefault(name, []).append(float(value.mean().item()))
    effort = np.stack(efforts)
    velocity = np.stack(velocities)
    peak = effort.max(axis=(0, 1))
    mean = effort.mean(axis=(0, 1))
    result = {
      "condition": condition,
      "checkpoint": str(checkpoint.resolve()),
      "reference": str(reference.resolve()),
      "num_envs": num_envs,
      "steps": steps,
      "seconds_per_env": steps * env.step_dt,
      "failure_resets": failures,
      "time_limit_resets": timeouts,
      "completed_episodes": len(completed),
      "mean_completed_episode_seconds": float(np.mean(completed))
      if completed
      else None,
      "surviving_episode_seconds_at_end": (env.episode_length_buf * env.step_dt)
      .cpu()
      .tolist(),
      "mean_reward_per_step": float(np.mean(rewards)),
      "joint_order": list(JOINTS),
      "peak_torque_nm": peak.tolist(),
      "mean_abs_torque_nm": mean.tolist(),
      "peak_torque_limit_nm": list(PEAK_TORQUE),
      "stalled_torque_rating_nm": list(STALLED_TORQUE),
      "peak_joint_speed_rad_s": velocity.max(axis=(0, 1)).tolist(),
      "motor_speed_limit_rad_s": list(MOTOR_SPEED_LIMIT_RAD_S),
      "deployed_target_slew_limit_rad_s": 1.0,
      "mean_tracking_errors": {k: float(np.mean(v)) for k, v in metrics.items()},
    }
    return result
  finally:
    env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--reference", type=Path)
  parser.add_argument(
    "--condition", choices=("nominal", "latency", "extreme"), required=True
  )
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  ref = args.reference or args.checkpoint.parent / "reference.npz"
  result = evaluate(args.checkpoint, ref, args.condition, args.num_envs, args.steps)
  if args.output:
    args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
