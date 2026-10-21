"""Evaluate a velocity policy on a fixed command grid.

Reports, per command: survival, achieved body-frame velocity and tracking
reward.  Commands are written directly into the velocity command term so every
environment sees the same instruction, which makes runs comparable.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from hardware_tracking_task import HardwareTrackingConfig
from velocity_task import make_triton_velocity_cfg

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

# (label, vx_body, vy_body, yaw_rate).  Body y is the robot's forward axis.
COMMANDS = (
  ("stand", 0.0, 0.0, 0.0),
  ("forward_slow", 0.0, 0.3, 0.0),
  ("forward_fast", 0.0, 0.6, 0.0),
  ("backward", 0.0, -0.3, 0.0),
  ("strafe_left", 0.2, 0.0, 0.0),
  ("strafe_right", -0.2, 0.0, 0.0),
  ("turn_left", 0.0, 0.0, 0.5),
  ("turn_right", 0.0, 0.0, -0.5),
  ("diagonal", 0.15, 0.4, 0.0),
)


def build_agent():
  agent = unitree_g1_ppo_runner_cfg()
  agent.actor.hidden_dims = (256, 128, 64)
  agent.critic.hidden_dims = (256, 128, 64)
  agent.actor.distribution_cfg["init_std"] = 0.5
  return agent


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--steps", type=int, default=400)
  parser.add_argument("--target-speed", type=float, default=3.0)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

  hardware = HardwareTrackingConfig(target_speed_rad_s=args.target_speed)
  cfg = make_triton_velocity_cfg(hardware, play=True)
  cfg.scene.num_envs = args.num_envs
  term_cfg = cfg.commands["twist"]
  # Fixed commands: no resampling, no heading controller, no standing envs.
  term_cfg.heading_command = False
  term_cfg.ranges.heading = None
  term_cfg.resampling_time_range = (1e9, 1e9)
  term_cfg.rel_heading_envs = 0.0
  term_cfg.rel_standing_envs = 0.0
  term_cfg.rel_forward_envs = 0.0
  term_cfg.rel_world_envs = 0.0
  term_cfg.init_velocity_prob = 0.0

  agent = build_agent()
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = VelocityOnPolicyRunner(wrapped, asdict(agent), str(args.checkpoint.parent), "cuda:0")
  runner.load(str(args.checkpoint))
  policy = runner.get_inference_policy(device="cuda:0")
  command_term = env.command_manager.get_term("twist")
  robot = env.scene["robot"]

  results = {}
  for label, vx, vy, yaw in COMMANDS:
    command = torch.tensor([vx, vy, yaw], device=env.device).repeat(args.num_envs, 1)
    obs, _ = wrapped.reset()
    command_term.vel_command_b[:] = command
    falls = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    lin = torch.zeros(args.num_envs, device=env.device)
    yaw_err = torch.zeros(args.num_envs, device=env.device)
    reward_sum = torch.zeros(args.num_envs, device=env.device)
    mean_vel = torch.zeros(3, device=env.device)
    mean_speed = torch.zeros(args.num_envs, device=env.device)
    with torch.no_grad():
      for step in range(args.steps):
        actions = policy(obs)
        obs, rewards, dones, _ = wrapped.step(actions)
        # Hold the command fixed across resets.
        command_term.vel_command_b[:] = command
        vel = robot.data.root_link_lin_vel_b
        err = torch.square(command[:, :2] - vel[:, :2]).sum(dim=1)
        lin += torch.exp(-err / 0.25)
        yaw_err += torch.abs(command[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
        mean_vel += robot.data.root_link_lin_vel_b.mean(dim=0)
        mean_speed += robot.data.root_link_lin_vel_b.norm(dim=1)
        reward_sum += rewards
        falls |= dones.bool()
    n = args.steps
    results[label] = {
      "command": [vx, vy, yaw],
      "fall_rate": float(falls.float().mean()),
      "achieved_vel_xy": [float(mean_vel[0] / n), float(mean_vel[1] / n)],
      "achieved_speed": float((mean_speed / n).mean()),
      "mean_linear_tracking_reward": float((lin / n).mean()),
      "mean_yaw_error": float((yaw_err / n).mean()),
      "mean_reward_per_step": float((reward_sum / n).mean()),
    }

  payload = {
    "checkpoint": str(args.checkpoint),
    "num_envs": args.num_envs,
    "steps": args.steps,
    "target_speed_rad_s": args.target_speed,
    "commands": results,
    "fall_rate": max(r["fall_rate"] for r in results.values()),
    "mean_linear_tracking_reward": sum(
      r["mean_linear_tracking_reward"] for r in results.values()
    )
    / len(results),
    "mean_forward_speed": results["forward_slow"]["achieved_vel_xy"][1],
  }
  text = json.dumps(payload, indent=2)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
  print(text)
  env.close()


if __name__ == "__main__":
  main()
