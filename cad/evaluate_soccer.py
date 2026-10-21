"""Evaluate a soccer policy: goals scored, ball speed, ball travel per episode."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from hardware_tracking_task import HardwareTrackingConfig
from soccer_task import GOAL_OFFSET, make_soccer_cfg

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def build_agent():
  agent = unitree_g1_ppo_runner_cfg()
  agent.actor.hidden_dims = (256, 128, 64)
  agent.critic.hidden_dims = (256, 128, 64)
  agent.actor.distribution_cfg["init_std"] = 0.6
  return agent


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--steps", type=int, default=1200)
  parser.add_argument("--target-speed", type=float, default=3.0)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

  hardware = HardwareTrackingConfig(target_speed_rad_s=args.target_speed)
  cfg = make_soccer_cfg(hardware, play=True)
  cfg.scene.num_envs = args.num_envs
  agent = build_agent()
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = VelocityOnPolicyRunner(wrapped, asdict(agent), str(args.checkpoint.parent), "cuda:0")
  runner.load(str(args.checkpoint))
  policy = runner.get_inference_policy(device="cuda:0")

  robot = env.scene["robot"]
  ball = env.scene["ball"]
  obs, _ = wrapped.reset()

  num_envs = args.num_envs
  goals = torch.zeros(num_envs, device=env.device)
  episodes = torch.zeros(num_envs, device=env.device)
  episode_peak = torch.zeros(num_envs, device=env.device)
  episode_travel = torch.zeros(num_envs, device=env.device)
  active_peak = torch.zeros(num_envs, device=env.device)
  # The goal is GOAL_OFFSET beyond the ball's start along the env-local y axis.
  def ball_local_y() -> torch.Tensor:
    return ball.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]

  start_local_y = ball_local_y().clone()
  goal_local_y = start_local_y + GOAL_OFFSET
  prev_local_y = start_local_y.clone()
  crossed = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
  low_base_steps = torch.zeros(num_envs, device=env.device)

  with torch.no_grad():
    for _ in range(args.steps):
      actions = policy(obs)
      obs, _, _, _ = wrapped.step(actions)
      low_base_steps += (robot.data.root_link_pos_w[:, 2] < 0.45).float()
      local_y = ball_local_y()
      speed = ball.data.root_link_lin_vel_w.norm(dim=-1)
      active_peak = torch.maximum(active_peak, speed)
      # The ball cannot move 15 cm in one 20 ms step unless it was kicked hard,
      # so a larger jump means the episode was reset and the ball re-placed.
      reset = (local_y - prev_local_y).abs() > 0.15
      scored = local_y > goal_local_y
      goals += (scored & ~crossed).float()
      crossed |= scored
      if reset.any():
        episode_peak = torch.where(
          reset, torch.maximum(episode_peak, active_peak), episode_peak
        )
        episode_travel = torch.where(
          reset, torch.maximum(episode_travel, local_y - start_local_y), episode_travel
        )
        episodes += reset.float()
        active_peak = torch.where(reset, torch.zeros_like(active_peak), active_peak)
        start_local_y = torch.where(reset, local_y, start_local_y)
        goal_local_y = torch.where(reset, local_y + GOAL_OFFSET, goal_local_y)
        crossed = torch.where(reset, torch.zeros_like(crossed), crossed)
      prev_local_y = local_y.clone()

  payload = {
    "checkpoint": str(args.checkpoint),
    "num_envs": num_envs,
    "steps": args.steps,
    "goals_per_episode": float(goals.sum() / max(1.0, float(episodes.sum()))),
    "goals_scored": int(goals.sum()),
    "episodes": int(episodes.sum()),
    "mean_ball_travel": float(episode_travel.mean()),
    "peak_ball_speed": float(torch.maximum(episode_peak, active_peak).max()),
    "mean_peak_ball_speed": float(torch.maximum(episode_peak, active_peak).mean()),
    "low_base_step_fraction": float((low_base_steps / args.steps).mean()),
    "fall_rate": float((low_base_steps / args.steps).mean()),
  }
  text = json.dumps(payload, indent=2)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
  print(text)
  env.close()


if __name__ == "__main__":
  main()
