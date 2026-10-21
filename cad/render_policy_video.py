"""Record a headless video of a velocity or soccer policy."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from dataclasses import asdict
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch
from hardware_tracking_task import HardwareTrackingConfig
from soccer_task import make_soccer_cfg
from velocity_task import make_triton_velocity_cfg

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=("velocity", "soccer"), default="velocity")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--out", type=Path, default=Path("policy_video.mp4"))
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--command", type=float, nargs=3, default=(0.0, 0.5, 0.0),
                      help="vx(body lateral), vy(body forward), yaw_rate")
  parser.add_argument("--azimuth", type=float, default=135.0)
  parser.add_argument("--distance", type=float, default=3.5)
  parser.add_argument("--target-speed", type=float, default=3.0)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--fps", type=int, default=50)
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)

  hardware = HardwareTrackingConfig(target_speed_rad_s=args.target_speed)
  cfg = (
    make_soccer_cfg(hardware, play=True)
    if args.task == "soccer"
    else make_triton_velocity_cfg(hardware, play=True)
  )
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cuda:0")

  agent = unitree_g1_ppo_runner_cfg()
  agent.actor.hidden_dims = (256, 128, 64)
  agent.critic.hidden_dims = (256, 128, 64)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = VelocityOnPolicyRunner(wrapped, asdict(agent), str(args.checkpoint.parent), "cuda:0")
  # Actor only: critic shapes differ between the velocity and soccer tasks.
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
    strict=True,
    map_location="cuda:0",
  )
  policy = runner.get_inference_policy(device="cuda:0")

  if args.task == "velocity":
    command = torch.tensor(args.command, device="cuda:0").repeat(1, 1)
    term = env.command_manager.get_term("twist")
    if hasattr(term, "cfg"):
      term.cfg.resampling_time_range = (1e9, 1e9)
  obs, _ = wrapped.reset()
  if args.task == "velocity":
    term.vel_command_b[:] = command

  cpu_model = env.sim.mj_model
  cpu_data = mujoco.MjData(cpu_model)
  renderer = mujoco.Renderer(cpu_model, height=args.height, width=args.width)
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.distance = args.distance
  camera.elevation = -15.0
  camera.azimuth = args.azimuth
  camera.lookat[:] = [0.0, 0.0, 0.6]

  frames = []
  print(f"Recording {args.steps} steps...")
  for step in range(args.steps):
    with torch.no_grad():
      action = policy(obs)
      obs, _, _, _ = wrapped.step(action)
    if args.task == "velocity":
      term.vel_command_b[:] = command
    gpu_data = env.sim.data
    cpu_data.qpos[:] = gpu_data.qpos[0].cpu().numpy()
    cpu_data.qvel[:] = gpu_data.qvel[0].cpu().numpy()
    mujoco.mj_forward(cpu_model, cpu_data)
    camera.lookat[:] = cpu_data.qpos[:3]
    renderer.update_scene(cpu_data, camera)
    frames.append(renderer.render().copy())
    if step % 100 == 0:
      print(f"  step {step}/{args.steps}")

  renderer.close()
  env.close()
  args.out.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(str(args.out), fps=args.fps, quality=8)
  for frame in frames:
    writer.append_data(frame)
  writer.close()
  print(f"Wrote {args.out}")


if __name__ == "__main__":
  main()
