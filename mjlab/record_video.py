"""Record a short video of the trained policy headlessly."""

import os

os.environ["MUJOCO_GL"] = "egl"

from dataclasses import asdict

import imageio
import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Tracking-Flat-T800"
CHECKPOINT = "logs/rsl_rl/t800_tracking/2026-05-26_00-25-41/model_14999.pt"
MOTION_FILE = "/home/thewheelneedsme/Projects/droids/URKL-Simulations/motion_retargeting/data/npz/kick_punch_combofix.npz"
OUTPUT_FILE = "policy_video.mp4"
NUM_STEPS = 300
WIDTH, HEIGHT = 1280, 720

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

env_cfg = load_env_cfg(TASK_ID, play=True)
agent_cfg = load_rl_cfg(TASK_ID)

# Disable terminations so we can watch the full motion.
env_cfg.terminations = {}

motion_cmd = env_cfg.commands["motion"]
assert isinstance(motion_cmd, MotionCommandCfg)
motion_cmd.motion_file = MOTION_FILE

env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = MjlabOnPolicyRunner(env_wrapped, asdict(agent_cfg), device=device)
runner.load(CHECKPOINT, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

# The sim keeps a host-side MjModel; use it directly for rendering.
cpu_model = env.sim.mj_model
cpu_data = mujoco.MjData(cpu_model)
renderer = mujoco.Renderer(cpu_model, height=HEIGHT, width=WIDTH)

# Camera: fixed lookat, no body tracking (avoids ID mismatch with GPU model).
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.distance = 3.0
camera.elevation = -15.0
camera.azimuth = 150.0
camera.lookat[:] = [0.0, 0.0, 0.9]

obs = env_wrapped.get_observations()
frames = []
print(f"Recording {NUM_STEPS} steps...")

for step in range(NUM_STEPS):
  action = policy(obs)
  obs, _, _, _ = env_wrapped.step(action)

  # Sync GPU sim state to CPU data for rendering.
  gpu_data = env.sim.data
  cpu_data.qpos[:] = gpu_data.qpos[0].cpu().numpy()
  cpu_data.qvel[:] = gpu_data.qvel[0].cpu().numpy()
  mujoco.mj_forward(cpu_model, cpu_data)

  # Update camera lookat to follow the robot's base.
  base_pos = cpu_data.qpos[:3]
  camera.lookat[:] = base_pos

  renderer.update_scene(cpu_data, camera)
  frame = renderer.render()
  frames.append(frame.copy())

  if step % 50 == 0:
    print(f"  step {step}/{NUM_STEPS}")

renderer.close()
env.close()

print(f"Saving {len(frames)} frames to {OUTPUT_FILE}...")
writer = imageio.get_writer(OUTPUT_FILE, fps=50, quality=8)
for f in frames:
  writer.append_data(f)
writer.close()
print(f"Done! Video saved to {OUTPUT_FILE}")
