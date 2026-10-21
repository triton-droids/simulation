"""Warm-start the velocity policy by imitating the retargeted walking reference.

PPO from scratch parks 10-DoF leg-only robots in a "stand still" optimum: it can
collect upright/posture reward with no risk of falling, and the first step is
expensive.  Instead of more reward shaping, this script clones the joint targets
of the reference clip (``logs/legs_tracking/*/reference.npz``, or any motion
produced by ``teleop_to_motion.py``) into the velocity actor, so PPO starts from
a walking gait and only has to learn to modulate it with the command.

The actor sees exactly the velocity task's observation layout, so the resulting
checkpoint can be used with ``train_velocity.py --resume``.

Usage:
  uv run --no-sync python cad/bc_warmstart.py \
      --motion logs/legs_tracking/20260914_170827/reference.npz \
      --output logs/legs_velocity/bc_init.pt --epochs 3000
"""

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from hardware_tracking_task import JOINTS, HardwareTrackingConfig
from tensordict import TensorDict
from velocity_task import FOOT_SITES, make_triton_velocity_cfg

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

STANCE = {"left_hip1_joint": 0.05, "left_hip2_joint": 0.0, "left_thigh_joint": 0.0,
          "left_knee_joint": -0.35, "left_ankle_joint": 0.30,
          "right_hip1_joint": 0.05, "right_hip2_joint": 0.0, "right_thigh_joint": 0.0,
          "right_knee_joint": -0.35, "right_ankle_joint": 0.30}
ACTION_SCALE = 0.2


def quat_rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  """Rotate world vectors into the base frame (quat is wxyz, shape [...,4])."""
  w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
  # Inverse rotation = conjugate.
  w, x, y, z = w, -x, -y, -z
  out = np.empty_like(vec)
  out[..., 0] = (1 - 2 * (y * y + z * z)) * vec[..., 0] + 2 * (x * y - w * z) * vec[..., 1] + 2 * (x * z + w * y) * vec[..., 2]
  out[..., 1] = 2 * (x * y + w * z) * vec[..., 0] + (1 - 2 * (x * x + z * z)) * vec[..., 1] + 2 * (y * z - w * x) * vec[..., 2]
  out[..., 2] = 2 * (x * z - w * y) * vec[..., 0] + 2 * (y * z + w * x) * vec[..., 1] + (1 - 2 * (x * x + y * y)) * vec[..., 2]
  return out


def build_dataset(motion: Path) -> tuple[np.ndarray, np.ndarray]:
  data = np.load(motion)
  fps = float(data["fps"])
  joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
  joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
  body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
  body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)
  body_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
  body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float64)

  frames = joint_pos.shape[0]
  default = np.array([STANCE[name] for name in JOINTS])

  gravity = quat_rotate_inverse(body_quat[:, 0], np.tile(np.array([0.0, 0.0, -1.0]), (frames, 1)))
  gyro = quat_rotate_inverse(body_quat[:, 0], body_ang_vel[:, 0])
  # Command = mean body-frame velocity of the clip (forward is body +y here).
  command = quat_rotate_inverse(body_quat[:, 0], body_lin_vel[:, 0]).mean(axis=0)
  command = np.array([command[0], command[1], 0.0])

  # Gait phase: the reference is a periodic gait, so derive its frequency from
  # the left foot height and line the clock up with the foot's peak.
  foot_z = body_pos[:, 7, 2]
  foot_z = foot_z - foot_z.mean()
  spectrum = np.abs(np.fft.rfft(foot_z))
  freqs = np.fft.rfftfreq(frames, d=1.0 / fps)
  frequency = float(freqs[1 + np.argmax(spectrum[1:])])
  peak_frame = int(np.argmax(foot_z))
  times = np.arange(frames) / fps
  phase = 2 * math.pi * frequency * (times - times[peak_frame])

  actions = np.concatenate(
    (np.zeros((1, joint_pos.shape[1])), (joint_pos[:-1] - default) / ACTION_SCALE), axis=0
  )
  obs = np.concatenate(
    (
      gyro,
      gravity,
      joint_pos - default,
      joint_vel,
      actions,
      np.tile(command, (frames, 1)),
      np.stack((np.cos(phase), np.sin(phase)), axis=-1),
    ),
    axis=-1,
  )
  # Target for frame t is the joint position at t, i.e. the action that would
  # have produced it at the next control step.
  labels = np.clip((joint_pos - default) / ACTION_SCALE, -1.0, 1.0)
  print(f"reference: {frames} frames at {fps:.0f} Hz, gait {frequency:.2f} Hz, command {command.round(2)}")
  return obs.astype(np.float32), labels.astype(np.float32)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--epochs", type=int, default=3000)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  obs, labels = build_dataset(args.motion)
  obs_t = torch.from_numpy(obs).cuda()
  label_t = torch.from_numpy(labels).cuda()

  # Build the runner so the checkpoint has the exact structure train_velocity.py
  # expects, then train its actor with a supervised loss.
  cfg = make_triton_velocity_cfg(HardwareTrackingConfig())
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  agent = unitree_g1_ppo_runner_cfg()
  agent.actor.hidden_dims = (256, 128, 64)
  agent.critic.hidden_dims = (256, 128, 64)
  agent.actor.distribution_cfg["init_std"] = 1.0
  agent.seed = args.seed
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  out_dir = args.output.parent.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  runner = VelocityOnPolicyRunner(wrapped, asdict(agent), str(out_dir), "cuda:0")
  actor = runner.alg.actor

  optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
  n = obs_t.shape[0]
  for epoch in range(args.epochs):
    perm = torch.randperm(n, device="cuda")
    total = 0.0
    for start in range(0, n, args.batch_size):
      idx = perm[start : start + args.batch_size]
      pred = actor(TensorDict({"actor": obs_t[idx]}, batch_size=[len(idx)]))
      loss = torch.mean(torch.square(pred - label_t[idx]))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total += float(loss) * len(idx)
    if epoch % 250 == 0 or epoch == args.epochs - 1:
      print(f"epoch {epoch:5d} mse {total / n:.5f}")
  with torch.no_grad():
    final = actor(TensorDict({"actor": obs_t}, batch_size=[n]))
    print("final action mse (rad^2):", float(torch.mean(torch.square(final - label_t))))

  runner.save(str(args.output))
  env.close()
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()
