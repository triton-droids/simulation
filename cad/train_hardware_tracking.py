"""Train the hardware-constrained reference-tracking policy from scratch."""

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from hardware_tracking_task import (
  JOINTS,
  HardwareTrackingConfig,
  make_hardware_leg_tracking_cfg,
)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--motion",
    type=Path,
    default=Path(__file__).resolve().parent.parent
    / "logs"
    / "legs_tracking"
    / "20260914_170827"
    / "reference.npz",
  )
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--target-speed", type=float, default=1.0)
  parser.add_argument(
    "--run-dir", type=Path, help="New output directory; must not exist"
  )
  parser.add_argument("--iterations", type=int, default=100000)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument(
    "--resume", type=Path, help="Continue policy and optimizer from a checkpoint"
  )
  args = parser.parse_args()
  if args.resume is not None and not args.resume.is_file():
    raise FileNotFoundError(args.resume)
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable in this execution environment")
  hardware = HardwareTrackingConfig(target_speed_rad_s=args.target_speed)
  cfg = make_hardware_leg_tracking_cfg(args.motion, hardware)
  cfg.scene.num_envs = args.num_envs
  ref = np.load(args.motion)
  frames = ref["joint_pos"].shape[0]
  if frames != 299:
    raise ValueError(f"Expected the deployed 299-frame reference, got {frames}")
  required = {
    "joint_pos": (10,),
    "joint_vel": (10,),
    "body_pos_w": (13, 3),
    "body_quat_w": (13, 4),
    "body_lin_vel_w": (13, 3),
    "body_ang_vel_w": (13, 3),
  }
  for key, trailing in required.items():
    if ref[key].shape != (frames, *trailing) or not np.isfinite(ref[key]).all():
      raise ValueError(f"Invalid reference array: {key}")
  log = args.run_dir or (
    Path(__file__).resolve().parent.parent
    / "logs"
    / "legs_tracking_hardware"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
  )
  log.mkdir(parents=True, exist_ok=False)
  shutil.copy2(args.motion, log / "reference.npz")
  for name in (
    "chrobot_hardware_candidate.xml",
    "chrobot_hardware_actuated.xml",
    "tracking_body_order.json",
    "tracking_task.py",
    "hardware_tracking_task.py",
    "train_hardware_tracking.py",
    "evaluate_hardware_tracking.py",
    "verify_tracking_onnx.py",
    "HARDWARE_TRACKING_CHANGES.md",
  ):
    shutil.copy2(Path(__file__).with_name(name), log / name)
  (log / "CHANGES.md").write_text(
    (Path(__file__).with_name("HARDWARE_TRACKING_CHANGES.md")).read_text()
  )
  (log / "run.json").write_text(
    json.dumps(
      {
        "motion": str(args.motion.resolve()),
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "resume": str(args.resume.resolve()) if args.resume else None,
        "hardware": asdict(hardware),
      },
      indent=2,
    )
  )
  agent = unitree_g1_tracking_ppo_runner_cfg()
  agent.experiment_name = "legs_tracking_hardware"
  agent.logger = "tensorboard"
  agent.actor.hidden_dims = (128, 64)
  agent.critic.hidden_dims = (128, 64)
  agent.actor.distribution_cfg["init_std"] = 0.2
  agent.save_interval = args.save_interval
  (log / "agent.json").write_text(json.dumps(asdict(agent), indent=2))
  print(f"Training directory: {log}", flush=True)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    assert tuple(env.scene["robot"].joint_names) == JOINTS
    assert tuple(env.action_manager.get_term("joint_pos").target_names) == JOINTS
    assert env.observation_manager.group_obs_dim["actor"] == (56,)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MotionTrackingOnPolicyRunner(wrapped, asdict(agent), str(log), "cuda:0")
    if args.resume is not None:
      runner.load(str(args.resume.resolve()), map_location="cuda:0")
      print(
        f"Resumed from {args.resume}; iteration {runner.current_learning_iteration}",
        flush=True,
      )
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    print(f"Training finished: {log}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
