"""Train the walk-to-ball and kick policy."""

import argparse
import json
import os
import platform
import shutil
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import torch
from budgeted_soccer_runner import BudgetedSoccerRunner, TrainingBudgetReached
from hardware_tracking_task import JOINTS, HardwareTrackingConfig
from soccer_task import make_soccer_cfg

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

# velocity observations (39) + ball position in base frame (2)
ACTOR_OBS_DIM = 3 + 3 + 10 + 10 + 10 + 3 + 2 + 2  # + gait phase, ball


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--target-speed", type=float, default=1.0)
  parser.add_argument("--run-dir", type=Path)
  parser.add_argument("--iterations", type=int, default=100000)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--evaluation-interval", type=int, default=500)
  parser.add_argument("--hours", type=float, default=None)
  parser.add_argument("--resume", type=Path)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()
  if args.hours is not None and args.hours <= 0:
    parser.error("--hours must be positive")
  if args.evaluation_interval % args.save_interval:
    parser.error("evaluation interval must be a multiple of save interval")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable in this execution environment")

  hardware = HardwareTrackingConfig(target_speed_rad_s=args.target_speed)
  cfg = make_soccer_cfg(hardware)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed

  log = args.run_dir or (
    Path(__file__).resolve().parent.parent
    / "logs"
    / "legs_soccer"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
  )
  log = Path(os.path.abspath(log))
  if args.resume is not None and log.is_dir():
    print(f"Resuming into existing run directory: {log}", flush=True)
  else:
    log.mkdir(parents=True, exist_ok=False)
  for name in (
    "triton_lower_body.xml",
    "triton_lower_body_actuated.xml",
    "hardware_tracking_task.py",
    "velocity_task.py",
    "soccer_task.py",
    "budgeted_velocity_runner.py",
    "budgeted_soccer_runner.py",
    "train_soccer.py",
    "evaluate_soccer.py",
    "HARDWARE_TRACKING_CHANGES.md",
  ):
    shutil.copy2(Path(__file__).with_name(name), log / name)
  (log / "run.json").write_text(
    json.dumps(
      {
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "hours": args.hours,
        "seed": args.seed,
        "environment": {
          "python": platform.python_version(),
          "torch": torch.__version__,
          "mujoco": version("mujoco"),
          "rsl_rl": version("rsl-rl-lib"),
        },
        "hardware": asdict(hardware),
      },
      indent=2,
    )
  )

  agent = unitree_g1_ppo_runner_cfg()
  agent.experiment_name = "legs_soccer"
  agent.logger = "tensorboard"
  agent.seed = args.seed
  agent.max_iterations = args.iterations
  agent.save_interval = args.save_interval
  agent.num_steps_per_env = 24
  agent.actor.hidden_dims = (256, 128, 64)
  agent.critic.hidden_dims = (256, 128, 64)
  agent.actor.distribution_cfg["init_std"] = 0.6
  (log / "agent.json").write_text(json.dumps(asdict(agent), indent=2))

  print(f"Training directory: {log}", flush=True)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    assert tuple(env.scene["robot"].joint_names) == JOINTS
    assert tuple(env.action_manager.get_term("joint_pos").target_names) == JOINTS
    assert env.observation_manager.group_obs_dim["actor"] == (ACTOR_OBS_DIM,), (
      env.observation_manager.group_obs_dim
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = BudgetedSoccerRunner(wrapped, asdict(agent), str(log), "cuda:0")
    runner.configure_budget(args.hours, args.evaluation_interval, log)
    if args.resume is not None:
      runner.load(str(args.resume.resolve()), map_location="cuda:0")
      print(
        f"Resumed from {args.resume}; iteration {runner.current_learning_iteration}",
        flush=True,
      )
    try:
      runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    except TrainingBudgetReached:
      print("Wall-clock budget reached; checkpoint saved.", flush=True)
    status_path = log / "status.json"
    status = json.loads(status_path.read_text())
    status["state"] = "training_finished"
    status_path.write_text(json.dumps(status, indent=2))
    (log / "TRAINING_FINISHED").write_text(
      "Training ended; inspect evaluation results before deployment.\n"
    )
    print(f"Training finished: {log}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
