"""Train the velocity-command locomotion policy for the Triton lower body."""

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
from budgeted_velocity_runner import BudgetedVelocityRunner, TrainingBudgetReached
from hardware_tracking_task import JOINTS, HardwareTrackingConfig
from velocity_task import make_triton_velocity_cfg

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

ACTOR_OBS_DIM = 3 + 3 + 10 + 10 + 10 + 3 + 2  # + gait phase


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument(
    "--target-speed",
    type=float,
    default=1.0,
    help="Deployed position-target slew cap in rad/s (hardware contract)",
  )
  parser.add_argument("--run-dir", type=Path, help="New output directory; must not exist")
  parser.add_argument("--iterations", type=int, default=100000)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--evaluation-interval", type=int, default=1000)
  parser.add_argument("--hours", type=float, default=None)
  parser.add_argument("--resume", type=Path)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--init-std",
    type=float,
    default=None,
    help=(
      "Override the policy action std after loading a warm start.  A BC actor is "
      "trained on the distribution mean only, so its std stays at the value from "
      "training time; leaving it at 1.0 injects ~0.2 rad of noise per joint and "
      "destroys a delicate gait on the first PPO update."
    ),
  )
  parser.add_argument(
    "--episode-seconds",
    type=float,
    default=20.0,
    help=(
      "Episode length.  A warm-started gait that survives ~8 s is better off with "
      "5-8 s episodes: otherwise every walking episode ends in a fall while "
      "standing survives to the time limit, and PPO learns to sit down."
    ),
  )
  args = parser.parse_args()
  if args.hours is not None and args.hours <= 0:
    parser.error("--hours must be positive")
  if args.evaluation_interval % args.save_interval:
    parser.error("evaluation interval must be a multiple of save interval")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable in this execution environment")

  hardware = HardwareTrackingConfig(target_speed_rad_s=args.target_speed)
  cfg = make_triton_velocity_cfg(hardware)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.episode_length_s = args.episode_seconds

  log = args.run_dir or (
    Path(__file__).resolve().parent.parent
    / "logs"
    / "legs_velocity"
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
    "budgeted_velocity_runner.py",
    "train_velocity.py",
    "evaluate_velocity.py",
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
        "control_hz": 1.0 / (cfg.sim.mujoco.timestep * cfg.decimation),
      },
      indent=2,
    )
  )

  agent = unitree_g1_ppo_runner_cfg()
  agent.experiment_name = "legs_velocity"
  agent.logger = "tensorboard"
  agent.seed = args.seed
  agent.max_iterations = args.iterations
  agent.save_interval = args.save_interval
  agent.num_steps_per_env = 24
  agent.actor.hidden_dims = (256, 128, 64)
  agent.critic.hidden_dims = (256, 128, 64)
  agent.actor.distribution_cfg["init_std"] = 1.0
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
    runner = BudgetedVelocityRunner(wrapped, asdict(agent), str(log), "cuda:0")
    runner.configure_budget(args.hours, args.evaluation_interval, log)
    if args.resume is not None:
      runner.load(str(args.resume.resolve()), map_location="cuda:0")
      print(
        f"Resumed from {args.resume}; iteration {runner.current_learning_iteration}",
        flush=True,
      )
    if args.init_std is not None:
      std_param = runner.alg.actor.distribution.std_param
      with torch.no_grad():
        std_param.fill_(args.init_std)
      print(f"Action std set to {args.init_std}", flush=True)
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
