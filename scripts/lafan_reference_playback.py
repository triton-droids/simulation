# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run CH LAFAN reference playback with zero residual actions."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="LAFAN reference playback smoke test.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Humanoid-Locomotion-Flat-Direct-v0", help="Task name.")
parser.add_argument(
    "--motion_dir",
    type=str,
    default="/cephfs/holosoma/data/lafan/retargeted/ch_robot_stance_flatfoot_locomotion_full_floor_norm_with_vel",
    help="Directory containing enriched LAFAN .npz files.",
)
parser.add_argument("--motion_manifest", type=str, default="", help="Optional manifest file with one .npz path per line.")
parser.add_argument("--num_steps", type=int, default=600, help="Number of policy steps to run; <=0 runs until closed.")
parser.add_argument("--random_start", action="store_true", default=False, help="Start each env from a random clip frame.")
parser.add_argument("--debug_print", action="store_true", default=False, help="Print motion/reference debug info.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import tritonhumanoid.tasks  # noqa: F401


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.motion_reference_dir = args_cli.motion_dir
    env_cfg.motion_manifest_file = args_cli.motion_manifest
    env_cfg.motion_reference_playback = True
    env_cfg.motion_random_start = args_cli.random_start
    env_cfg.motion_reference_debug_print = args_cli.debug_print
    env_cfg.enable_adr = False
    env_cfg.use_curriculum = False
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_joint_vel_noise = 0.0
    env_cfg.push_force_range = (0.0, 0.0)
    env_cfg.action_max_latency = 0
    env_cfg.obs_max_latency = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")
    obs, _ = env.reset()
    print(f"[INFO] Reset complete. Initial policy obs shape: {obs['policy'].shape}")

    step = 0
    while simulation_app.is_running() and (args_cli.num_steps <= 0 or step < args_cli.num_steps):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
        step += 1

    env.close()
    print(f"[INFO] Playback finished after {step} policy steps.")


if __name__ == "__main__":
    main()
    simulation_app.close()
