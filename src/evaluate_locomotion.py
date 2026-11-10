"""Evaluate a trained checkpoint on one fixed velocity command in PhysX.

Example:
    python src/evaluate_locomotion.py --checkpoint outputs/locomotion_256x300/humanoid_legs_5_latest.pt --vx 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import antioch

from train_locomotion import REWARD_PROFILES, build_env_cfg
from trajectory_metrics import TrajectoryMetrics
from locomotion_rewards import lateral_sole_spacing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--standing-pose", choices=["neutral", "crouch", "legacy_gait"], default="neutral", help="Must match the checkpoint's training pose")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="baseline", help="Must match the checkpoint's observation layout")
    parser.add_argument("--action-scale", type=float, default=0.40)
    parser.add_argument("--actor-base-velocity", action="store_true", help="Include body linear velocity for checkpoints trained with the 44-value privileged observation")
    parser.add_argument("--gait-period", type=float, default=0.80)
    parser.add_argument("--reset-noise-scale", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--output", default="outputs/locomotion/evaluation.json")
    args = parser.parse_args()
    if args.duration <= 0 or args.num_envs < 1:
        parser.error("duration and num-envs must be positive")

    antioch.start_simulation()
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    cfg = build_env_cfg(
        args.num_envs, standing_pose=args.standing_pose,
        reward_profile=args.reward_profile, action_scale=args.action_scale,
        actor_base_velocity=args.actor_base_velocity,
        gait_period=args.gait_period,
        reset_noise_scale=args.reset_noise_scale,
    )
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.vx, args.vx)
    cfg.commands.base_velocity.ranges.lin_vel_y = (args.vy, args.vy)
    cfg.commands.base_velocity.ranges.ang_vel_z = (args.yaw_rate, args.yaw_rate)
    cfg.episode_length_s = max(cfg.episode_length_s, args.duration + 1.0)
    env = ManagerBasedRLEnv(cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    train_cfg = {
        "seed": 42, "device": str(env.device), "num_steps_per_env": 24,
        "max_iterations": 1, "save_interval": 1,
        "empirical_normalization": True,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "scalar"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": None},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2, "entropy_coef": 0.01, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 3e-4, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
        "logger": "tensorboard", "experiment_name": "humanoid_legs_5_eval", "run_name": "eval", "log_dir": None,
    }
    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=str(env.device))
    checkpoint = Path(args.checkpoint)
    runner.load(str(checkpoint), map_location=str(env.device))
    policy = runner.get_inference_policy(device=str(env.device))
    robot = env.scene["robot"]
    body_names = list(robot.body_names)
    left_foot_id = body_names.index("left_foot") if "left_foot" in body_names else None
    right_foot_id = body_names.index("right_foot") if "right_foot" in body_names else None
    obs = wrapped.get_observations()
    hip_id = body_names.index("hip")
    trajectory = TrajectoryMetrics(robot.data.body_com_pos_w[:, hip_id], robot.data.root_quat_w)
    ticks = round(args.duration / 0.02)
    sample_start = ticks // 2
    sum_vel_error = 0.0
    sum_vx = 0.0
    sum_vy = 0.0
    sum_action_abs = 0.0
    sum_yaw_error = 0.0
    sum_upright = 0.0
    sum_height = 0.0
    sum_left_foot_z = 0.0
    sum_right_foot_z = 0.0
    sum_stance_width = 0.0
    sample_count = 0
    done_count = 0
    with torch.inference_mode():
        for i in range(ticks):
            actions = policy(obs)
            obs, _, done, _ = wrapped.step(actions)
            done_count += int(done.sum())
            trajectory.update(robot.data.body_com_pos_w[:, hip_id], robot.data.root_quat_w, done, env.step_dt)
            if i >= sample_start:
                vel = robot.data.root_lin_vel_b[:, :2]
                sum_vx += float(vel[:, 0].mean())
                sum_vy += float(vel[:, 1].mean())
                sum_action_abs += float(actions.abs().mean())
                yaw_rate = robot.data.root_ang_vel_b[:, 2]
                target_xy = torch.tensor([args.vx, args.vy], device=env.device)
                sum_vel_error += float(torch.linalg.vector_norm(vel - target_xy, dim=1).mean())
                sum_yaw_error += float((yaw_rate - args.yaw_rate).abs().mean())
                sum_upright += float((-robot.data.projected_gravity_b[:, 2]).mean())
                sum_height += float(robot.data.root_pos_w[:, 2].mean())
                if left_foot_id is not None:
                    sum_left_foot_z += float(robot.data.body_pos_w[:, left_foot_id, 2].mean())
                if right_foot_id is not None:
                    sum_right_foot_z += float(robot.data.body_pos_w[:, right_foot_id, 2].mean())
                sum_stance_width += float(lateral_sole_spacing(env).mean())
                sample_count += 1
    result = {
        "standing_pose": args.standing_pose,
        "reward_profile": args.reward_profile,
        "checkpoint": str(checkpoint),
        "command_mps_radps": {"vx": args.vx, "vy": args.vy, "yaw_rate": args.yaw_rate},
        "duration_s": args.duration,
        "num_envs": args.num_envs,
        "mean_horizontal_velocity_error_mps": sum_vel_error / sample_count,
        "mean_root_velocity_body_xy_mps": {"vx": sum_vx / sample_count, "vy": sum_vy / sample_count},
        "mean_absolute_policy_action": sum_action_abs / sample_count,
        "mean_yaw_rate_error_radps": sum_yaw_error / sample_count,
        "mean_upright_cosine": sum_upright / sample_count,
        "mean_root_height_m": sum_height / sample_count,
        "articulation_body_names": body_names,
        "mean_foot_body_height_m": {
            "left": sum_left_foot_z / sample_count if left_foot_id is not None else None,
            "right": sum_right_foot_z / sample_count if right_foot_id is not None else None,
        },
        "mean_lateral_sole_spacing_m": sum_stance_width / sample_count,
        "terminated_env_steps": done_count,
        "trajectory": trajectory.summary(),
        "hardware_ready": False,
        "scope": "fixed-command Isaac Lab PhysX evaluation; not hardware validation",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
