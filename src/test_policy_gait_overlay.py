"""Try simple gait references overlaid on a trained balance policy in PhysX."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import antioch

from train_locomotion import DRIVEN_JOINTS, build_env_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--command-y", type=float, default=0.16)
    parser.add_argument("--action-scale", type=float, default=0.22)
    parser.add_argument("--tracking-std", type=float, default=0.15)
    parser.add_argument("--reset-noise-scale", type=float, default=0.35)
    parser.add_argument("--hip-amp", type=float)
    parser.add_argument("--knee-amp", type=float)
    parser.add_argument("--frequency", type=float)
    parser.add_argument("--hip-sign", type=float, choices=(-1.0, 1.0))
    parser.add_argument("--knee-phase", type=float)
    parser.add_argument("--balance-blend", type=float)
    parser.add_argument("--output", default=None, help="Optional JSON path for the sweep report")
    args = parser.parse_args()

    antioch.start_simulation()
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    cfg = build_env_cfg(
        args.num_envs,
        standing_pose="legacy_gait",
        reward_profile="stepping_wide",
        action_scale=args.action_scale,
        actor_base_velocity=True,
        gait_period=1.0 / 0.35,
        reset_noise_scale=args.reset_noise_scale,
    )
    cfg.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (args.command_y, args.command_y)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env = ManagerBasedRLEnv(cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    train_cfg = {
        "seed": 42, "device": str(env.device), "num_steps_per_env": 24,
        "max_iterations": 1, "save_interval": 1, "empirical_normalization": True,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "scalar"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": None},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2, "entropy_coef": 0.01, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 3e-4, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
        "logger": "tensorboard", "experiment_name": "gait_overlay_eval", "run_name": "eval", "log_dir": None,
    }
    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=str(env.device))
    runner.load(
        str(Path(args.checkpoint)),
        load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
        map_location=str(env.device),
    )
    policy = runner.get_inference_policy(device=str(env.device))
    robot = env.scene["robot"]
    from trajectory_metrics import TrajectoryMetrics
    body_names = list(robot.body_names)
    hip_body_id = body_names.index("hip")

    specified = (args.hip_amp, args.knee_amp, args.frequency, args.hip_sign,
                 args.knee_phase, args.balance_blend)
    if any(value is not None for value in specified):
        if any(value is None for value in specified):
            parser.error("single-variant mode requires all six gait parameters")
        variants = [specified]
    else:
        variants = list(itertools.product(
            (0.30, 0.35, 0.40),  # cycle frequency
            (1.0, -1.0),          # hip swing sign
            (0.0, math.pi / 2),   # knee flexion phase
            (0.45, 0.515, 0.58), # balance action blend
        ))
        variants = [(0.10, 0.08, freq, sign, knee_phase, blend)
                    for freq, sign, knee_phase, blend in variants]
    reports = []
    ticks = round(args.seconds / 0.02)
    with torch.inference_mode():
        for variant_i, (hip_amp, knee_amp, freq, hip_sign, knee_phase, balance_blend) in enumerate(variants):
            if variant_i:
                env.reset()
            obs = wrapped.get_observations()
            trajectory = TrajectoryMetrics(
                robot.data.body_com_pos_w[:, hip_body_id], robot.data.root_quat_w
            )
            sums = torch.zeros(7, device=env.device)
            done_count = 0
            samples = 0
            for step in range(ticks):
                actions = policy(obs)
                phase = 2.0 * math.pi * freq * env.episode_length_buf * env.step_dt
                swing = torch.sin(phase)
                knee_swing = torch.sin(phase + knee_phase)
                left_knee = -knee_amp * knee_swing.clamp_min(0.0)
                right_knee = -knee_amp * (-knee_swing).clamp_min(0.0)
                offsets = {
                    "left_hip1_joint": hip_sign * hip_amp * swing,
                    "right_hip1_joint": -hip_sign * hip_amp * swing,
                    "left_knee_joint": left_knee,
                    "right_knee_joint": right_knee,
                }
                for name, delta in offsets.items():
                    jid = DRIVEN_JOINTS.index(name)
                    ref_action = delta / args.action_scale
                    # Keep the balance policy everywhere else; use its action
                    # as a center and smoothly blend in the joint trajectory.
                    actions[:, jid] = balance_blend * actions[:, jid] + (1.0 - balance_blend) * ref_action
                obs, _, done, _ = wrapped.step(actions)
                done_count += int(done.sum())
                trajectory.update(
                    robot.data.body_com_pos_w[:, hip_body_id],
                    robot.data.root_quat_w,
                    done,
                    env.step_dt,
                )
                if step >= ticks // 2:
                    com_vel_yaw = quat_apply_inverse(
                        yaw_quat(robot.data.root_quat_w.torch), robot.data.body_com_lin_vel_w.torch[:, 0]
                    )
                    feet = robot.data.body_pos_w[:, [robot.body_names.index("left_foot"), robot.body_names.index("right_foot")], 2]
                    sums += torch.tensor([
                        float(com_vel_yaw[:, 0].mean()), float(com_vel_yaw[:, 1].mean()),
                        float((-robot.data.projected_gravity_b[:, 2]).mean()),
                        float(robot.data.root_pos_w[:, 2].mean()), float(feet[:, 0].mean()),
                        float(feet[:, 1].mean()), float(actions.abs().mean()),
                    ], device=env.device)
                    samples += 1
            reports.append({
                "hip_amp": hip_amp, "knee_amp": knee_amp, "cycle_hz": freq,
                "hip_sign": hip_sign, "knee_phase": knee_phase,
                "balance_blend": balance_blend,
                "mean_com_velocity_yaw_xy_mps": (sums[:2] / samples).tolist(),
                "mean_upright_cosine": float(sums[2] / samples),
                "mean_root_height_m": float(sums[3] / samples),
                "mean_foot_body_height_m": (sums[4:6] / samples).tolist(),
                "mean_abs_raw_action": float(sums[6] / samples),
                "terminated_env_steps": done_count,
                "trajectory": trajectory.summary(),
            })
    result = {"checkpoint": args.checkpoint, "command_y_mps": args.command_y,
              "duration_s": args.seconds, "num_envs": args.num_envs,
              "reset_noise_scale": args.reset_noise_scale,
              "variants": reports}
    encoded = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n")
    print(encoded)
    env.close()


if __name__ == "__main__":
    main()
