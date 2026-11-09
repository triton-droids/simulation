"""Try simple gait references overlaid on a trained balance policy in PhysX."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    antioch.start_simulation()
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    cfg = build_env_cfg(
        args.num_envs,
        action_scale=args.action_scale,
        command_min=args.command_y,
        command_max=args.command_y,
        tracking_std=args.tracking_std,
        privileged_critic=True,
        actor_base_velocity=True,
        phase_observation=True,
        gait_cycle_hz=0.35,
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
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "scalar"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": None},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2, "entropy_coef": 0.01, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 3e-4, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
        "logger": "tensorboard", "experiment_name": "gait_overlay_eval", "run_name": "eval", "log_dir": None,
    }
    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=str(env.device))
    runner.load(str(Path(args.checkpoint)), map_location=str(env.device))
    policy = runner.get_inference_policy(device=str(env.device))
    robot = env.scene["robot"]

    variants = [
        (0.10, 0.08, 0.35, 1.0, 0.0, 0.51),
        (0.10, 0.08, 0.35, 1.0, 0.0, 0.515),
        (0.10, 0.08, 0.35, 1.0, 0.0, 0.525),
    ]
    reports = []
    ticks = round(args.seconds / 0.02)
    with torch.inference_mode():
        for variant_i, (hip_amp, knee_amp, freq, hip_sign, knee_phase, balance_blend) in enumerate(variants):
            if variant_i:
                env.reset()
            obs = wrapped.get_observations()
            sums = torch.zeros(7, device=env.device)
            done_count = 0
            samples = 0
            for step in range(ticks):
                actions = policy(obs)
                phase = 2.0 * math.pi * freq * step * 0.02
                s = math.sin(phase)
                left_knee = -knee_amp * max(0.0, math.sin(phase + knee_phase))
                right_knee = -knee_amp * max(0.0, -math.sin(phase + knee_phase))
                offsets = {
                    "left_hip1_joint": hip_sign * hip_amp * s,
                    "right_hip1_joint": -hip_sign * hip_amp * s,
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
            })
    print(json.dumps({"checkpoint": args.checkpoint, "variants": reports}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
