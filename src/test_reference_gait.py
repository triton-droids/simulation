"""PhysX check of a simple alternating hip/knee reference gait."""

from __future__ import annotations

import argparse
import json
import math

import antioch

from train_locomotion import DRIVEN_JOINTS, build_env_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--cycle-hz", type=float, default=0.45)
    parser.add_argument("--hip-amp", type=float, default=0.10)
    parser.add_argument("--knee-amp", type=float, default=0.08)
    parser.add_argument("--hip-sign", type=float, default=1.0)
    parser.add_argument("--knee-phase", type=float, default=0.0)
    args = parser.parse_args()

    antioch.start_simulation()
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    scale = 0.22
    cfg = build_env_cfg(
        args.num_envs,
        standing_pose="legacy_gait",
        reward_profile="stepping_wide",
        action_scale=scale,
    )
    cfg.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)
    env = ManagerBasedRLEnv(cfg=cfg)
    robot = env.scene["robot"]
    ids = {n: robot.joint_names.index(n) for n in DRIVEN_JOINTS}
    obs = env.observation_manager.compute()
    variants = [
        (args.cycle_hz, args.hip_amp, args.knee_amp, args.hip_sign, args.knee_phase),
        (0.30, 0.05, 0.04, args.hip_sign, 0.0),
        (0.30, 0.10, 0.04, args.hip_sign, 0.0),
        (0.45, 0.05, 0.04, args.hip_sign, 0.0),
        (0.45, 0.10, 0.08, -args.hip_sign, math.pi),
    ]
    reports = []
    with torch.inference_mode():
        for variant_i, (cycle_hz, hip_amp, knee_amp, hip_sign, knee_phase) in enumerate(variants):
            if variant_i:
                env.reset()
            vel_sum = torch.zeros(2, device=env.device)
            upright_sum = torch.zeros((), device=env.device)
            height_sum = torch.zeros((), device=env.device)
            lift_sum = torch.zeros(2, device=env.device)
            done_count = 0
            n = 0
            ticks = round(args.seconds / 0.02)
            for step in range(ticks):
                phase = 2.0 * math.pi * cycle_hz * step * 0.02
                s = math.sin(phase)
                action = torch.zeros((args.num_envs, len(DRIVEN_JOINTS)), device=env.device)
                left_hip = hip_sign * hip_amp * s
                right_hip = -left_hip
                left_flex = max(0.0, math.sin(phase + knee_phase))
                right_flex = max(0.0, -math.sin(phase + knee_phase))
                action[:, DRIVEN_JOINTS.index("left_hip1_joint")] = left_hip / scale
                action[:, DRIVEN_JOINTS.index("right_hip1_joint")] = right_hip / scale
                action[:, DRIVEN_JOINTS.index("left_knee_joint")] = -knee_amp * left_flex / scale
                action[:, DRIVEN_JOINTS.index("right_knee_joint")] = -knee_amp * right_flex / scale
                _, _, terminated, truncated, _ = env.step(action)
                done_count += int((terminated | truncated).sum())
                if step >= ticks // 2:
                    com_vel_yaw = quat_apply_inverse(
                        yaw_quat(robot.data.root_quat_w.torch), robot.data.body_com_lin_vel_w.torch[:, 0]
                    )
                    vel_sum += com_vel_yaw[:, :2].mean(dim=0)
                    upright_sum += (-robot.data.projected_gravity_b[:, 2]).mean()
                    height_sum += robot.data.root_pos_w[:, 2].mean()
                    foot_ids = [robot.body_names.index("left_foot"), robot.body_names.index("right_foot")]
                    lift_sum += robot.data.body_pos_w[:, foot_ids, 2].mean(dim=0)
                    n += 1
            reports.append({
                "seconds": args.seconds, "cycle_hz": cycle_hz, "hip_amp": hip_amp,
                "knee_amp": knee_amp, "hip_sign": hip_sign, "knee_phase": knee_phase,
                "mean_com_velocity_yaw_xy_mps": (vel_sum / n).tolist(),
                "mean_upright_cosine": float(upright_sum / n),
                "mean_root_height_m": float(height_sum / n),
                "mean_foot_body_height_m": (lift_sum / n).tolist(),
                "terminated_env_steps": done_count,
            })
    print(json.dumps({"variants": reports, "scope": "scripted joint references; exploratory model check, not learned walking"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
