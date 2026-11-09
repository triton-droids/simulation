"""Record a streamed Isaac Sim preview of a locomotion checkpoint.

Run in Antioch with `--stream` so the Isaac viewport is available to capture.
The video is a simulation preview and does not establish hardware readiness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import antioch

from train_locomotion import DRIVEN_JOINTS, build_env_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="robots/humanoid_legs_5/locomotion_preview.mp4")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--command-y", type=float, default=0.35)
    parser.add_argument("--action-scale", type=float, default=0.40)
    parser.add_argument("--tracking-std", type=float, default=0.60)
    parser.add_argument("--privileged-critic", action="store_true")
    parser.add_argument("--actor-base-velocity", action="store_true")
    parser.add_argument("--phase-observation", action="store_true")
    parser.add_argument("--gait-cycle-hz", type=float, default=0.5)
    parser.add_argument("--hybrid-gait", action="store_true", help="Overlay a phase gait while the checkpoint balances the other joints")
    parser.add_argument("--gait-hip-amplitude", type=float, default=0.10)
    parser.add_argument("--gait-knee-amplitude", type=float, default=0.08)
    parser.add_argument("--gait-balance-blend", type=float, default=0.50)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=5, help="Policy steps between captured frames")
    args = parser.parse_args()

    antioch.start_simulation()
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaacsim.core.utils.viewports import set_camera_view
    from rsl_rl.runners import OnPolicyRunner

    cfg = build_env_cfg(
        1,
        action_scale=args.action_scale,
        tracking_std=args.tracking_std,
        privileged_critic=args.privileged_critic,
        actor_base_velocity=args.actor_base_velocity,
        phase_observation=args.phase_observation,
        gait_cycle_hz=args.gait_cycle_hz,
    )
    cfg.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (args.command_y, args.command_y)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env = ManagerBasedRLEnv(cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner_cfg = {
        "seed": 42, "device": str(env.device), "num_steps_per_env": 24,
        "max_iterations": 1, "save_interval": 1,
        "empirical_normalization": True,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"] if args.privileged_critic else ["policy"]},
        "actor": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "scalar"}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [256, 128, 128], "activation": "elu", "obs_normalization": True, "distribution_cfg": None},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2, "entropy_coef": 0.01, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 3e-4, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
        "logger": "tensorboard", "experiment_name": "humanoid_legs_5_preview", "run_name": "preview", "log_dir": None,
    }
    runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=None, device=str(env.device))
    checkpoint = Path(args.checkpoint)
    runner.load(str(checkpoint), map_location=str(env.device))
    policy = runner.get_inference_policy(device=str(env.device))
    set_camera_view(eye=[1.2, 1.2, 0.85], target=[0.0, 0.0, 0.45], camera_prim_path="/OmniverseKit_Persp")
    obs = wrapped.get_observations()

    # Discard the initial viewport request: Antioch may return the Kit grid
    # before the first robot frame has rendered.
    antioch.capture_viewport()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=args.fps, codec="libx264", quality=8)
    body_y_sum = 0.0
    sample_count = 0
    done_count = 0
    total_steps = max(1, round(args.seconds * 50.0))
    try:
        with torch.inference_mode():
            for step in range(total_steps):
                actions = policy(obs)
                if args.hybrid_gait:
                    phase = 2.0 * np.pi * args.gait_cycle_hz * env.episode_length_buf * env.step_dt
                    swing = torch.sin(phase)
                    offsets = {
                        "left_hip1_joint": args.gait_hip_amplitude * swing,
                        "right_hip1_joint": -args.gait_hip_amplitude * swing,
                        "left_knee_joint": -args.gait_knee_amplitude * torch.clamp(swing, min=0.0),
                        "right_knee_joint": -args.gait_knee_amplitude * torch.clamp(-swing, min=0.0),
                    }
                    for name, delta in offsets.items():
                        jid = DRIVEN_JOINTS.index(name)
                        ref_action = delta / args.action_scale
                        actions[:, jid] = args.gait_balance_blend * actions[:, jid] + (1.0 - args.gait_balance_blend) * ref_action
                obs, _, done, _ = wrapped.step(actions)
                done_count += int(done.sum())
                if step >= total_steps // 2:
                    robot = env.scene["robot"]
                    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
                    com_vel_yaw = quat_apply_inverse(
                        yaw_quat(robot.data.root_quat_w.torch), robot.data.body_com_lin_vel_w.torch[:, 0]
                    )
                    body_y_sum += float(com_vel_yaw[:, 1].mean())
                    sample_count += 1
                if step % args.frame_stride == 0:
                    frame = antioch.capture_viewport()
                    if frame is None:
                        continue
                    rgb = np.asarray(frame)
                    if rgb.ndim == 3 and rgb.shape[2] >= 3:
                        image = Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB")
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, image.width, 44), fill=(12, 18, 28))
                        label = "hybrid gait" if args.hybrid_gait else "policy"
                        draw.text((14, 12), f"humanoid_legs_5 | {label} | COM command +{args.command_y:.2f} m/s | Isaac Lab", fill=(255, 255, 255))
                        writer.append_data(np.asarray(image))
    finally:
        writer.close()
        env.close()

    print(f"Video: {output} ({output.stat().st_size} bytes)", flush=True)
    print(f"Mean body-Y velocity in final half: {body_y_sum / max(1, sample_count):.4f} m/s", flush=True)
    print(f"Fall/reset events during preview: {done_count}", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
