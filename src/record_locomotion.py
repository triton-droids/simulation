"""Record a streamed Isaac Sim preview of a trained locomotion checkpoint.

Run in Antioch with `--stream` so the Isaac viewport is available to capture.
The current candidate is not a walking policy; the overlay labels that clearly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import antioch

from train_locomotion import DRIVEN_JOINTS, REWARD_PROFILES, build_env_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--standing-pose", choices=["neutral", "crouch"], default="neutral", help="Must match the checkpoint's training pose")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="baseline", help="Must match the checkpoint's observation layout")
    parser.add_argument("--original-preview-settings", action="store_true", help="Replay the original candidate with its historical knee reset and ankle gains")
    parser.add_argument("--output", default="robots/humanoid_legs_5/locomotion_preview.mp4")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--command-y", type=float, default=0.35)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--follow-camera", action="store_true", help="Translate the camera with the robot to keep it in frame")
    parser.add_argument("--gait-diagnostics", action="store_true", help="Save 50 Hz foot forces, sole clearance, motion, and actual action targets")
    parser.add_argument("--frame-stride", type=int, default=5, help="Policy steps between captured frames")
    args = parser.parse_args()
    if args.original_preview_settings and (args.standing_pose != "neutral" or args.reward_profile != "baseline"):
        parser.error("Original preview settings require the baseline profile and neutral CLI pose")

    antioch.start_simulation()
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaacsim.core.utils.viewports import set_camera_view
    from rsl_rl.runners import OnPolicyRunner

    cfg = build_env_cfg(1, standing_pose=args.standing_pose, reward_profile=args.reward_profile)
    if args.original_preview_settings:
        cfg.scene.robot.init_state.joint_pos = {"left_knee_joint": -0.1, "right_knee_joint": -0.1}
        cfg.scene.robot.actuators["ankle"].stiffness = 20.0
        cfg.scene.robot.actuators["ankle"].damping = 2.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (args.command_y, args.command_y)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 1.0)
    env = ManagerBasedRLEnv(cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner_cfg = {
        "seed": 42, "device": str(env.device), "num_steps_per_env": 24,
        "max_iterations": 1, "save_interval": 1,
        "empirical_normalization": True,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
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
    camera_origin = env.scene["robot"].data.root_pos_w[0].clone()
    from trajectory_metrics import TrajectoryMetrics
    robot = env.scene["robot"]
    hip_id = robot.body_names.index("hip")
    trajectory = TrajectoryMetrics(robot.data.body_com_pos_w[:, hip_id], robot.data.root_quat_w)
    diagnostics = None
    if args.gait_diagnostics:
        from gait_diagnostics import GaitDiagnostics
        diagnostics = GaitDiagnostics(env, DRIVEN_JOINTS)

    # Discard the initial viewport request: Antioch may return the Kit grid
    # before the first robot frame has rendered.
    antioch.capture_viewport()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=args.fps, codec="libx264", quality=8)
    body_y_sum = 0.0
    sample_count = 0
    joint_names = list(env.scene["robot"].joint_names)
    samples = []
    resets = 0
    total_steps = max(1, round(args.seconds * 50.0))
    try:
        with torch.inference_mode():
            for step in range(total_steps):
                actions = policy(obs)
                raw_actions = actions.clone() if diagnostics is not None else None
                obs, _, done, _ = wrapped.step(actions)
                if diagnostics is not None:
                    diagnostics.sample((step + 1) * env.step_dt, raw_actions, done)
                resets += int(done.sum())
                trajectory.update(robot.data.body_com_pos_w[:, hip_id], robot.data.root_quat_w, done, env.step_dt)
                if step >= total_steps // 2:
                    body_y_sum += float(env.scene["robot"].data.root_lin_vel_b[:, 1].mean())
                    sample_count += 1
                if step % args.frame_stride == 0:
                    robot = env.scene["robot"]
                    samples.append({"t_s": (step + 1) * env.step_dt, "joint_pos": robot.data.joint_pos[0].tolist(), "raw_actions": actions[0].tolist(), "root_quat_xyzw": robot.data.root_quat_w[0].tolist()})
                    if args.follow_camera:
                        delta = (robot.data.root_pos_w[0] - camera_origin).tolist()
                        set_camera_view(
                            eye=[1.2 + delta[0], 1.2 + delta[1], 0.85],
                            target=[delta[0], delta[1], 0.45],
                            camera_prim_path="/OmniverseKit_Persp",
                        )
                        env.sim.render()
                    frame = antioch.capture_viewport()
                    if frame is None:
                        continue
                    rgb = np.asarray(frame)
                    if rgb.ndim == 3 and rgb.shape[2] >= 3:
                        image = Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB")
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, image.width, 44), fill=(12, 18, 28))
                        draw.text((14, 12), f"PPO checkpoint | cmd +{args.command_y:.2f} m/s body Y | not validated walking", fill=(255, 255, 255))
                        writer.append_data(np.asarray(image))
    finally:
        writer.close()
        if diagnostics is not None:
            output.with_suffix(".gait.json").write_text(json.dumps(diagnostics.report(), indent=2) + "\n")
        env.close()

    print(f"Video: {output} ({output.stat().st_size} bytes)", flush=True)
    print(f"Mean body-Y velocity in final half: {body_y_sum / max(1, sample_count):.4f} m/s", flush=True)
    print(f"Checkpoint: {checkpoint}", flush=True)
    output.with_suffix(".json").write_text(json.dumps({"checkpoint": str(checkpoint), "standing_pose": args.standing_pose, "reward_profile": args.reward_profile, "original_preview_settings": args.original_preview_settings, "joint_names": joint_names, "resets": resets, "trajectory": trajectory.summary(), "samples": samples}, indent=2) + "\n")


if __name__ == "__main__":
    main()
