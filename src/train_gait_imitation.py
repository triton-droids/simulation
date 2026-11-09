"""Clone a stable balance policy plus a tested phase gait into one actor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import antioch

from train_locomotion import DRIVEN_JOINTS, build_env_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Stable PPO balance checkpoint")
    parser.add_argument("--output", default="outputs/locomotion_gait_imitation/humanoid_legs_5_gait.pt")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--collect-seconds", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--command-y", type=float, default=0.16)
    parser.add_argument("--action-scale", type=float, default=0.22)
    parser.add_argument("--cycle-hz", type=float, default=0.35)
    parser.add_argument("--hip-amplitude", type=float, default=0.10)
    parser.add_argument("--knee-amplitude", type=float, default=0.08)
    parser.add_argument("--balance-blend", type=float, default=0.35)
    parser.add_argument("--filter-upright", type=float, default=0.80, help="Discard teacher samples below this upright cosine")
    args = parser.parse_args()

    antioch.start_simulation()
    import torch
    import torch.nn.functional as F
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    cfg = build_env_cfg(
        args.num_envs,
        action_scale=args.action_scale,
        command_min=args.command_y,
        command_max=args.command_y,
        tracking_std=0.10,
        privileged_critic=True,
        actor_base_velocity=True,
        phase_observation=True,
        gait_cycle_hz=args.cycle_hz,
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
        "logger": "tensorboard", "experiment_name": "gait_imitation", "run_name": "student", "log_dir": None,
    }
    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=str(env.device))
    runner.load(args.checkpoint, map_location=str(env.device))
    actor = runner.alg.actor
    actor.eval()  # Keep the PPO observation normalizer fixed during cloning.
    teacher_policy = runner.get_inference_policy(device=str(env.device))

    obs = wrapped.get_observations()
    obs_samples: list[torch.Tensor] = []
    action_samples: list[torch.Tensor] = []
    collected_sample_count = 0
    ticks = round(args.collect_seconds / 0.02)
    hip_left = DRIVEN_JOINTS.index("left_hip1_joint")
    hip_right = DRIVEN_JOINTS.index("right_hip1_joint")
    knee_left = DRIVEN_JOINTS.index("left_knee_joint")
    knee_right = DRIVEN_JOINTS.index("right_knee_joint")

    print(f"Collecting hybrid teacher states for {args.collect_seconds:g}s across {args.num_envs} environments", flush=True)
    with torch.inference_mode():
        for step in range(ticks):
            teacher_action = teacher_policy(obs)
            phase = 2.0 * math.pi * args.cycle_hz * env.episode_length_buf * env.step_dt
            swing = torch.sin(phase)
            reference = teacher_action.clone()
            targets = {
                hip_left: args.hip_amplitude * swing / args.action_scale,
                hip_right: -args.hip_amplitude * swing / args.action_scale,
                knee_left: -args.knee_amplitude * torch.clamp(swing, min=0.0) / args.action_scale,
                knee_right: -args.knee_amplitude * torch.clamp(-swing, min=0.0) / args.action_scale,
            }
            for joint_id, target in targets.items():
                reference[:, joint_id] = args.balance_blend * teacher_action[:, joint_id] + (1.0 - args.balance_blend) * target
            reference[:, hip_left] = reference[:, hip_left].clamp(-1.0, 1.0)
            reference[:, hip_right] = reference[:, hip_right].clamp(-1.0, 1.0)
            reference[:, knee_left] = reference[:, knee_left].clamp(-1.0, 1.0)
            reference[:, knee_right] = reference[:, knee_right].clamp(-1.0, 1.0)
            upright = -env.scene["robot"].data.projected_gravity_b[:, 2]
            keep = upright >= args.filter_upright
            obs_samples.append(obs["policy"][keep].detach().clone())
            action_samples.append(reference[keep].detach().clone())
            collected_sample_count += int(keep.sum())
            obs, _, _, _ = wrapped.step(reference)

    observations = torch.cat(obs_samples, dim=0)
    labels = torch.cat(action_samples, dim=0)
    del obs_samples, action_samples
    print(f"Collected {observations.shape[0]} state-action samples", flush=True)

    optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=args.learning_rate)
    sample_count = observations.shape[0]
    batch_size = min(args.batch_size, sample_count)
    losses = []
    for epoch in range(args.epochs):
        permutation = torch.randperm(sample_count, device=env.device)
        epoch_loss = 0.0
        batches = 0
        for indices in permutation.split(batch_size):
            td = TensorDict({"policy": observations[indices]}, batch_size=[len(indices)], device=env.device)
            prediction = actor(td, stochastic_output=False)
            loss = F.mse_loss(prediction, labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        mean_loss = epoch_loss / max(1, batches)
        losses.append(mean_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(json.dumps({"imitation_epoch": epoch + 1, "action_mse": mean_loss}), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    saved = runner.alg.save()
    saved["iter"] = runner.current_learning_iteration
    saved["infos"] = {"training": "supervised gait imitation"}
    torch.save(saved, output)
    metadata = {
        "teacher_checkpoint": args.checkpoint,
        "samples": sample_count,
        "discarded_tilted_samples": args.num_envs * ticks - collected_sample_count,
        "epochs": args.epochs,
        "final_action_mse": losses[-1],
        "command_y_mps": args.command_y,
        "cycle_hz": args.cycle_hz,
        "hip_amplitude_rad": args.hip_amplitude,
        "knee_amplitude_rad": args.knee_amplitude,
        "balance_blend": args.balance_blend,
        "output_checkpoint": str(output),
        "scope": "supervised imitation of a hybrid balance-plus-gait controller; requires fixed-command PhysX validation",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
