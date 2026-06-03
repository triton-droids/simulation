#!/usr/bin/env python3
"""Evaluate the Isaac locomotion policy in MuJoCo physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "tritonhumanoid"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tritonhumanoid.eval.mujoco_locomotion import (  # noqa: E402
    ACTION_DIM,
    CLIP_OBSERVATIONS,
    CONTROL_DT,
    DEFAULT_ACTIVE_URDF,
    DEFAULT_MODEL_CACHE,
    DEFAULT_SOURCE_MJCF,
    MujocoLocomotionEnv,
    PlaceholderMJCFError,
    command_profile,
    ensure_isaac_locomotion_mjcf,
    load_torchscript_policy,
    sleep_for_realtime,
    validate_mjcf_against_urdf,
)


DEFAULT_POLICY = REPO_ROOT / "logs/rl_games/humanoid_flat_direct/exported_policy/ppo_policy.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--model-xml", type=Path, default=None, help="Override generated Isaac-locomotion MJCF path.")
    parser.add_argument("--source-xml", type=Path, default=DEFAULT_SOURCE_MJCF)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--refresh-model", action="store_true")
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional rollout .npz path.")
    parser.add_argument("--command-profile", choices=["none", "stand_forward_yaw"], default="stand_forward_yaw")
    parser.add_argument(
        "--joint-velocity-limit",
        type=float,
        default=None,
        help="Actuated joint velocity clip in rad/s. Defaults to MUJOCO_JOINT_VELOCITY_LIMIT or 15.0; use 0 to disable.",
    )
    return parser.parse_args()


def run_validate_only(args: argparse.Namespace) -> None:
    xml_path = args.model_xml
    if xml_path is None:
        xml_path = ensure_isaac_locomotion_mjcf(
            source_xml=args.source_xml,
            model_dir=DEFAULT_MODEL_CACHE,
            urdf_path=DEFAULT_ACTIVE_URDF,
            refresh=args.refresh_model,
        )
    validate_mjcf_against_urdf(xml_path, DEFAULT_ACTIVE_URDF)
    print("MuJoCo locomotion eval validation OK")
    print(f"  source MJCF: {args.source_xml}")
    print(f"  patched MJCF: {xml_path}")
    print(f"  URDF: {DEFAULT_ACTIVE_URDF}")


def summarize(infos: list[dict]) -> dict:
    if not infos:
        return {"steps": 0}
    root_z = [float(info["root_pos_w"][2]) for info in infos]
    joint_speed = []
    torque_abs = []
    for info in infos:
        joint_speed.extend(abs(float(x)) for x in info["joint_vel"])
        torque_abs.extend(abs(float(x)) for x in info["torque"])
    return {
        "steps": len(infos),
        "done_reason": infos[-1].get("done_reason", ""),
        "final_root_z": root_z[-1],
        "min_root_z": min(root_z),
        "mean_abs_joint_velocity": sum(joint_speed) / max(1, len(joint_speed)),
        "max_abs_joint_velocity": max(joint_speed) if joint_speed else 0.0,
        "mean_abs_torque": sum(torque_abs) / max(1, len(torque_abs)),
        "max_abs_torque": max(torque_abs) if torque_abs else 0.0,
    }


def _append(logs: dict[str, list], info: dict) -> None:
    for key in (
        "qpos",
        "qvel",
        "joint_pos",
        "joint_vel",
        "q_des",
        "actions",
        "commands",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "obs_latest",
        "torque",
    ):
        logs.setdefault(key, []).append(info[key])


def run_rollout(args: argparse.Namespace) -> tuple[list[dict], dict[str, list]]:
    torch, policy = load_torchscript_policy(args.policy)
    env = MujocoLocomotionEnv(
        model_xml=args.model_xml,
        seed=args.seed,
        render=args.render,
        refresh_model=args.refresh_model,
        joint_velocity_limit=args.joint_velocity_limit,
    )
    infos: list[dict] = []
    logs: dict[str, list] = {"time_s": []}
    try:
        obs = env.reset(command=command_profile(0, CONTROL_DT, args.command_profile))
        if hasattr(policy, "reset_memory"):
            policy.reset_memory()
        for step in range(int(args.max_steps)):
            start = time.time()
            cmd = command_profile(step, CONTROL_DT, args.command_profile)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            obs_tensor = torch.clamp(obs_tensor, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS)
            with torch.inference_mode():
                action_tensor = policy(obs_tensor)
            if isinstance(action_tensor, (tuple, list)):
                action_tensor = action_tensor[0]
            action = action_tensor.detach().cpu().numpy().reshape(-1)[:ACTION_DIM]
            obs, _, done, info = env.step(action, command=cmd)
            infos.append(info)
            logs["time_s"].append(step * CONTROL_DT)
            _append(logs, info)
            if args.render:
                env.render()
            if args.real_time:
                sleep_for_realtime(start, CONTROL_DT)
            if done:
                break
    finally:
        env.close()
    return infos, logs


def main() -> None:
    args = parse_args()
    try:
        if args.validate_only:
            run_validate_only(args)
            return
        infos, logs = run_rollout(args)
    except PlaceholderMJCFError as exc:
        raise SystemExit(str(exc)) from exc

    summary = summarize(infos)
    print(
        "steps={steps} done={done_reason} root_z={final_root_z:.3f} "
        "max_qd={max_abs_joint_velocity:.3f} max_tau={max_abs_torque:.3f}".format(**summary)
    )
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output is not None:
        np = __import__("numpy")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **{key: np.asarray(value) for key, value in logs.items()})


if __name__ == "__main__":
    main()

