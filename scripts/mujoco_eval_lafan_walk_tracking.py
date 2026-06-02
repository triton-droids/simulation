#!/usr/bin/env python3
"""Evaluate the LAFAN walk-tracking policy in MuJoCo."""

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

from tritonhumanoid.eval.mujoco_lafan import (  # noqa: E402
    ACTION_DIM,
    CLIP_OBSERVATIONS,
    CONTROL_DT,
    DEFAULT_ACTIVE_URDF,
    DEFAULT_MODEL_CACHE,
    OBS_DIM,
    MujocoLafanWalkTrackingEnv,
    ensure_isaac_trained_mjcf,
    sleep_for_realtime,
    validate_mjcf_against_urdf,
)


DEFAULT_POLICY = REPO_ROOT / "logs/rl_games/humanoid_flat_direct/lafan_walk_tracking/exported_policy/ppo_policy.pt"
DEFAULT_MOTION_DIR = Path(
    "/cephfs/holosoma/data/lafan/retargeted/ch_robot_stance_flatfoot_locomotion_full_floor_norm_with_vel"
)
DEFAULT_MOTION_MANIFEST = REPO_ROOT / "manifests/lafan_walk_tracking_manifest.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--motion-dir", type=Path, default=DEFAULT_MOTION_DIR)
    parser.add_argument("--motion-manifest", type=Path, default=DEFAULT_MOTION_MANIFEST)
    parser.add_argument("--model-xml", type=Path, default=None, help="Override generated Isaac-trained MJCF path.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--refresh-model", action="store_true", help="Regenerate the cached Isaac-trained MJCF.")
    return parser.parse_args()


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required to run the exported TorchScript policy. "
            "Run this script with the project/Isaac Python environment."
        ) from exc
    return torch


def validate_policy_metadata(policy_path: Path) -> dict:
    metadata_path = policy_path.with_name("ppo_metadata.pt")
    if not metadata_path.exists():
        return {"metadata_found": False}

    torch = _require_torch()
    metadata = torch.load(metadata_path, map_location="cpu")
    errors = []
    if int(metadata.get("num_observations", -1)) != OBS_DIM:
        errors.append(f"num_observations={metadata.get('num_observations')} expected {OBS_DIM}")
    if int(metadata.get("num_actions", -1)) != ACTION_DIM:
        errors.append(f"num_actions={metadata.get('num_actions')} expected {ACTION_DIM}")
    if bool(metadata.get("normalize_input", True)):
        errors.append("normalize_input must be false for this checkpoint")
    if errors:
        raise RuntimeError("Policy metadata mismatch: " + "; ".join(errors))
    return {"metadata_found": True, "metadata": metadata}


def load_policy(policy_path: Path):
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Exported policy not found: {policy_path}\n"
            "Export it with, for example:\n"
            "  HEADLESS=1 ./run_play_lafan_walk_tracking.sh --video_length 1"
        )
    validate_policy_metadata(policy_path)
    torch = _require_torch()
    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
    return torch, policy


def run_validate_only(args: argparse.Namespace) -> None:
    xml_path = args.model_xml
    if xml_path is None:
        xml_path = ensure_isaac_trained_mjcf(
            model_dir=DEFAULT_MODEL_CACHE,
            urdf_path=DEFAULT_ACTIVE_URDF,
            refresh=args.refresh_model,
        )
    validate_mjcf_against_urdf(xml_path, DEFAULT_ACTIVE_URDF)
    print("MuJoCo LAFAN eval validation OK")
    print(f"  MJCF : {xml_path}")
    print(f"  URDF : {DEFAULT_ACTIVE_URDF}")
    print(f"  cache: {DEFAULT_MODEL_CACHE}")
    if args.policy.exists():
        try:
            result = validate_policy_metadata(args.policy)
        except ModuleNotFoundError:
            print(f"  policy metadata: skipped because torch is unavailable ({args.policy})")
        else:
            print(f"  policy metadata: {'found' if result.get('metadata_found') else 'not found'}")
    else:
        print(f"  policy: missing ({args.policy})")


def summarize_episode(infos: list[dict], episode: int) -> dict:
    if not infos:
        return {"episode": episode, "steps": 0}
    joint_rmse = [float(info["joint_pos_rmse"]) for info in infos]
    height_err = [abs(float(info["root_height_error"])) for info in infos]
    yaw_err = [abs(float(info["root_yaw_error"])) for info in infos]
    return {
        "episode": episode,
        "steps": len(infos),
        "motion_id": int(infos[-1]["motion_id"]),
        "end_frame": int(infos[-1]["frame"]),
        "done_reason": infos[-1].get("done_reason", ""),
        "mean_joint_pos_rmse": sum(joint_rmse) / len(joint_rmse),
        "max_joint_pos_rmse": max(joint_rmse),
        "mean_abs_root_height_error": sum(height_err) / len(height_err),
        "mean_abs_root_yaw_error": sum(yaw_err) / len(yaw_err),
        "final_root_z": float(infos[-1]["root_z"]),
    }


def run_rollout(args: argparse.Namespace) -> list[dict]:
    torch, policy = load_policy(args.policy)
    env = MujocoLafanWalkTrackingEnv(
        motion_dir=args.motion_dir,
        motion_manifest=args.motion_manifest,
        model_xml=args.model_xml,
        seed=args.seed,
        render=args.render,
        refresh_model=args.refresh_model,
    )

    results = []
    try:
        for episode in range(int(args.episodes)):
            obs = env.reset(random_start=True)
            if hasattr(policy, "reset_memory"):
                policy.reset_memory()
            infos = []
            for _ in range(int(args.max_steps)):
                start = time.time()
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                obs_tensor = torch.clamp(obs_tensor, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS)
                with torch.inference_mode():
                    action_tensor = policy(obs_tensor)
                if isinstance(action_tensor, (tuple, list)):
                    action_tensor = action_tensor[0]
                action = action_tensor.detach().cpu().numpy().reshape(-1)[:ACTION_DIM]
                obs, _, done, info = env.step(action)
                infos.append(info)
                if args.render:
                    env.render()
                if args.real_time:
                    sleep_for_realtime(start, CONTROL_DT)
                if done:
                    break
            summary = summarize_episode(infos, episode)
            results.append(summary)
            print(
                "episode={episode} steps={steps} motion={motion_id} done={done_reason} "
                "joint_rmse={mean_joint_pos_rmse:.4f} root_z={final_root_z:.3f}".format(**summary)
            )
    finally:
        env.close()
    return results


def main() -> None:
    args = parse_args()
    if args.validate_only:
        run_validate_only(args)
        return

    results = run_rollout(args)
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
