#!/usr/bin/env python3
"""Kinematic MuJoCo playback for Isaac locomotion traces."""

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
    CONTROL_DT,
    DEFAULT_ACTIVE_URDF,
    DEFAULT_MODEL_CACHE,
    DEFAULT_SOURCE_MJCF,
    MujocoLocomotionEnv,
    PlaceholderMJCFError,
    ensure_isaac_locomotion_mjcf,
    sleep_for_realtime,
    validate_mjcf_against_urdf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Isaac sim2sim .npz trace from parity_log.py.")
    parser.add_argument("--model-xml", type=Path, default=None)
    parser.add_argument("--source-xml", type=Path, default=DEFAULT_SOURCE_MJCF)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means play the whole trace.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--refresh-model", action="store_true")
    parser.add_argument("--command-profile", choices=["none", "stand_forward_yaw", "forward"], default="stand_forward_yaw")
    parser.add_argument("--enable-contact", action="store_true", help="Keep MuJoCo contact detection enabled during kinematic playback.")
    parser.add_argument("--enable-gravity", action="store_true", help="Keep MuJoCo gravity enabled during kinematic playback.")
    parser.add_argument("--hide-floor", action="store_true", help="Hide floor/ground geoms during kinematic playback.")
    parser.add_argument("--output", type=Path, default=None, help="Optional copied playback .npz path.")
    return parser.parse_args()


def validate_trace(trace) -> None:
    for key in ("qpos", "qvel"):
        if key not in trace:
            raise RuntimeError(f"Trace is missing required key {key!r}. Re-record with --sim2sim-log.")
    if trace["qpos"].ndim != 2 or trace["qvel"].ndim != 2:
        raise RuntimeError("Trace qpos/qvel must be rank-2 arrays.")


def load_trace_joint_order(trace_path: Path) -> tuple[str, ...] | None:
    metadata_path = trace_path.with_suffix(".json")
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    joint_order = metadata.get("joint_order")
    if joint_order is None:
        return None
    return tuple(str(name) for name in joint_order)


def run_validate_only(args: argparse.Namespace) -> None:
    np = __import__("numpy")
    trace = np.load(args.trace)
    validate_trace(trace)
    xml_path = args.model_xml
    if xml_path is None:
        xml_path = ensure_isaac_locomotion_mjcf(
            source_xml=args.source_xml,
            model_dir=DEFAULT_MODEL_CACHE,
            urdf_path=DEFAULT_ACTIVE_URDF,
            refresh=args.refresh_model,
        )
    validate_mjcf_against_urdf(xml_path, DEFAULT_ACTIVE_URDF)
    print("MuJoCo locomotion playback validation OK")
    print(f"  trace: {args.trace}")
    print(f"  patched MJCF: {xml_path}")
    print(f"  frames: {trace['qpos'].shape[0]}")
    joint_order = load_trace_joint_order(args.trace)
    if joint_order is not None:
        print(f"  trace joint order: {list(joint_order)}")


def main() -> None:
    args = parse_args()
    try:
        if args.validate_only:
            run_validate_only(args)
            return
        np = __import__("numpy")
        trace = np.load(args.trace)
        validate_trace(trace)
        trace_joint_order = load_trace_joint_order(args.trace)
        model_xml = args.model_xml
        if model_xml is None:
            model_xml = ensure_isaac_locomotion_mjcf(
                source_xml=args.source_xml,
                model_dir=DEFAULT_MODEL_CACHE,
                urdf_path=DEFAULT_ACTIVE_URDF,
                refresh=args.refresh_model,
            )
        env = MujocoLocomotionEnv(
            model_xml=model_xml,
            seed=args.seed,
            render=args.render,
        )
        if not args.enable_contact:
            env.disable_contact()
        if not args.enable_gravity:
            env.disable_gravity()
        if args.hide_floor:
            env.hide_geoms(("floor", "ground"))
        frames = int(trace["qpos"].shape[0])
        if args.max_steps > 0:
            frames = min(frames, int(args.max_steps))
        logs = {"time_s": [], "qpos": [], "qvel": []}
        try:
            env.reset()
            for step in range(frames):
                start = time.time()
                env.set_state_from_trace(trace, step, joint_names=trace_joint_order)
                logs["time_s"].append(step * CONTROL_DT)
                logs["qpos"].append(env.data.qpos.copy())
                logs["qvel"].append(env.data.qvel.copy())
                if args.render:
                    env.render()
                if args.real_time:
                    sleep_for_realtime(start, CONTROL_DT)
        finally:
            env.close()
    except PlaceholderMJCFError as exc:
        raise SystemExit(str(exc)) from exc

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **{key: np.asarray(value) for key, value in logs.items()})
    print(f"played_frames={frames}")


if __name__ == "__main__":
    main()
