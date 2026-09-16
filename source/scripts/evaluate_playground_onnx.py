# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright 2026 Triton Droids.
# Adapted from MuJoCo Playground's play_g1_joystick.py at commit
# 8a4b4642d8eba8a80ac99ed125cb62c16e1457ad.

"""Evaluate MuJoCo Playground's exact shipped G1 ONNX policy locally.

This is a deterministic behavioral oracle, not a locally trained policy and
not Gate 4 evidence.  It verifies that the pinned model and 103-to-29
observation/action interface can support stable locomotion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "win32" else "egl")

import mediapy as media
import mujoco
import numpy as np
from omegaconf import OmegaConf

from source.config.g1_playground import G1PlaygroundMJXConfig
from source.locomotion.unitree_g1.playground_joystick import Joystick
from source.robots.unitree_g1 import UnitreeG1Model


EXPECTED_ONNX_SHA256 = (
    "db2eb258494c1297c43d2b9ffa94cdbde97654c2a44cbab0b40fd4b990752a5b"
)
COMMANDS = (
    ("stand", (0.0, 0.0, 0.0)),
    ("forward", (0.5, 0.0, 0.0)),
    ("backward", (-0.3, 0.0, 0.0)),
    ("left", (0.0, 0.3, 0.0)),
    ("right", (0.0, -0.3, 0.0)),
    ("turn_left", (0.0, 0.0, 0.5)),
    ("turn_right", (0.0, 0.0, -0.5)),
    ("combined", (0.4, 0.2, 0.35)),
)
ORIGINAL_ARGV = tuple(sys.argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_record() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-c", "core.autocrlf=true", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"available": True, "commit": commit, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "commit": None, "branch": None, "dirty": None}


def _default_policy_path() -> Path:
    return (
        PROJECT_ROOT
        / ".cache"
        / "mujoco_playground"
        / "mujoco_playground"
        / "experimental"
        / "sim2sim"
        / "onnx"
        / "g1_policy.onnx"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--policy",
        type=Path,
        default=_default_policy_path(),
        help="Exact Playground-shipped g1_policy.onnx path.",
    )
    return parser.parse_args()


def _observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    default_pose: np.ndarray,
    last_action: np.ndarray,
    phase: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    linvel = data.sensor("local_linvel_pelvis").data.copy()
    gyro = data.sensor("gyro_pelvis").data.copy()
    imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0.0, 0.0, -1.0])
    value = np.hstack(
        [
            linvel,
            gyro,
            gravity,
            command,
            data.qpos[7:] - default_pose,
            data.qvel[6:],
            last_action,
            np.concatenate([np.cos(phase), np.sin(phase)]),
        ]
    ).astype(np.float32)
    if value.shape != (103,):
        raise RuntimeError(f"Unexpected oracle observation shape: {value.shape}")
    return value


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> str:
    if not media.video_is_available():
        try:
            import imageio_ffmpeg

            media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        except (ImportError, RuntimeError):
            pass
    media.write_video(path, frames, fps=fps)
    return "mediapy"


def _is_terminal(
    torso_up_z: float,
    illegal_contact: bool,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> bool:
    """Match Playground's upright sign and reject illegal/nonfinite states."""

    return bool(
        torso_up_z < 0.0
        or illegal_contact
        or not (np.isfinite(qpos).all() and np.isfinite(qvel).all())
    )


def main() -> None:
    args = _parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "The optional oracle requires onnxruntime==1.22.1. "
            "Install it in the local WSL environment before running this script."
        ) from error

    policy_path = args.policy.resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(
            f"Missing shipped policy: {policy_path}. Expand the pinned Playground "
            "sparse checkout as documented in ASTRA_HANDOFF.md."
        )
    policy_sha256 = _sha256(policy_path)
    if policy_sha256 != EXPECTED_ONNX_SHA256:
        raise RuntimeError(
            f"Refusing unverified ONNX policy {policy_path}: {policy_sha256}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    started_utc = _utc_now()

    cfg = OmegaConf.structured(G1PlaygroundMJXConfig())
    cfg.playground.fetch_source = False
    cfg.reset.randomize = False
    cfg.noise.add_noise = False
    cfg.push.add_push = False
    env = Joystick("unitree_g1", UnitreeG1Model(fetch=False), "flat", cfg)
    model = env.mj_model
    default_pose = np.asarray(model.keyframe("knees_bent").qpos[7:]).copy()
    keyframe_id = model.key("knees_bent").id
    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_signature = [(item.name, item.shape) for item in session.get_inputs()]
    output_signature = [(item.name, item.shape) for item in session.get_outputs()]
    if input_signature != [("obs", [1, 103])]:
        raise RuntimeError(f"Unexpected ONNX input signature: {input_signature}")
    if output_signature != [("continuous_actions", [1, 29])]:
        raise RuntimeError(f"Unexpected ONNX output signature: {output_signature}")

    foot_sensor_addresses = [
        model.sensor_adr[sensor_id] for sensor_id in env._env._feet_floor_found_sensor
    ]
    illegal_sensor_addresses = [
        model.sensor_adr[sensor_id]
        for sensor_id in (
            env._env._right_foot_left_foot_found_sensor,
            env._env._left_foot_right_shin_found_sensor,
            env._env._right_foot_left_shin_found_sensor,
        )
    ]

    rows: list[dict[str, Any]] = []
    videos: list[str] = []
    for name, command_values in COMMANDS:
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        data.ctrl[:] = default_pose
        mujoco.mj_forward(model, data)
        command = np.asarray(command_values, dtype=np.float32)
        phase = np.array([0.0, np.pi])
        phase_dt = 2 * np.pi * 1.5 * 0.02
        last_action = np.zeros(29, dtype=np.float32)
        velocities: list[np.ndarray] = []
        gyros: list[np.ndarray] = []
        heights: list[float] = []
        contacts: list[list[bool]] = []
        actions: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        renderer = (
            mujoco.Renderer(model, height=480, width=640)
            if args.video and name == "combined"
            else None
        )
        terminal = False

        for step in range(args.steps):
            obs = _observation(model, data, default_pose, last_action, phase, command)
            action = session.run(["continuous_actions"], {"obs": obs[None]})[0][0]
            data.ctrl[:] = action * 0.5 + default_pose
            for _ in range(10):
                mujoco.mj_step(model, data)
            last_action = action.copy()
            phase = np.fmod(phase + phase_dt + np.pi, 2 * np.pi) - np.pi

            velocities.append(data.sensor("local_linvel_pelvis").data.copy())
            gyros.append(data.sensor("gyro_pelvis").data.copy())
            heights.append(float(data.qpos[2]))
            contacts.append(
                [data.sensordata[address] > 0 for address in foot_sensor_addresses]
            )
            actions.append(action.copy())
            if renderer is not None and step % 2 == 0:
                renderer.update_scene(data, camera="track")
                frames.append(renderer.render())

            torso_up_z = float(data.sensor("upvector_torso").data[2])
            illegal = any(
                data.sensordata[address] > 0 for address in illegal_sensor_addresses
            )
            terminal = _is_terminal(torso_up_z, illegal, data.qpos, data.qvel)
            if terminal:
                break

        if renderer is not None:
            renderer.close()
            video_path = output_dir / "onnx_combined.mp4"
            _write_video(video_path, frames, fps=25)
            videos.append(video_path.name)

        velocity = np.asarray(velocities)
        gyro = np.asarray(gyros)
        contact = np.asarray(contacts, dtype=bool)
        action_values = np.asarray(actions)
        transitions = (
            np.sum(contact[1:] != contact[:-1], axis=0)
            if len(contact) > 1
            else np.zeros(2)
        )
        row = {
            "command_name": name,
            "command": command_values,
            "completed_steps": len(velocity),
            "duration_seconds": len(velocity) * 0.02,
            "terminal": bool(terminal),
            "linear_velocity_mean": np.mean(velocity[:, :2], axis=0).tolist(),
            "linear_velocity_vector_rmse": float(
                np.sqrt(
                    np.mean(np.sum(np.square(velocity[:, :2] - command[:2]), axis=1))
                )
            ),
            "yaw_rate_mean": float(np.mean(gyro[:, 2])),
            "yaw_rate_rmse": float(
                np.sqrt(np.mean(np.square(gyro[:, 2] - command[2])))
            ),
            "minimum_pelvis_height": float(np.min(heights)),
            "mean_pelvis_height": float(np.mean(heights)),
            "left_contact_duty": float(np.mean(contact[:, 0])),
            "right_contact_duty": float(np.mean(contact[:, 1])),
            "single_support_fraction": float(
                np.mean(np.logical_xor(contact[:, 0], contact[:, 1]))
            ),
            "double_support_fraction": float(
                np.mean(np.logical_and(contact[:, 0], contact[:, 1]))
            ),
            "flight_fraction": float(
                np.mean(~np.logical_or(contact[:, 0], contact[:, 1]))
            ),
            "contact_transitions": transitions.astype(int).tolist(),
            "maximum_abs_action": float(np.max(np.abs(action_values))),
            "finite": bool(
                np.isfinite(velocity).all()
                and np.isfinite(gyro).all()
                and np.isfinite(action_values).all()
            ),
        }
        rows.append(row)
        print(name, row)

    result = {
        "kind": "playground_shipped_g1_onnx_behavioral_oracle",
        "command": list(ORIGINAL_ARGV),
        "started_at_utc": started_utc,
        "ended_at_utc": _utc_now(),
        "wall_time_seconds": time.perf_counter() - started,
        "git": _git_record(),
        "playground_revision": env.playground_source.revision,
        "menagerie_revision": env.model_source.revision,
        "onnx_policy": str(policy_path),
        "onnx_sha256": policy_sha256,
        "onnxruntime": ort.__version__,
        "mujoco": mujoco.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_signature": input_signature,
        "output_signature": output_signature,
        "steps_requested": args.steps,
        "control_dt_seconds": 0.02,
        "rows": rows,
        "videos": videos,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
