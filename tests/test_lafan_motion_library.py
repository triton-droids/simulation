from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("numpy") is None,
    reason="torch or numpy is not installed",
)

if importlib.util.find_spec("numpy") is not None:
    import numpy as np

if importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None:
    import torch

    from tritonhumanoid.tasks.direct.tritonhumanoid.lafan_motion_library import LafanMotionLibrary


ISAAC_ACTION_JOINT_NAMES = [
    "left_hip1_joint",
    "right_hip1_joint",
    "left_hip2_joint",
    "right_hip2_joint",
    "left_thigh_joint",
    "right_thigh_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_joint",
    "right_ankle_joint",
]


def write_motion(path, frames: int = 35, fps: int = 30, cost: float | None = None) -> None:
    t = np.arange(frames, dtype=np.float32)
    qpos = np.zeros((frames, 17), dtype=np.float32)
    qpos[:, 0] = t / float(fps)
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    qpos[:, 7:17] = t[:, None] + np.arange(10, dtype=np.float32)[None, :]

    qvel = np.zeros((frames, 16), dtype=np.float32)
    qvel[:, 0] = 1.0
    joint_vel = np.ones((frames, 10), dtype=np.float32)
    root_lin_vel_w = np.zeros((frames, 3), dtype=np.float32)
    root_lin_vel_w[:, 0] = 1.0
    root_lin_vel_b = root_lin_vel_w.copy()
    root_ang_vel_w = np.zeros((frames, 3), dtype=np.float32)
    root_ang_vel_b = np.zeros((frames, 3), dtype=np.float32)
    root_yaw = np.zeros((frames,), dtype=np.float32)
    yaw_rate_ref = np.zeros((frames,), dtype=np.float32)

    arrays = {
        "qpos": qpos,
        "qvel": qvel,
        "joint_vel": joint_vel,
        "root_lin_vel_w": root_lin_vel_w,
        "root_lin_vel_b": root_lin_vel_b,
        "root_ang_vel_w": root_ang_vel_w,
        "root_ang_vel_b": root_ang_vel_b,
        "root_yaw": root_yaw,
        "yaw_rate_ref": yaw_rate_ref,
        "fps": np.array(fps, dtype=np.int32),
    }
    if cost is not None:
        arrays["cost"] = np.array(cost, dtype=np.float32)
    np.savez(path, **arrays)


def test_loads_and_remaps_joint_order(tmp_path):
    write_motion(tmp_path / "motion.npz")

    lib = LafanMotionLibrary(str(tmp_path), "cpu", ISAAC_ACTION_JOINT_NAMES)
    frames = lib.get_frames(torch.tensor([0]), torch.tensor([0]))

    assert lib.num_motions == 1
    assert lib.mujoco_to_isaac == [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    assert torch.allclose(
        frames["joint_pos"][0],
        torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0]),
    )


def test_filters_short_and_high_cost_motions(tmp_path):
    write_motion(tmp_path / "short.npz", frames=10)
    write_motion(tmp_path / "expensive.npz", frames=35, cost=10.0)
    write_motion(tmp_path / "kept.npz", frames=35, cost=0.5)

    lib = LafanMotionLibrary(
        str(tmp_path),
        "cpu",
        ISAAC_ACTION_JOINT_NAMES,
        motion_min_length_s=1.0,
        motion_max_cost=1.0,
    )

    assert lib.num_motions == 1
    assert lib.paths[0].name == "kept.npz"


def test_manifest_relative_paths(tmp_path):
    motion_dir = tmp_path / "motions"
    motion_dir.mkdir()
    write_motion(motion_dir / "motion.npz")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("# comments are ignored\nmotions/motion.npz\n")

    lib = LafanMotionLibrary(str(tmp_path / "unused"), "cpu", ISAAC_ACTION_JOINT_NAMES, motion_manifest_file=str(manifest))

    assert lib.num_motions == 1
    assert lib.paths[0].name == "motion.npz"


def test_sampling_uses_legal_start_frames(tmp_path):
    write_motion(tmp_path / "motion.npz", frames=35)
    lib = LafanMotionLibrary(str(tmp_path), "cpu", ISAAC_ACTION_JOINT_NAMES)

    _, starts, ends = lib.sample(128, random_start=True)

    assert torch.all(starts >= 0)
    assert torch.all(starts < ends - 1)


def test_future_frames_clamp_without_wrapping(tmp_path):
    write_motion(tmp_path / "motion.npz", frames=35)
    lib = LafanMotionLibrary(str(tmp_path), "cpu", ISAAC_ACTION_JOINT_NAMES)

    future = lib.get_future_joint_pos(
        torch.tensor([0]),
        torch.tensor([33]),
        future_offsets=(1, 2, 4, 6),
    )

    assert future.shape == (1, 4, 10)
    assert torch.allclose(future[0, 0], future[0, 1])
    assert torch.allclose(future[0, 1], future[0, 2])
    assert torch.allclose(future[0, 2], future[0, 3])


def test_missing_required_key_fails(tmp_path):
    write_motion(tmp_path / "bad.npz")
    with np.load(tmp_path / "bad.npz") as data:
        arrays = {key: data[key] for key in data.files if key != "yaw_rate_ref"}
    np.savez(tmp_path / "bad.npz", **arrays)

    with pytest.raises(RuntimeError, match="missing required keys"):
        LafanMotionLibrary(str(tmp_path), "cpu", ISAAC_ACTION_JOINT_NAMES)
