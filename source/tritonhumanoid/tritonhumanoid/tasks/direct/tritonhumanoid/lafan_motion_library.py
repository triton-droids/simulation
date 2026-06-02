from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


CH_MUJOCO_JOINT_NAMES = [
    "left_hip1_joint",
    "left_hip2_joint",
    "left_thigh_joint",
    "left_knee_joint",
    "left_ankle_joint",
    "right_hip1_joint",
    "right_hip2_joint",
    "right_thigh_joint",
    "right_knee_joint",
    "right_ankle_joint",
]


class LafanMotionLibrary:
    """Loads enriched CH LAFAN references and serves batched frame tensors."""

    REQUIRED_KEYS = {
        "qpos",
        "qvel",
        "joint_vel",
        "root_lin_vel_w",
        "root_lin_vel_b",
        "root_ang_vel_w",
        "root_ang_vel_b",
        "root_yaw",
        "yaw_rate_ref",
        "fps",
    }

    def __init__(
        self,
        motion_reference_dir: str,
        device: str | torch.device,
        isaac_action_joint_names: list[str],
        motion_fps: int = 30,
        motion_min_length_s: float = 1.0,
        motion_max_cost: float | None = None,
        cache_on_gpu: bool = True,
        motion_manifest_file: str = "",
    ) -> None:
        self.device = torch.device(device)
        self.storage_device = self.device if cache_on_gpu else torch.device("cpu")
        self.motion_fps = int(motion_fps)
        self.motion_min_length_s = float(motion_min_length_s)
        self.motion_max_cost = motion_max_cost
        self.cache_on_gpu = bool(cache_on_gpu)

        self.action_joint_names = [str(name) for name in isaac_action_joint_names]
        missing = [name for name in self.action_joint_names if name not in CH_MUJOCO_JOINT_NAMES]
        if missing:
            raise RuntimeError(
                "Isaac action joints are not covered by the CH MuJoCo joint contract: "
                f"{missing}. Known joints: {CH_MUJOCO_JOINT_NAMES}"
            )
        self.mujoco_to_isaac = [CH_MUJOCO_JOINT_NAMES.index(name) for name in self.action_joint_names]

        motion_paths = self._resolve_motion_paths(motion_reference_dir, motion_manifest_file)
        loaded = [self._load_motion(path) for path in motion_paths]
        loaded = [motion for motion in loaded if motion is not None]
        if not loaded:
            raise RuntimeError(
                "No valid LAFAN motion references were loaded. "
                f"dir={motion_reference_dir!r} manifest={motion_manifest_file!r}"
            )

        self.paths = [motion["path"] for motion in loaded]
        self.costs = [motion["cost"] for motion in loaded]
        self.lengths = torch.as_tensor(
            [motion["length"] for motion in loaded], dtype=torch.long, device=self.device
        )
        self._lengths_storage = self.lengths.to(self.storage_device)
        self.num_motions = len(loaded)
        self.max_length = int(max(motion["length"] for motion in loaded))

        self.root_pos = self._pad_and_stack([motion["root_pos"] for motion in loaded])
        self.root_quat = self._pad_and_stack([motion["root_quat"] for motion in loaded])
        self.joint_pos = self._pad_and_stack([motion["joint_pos"] for motion in loaded])
        self.joint_vel = self._pad_and_stack([motion["joint_vel"] for motion in loaded])
        self.root_lin_vel_w = self._pad_and_stack([motion["root_lin_vel_w"] for motion in loaded])
        self.root_lin_vel_b = self._pad_and_stack([motion["root_lin_vel_b"] for motion in loaded])
        self.root_ang_vel_w = self._pad_and_stack([motion["root_ang_vel_w"] for motion in loaded])
        self.root_ang_vel_b = self._pad_and_stack([motion["root_ang_vel_b"] for motion in loaded])
        self.root_yaw = self._pad_and_stack([motion["root_yaw"] for motion in loaded])
        self.yaw_rate_ref = self._pad_and_stack([motion["yaw_rate_ref"] for motion in loaded])

    def sample(self, num_samples: int, random_start: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_ids = torch.randint(0, self.num_motions, (num_samples,), device=self.device)
        lengths = self.lengths[motion_ids]
        if random_start:
            # Leave at least one frame available after reset so clips do not end immediately.
            high = torch.clamp(lengths - 1, min=1)
            start_frames = torch.floor(torch.rand(num_samples, device=self.device) * high.float()).long()
        else:
            start_frames = torch.zeros(num_samples, dtype=torch.long, device=self.device)
        end_frames = lengths
        return motion_ids, start_frames, end_frames

    def get_frames(self, motion_ids: torch.Tensor, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        frames = self.clamp_frames(motion_ids, frames)
        return {
            "root_pos": self._select(self.root_pos, motion_ids, frames),
            "root_quat": self._select(self.root_quat, motion_ids, frames),
            "joint_pos": self._select(self.joint_pos, motion_ids, frames),
            "joint_vel": self._select(self.joint_vel, motion_ids, frames),
            "root_lin_vel_w": self._select(self.root_lin_vel_w, motion_ids, frames),
            "root_lin_vel_b": self._select(self.root_lin_vel_b, motion_ids, frames),
            "root_ang_vel_w": self._select(self.root_ang_vel_w, motion_ids, frames),
            "root_ang_vel_b": self._select(self.root_ang_vel_b, motion_ids, frames),
            "root_yaw": self._select(self.root_yaw, motion_ids, frames),
            "yaw_rate_ref": self._select(self.yaw_rate_ref, motion_ids, frames),
        }

    def get_future_joint_pos(
        self, motion_ids: torch.Tensor, frames: torch.Tensor, future_offsets: tuple[int, ...]
    ) -> torch.Tensor:
        if len(future_offsets) == 0:
            return torch.empty((motion_ids.shape[0], 0, len(self.action_joint_names)), device=self.device)

        offsets = torch.as_tensor(future_offsets, dtype=torch.long, device=self.device)
        future_frames = frames.unsqueeze(1) + offsets.unsqueeze(0)
        max_frames = self.lengths[motion_ids].unsqueeze(1) - 1
        future_frames = torch.minimum(future_frames, max_frames)

        flat_motion_ids = motion_ids.unsqueeze(1).expand_as(future_frames).reshape(-1)
        flat_frames = future_frames.reshape(-1)
        future = self._select(self.joint_pos, flat_motion_ids, flat_frames)
        return future.reshape(motion_ids.shape[0], len(future_offsets), len(self.action_joint_names))

    def clamp_frames(self, motion_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        return torch.minimum(frames, self.lengths[motion_ids] - 1)

    def _resolve_motion_paths(self, motion_reference_dir: str, motion_manifest_file: str) -> list[Path]:
        if motion_manifest_file:
            manifest_path = Path(motion_manifest_file).expanduser()
            if not manifest_path.exists():
                raise FileNotFoundError(f"LAFAN motion manifest does not exist: {manifest_path}")
            paths: list[Path] = []
            for raw_line in manifest_path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                path = Path(line).expanduser()
                if not path.is_absolute():
                    path = manifest_path.parent / path
                paths.append(path)
            return sorted(paths)

        motion_dir = Path(motion_reference_dir).expanduser()
        if not motion_dir.exists():
            raise FileNotFoundError(f"LAFAN motion directory does not exist: {motion_dir}")
        return sorted(motion_dir.rglob("*.npz"))

    def _load_motion(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            raise FileNotFoundError(f"LAFAN motion file listed but not found: {path}")

        with np.load(str(path), allow_pickle=True) as data:
            missing = sorted(self.REQUIRED_KEYS.difference(data.files))
            if missing:
                raise RuntimeError(f"LAFAN motion {path} is missing required keys: {missing}")

            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if abs(fps - float(self.motion_fps)) > 1e-6:
                raise RuntimeError(f"LAFAN motion {path} has fps={fps:g}; expected {self.motion_fps}.")

            qpos = self._array(data, "qpos", 2)
            qvel = self._array(data, "qvel", 2)
            joint_vel = self._array(data, "joint_vel", 2)
            length = int(qpos.shape[0])

            if qpos.shape != (length, 17):
                raise RuntimeError(f"LAFAN motion {path} expected qpos shape (T, 17), got {qpos.shape}.")
            if qvel.shape != (length, 16):
                raise RuntimeError(f"LAFAN motion {path} expected qvel shape (T, 16), got {qvel.shape}.")
            if joint_vel.shape != (length, 10):
                raise RuntimeError(f"LAFAN motion {path} expected joint_vel shape (T, 10), got {joint_vel.shape}.")

            min_length = max(1, int(np.ceil(self.motion_min_length_s * self.motion_fps)))
            if length < min_length:
                return None

            cost = float(np.asarray(data["cost"]).reshape(-1)[0]) if "cost" in data.files else None
            if self.motion_max_cost is not None and cost is not None and cost > self.motion_max_cost:
                return None

            root_lin_vel_w = self._array(data, "root_lin_vel_w", 2, expected_shape=(length, 3))
            root_lin_vel_b = self._array(data, "root_lin_vel_b", 2, expected_shape=(length, 3))
            root_ang_vel_w = self._array(data, "root_ang_vel_w", 2, expected_shape=(length, 3))
            root_ang_vel_b = self._array(data, "root_ang_vel_b", 2, expected_shape=(length, 3))
            root_yaw = self._array(data, "root_yaw", 1, expected_shape=(length,))
            yaw_rate_ref = self._array(data, "yaw_rate_ref", 1, expected_shape=(length,))

        remap = self.mujoco_to_isaac
        return {
            "path": path,
            "cost": cost,
            "length": length,
            "root_pos": qpos[:, 0:3],
            "root_quat": qpos[:, 3:7],
            "joint_pos": qpos[:, 7:17][:, remap],
            "joint_vel": joint_vel[:, remap],
            "root_lin_vel_w": root_lin_vel_w,
            "root_lin_vel_b": root_lin_vel_b,
            "root_ang_vel_w": root_ang_vel_w,
            "root_ang_vel_b": root_ang_vel_b,
            "root_yaw": root_yaw,
            "yaw_rate_ref": yaw_rate_ref,
        }

    def _array(
        self,
        data: np.lib.npyio.NpzFile,
        key: str,
        ndim: int,
        expected_shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        array = np.asarray(data[key], dtype=np.float32)
        if array.ndim != ndim:
            raise RuntimeError(f"LAFAN field {key!r} expected ndim={ndim}, got shape {array.shape}.")
        if expected_shape is not None and array.shape != expected_shape:
            raise RuntimeError(f"LAFAN field {key!r} expected shape {expected_shape}, got {array.shape}.")
        if not np.isfinite(array).all():
            raise RuntimeError(f"LAFAN field {key!r} contains non-finite values.")
        return array

    def _pad_and_stack(self, arrays: list[np.ndarray]) -> torch.Tensor:
        padded = []
        for array in arrays:
            pad_count = self.max_length - array.shape[0]
            if pad_count > 0:
                pad = np.repeat(array[-1:], pad_count, axis=0)
                array = np.concatenate([array, pad], axis=0)
            padded.append(array)
        return torch.as_tensor(np.stack(padded, axis=0), dtype=torch.float32, device=self.storage_device)

    def _select(self, tensor: torch.Tensor, motion_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        motion_ids_storage = motion_ids.to(self.storage_device)
        frames_storage = frames.to(self.storage_device)
        frames_storage = torch.minimum(frames_storage, self._lengths_storage[motion_ids_storage] - 1)
        selected = tensor[motion_ids_storage, frames_storage]
        if selected.device != self.device:
            selected = selected.to(self.device)
        return selected
