"""Straight-line metrics in each robot's initial heading; XYZW quaternions."""

import torch


def yaw_xyzw(quat):
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class TrajectoryMetrics:
    """Track only the first episode, excluding reset jumps and terminal frames."""

    def __init__(self, positions, quaternions):
        self.start = positions[:, :2].clone()
        self.last = self.start.clone()
        self.initial_yaw = yaw_xyzw(quaternions).clone()
        self.last_yaw = self.initial_yaw.clone()
        self.heading_change = torch.zeros_like(self.initial_yaw)
        self.max_heading = torch.zeros_like(self.initial_yaw)
        self.path_length = torch.zeros_like(self.initial_yaw)
        self.max_lateral = torch.zeros_like(self.initial_yaw)
        self.alive = torch.ones_like(self.initial_yaw, dtype=torch.bool)
        self.valid_seconds = torch.zeros_like(self.initial_yaw)

    def update(self, positions, quaternions, done, dt):
        self.alive &= ~done.bool()
        yaw = yaw_xyzw(quaternions)
        delta_yaw = torch.atan2(torch.sin(yaw - self.last_yaw), torch.cos(yaw - self.last_yaw))
        delta_pos = positions[:, :2] - self.last
        self.heading_change += torch.where(self.alive, delta_yaw, 0.)
        self.max_heading = torch.maximum(self.max_heading, self.heading_change.abs())
        self.path_length += torch.where(self.alive, delta_pos.norm(dim=-1), 0.)
        self.last = torch.where(self.alive[:, None], positions[:, :2], self.last)
        self.last_yaw = torch.where(self.alive, yaw, self.last_yaw)
        self.valid_seconds += self.alive * dt
        self.max_lateral = torch.maximum(self.max_lateral, self.displacements()[0].abs())

    def displacements(self):
        delta = self.last - self.start
        c, s = self.initial_yaw.cos(), self.initial_yaw.sin()
        lateral = c * delta[:, 0] + s * delta[:, 1]
        forward = -s * delta[:, 0] + c * delta[:, 1]  # Body Y is forward.
        return lateral, forward

    def summary(self):
        lateral, forward = self.displacements()
        return {
            "tracked_point": "hip center of mass",
            "scope": "first episode only; terminal/reset transitions excluded",
            "survived_full_trial": int(self.alive.sum()),
            "num_envs": len(self.alive),
            "mean_valid_seconds": float(self.valid_seconds.mean()),
            "mean_forward_displacement_m": float(forward.mean()),
            "mean_abs_lateral_displacement_m": float(lateral.abs().mean()),
            "mean_max_abs_lateral_displacement_m": float(self.max_lateral.mean()),
            "mean_world_path_length_m": float(self.path_length.mean()),
            "mean_abs_final_heading_drift_deg": float(self.heading_change.abs().mean().rad2deg()),
            "mean_max_abs_heading_drift_deg": float(self.max_heading.mean().rad2deg()),
            "max_abs_heading_drift_deg": float(self.max_heading.max().rad2deg()),
            "per_env_forward_m": forward.tolist(),
            "per_env_lateral_m": lateral.tolist(),
            "per_env_heading_drift_deg": self.heading_change.rad2deg().tolist(),
        }
