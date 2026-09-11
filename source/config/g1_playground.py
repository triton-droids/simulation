"""Pinned MuJoCo Playground configuration for the G1 baseline adapter."""

from dataclasses import dataclass, field
from typing import Optional

from source.config.g1 import G1MJXConfig


@dataclass
class G1PlaygroundMJXConfig(G1MJXConfig):
    """Expose the authoritative G1 task without changing the native profile."""

    @dataclass
    class RewardScales:
        tracking_lin_vel: float = 1.0
        tracking_ang_vel: float = 0.75
        lin_vel_z: float = 0.0
        ang_vel_xy: float = -0.15
        orientation: float = -2.0
        base_height: float = 0.0
        torques: float = 0.0
        action_rate: float = 0.0
        energy: float = 0.0
        dof_acc: float = 0.0
        feet_clearance: float = 0.0
        feet_air_time: float = 2.0
        feet_slip: float = -0.25
        feet_height: float = 0.0
        feet_phase: float = 1.0
        alive: float = 0.0
        stand_still: float = -1.0
        termination: float = -100.0
        collision: float = -0.1
        contact_force: float = -0.01
        joint_deviation_knee: float = -0.1
        joint_deviation_hip: float = -0.25
        dof_pos_limits: float = -1.0
        pose: float = -0.1

    @dataclass
    class CommandsConfig(G1MJXConfig.CommandsConfig):
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-0.5, 0.5)
        ang_vel_yaw: tuple[float, float] = (-1.0, 1.0)

    @dataclass
    class PushConfig(G1MJXConfig.PushConfig):
        add_push: bool = True

    @dataclass
    class PlaygroundSourceConfig:
        source_root: Optional[str] = None
        cache_root: Optional[str] = None
        fetch_source: bool = True
        implementation: str = "jax"

    reward_scales: RewardScales = field(default_factory=RewardScales)
    commands: CommandsConfig = field(default_factory=CommandsConfig)
    push: PushConfig = field(default_factory=PushConfig)
    playground: PlaygroundSourceConfig = field(default_factory=PlaygroundSourceConfig)
