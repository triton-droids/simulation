"""MJX simulation, observation, reward, noise, and randomization defaults."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MJXConfig:
    """Configuration class for the MJX environment."""

    @dataclass
    class SimConfig:
        """MuJoCo/MJX solver and timestep settings."""

        timestep: float = 0.002
        solver: int = 2
        iterations: int = 3 
        ls_iterations: int = 5

    @dataclass
    class ObsConfig:
        """Observation sizes and optional frame-stacking controls."""

        stack_obs: bool = False
        frame_stack: int = 15
        c_frame_stack: int = 15
        num_single_obs: int = 52
        # 52 policy values plus 60 clean/privileged values. Keep this explicit:
        # the default task is intentionally still a 12-actuator regression target.
        num_single_privileged_obs: int = 112

    @dataclass
    class ActionConfig:
        """Action scaling and frame-repeat settings for control inputs."""

        action_scale: float = 0.5
        n_frames: int = 5

    @dataclass
    class RewardConfig:
        """Shared scalar targets used by reward functions."""

        healthy_z_range: Tuple[float, float] = (0.7, 1.3)
        tracking_sigma: float = 0.5
        max_foot_height: float = 0.2
        base_height_target: float = 0.9

    @dataclass
    class RewardScales:
        """Per-term reward weights applied during training."""

        #tracking related rewards
        lin_vel: float = 1.5
        ang_vel: float = 1.0
        #base related rewards
        lin_vel_z: float = 0.0
        ang_vel_xy: float = -0.05
        base_height: float = 0.0
        orientation: float = -2.0
        #energy related rewards
        torques: float = -2.5e-5
        action_rate: float = -0.01
        energy: float = 0.0
        #feet related rewards
        feet_slip: float = -0.25
        feet_clearance: float = 0.0
        feet_height: float = 0.0
        feet_phase: float = 3.0
        feet_air_time: float = 2.0
        #pose related rewards
        joint_deviation_hip: float = -0.1
        joint_deviation_knee: float = -0.25
        pose: float = -1.0
        #other rewards
        stand_still: float = 0.0
        survival: float = -1.0
        
        
    @dataclass
    class CommandsConfig:
        """Command resampling and episode reset timing."""

        resample_time: float = 10.0 #no resampling by default
        reset_time: float = 100.0  # No resetting by default

    @dataclass
    class DomainRandConfig:
        """Domain randomization ranges for MJX training."""

        add_domain_rand: bool = True
        lin_vel_x: Tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: Tuple[float, float] = (-1.0, 1.0)
        ang_vel_yaw: Tuple[float, float] = (-1.0, 1.0)
        friction_range: Tuple[float] = (0.5, 2.0)
        frictionloss_range: Tuple[float] = (0.9, 1.1)
        armature_range: Tuple[float] = (1.0, 1.05)
        body_mass_range: Tuple[float] = (0.9, 1.1)
        torso_mass_range: Tuple[float] = (-1.0, 1.0)
        qpos0_range: Tuple[float] = (-0.05, 0.05)

    @dataclass
    class NoiseConfig:
        """Sensor and state noise settings for robustness training."""

        add_noise: bool = True
        level: float = 1.0
        hip_pos: float = 0.03
        kfe_pos: float = 0.05
        ffe_pos: float = 0.08
        faa_pos: float = 0.03
        lin_vel: float = 0.1
        gyro: float = 0.2
        gravity: float = 0.05
        joint_vel: float = 1.5

    @dataclass
    class PushConfig:
        """Random external push settings used during training."""

        add_push: bool = True
        interval_range: Tuple[float, float] = (5.0, 10.0)
        magnitude_range: Tuple[float, float] = (0.1, 2.0)
       

    sim: SimConfig = field(default_factory=SimConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    reward_scales: RewardScales = field(default_factory=RewardScales)
    commands: CommandsConfig = field(default_factory=CommandsConfig)
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    push: PushConfig = field(default_factory=PushConfig)
