"""Unitree G1-specific MJX configuration for the velocity prototype."""

from dataclasses import dataclass, field


@dataclass
class G1MJXConfig:
    """Configuration kept separate from the 12-actuator regression task."""

    @dataclass
    class SimConfig:
        timestep: float = 0.002
        solver: int = 2
        iterations: int = 3
        ls_iterations: int = 5

    @dataclass
    class ObsConfig:
        stack_obs: bool = False
        frame_stack: int = 1
        c_frame_stack: int = 1
        num_single_obs: int = 103
        num_single_privileged_obs: int = 216

    @dataclass
    class ActionConfig:
        action_scale: float = 0.5
        n_frames: int = 10

    @dataclass
    class RewardConfig:
        tracking_sigma: float = 0.25
        soft_joint_pos_limit_factor: float = 0.95
        max_foot_height: float = 0.15
        max_contact_force: float = 500.0

    @dataclass
    class RewardScales:
        # Tracking is deliberately dominant.  A previous exploratory profile
        # used alive=3.0 and learned an unstable survival maneuver that ignored
        # yaw commands; these scales retain a smaller survival bridge while
        # keeping the required regularizers and Playground reward family.
        tracking_lin_vel: float = 2.0
        tracking_ang_vel: float = 1.5
        lin_vel_z: float = -0.01
        ang_vel_xy: float = -0.15
        orientation: float = -2.0
        torques: float = -1.0e-5
        energy: float = -1.0e-5
        action_rate: float = -0.005
        dof_acc: float = -2.5e-8
        dof_pos_limits: float = -1.0
        feet_slip: float = -0.25
        collision: float = -0.1
        termination: float = -100.0
        alive: float = 0.5
        feet_air_time: float = 2.0
        feet_phase: float = 1.0
        stand_still: float = -1.0
        pose: float = -0.1

    @dataclass
    class CommandsConfig:
        resample_time: float = 10.0
        # A conservative subset of Playground's supported G1 ranges, matched
        # to the fixed Gate 4 command suite.
        lin_vel_x: tuple[float, float] = (-0.5, 0.5)
        lin_vel_y: tuple[float, float] = (-0.3, 0.3)
        ang_vel_yaw: tuple[float, float] = (-0.5, 0.5)
        zero_probability: float = 0.1

    @dataclass
    class ResetConfig:
        randomize: bool = True
        xy_range: float = 0.5
        yaw_range: float = 3.14
        joint_scale_range: tuple[float, float] = (0.5, 1.5)
        base_velocity_range: float = 0.5

    @dataclass
    class NoiseConfig:
        add_noise: bool = True
        level: float = 1.0
        joint_pos: float = 0.03
        joint_vel: float = 1.5
        gravity: float = 0.05
        lin_vel: float = 0.1
        gyro: float = 0.2

    @dataclass
    class PushConfig:
        add_push: bool = False
        interval_range: tuple[float, float] = (5.0, 10.0)
        magnitude_range: tuple[float, float] = (0.1, 2.0)

    @dataclass
    class DomainRandConfig:
        add_domain_rand: bool = False
        friction_range: tuple[float, float] = (0.4, 1.0)
        frictionloss_range: tuple[float, float] = (0.5, 2.0)
        armature_range: tuple[float, float] = (1.0, 1.05)
        body_mass_range: tuple[float, float] = (0.9, 1.1)
        torso_mass_range: tuple[float, float] = (-1.0, 1.0)
        qpos0_range: tuple[float, float] = (-0.05, 0.05)

    @dataclass
    class TerminationConfig:
        # The knees-bent keyframe starts at 0.755 m. A 0.45 m cutoff ended
        # nominal rollouts while the torso was still near-upright; 0.25 m
        # catches a collapsed body without excluding a future safe squat.
        min_pelvis_height: float = 0.25
        max_pelvis_height: float = 1.05
        min_torso_up_z: float = 0.0

    sim: SimConfig = field(default_factory=SimConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    reward_scales: RewardScales = field(default_factory=RewardScales)
    commands: CommandsConfig = field(default_factory=CommandsConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    push: PushConfig = field(default_factory=PushConfig)
    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
