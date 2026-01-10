# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg

from ....assets.humanoid import HUMANOID_CFG


@configclass
class EventCfg:
    """Domain randomization terms (EventManager-based)."""

    # --- robot material ---
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    # --- actuator PD gains (implicit actuators) ---
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "distribution": "uniform",
        },
    )

    # --- joint physics params (friction/armature/limits/etc.) ---
    robot_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            # start as no-op, ADR widens
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (1.0, 1.0),
            # optional: avoid changing limits unless you really want to
            # "lower_limit_distribution_params": (1.0, 1.0),
            # "upper_limit_distribution_params": (1.0, 1.0),
            "distribution": "uniform",
        },
    )

    # --- gravity scale ---
    gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "gravity_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    """Standing disturbance-rejection env with ADR."""

    # === RL timing ===
    episode_length_s = 15.0
    decimation = 2

    # === actions / obs ===
    action_scale = 5.0
    action_space = 10

    # position control parameters
    residual_pos_scale: float = 0.25  # scale actions to position deltas (radians)
    action_filter_alpha: float = 0.2  # first-order filter for motor dynamics
    action_rate_scale: float = 0.05   # penalty for action changes

    # observation parameters (hardware-only sensors)
    num_dofs: int = 10
    angular_velocity_scale: float = 0.25
    dof_vel_scale: float = 0.1
    torque_scale: float = 0.01  # scale for joint torques in observation
    
    # observation stacking for memory (set >1 to enable)
    obs_stack_frames: int = 3  # stack last 3 observations
    
    # single-frame observation dimension (hardware-only)
    observation_space_single = (
        3            # gravity direction in body frame (IMU)
        + 3          # gyro (IMU angular velocity)
        + num_dofs   # joint pos (scaled)
        + num_dofs   # joint vel
        + num_dofs   # joint torque
        + action_space  # previous actions
    )
    
    # total observation space (with stacking)
    observation_space = observation_space_single * obs_stack_frames
    state_space = 0

    # === sim ===
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # === scene ===
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # === robot + contacts ===
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # === events ===
    events: EventCfg = EventCfg()

    # === pushes (interpreted as delta-velocity kicks in XY) ===
    push_force_range = (0.0, 0.0)   # start with none if doing ADR curriculum
    min_push_interval_s = 1.0
    max_push_interval_s = 3.0

    # === reward scales ===
    up_weight: float = 2.0
    target_root_height: float = 1.0
    base_height_scale: float = 5.0
    base_xy_scale: float = 1.0
    lin_vel_l2_scale: float = 0.1
    ang_vel_l2_scale: float = 0.1

    step_width_scale: float = 0.5
    max_stride_length: float = 0.30
    stride_penalty_scale: float = 1.0
    hip_posture_scale: float = 0.5

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    dof_vel_scale: float = 0.1
    angular_velocity_scale: float = 0.25

    alive_reward_scale: float = 0.1
    death_cost: float = -2.0
    termination_height: float = 0.6
    termination_up_proj: float = 0.5
    max_xy_displacement: float = 0.6

    # =========================
    # Adaptive Domain Randomization
    # =========================
    enable_adr: bool = True

    # ADR “difficulty” = num_increments / num_adr_increments
    starting_adr_increments: int = 0
    num_adr_increments: int = 20

    # criteria & schedule
    adr_success_rate_to_increase: float = 0.70   # EMA survival-rate threshold
    adr_success_rate_to_decrease: float = 0.25   # if struggling badly
    adr_ema_factor: float = 0.10                 # EMA update factor
    adr_update_interval_steps: int = 10_000      # policy steps between ADR updates
    adr_min_steps_before_decrease: int = 10_000  # cooldown before decreasing too
    adr_print_every_update: bool = True

    # --- ADR ranges for EventManager terms (max difficulty endpoints) ---
    # The “min” endpoints come from your EventCfg above.
    adr_event_cfg_dict = {
        "robot_physics_material": {
            "static_friction_range": (0.4, 1.4),
            "dynamic_friction_range": (0.3, 1.3),
            "restitution_range": (0.0, 0.2),
        },
        "robot_joint_stiffness_and_damping": {
            # scale factors applied to existing actuator gains
            "stiffness_distribution_params": (0.6, 1.6),
            "damping_distribution_params": (0.6, 1.6),
        },
        "robot_joint_parameters": {
            "friction_distribution_params": (0.5, 1.8),
            "armature_distribution_params": (0.7, 1.5),
        },
        "gravity": {
            "gravity_distribution_params": (0.9, 1.1),
        },
    }

    # --- ADR ranges for custom (non-event) randomizations ---
    adr_custom_cfg_dict = {
        "push": {
            "push_force_range": ((0.0, 0.0), (0.2, 1.2)),  # (min_range, max_range)
        },
        "motor_strength": {
            "per_joint_mult_range": (0.0, 0.25),  # at max: mult ~ U[1-r, 1+r]
        },
        "action_noise": {
            "std": (0.0, 0.12),
        },
        "latency": {
            "act_steps": (0, 3),
            "obs_steps": (0, 2),
        },
        "obs_noise": {
            "gravity_std": (0.0, 0.02),      # IMU gravity direction noise
            "gyro_std": (0.0, 0.08),         # IMU gyroscope noise
            "joint_pos_std": (0.0, 0.02),
            "joint_vel_std": (0.0, 0.10),
            "joint_torque_std": (0.0, 0.5),  # torque sensor noise
        },
        "imu_bias": {
            "gravity_bias_range": (0.0, 0.03),  # constant bias on gravity direction
            "gyro_bias_range": (0.0, 0.05),     # constant bias on gyroscope
        },
    }

    # allocate buffers using max latency possible
    act_max_latency: int = int(adr_custom_cfg_dict["latency"]["act_steps"][1])
    obs_max_latency: int = int(adr_custom_cfg_dict["latency"]["obs_steps"][1])
