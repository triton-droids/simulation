# Copyright (c) 2022-2025, The Isaac Lab Project Developers
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
    """Domain randomization for humanoid standing disturbance env."""

    # Robot material
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",  # randomize each reset
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.1),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 128,
        },
    )

    # Joint stiffness & damping
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "distribution": "uniform",
        },
    )

    # Small gravity variations (sim2real robustness)
    gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            # e.g. uniform random additive perturbation in z only
            "gravity_distribution_params": (0.9, 1.1),  # scale factor range
            "operation": "scale",                       # or "add"
            "distribution": "uniform",
        },
    )


    # You could also add joint parameter randomization, COM shifts, etc. later:
    # joint_params = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     min_step_count_between_reset=0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.8, 1.2),
    #         "damping_distribution_params": (0.8, 1.2),
    #         "friction_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    """Config for humanoid standing disturbance rejection."""

    # === RL env timing ===
    episode_length_s = 15.0
    decimation = 2

    # === Actions / observations ===
    # 10 leg DOFs, velocity control
    action_scale = 5.0
    action_space = 10

    # Observation layout: see HumanoidDisturbanceEnv._get_observations
    #  Hardware-only sensors (IMU + joints):
    #+ 3 (gravity in body frame - from IMU)
    #+ 3 (angular velocity in body frame - from IMU gyro)
    #+ num_dofs (joint pos scaled)
    #+ num_dofs (joint vel)
    #+ num_dofs (joint torques)
    #+ num_actions (last actions)
    num_dofs: int = 10
    
    # Single frame observation dimension
    observation_space_single = 3 + 3 + num_dofs + num_dofs + num_dofs + action_space
    
    # Observation stacking for memory (frames)
    obs_stack_frames: int = 1
    
    # Total observation space (accounting for stacking)
    observation_space = observation_space_single * obs_stack_frames
    
    state_space = 0

    # === Action processing ===
    residual_pos_scale: float = 0.25  # scaling for residual position control
    action_filter_alpha: float = 0.2  # first-order filter coefficient
    action_rate_scale: float = 0.05   # penalty for action changes
    
    # === Observation scaling ===
    angular_velocity_scale: float = 0.25
    dof_vel_scale: float = 0.1
    torque_scale: float = 0.01

    # === Simulation ===
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

    # === Scene ===
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # === Robot & contacts ===
    robot: ArticulationCfg = HUMANOID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # === Domain randomization events ===
    events: EventCfg = EventCfg()

    # === Disturbance (push) parameters ===
    # Forces are in Newtons, applied in random XY directions at random intervals.
    # Tune based on your robot’s mass and actuator limits.
    # treat as delta-velocity, not force
    push_force_range = (0.2, 0.8)   # start small
    min_push_interval_s = 1.0
    max_push_interval_s = 3.0


    # === Reward / penalty scales ===
    up_weight: float = 2.0

    # keep CoM height near nominal (tune target_root_height from initial pose)
    target_root_height: float = 1.0
    base_height_scale: float = 5.0

    # penalize wandering in XY (soft to allow small steps)
    base_xy_scale: float = 1.0

    # velocity penalties
    lin_vel_l2_scale: float = 0.1
    ang_vel_l2_scale: float = 0.1

    # stance / stepping regularizers
    step_width_scale: float = 0.5
    max_stride_length: float = 0.30  # meters
    stride_penalty_scale: float = 1.0
    hip_posture_scale: float = 0.5

    # action/energy regularization
    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    dof_vel_scale: float = 0.1

    # alive / termination
    alive_reward_scale: float = 0.1
    death_cost: float = -2.0
    termination_height: float = 0.6
    termination_up_proj: float = 0.5  # ~60 degrees from upright
    max_xy_displacement: float = 0.6  # meters from env origin

    # === ADR configuration ===
    enable_adr: bool = False
    num_adr_increments: int = 100
    starting_adr_increments: int = 0
    adr_update_interval_steps: int = 10000
    adr_success_rate_to_increase: float = 0.85
    adr_success_rate_to_decrease: float = 0.50
    adr_min_steps_before_decrease: int = 50000
    adr_ema_factor: float = 0.05
    adr_print_every_update: bool = True
    
    # ADR event randomization ranges (max difficulty)
    adr_event_cfg_dict: dict = {
        "robot_physics_material": {
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.3, 1.4),
            "restitution_range": (0.0, 0.4),
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
        },
        "gravity": {
            "gravity_distribution_params": (0.8, 1.2),
        },
    }
    
    # ADR custom parameters (max difficulty)
    adr_custom_cfg_dict: dict = {
        "push": {
            "push_force_range": ((0.2, 0.8), (0.5, 2.0)),  # (min_dv, max_dv) at difficulty=1
        },
        "action_noise": {
            "std": (0.0, 0.1),
        },
        "obs_noise": {
            "gravity_std": (0.0, 0.05),
            "gyro_std": (0.0, 0.1),
            "joint_pos_std": (0.0, 0.02),
            "joint_vel_std": (0.0, 0.5),
            "joint_torque_std": (0.0, 0.5),
        },
        "motor_strength": {
            "per_joint_mult_range": (0.0, 0.3),  # +/- 30% at max difficulty
        },
        "latency": {
            "act_steps": (0, 5),
            "obs_steps": (0, 3),
        },
        "imu_bias": {
            "gravity_bias_range": (0.0, 0.1),
            "gyro_bias_range": (0.0, 0.2),
        },
    }
    
    # Latency buffer sizes (must be >= max ADR latency)
    act_max_latency: int = 5
    obs_max_latency: int = 3
