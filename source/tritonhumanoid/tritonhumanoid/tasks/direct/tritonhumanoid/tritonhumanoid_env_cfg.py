# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from ....assets.humanoid import HUMANOID_LOCOMOTION_DELAYED_PD_CFG

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

@configclass
class EventCfg:
    """Domain randomization terms for humanoid locomotion."""

    # Robot material
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",  # randomize each reset
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 3.0),
            "dynamic_friction_range": (0.1, 3.0),
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
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "distribution": "uniform",
        },
    )

    # Small gravity variations (sim2real robustness)
    gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "gravity_distribution_params": ([1.0, 1.0, 0.9], [1.0, 1.0, 1.1]),  # scale factor range
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Link mass scaling
    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (1.0, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Link COM offset (if available in this Isaac Lab version)
    if hasattr(mdp, "randomize_rigid_body_com"):
        robot_com_offset = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="reset",
            min_step_count_between_reset=0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)},
            },
        )

    # Inertia scaling (if available in this Isaac Lab version)
    if hasattr(mdp, "randomize_rigid_body_inertia"):
        robot_inertia_scale = EventTerm(
            func=mdp.randomize_rigid_body_inertia,
            mode="reset",
            min_step_count_between_reset=0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "inertia_distribution_params": (0.85, 1.15),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

    # Joint friction/armature
    robot_joint_friction_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.5, 1.5),
            "armature_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    min_episode_length_s: float = 5.0
    randomize_episode_length: bool = True
    decimation = 4

    # Position control: actions map to POSITION OFFSETS (radians) around default pose
    # action_scale determines the maximum offset: actions in [-1, 1] → [-action_scale, +action_scale] radians
    # The actual position command is: q_target = default_pose + action_scale * action
    action_scale: float = 0.8  # Lower for early training stability; increase to 0.5 after initial learning
    # Per-joint multipliers applied on top of action_scale
    action_scale_by_joint: dict[str, float] = {
        "left_hip2_joint": 0.50,
        "right_hip2_joint": 0.50,
        "left_thigh_joint": 0.3,
        "right_thigh_joint": 0.3,
    }

    # 10 actuated leg joints
    action_space = 10

    # obs_dim = 3 (lin_vel) + 3 (ang_vel) + 3 (up_b) + 3 (commands) + 10 (pos) + 10 (vel) + 10 (prev_actions) = 42
    # + 2 (phase clock if use_phase_obs=True) = 44
    observation_space_single = 42
    obs_stack_frames: int = 3
    observation_space = observation_space_single * obs_stack_frames

    state_space = 0

    # simulation
    sim = SimulationCfg(
        dt=1/200,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_external_forces_every_iteration=True,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # 4096 envs -> 64
    _grid = int(math.sqrt(4096))

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=TerrainGeneratorCfg(
    #         size=(4.0, 4.0),        # REQUIRED: (x_width, y_length) per tile :contentReference[oaicite:1]{index=1}
    #         num_rows=_grid,         # 64
    #         num_cols=_grid,         # 64
    #         horizontal_scale=0.2,   # optional (defaults exist)
    #         vertical_scale=0.005,
    #         sub_terrains={
    #             "flat": HfRandomUniformTerrainCfg(
    #                 proportion=0.3, noise_range=(0.0, 0.0), noise_step=0.005
    #             ),
    #             "rough": HfRandomUniformTerrainCfg(
    #                 proportion=0.7, noise_range=(0.0, 0.04), noise_step=0.005
    #             ),
    #         },
    #     ),
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="average",
    #         restitution_combine_mode="average",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )

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


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

     # events
    events: EventCfg = EventCfg()

    # robot
    # Locomotion uses a dedicated delayed-PD asset with motor-pair-specific latency groups
    # derived from the embedded motor dataset analysis.
    robot: ArticulationCfg = HUMANOID_LOCOMOTION_DELAYED_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    disable_non_foot_collisions: bool = False
    non_foot_collision_body_names: tuple[str, ...] = (
        "torso",
        "hip",
        "left_leg1",
        "left_leg2",
        "left_leg3",
        "left_leg4",
        "right_leg1",
        "right_leg2",
        "right_leg3",
        "right_leg4",
    )
    enable_contact_sensor: bool = False
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot", history_length=3, update_period=0.005, track_air_time=True
    )

    # scene.ee_site = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/world",   # source frame
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/Robot/torso",  # parent body of your site
    #             name="top",
    #             offset=OffsetCfg(
    #                 pos=(-0.155, -0.016, 0.765),            # your trial site position (m)
    #                 rot=(1.0, 0.0, 0.0, 0.0),        # quaternion (w,x,y,z)
    #             ),
    #         ),
    #     ],
    #     debug_vis=True,   # show frame markers
    # )

    ee_site: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/world",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/torso",
                name="top",
                offset=OffsetCfg(
                    pos=(-0.155, -0.016, 0.6996),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        ],
        debug_vis=False,  # keep off during training/headless
    )
    use_frame_transformer_sensor: bool = False
    track_body_name: str = "torso"
    track_body_offset: tuple[float, float, float] = (-0.155, -0.016, 0.6996)

    # Observation scales
    ang_vel_scale: float = 0.25
    dof_vel_scale: float = 0.1
    joint_pos_obs_noise_std_rad: float = 0.0
    joint_vel_obs_noise_std_rad_s: float = 0.0
    command_yaw_offset: float = -math.pi / 2.0  # rotate body-frame vectors to align +Y forward with +X commands

    # Observation term config (uniform noise, scale, clip)
    obs_term_cfg: dict = {
        "projected_gravity": {"noise": 0.01, "scale": 1.0, "clip": 1.0},
        "base_ang_vel": {"noise": 0.05, "scale": 0.25, "clip": 5.0},
        "base_lin_vel": {"noise": 0.05, "scale": 1.0, "clip": 5.0},
        "dof_pos_delta": {"noise": 0.01, "scale": 1.0, "clip": 2.0},
        "dof_vel": {"noise": 0.1, "scale": 0.1, "clip": 5.0},
        "prev_actions": {"noise": 0.0, "scale": 1.0, "clip": 1.0},
        "commands": {"noise": 0.0, "scale": 1.0, "clip": 1.0},
        "phase": {"noise": 0.0, "scale": 1.0, "clip": 1.0},
    }

    # Termination
    termination_height: float = 0.4
    upright_threshold: float = 0.5  # up_b.z

    # -------------------------
    # HOMIE-inspired locomotion reward
    # -------------------------

    # Tracking sharpness
    tracking_sigma: float = 0.25
    height_tracking_sigma: float = 0.02

    # Command tracking
    tracking_x_vel_scale: float = 1.5
    tracking_y_vel_scale: float = 1.0
    tracking_ang_vel_scale: float = 2.0

    # HOMIE-style fixed walking height target. This is a privileged reward term only.
    base_height_target: float = 0.70
    tracking_base_height_scale: float = 1.0

    # Base stability
    lin_vel_z_scale: float = -0.5
    ang_vel_xy_scale: float = -0.025
    orientation_scale: float = -1.5

    # Joint posture regularization
    deviation_hip_joint_scale: float = -0.10
    deviation_knee_joint_scale: float = -0.15
    deviation_ankle_joint_scale: float = -0.20

    # Smoothness / effort
    action_rate_scale: float = -0.01
    dof_vel_reward_scale: float = -1.0e-4
    dof_acc_scale: float = -2.5e-7
    torques_scale: float = -2.5e-6
    joint_power_scale: float = -2.0e-5

    # Limits
    dof_pos_limits_scale: float = -2.0
    soft_dof_pos_limit: float = 0.975

    # Distance-based feet shaping. These replace contact/air-time rewards so training does
    # not depend on contact sensor signals or non-foot self-collision.
    foot_body_regex: str = "left_foot|right_foot"
    knee_body_regex: str = "left_leg4|right_leg4"
    foot_ground_height_target: float = 0.035
    foot_clearance_target: float = 0.10
    foot_height_sigma: float = 0.01
    foot_swing_speed_threshold: float = 0.15
    feet_support_height_scale: float = 0.50
    feet_clearance_scale: float = -0.75
    feet_near_ground_velocity_scale: float = -0.05
    feet_distance_scale: float = -0.20
    feet_distance_min: float = 0.12
    feet_distance_max: float = 0.38
    knee_distance_scale: float = -0.15
    knee_distance_min: float = 0.08
    knee_distance_max: float = 0.32

    # Stand behavior
    stand_still_scale: float = -0.15
    stand_cmd_lin_thresh: float = 0.08
    stand_cmd_yaw_thresh: float = 0.10

    # Death
    death_cost: float = -1.0

    # Penalty curriculum (episode-length driven)
    penalty_curriculum_enabled: bool = False
    penalty_curriculum_mode: str = "smooth"  # "smooth" or "threshold"
    penalty_curriculum_min_scale: float = 1.0
    penalty_curriculum_max_scale: float = 1.0
    penalty_curriculum_use_ema: bool = True
    penalty_curriculum_ema_alpha: float = 0.05
    penalty_curriculum_window: int = 256
    penalty_curriculum_exclude_timeouts: bool = True
    # Threshold mode params (fractions of max episode steps)
    penalty_curriculum_degree: float = 0.02
    penalty_curriculum_low_len_frac: float = 0.2
    penalty_curriculum_high_len_frac: float = 0.8
    # Smooth mode params (fraction of max episode steps)
    penalty_curriculum_target_len_frac: float = 0.8
    penalty_curriculum_power: float = 2.0

    # Optional: gait phase for observation/debug timing only.
    use_phase_obs: bool = False
    gait_period_s: float = 1.0
    gait_period_randomization_width: float = 0.0
    randomize_phase: bool = True
    phase_offset_default: tuple[float, float] = (0.0, math.pi)
    freeze_phase_when_standing: bool = True
    stand_phase_lin_threshold: float = 0.08
    stand_phase_yaw_threshold: float = 0.10
    stand_phase_value: float = math.pi

    # Command curriculum (progressive difficulty) - based on per-env steps
    use_curriculum: bool = True
    curriculum_stage1_steps_per_env: int = 10000   # per-env steps before adding yaw (stage 0 -> 1)
    curriculum_stage2_steps_per_env: int = 30000  # per-env steps before adding lateral (stage 1 -> 2)
    
    # Stage 0: encourage forward motion (not standing still)
    curriculum_stage0_vx_min: float = 0.2  # minimum forward velocity command in stage 0
    curriculum_stage0_vx_max: float = 0.6  # maximum forward velocity command in stage 0
    zero_command_probability: float = 0.10  # chance to sample a standstill command (vx=vy=yaw=0)
    turn_in_place_probability: float = 0.02  # chance to sample vx=vy=0, yaw!=0
    turn_in_place_yaw_min: float = 0.3
    turn_in_place_yaw_max: float = 1.0
    turn_in_place_min_stage: int = 1  # only allow in-place turns once yaw commands are introduced
    
    # Command resampling (per-episode step count)
    command_resample_interval_s: float = 4.0
    command_resample_interval_steps: int = 0  # if >0, overrides seconds-based interval
    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_yaw_range: tuple[float, float] = (-1.0, 1.0)

    # --- Per-episode reset randomization (always-on, ADR or not) ---
    # Joint state noise at reset (actuated joints only)
    reset_joint_pos_noise: float = 0.05
    reset_joint_vel_noise: float = 0.2

    # === Disturbance (push) parameters ===
    # treat as delta-velocity, not force
    push_force_range = (0.2, 0.8)
    # Push timing profiles (in seconds)
    push_strong: bool = False  # True -> 1–3s, False -> 5–10s
    strong_min_push_interval_s = 1.0
    strong_max_push_interval_s = 3.0
    min_push_interval_s = 5.0
    max_push_interval_s = 10.0
    push_z_fraction: float = 0.25
    push_angvel_scale: float = 0.8

    # === ADR configuration ===
    enable_adr: bool = True
    num_adr_increments: int = 100
    starting_adr_increments: int = 0
    adr_update_interval_steps: int = 500
    adr_success_rate_to_increase: float = 0.85
    adr_success_rate_to_decrease: float = 0.2
    adr_min_steps_before_decrease: int = 10000
    adr_ema_factor: float = 0.05
    adr_warmup_steps: int = 20000
    adr_min_stage: int = 2
    adr_track_err_lin_increase_threshold: float = 0.25
    adr_track_err_lin_decrease_threshold: float = 0.45
    adr_track_err_yaw_increase_threshold: float = 0.25
    adr_track_err_yaw_decrease_threshold: float = 0.6
    adr_command_scale_min_stage: int = 2
    adr_push_start_difficulty: float = 0.3
    adr_push_ramp_difficulty: float = 0.3
    adr_micro_wrench_start_difficulty: float = 0.4
    adr_micro_wrench_ramp_difficulty: float = 0.3
    adr_print_every_update: bool = False
    adr_debug_print: bool = False
    adr_debug_print_every_steps: int = 2000   # print cadence

    # ADR event randomization ranges (max difficulty)
    adr_event_cfg_dict: dict = {
        "robot_physics_material": {
            "static_friction_range": (0.1, 3.0),
            "dynamic_friction_range": (0.1, 3.0),
            "restitution_range": (0.0, 0.4),
        },
        "robot_joint_stiffness_and_damping": {
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
        },
        "gravity": {
            "gravity_distribution_params": ([1.0, 1.0, 0.8], [1.0, 1.0, 1.2]),
        },
        "robot_scale_mass": {
            "mass_distribution_params": (0.7, 1.3),
        },
        "robot_joint_friction_armature": {
            "friction_distribution_params": (0.5, 1.5),
            "armature_distribution_params": (0.5, 1.5),
        },
    }
    if hasattr(mdp, "randomize_rigid_body_com"):
        adr_event_cfg_dict["robot_com_offset"] = {
            "com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)},
        }
    if hasattr(mdp, "randomize_rigid_body_inertia"):
        adr_event_cfg_dict["robot_inertia_scale"] = {
            "inertia_distribution_params": (0.85, 1.15),
        }

    adr_custom_cfg_dict: dict = {
        "push": {
            "push_force_range": ((0.2, 0.8), (0.5, 2.0)),
        },
        "robot_spawn": {
            "joint_pos_noise": (0.0, 0.06),
            "joint_vel_noise": (0.0, 0.20),
        },
        "action_noise": {
            "std": (0.0, 0.03),
        },
        "obs_noise": {
            "gravity_std": (0.0, 0.03),
            "gyro_std": (0.0, 0.85),
            "joint_pos_std": (0.0, 0.01),
            "joint_vel_std": (0.0, 0.20),
            "joint_torque_std": (0.0, 0.20),
        },
        "latency": {
            "act_steps": (0, 2),
            "obs_steps": (0, 1),
        },
        "motor_strength": {
            "per_joint_mult_range": (0.0, 0.3),
        },
        "imu_bias": {
            "gravity_bias_range": (0.0, 0.1),
            "gyro_bias_range": (0.0, 0.2),
        },
        "micro_wrench": {
            "lin_acc_std": (0.0, 0.01),
            "ang_acc_std": (0.0, 0.02),
            "rho": (0.0, 0.6),
            "max_lin_acc": (0.0, 0.03),
            "max_ang_acc": (0.0, 0.05),
        },
        # command magnitude scaling
        "command_scale": {
            "scale": (1.0, 1.5),
        },
    }

    # Action/observation latency buffer sizes in policy steps.
    action_max_latency: int = 2
    obs_max_latency: int = 1

    # Debug visualization (draw velocity arrows for a single env)
    debug_vel_vis: bool = False  # disable during training for performance
    debug_vel_vis_all_envs: bool = True  # override to visualize all envs
    debug_vel_vis_all_envs_max_envs: int = 64  # auto-enable all-envs when num_envs is small
    debug_env_id: int = 0            # which env to draw (0..num_envs-1)
    vel_vis_scale: float = 0.5       # meters of arrow per 1 m/s
    vel_vis_height: float = 0.25     # arrow origin above torso (m)
    vel_vis_every_n: int = 2         # draw every N sim steps to reduce overhead

    # Debug observation printing
    debug_obs_print: bool = False
    debug_obs_print_steps: int = 5
    debug_obs_print_every: int = 1
    debug_obs_print_env: int = 0
    debug_print_orderings: bool = False
    enable_reward_logging: bool = False


""" ORDERINGS OF STUFF:

[DebugOrder] action/act_pos order (index -> joint name):
  0: left_hip1_joint
  1: right_hip1_joint
  2: left_hip2_joint
  3: right_hip2_joint
  4: left_thigh_joint
  5: right_thigh_joint
  6: left_knee_joint
  7: right_knee_joint
  8: left_ankle_joint
  9: right_ankle_joint
[DebugOrder] commands order: [vx, vy, yaw_rate]
[DebugOrder] observation slices (single frame):
    lin_vel_cmd: [0, 3)
    ang_vel_cmd_scaled: [3, 6)
    up_cmd: [6, 9)
    commands: [9, 12)
    act_pos_scaled: [12, 22)
    act_vel_scaled: [22, 32)
    prev_actions: [32, 42)


[JointLimits]
  left_hip1_joint   axis=1.0 0.0 0.0   limits=[-1.57, 1.57]
  left_hip2_joint   axis=0.0 1.0 0.0   limits=[-1.57, 0.436332]
  left_thigh_joint  axis=0.0 0.0 -1.0  limits=[-0.785398, 0.785398]
  left_knee_joint   axis=1.0 0.0 0.0   limits=[-2.0944, 0]
  left_ankle_joint  axis=1.0 0.0 0.0   limits=[-0.6, 0.6]

  right_hip1_joint  axis=1.0 0.0 0.0   limits=[-1.57, 1.57]
  right_hip2_joint  axis=0.0 1.0 0.0   limits=[-0.436332, 1.57]
  right_thigh_joint axis=0.0 0.0 1.0   limits=[-0.785398, 0.785398]
  right_knee_joint  axis=1.0 0.0 0.0   limits=[-2.0944, 0]
  right_ankle_joint axis=1.0 0.0 0.0   limits=[-0.6, 0.6]

"""
