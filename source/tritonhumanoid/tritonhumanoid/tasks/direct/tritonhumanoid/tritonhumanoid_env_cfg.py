# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from ....assets.humanoid import HUMANOID_CFG

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
            "gravity_distribution_params": (0.9, 1.1),  # scale factor range
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
            "mass_distribution_params": (0.8, 1.2),
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
                "com_distribution_params": (0.0, 0.005),
                "operation": "add",
                "distribution": "uniform",
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
    decimation = 5

    # Position control: actions map to POSITION OFFSETS (radians) around default pose
    # action_scale determines the maximum offset: actions in [-1, 1] → [-action_scale, +action_scale] radians
    # The actual position command is: q_target = default_pose + action_scale * action
    action_scale = 1.0  # Lower for early training stability; increase to 0.5 after initial learning
    # Per-joint multipliers applied on top of action_scale
    action_scale_by_joint: dict[str, float] = {
        "left_thigh_joint": 0.3,
        "right_thigh_joint": 0.3,
    }
    # Joint-limit-aware action bounds margin (to avoid hard stops)
    action_limit_margin: float = 0.02          # radians, added each side
    action_limit_margin_frac: float = 0.05     # fraction of joint range, added each side

    # 10 actuated leg joints
    action_space = 10

    # obs_dim = 3 (lin_vel) + 3 (ang_vel) + 3 (up_b) + 3 (commands) + 10 (pos) + 10 (vel) + 10 (prev_actions) = 42
    # + 2 (phase clock if use_phase_obs=True) = 44
    observation_space_single = 42
    obs_stack_frames: int = 3
    observation_space = observation_space_single * obs_stack_frames

    state_space = 0

    # simulation
    sim_cfg = SimulationCfg(
        dt=1/250,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.2,
            dynamic_friction=0.4,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            sub_terrains={
                "flat": HfRandomUniformTerrainCfg(
                    proportion=0.3,
                    height_range=(0.0, 0.0),
                    slope_range=(0.0, 0.0),
                    step_height_range=(0.0, 0.0),
                ),
                "rough": HfRandomUniformTerrainCfg(
                    proportion=0.7,
                    height_range=(0.0, 0.04),   # ~4cm variation to start
                    slope_range=(0.0, 0.05),    # gentle slopes
                    step_height_range=(0.0, 0.0),
                ),
            }
        ),
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
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # Observation scales
    ang_vel_scale: float = 0.25
    dof_vel_scale: float = 0.1
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

    # Reward shaping (minimal)
    lin_vel_reward_scale: float = 3.0
    yaw_rate_reward_scale: float = 2.0
    upright_reward_scale: float = 1.0
    alive_reward: float = 0.05

    action_cost_scale: float = 0.005
    joint_limit_cost_scale: float = 0.2
    death_cost: float = -1.0

    # Tracking sharpness (bigger = easier / smoother)
    lin_vel_sigma: float = 0.25  # in (m/s)^2 units inside exp; sharper tracking
    yaw_rate_sigma: float = 0.5  # in (rad/s)^2 units inside exp

    # Stability costs (prevent hopping/rolling)
    lin_vel_z_cost_scale: float = 0.02
    ang_vel_xy_cost_scale: float = 0.01
    flat_ori_cost_scale: float = 0.1

    # Penalize standing still when a command asks for motion
    command_speed_threshold: float = 0.2  # m/s, only apply penalty above this command
    standstill_speed_threshold: float = 0.15  # m/s, penalize if actual speed below this
    standstill_penalty_scale: float = 0.2

    # Smoothness costs (reduce jitter)
    action_rate_cost_scale: float = 0.01
    dof_vel_cost_scale: float = 0.0001
    dof_vel_delta_cost_scale: float = 0.01  # penalize velocity changes (instead of acceleration)
    energy_cost_scale: float = 0.0005  # penalize mechanical power |tau * qdot|

    # Return-to-default pose penalty (actuated joints)
    pose_return_scale: float = 0.3
    pose_return_upright_threshold: float = 0.7

    # Left/right symmetry penalty (actuated joints)
    symmetry_cost_scale: float = 0.1
    thigh_pose_cost_scale: float = 0.1  # keep thigh joints near neutral to avoid inward twisting

    # Gate smoothness/energy penalties to swing phase (set swing_gate_alpha=0.0 to disable gating)
    gate_smoothness_to_swing: bool = True
    swing_gate_alpha: float = 1.0

    # Contact-based rewards
    foot_body_regex: str = "left_foot|right_foot"
    foot_contact_force_thresh: float = 30.0  # N (20-80N typical for humanoid ground contact)
    min_air_time: float = 0.3  # seconds
    feet_air_time_reward_scale: float = 0.4
    air_time_symmetry_cost_scale: float = 0.05
    foot_slip_cost_scale: float = 0.02
    undesired_contact_force_thresh: float = 80.0  # N (50-200N typical)
    undesired_contact_cost_scale: float = 0.1

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

    # Optional: gait phase for timing
    use_phase_obs: bool = False
    gait_period_s: float = 1.0
    gait_period_randomization_width: float = 0.0
    randomize_phase: bool = False
    phase_offset_default: tuple[float, float] = (0.0, math.pi)
    stand_phase_value: float = math.pi
    stand_phase_lin_threshold: float = 0.01
    stand_phase_yaw_threshold: float = 0.01

    # Command curriculum (progressive difficulty) - based on per-env steps
    use_curriculum: bool = False
    curriculum_stage1_steps_per_env: int = 5000   # per-env steps before adding yaw (stage 0 -> 1)
    curriculum_stage2_steps_per_env: int = 10000  # per-env steps before adding lateral (stage 1 -> 2)
    
    # Stage 0: encourage forward motion (not standing still)
    curriculum_stage0_vx_min: float = 0.3  # minimum forward velocity command in stage 0
    curriculum_stage0_vx_max: float = 1.0  # maximum forward velocity command in stage 0
    zero_command_probability: float = 0.1  # chance to sample a standstill command (vx=vy=yaw=0)
    turn_in_place_probability: float = 0.05  # chance to sample vx=vy=0, yaw!=0
    turn_in_place_yaw_min: float = 0.3
    turn_in_place_yaw_max: float = 1.0
    turn_in_place_min_stage: int = 1  # only allow in-place turns once yaw commands are introduced
    
    # Command resampling (per-episode step count)
    command_resample_interval_s: float = 10.0
    command_resample_interval_steps: int = 0  # if >0, overrides seconds-based interval
    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_yaw_range: tuple[float, float] = (-1.0, 1.0)
    stand_prob: float = 0.2  # chance to force standstill (vx=vy=yaw=0)

    # --- Per-episode reset randomization (always-on, ADR or not) ---
    # Action latency sampling (reuses act_hist_buf path)
    act_latency_reset_range: tuple[int, int] = (0, 3)
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
    adr_warmup_steps: int = 2000
    adr_min_stage: int = 1
    adr_track_err_lin_increase_threshold: float = 0.25
    adr_track_err_lin_decrease_threshold: float = 0.45
    adr_track_err_yaw_increase_threshold: float = 0.25
    adr_track_err_yaw_decrease_threshold: float = 0.6
    adr_command_scale_min_stage: int = 2
    adr_push_start_difficulty: float = 0.3
    adr_push_ramp_difficulty: float = 0.3
    adr_micro_wrench_start_difficulty: float = 0.4
    adr_micro_wrench_ramp_difficulty: float = 0.3
    adr_print_every_update: bool = True
    adr_debug_print: bool = True
    adr_debug_print_every_steps: int = 2000   # print cadence

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
            "com_distribution_params": (0.0, 0.005),
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
        "sensor_extrinsics": {
            "imu_mount_deg": (0.0, 5.0),
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
            "per_joint_mult_range": (0.0, 0.3),
        },
        "latency": {
            "act_steps": (0, 5),
            "obs_steps": (0, 3),
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

    # Latency buffer sizes (must be >= max ADR latency)
    act_max_latency: int = 5
    obs_max_latency: int = 3

    # Debug visualization (draw velocity arrows for a single env)
    debug_vel_vis: bool = False  # disable during training for performance
    debug_vel_vis_all_envs: bool = False  # override to visualize all envs
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
