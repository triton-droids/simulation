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
    #  1 (height)
    #+ 3 (local lin vel)
    #+ 3 (local ang vel)
    #+ 1 (yaw)
    #+ 1 (roll)
    #+ 1 (pitch)
    #+ 1 (up_proj)
    #+ 1 (heading_proj)
    #+ 1 (angle_to_target)
    #+ num_dofs (joint pos scaled)
    #+ num_dofs (joint vel)
    #+ num_actions (last actions)
    # For your humanoid, num_dofs is likely 16; adjust if needed.
    num_dofs: int = 10
    observation_space = (
        1 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + num_dofs + num_dofs + action_space
    )
    state_space = 0

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

    angular_velocity_scale: float = 0.25  # for obs scaling
