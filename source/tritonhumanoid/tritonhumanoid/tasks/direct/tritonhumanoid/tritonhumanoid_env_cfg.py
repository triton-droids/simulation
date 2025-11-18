# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from ....assets.humanoid import HUMANOID_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         # base link name from your URDF
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "mass_distribution_params": (-2, 2),
    #         "operation": "add",
    #     },
    # )


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2

    # Velocity control: actions map to desired joint velocities via action_scale
    # If HUMANOID_CFG.actuators["legs"].velocity_limit_sim = 5.0, this makes
    # actions in [-1, 1] → [-5, 5] rad/s
    action_scale = 2.0

    # 10 actuated leg joints
    action_space = 10

    # If the robot has 10 total DOFs:
    # obs_dim = 12 + 3 * num_dofs = 12 + 30 = 42
    observation_space = 42

    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
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
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward weights
    heading_weight: float = 1.0
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.55

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

    feet_air_time_reward_scale: float = 0.1

    leg_heading_scale: float = 0.3
    yaw_penalty_scale: float = 0.5      # new: torso yaw alignment
    lateral_vel_scale: float = 0.2      # new: penalize sideways CoM velocity

    orient_vel_weight: float = 0.5

    feet_slide_scale: float = 0.2      # tune
    feet_air_asym_scale: float = 2.0   # for symmetry term below

    single_leg_scale: float = 1.0

    hip_posture_scale: float = 0.5   # or 0.1 to start, then tune
    hop_penalty_scale: float = 0.5

    sym_scale: float = 0.2

