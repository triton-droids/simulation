# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for my custom humanoid robot."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

import math

qz_minus_90 = (
    math.cos(math.pi / 4.0),  # ≈ 0.7071
    0.0,
    0.0,
    -math.sin(math.pi / 4.0), # ≈ -0.7071
)

HUMANOID_CFG = ArticulationCfg(
    prim_path="",  # you usually override this in the scene cfg
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{Path(__file__).parent}/human/human.usd",
        usd_path=f"{Path(__file__).parent}/human_offset/human_offset.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),          # tweak height so feet just touch the ground
        rot=qz_minus_90,
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),

    soft_joint_pos_limit_factor=0.95,

    actuators={
        # Single group for all leg joints (velocity control)
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip1_joint",
                "left_hip2_joint",
                "left_thigh_joint",
                "left_knee_joint",
                "left_ankle_joint",
                "right_hip1_joint",
                "right_hip2_joint",
                "right_thigh_joint",
                "right_knee_joint",
                "right_ankle_joint",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=20.0,

            # Velocity-control regime: P ≈ 0, D > 0
            stiffness=0.0,
            damping={
                ".*hip.*": 30.0,              # hip joints slightly stronger
                ".*thigh.*": 28.0,
                ".*knee.*": 22.0,
                ".*ankle.*": 16.0,
            },

            # Optional: very small joint friction/armature so it’s not totally ideal
            friction=0.05,
            armature=0.0,
        ),
    },
)
