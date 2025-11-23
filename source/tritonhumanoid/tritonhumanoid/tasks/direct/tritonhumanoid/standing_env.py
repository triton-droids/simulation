# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Sequence

import torch
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensor

from .standing_env_cfg import HumanoidEnvCfg


def normalize_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


class StandingEnv(DirectRLEnv):
    """Standing disturbance-rejection environment for the custom humanoid.

    Goal: stay upright, roughly over the environment origin, while rejecting random pushes.
    Small corrective steps are allowed and lightly penalized, but falling is heavily penalized.
    """

    cfg: HumanoidEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---------------------------------------------------------
        # Control setup (velocity-based, same 10 actuated joints)
        # ---------------------------------------------------------
        self.action_scale = self.cfg.action_scale

        # 10 actuated leg joints (same regex as locomotion env)
        actuated_joint_regex = (
            "left_hip1_joint|left_hip2_joint|left_thigh_joint|left_knee_joint|left_ankle_joint|"
            "right_hip1_joint|right_hip2_joint|right_thigh_joint|right_knee_joint|right_ankle_joint"
        )
        self._joint_dof_idx, _ = self.robot.find_joints(actuated_joint_regex)
        self.num_actions = len(self._joint_dof_idx)

        # Hip joints (for posture penalty & keeping stance)
        hip_joint_regex = "left_hip1_joint|left_hip2_joint|right_hip1_joint|right_hip2_joint"
        hip_dof_idx, _ = self.robot.find_joints(hip_joint_regex)
        self._hip_dof_idx = torch.as_tensor(
            hip_dof_idx, device=self.sim.device, dtype=torch.long
        )

        # Default joint pose (for posture regularization)
        self.default_joint_pos = self.robot.data.default_joint_pos[0]

        # Action buffer
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

        # Simple effort ratio (optional if you want energy cost)
        self.motor_effort_ratio = torch.ones(
            self.num_actions, dtype=torch.float32, device=self.sim.device
        )

        # ---------------------------------------------------------
        # Important bodies: torso + feet
        # ---------------------------------------------------------
        torso_body_indices, _ = self.robot.find_bodies("torso")
        self._torso_body_idx = int(torso_body_indices[0])

        right_foot_body_indices, _ = self.robot.find_bodies("right_foot")
        self._right_foot_body_idx = int(right_foot_body_indices[0])

        left_foot_body_indices, _ = self.robot.find_bodies("left_foot")
        self._left_foot_body_idx = int(left_foot_body_indices[0])

        # Default step width at initialization (used as reference; small deviations ok)
        feet_pos_all = self.robot.data.body_pos_w[:]  # [M, num_bodies, 3]
        left_xy = feet_pos_all[:, self._left_foot_body_idx, :2]
        right_xy = feet_pos_all[:, self._right_foot_body_idx, :2]
        torso_xy = feet_pos_all[:, self._torso_body_idx, :2]
        left_lat = left_xy[:, 1] - torso_xy[:, 1]
        right_lat = right_xy[:, 1] - torso_xy[:, 1]
        step_width = torch.abs(left_lat - right_lat)  # [M]
        self.default_step_width = step_width

        # Foot indices for contact-related rewards
        self._feet_ids, _ = self._contact_sensor.find_bodies("left_foot|right_foot")

        # ---------------------------------------------------------
        # Basis and orientation helpers (similar to locomotion env)
        # ---------------------------------------------------------
        # Use same "start rotation" as your locomotion env, but we will not
        # give any forward-progress reward. We only care about being upright and stable.
        qz_minus_90 = torch.tensor(
            [
                math.cos(math.pi / 4.0),  # w
                0.0,                      # x
                0.0,                      # y
                -math.sin(math.pi / 4.0), # z
            ],
            device=self.sim.device,
            dtype=torch.float32,
        )
        self.start_rotation = qz_minus_90
        self.inv_start_rot = quat_conjugate(self.start_rotation).unsqueeze(0).repeat(
            self.num_envs, 1
        )

        # local +Z is up, +X is "forward"
        self.basis_vec1 = torch.tensor(
            [0.0, 0.0, 1.0], device=self.sim.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)
        self.basis_vec0 = torch.tensor(
            [1.0, 0.0, 0.0], device=self.sim.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)

        # A dummy "target" just to reuse compute_heading_and_up/compute_rot;
        # for standing, angle_to_target is not used in the reward.
        self.targets = self.scene.env_origins + torch.tensor(
            [1.0, 0.0, 0.0], device=self.sim.device
        )  # 1m in +x per env

        self.potentials = torch.zeros(self.num_envs, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

        # ---------------------------------------------------------
        # Disturbance / push scheduling
        # ---------------------------------------------------------
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        min_steps = max(1, int(self.cfg.min_push_interval_s / self.dt))
        max_steps = max(min_steps, int(self.cfg.max_push_interval_s / self.dt))

        self._push_interval_steps_min = min_steps
        self._push_interval_steps_max = max_steps

        self.push_counters = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.sim.device
        )
        self.next_push_steps = torch.randint(
            self._push_interval_steps_min,
            self._push_interval_steps_max + 1,
            (self.num_envs,),
            device=self.sim.device,
        )

        # Buffers for commonly-used quantities
        self.torso_position = torch.zeros(self.num_envs, 3, device=self.sim.device)
        self.torso_rotation = torch.zeros(self.num_envs, 4, device=self.sim.device)
        self.velocity = torch.zeros(self.num_envs, 3, device=self.sim.device)
        self.ang_velocity = torch.zeros(self.num_envs, 3, device=self.sim.device)
        self.dof_pos = torch.zeros(
            self.num_envs, self.robot.num_joints, device=self.sim.device
        )
        self.dof_vel = torch.zeros_like(self.dof_pos)

        # Fields computed in _compute_intermediate_values
        self.up_proj = torch.zeros(self.num_envs, device=self.sim.device)
        self.heading_proj = torch.zeros_like(self.up_proj)
        self.up_vec = torch.zeros(self.num_envs, 3, device=self.sim.device)
        self.heading_vec = torch.zeros_like(self.up_vec)
        self.vel_loc = torch.zeros_like(self.velocity)
        self.angvel_loc = torch.zeros_like(self.ang_velocity)
        self.roll = torch.zeros(self.num_envs, device=self.sim.device)
        self.pitch = torch.zeros_like(self.roll)
        self.yaw = torch.zeros_like(self.roll)
        self.angle_to_target = torch.zeros_like(self.roll)
        self.dof_pos_scaled = torch.zeros_like(self.dof_pos)

    # ----------------------------------------------------------------------
    # Scene setup
    # ----------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Terrain (plane)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone envs
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lights
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.75, 0.75, 0.75)
        )
        light_cfg.func("/World/Light", light_cfg)

    # ----------------------------------------------------------------------
    # Disturbance logic
    # ----------------------------------------------------------------------
    def _maybe_apply_pushes(self):
        """Apply random horizontal velocity kicks to the base at random intervals."""

        # If upper bound is non-positive, treat pushes as disabled
        if self.cfg.push_force_range[1] <= 0.0:
            return

        # Update per-env push counters
        self.push_counters += 1
        ready = self.push_counters >= self.next_push_steps
        if not torch.any(ready):
            return

        env_ids = torch.nonzero(ready, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return

        num_push = env_ids.numel()
        device = self.sim.device

        # --- Sample random directions in the XY plane ---
        dirs_xy = torch.randn((num_push, 2), device=device)
        dirs_xy /= (torch.norm(dirs_xy, dim=-1, keepdim=True) + 1e-6)

        # --- Interpret push_force_range as delta-velocity (m/s) in XY ---
        min_dv, max_dv = self.cfg.push_force_range
        mags = torch.empty((num_push, 1), device=device).uniform_(min_dv, max_dv)

        delta_v_xy = dirs_xy * mags  # [K, 2]

        # --- Read current root velocity and inject the kick in XY ---
        root_vel = self.robot.data.root_vel_w[env_ids].clone()  # shape [K, 6] (lin[0:3], ang[3:6])
        root_vel[:, 0:2] += delta_v_xy  # modify linear x, y

        self.robot.write_root_velocity_to_sim(root_vel, env_ids)

        # --- Resample timers for these envs ---
        self.push_counters[env_ids] = 0
        self.next_push_steps[env_ids] = torch.randint(
            self._push_interval_steps_min,
            self._push_interval_steps_max + 1,
            (num_push,),
            device=self.sim.device,
        )


    # ----------------------------------------------------------------------
    # RL interface
    # ----------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        # store actions
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # schedule potential pushes
        self._maybe_apply_pushes()

    def _apply_action(self):
        vel_targets = self.action_scale * self.actions
        self.robot.set_joint_velocity_target(vel_targets, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        torso = self._torso_body_idx

        self.torso_position = self.robot.data.body_pos_w[:, torso]
        self.torso_rotation = self.robot.data.body_quat_w[:, torso]
        self.velocity = self.robot.data.body_lin_vel_w[:, torso]
        self.ang_velocity = self.robot.data.body_ang_vel_w[:, torso]

        self.dof_pos = self.robot.data.joint_pos
        self.dof_vel = self.robot.data.joint_vel

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        obs = torch.cat(
            (
                self.torso_position[:, 2].view(-1, 1),  # height
                self.vel_loc,  # local linear vel
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.pitch).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                # we keep heading_proj and angle_to_target for compatibility,
                # but they are not used in the reward for standing.
                self.heading_proj.unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # compute intermediate stuff (if not already)
        self._compute_intermediate_values()

        # Only actuated DOFs for energy penalty
        actuated_dof_vel = self.dof_vel[:, self._joint_dof_idx]

        # --------------------------------------
        # Base / posture rewards & penalties
        # --------------------------------------
        # upright: up_proj ≈ cos(angle between torso-up and world-up)
        up_clamped = torch.clamp(self.up_proj, min=0.0)
        up_reward = self.cfg.up_weight * up_clamped

        # keep torso height near target
        height = self.torso_position[:, 2]
        height_error = height - self.cfg.target_root_height
        base_height_penalty = self.cfg.base_height_scale * (height_error**2)

        # keep base near env origin in XY (allows small wandering)
        torso_xy = self.torso_position[:, :2] - self.scene.env_origins[:, :2]
        xy_dist = torch.norm(torso_xy, dim=-1)
        base_xy_penalty = self.cfg.base_xy_scale * (xy_dist**2)

        # penalize large linear / angular velocities (we want "quiet" standing)
        lin_vel = self.velocity  # world frame
        ang_vel = self.ang_velocity

        lin_vel_penalty = self.cfg.lin_vel_l2_scale * torch.sum(lin_vel**2, dim=-1)
        ang_vel_penalty = self.cfg.ang_vel_l2_scale * torch.sum(ang_vel**2, dim=-1)

        # --------------------------------------
        # Step-size / stance regularization
        # --------------------------------------
        # Encourage feet not to stray too far from default lateral stance,
        # but keep this soft so small step corrections are allowed.
        feet_pos = self.robot.data.body_pos_w[:, self._feet_ids, :]  # [N, 2, 3]
        feet_xy = feet_pos[:, :, :2]
        torso_xy_rep = torso_xy.unsqueeze(1) + self.scene.env_origins[:, :2].unsqueeze(1)
        # lateral positions (y) of each foot relative to torso
        foot_lats = feet_xy[:, :, 1] - self.torso_position[:, 1].unsqueeze(1)  # [N, 2]
        step_width = torch.abs(foot_lats[:, 0] - foot_lats[:, 1])  # [N]
        target_width = self.default_step_width
        width_deviation = step_width - target_width
        step_width_penalty = self.cfg.step_width_scale * (width_deviation**2)

        # penalize extremely large forward/backward foot offsets (huge steps)
        heading_xy = self.heading_vec[:, :2]
        heading_xy = heading_xy / (
            torch.norm(heading_xy, dim=-1, keepdim=True) + 1e-6
        )
        foot_vecs = feet_xy - self.torso_position[:, :2].unsqueeze(1)  # [N, 2, 2]
        forward_offsets = (foot_vecs * heading_xy.unsqueeze(1)).sum(dim=-1)  # [N, 2]
        max_forward = forward_offsets.abs().max(dim=1).values
        excess_stride = torch.clamp(
            max_forward - self.cfg.max_stride_length, min=0.0
        )
        stride_penalty = self.cfg.stride_penalty_scale * (excess_stride**2)

        # --------------------------------------
        # Hip posture regularization (reduce odd crouches)
        # --------------------------------------
        hip_angles = self.dof_pos[:, self._hip_dof_idx]
        hip_default = self.default_joint_pos[self._hip_dof_idx]
        hip_deviation = hip_angles - hip_default.unsqueeze(0)
        hip_posture_cost = (hip_deviation**2).mean(dim=1)
        hip_posture_penalty = self.cfg.hip_posture_scale * hip_posture_cost

        # --------------------------------------
        # Action / energy regularization
        # --------------------------------------
        actions_cost = torch.sum(self.actions**2, dim=-1)
        electricity_cost = torch.sum(
            torch.abs(self.actions * actuated_dof_vel * self.cfg.dof_vel_scale)
            * self.motor_effort_ratio.unsqueeze(0),
            dim=-1,
        )

        # --------------------------------------
        # Alive / fall penalties
        # --------------------------------------
        alive_reward = torch.ones_like(self.up_proj) * self.cfg.alive_reward_scale

        fell_height = self.torso_position[:, 2] < self.cfg.termination_height
        too_tilted = self.up_proj < self.cfg.termination_up_proj
        died = fell_height | too_tilted
        death_penalty = torch.where(
            died, torch.ones_like(self.up_proj) * self.cfg.death_cost, 0.0
        )

        total_reward = (
            alive_reward
            + up_reward
            - base_height_penalty
            - base_xy_penalty
            - lin_vel_penalty
            - ang_vel_penalty
            - step_width_penalty
            - stride_penalty
            - hip_posture_penalty
            - self.cfg.actions_cost_scale * actions_cost
            - self.cfg.energy_cost_scale * electricity_cost
            + death_penalty
        )

        return total_reward

    def _get_dones(self):
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        fell_height = self.torso_position[:, 2] < self.cfg.termination_height
        too_tilted = self.up_proj < self.cfg.termination_up_proj
        too_far_xy = (
            torch.norm(
                self.torso_position[:, :2] - self.scene.env_origins[:, :2], dim=-1
            )
            > self.cfg.max_xy_displacement
        )

        died = fell_height | too_tilted | too_far_xy
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.sim.device, dtype=torch.long)

        # Reset robot to default standing pose
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset actions
        self.actions[env_ids] = 0.0

        # Reset push timers
        self.push_counters[env_ids] = 0
        self.next_push_steps[env_ids] = torch.randint(
            self._push_interval_steps_min,
            self._push_interval_steps_max + 1,
            (env_ids.numel(),),
            device=self.sim.device,
        )

        # Recompute orientation-related stuff
        self._compute_intermediate_values()
        # potentials/targets exist but are not used in reward
        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(
            to_target, p=2, dim=-1
        ) / self.cfg.sim.dt


# ========= Helper (copied from locomotion env) =========

@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(
        dof_pos, dof_lower_limits, dof_upper_limits
    )

    prev_potentials[:] = potentials
    # We still update potentials but don't use them in the reward for this env
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
