# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensor


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class LocomotionEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- Control setup (velocity-based, 10 actuated joints) ----
        self.action_scale = self.cfg.action_scale

        # Only control the 10 actuated leg joints
        actuated_joint_regex = (
            "left_hip1_joint|left_hip2_joint|left_thigh_joint|left_knee_joint|left_ankle_joint|"
            "right_hip1_joint|right_hip2_joint|right_thigh_joint|right_knee_joint|right_ankle_joint"
        )
        self._joint_dof_idx, _ = self.robot.find_joints(actuated_joint_regex)
        self.num_actions = len(self._joint_dof_idx)

        # Hip joints only (both sides)
        hip_joint_regex = "left_hip1_joint|left_hip2_joint|right_hip1_joint|right_hip2_joint"
        hip_dof_idx, _ = self.robot.find_joints(hip_joint_regex)

        # store as tensor on the correct device
        self._hip_dof_idx = torch.as_tensor(
            hip_dof_idx,
            device=self.sim.device,
            dtype=torch.long,
        )

        # default pose for posture penalty
        self.default_joint_pos = self.robot.data.default_joint_pos[0]

        # actions: desired joint velocities (scaled by action_scale)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

        # used only for energy penalty; keep as simple scale for now
        self.motor_effort_ratio = torch.ones(self.num_actions, dtype=torch.float32, device=self.sim.device)

        # ---- Identify important body indices ----
        hip_body_indices, _ = self.robot.find_bodies("base_link")
        self._hip_body_idx = int(hip_body_indices[0])

        self._feet_ids, _ = self._contact_sensor.find_bodies("left_foot|right_foot")

        # ---- Locomotion targets and cached buffers ----
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([0, 1000, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([0, 1, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        # add contact sensors
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # actions in [-1, 1] → desired joint velocities
        self.actions = actions.clone()

    def _apply_action(self):
        # Velocity-based control:
        # actions \in [-1, 1] → desired velocity in [-action_scale, action_scale] rad/s
        vel_targets = self.action_scale * self.actions
        self.robot.set_joint_velocity_target(vel_targets, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        hip = self._hip_body_idx

        self.torso_position = self.robot.data.body_pos_w[:, hip]
        self.torso_rotation = self.robot.data.body_quat_w[:, hip]
        self.velocity = self.robot.data.body_lin_vel_w[:, hip]
        self.ang_velocity = self.robot.data.body_ang_vel_w[:, hip]

        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

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
        obs = torch.cat(
            (
                self.torso_position[:, 2].view(-1, 1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # use only actuated joint velocities for energy cost
        actuated_dof_vel = self.dof_vel[:, self._joint_dof_idx]

        base_reward = compute_rewards(
            self.actions,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            actuated_dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.motor_effort_ratio,
            self.velocity,
            self.heading_vec,
            self.cfg.orient_vel_weight,
        )

        # =========================
        # Feet air-time (biped aware)
        # =========================
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]  # [N, 2]

        air_time_threshold = 0.05
        per_foot = torch.clamp(last_air_time - air_time_threshold, min=0.0) * first_contact  # [N, 2]

        air_mean = per_foot.mean(dim=1)                        # [N]
        air_asym = torch.abs(per_foot[:, 0] - per_foot[:, 1])  # [N]

        speed = torch.norm(self.velocity[:, :2], dim=1)        # world-frame XY
        moving_mask = (speed > 0.1).float()

        raw_air = air_mean - self.cfg.feet_air_asym_scale * air_asym
        raw_air = torch.clamp(raw_air, min=0.0)

        feet_air_time_reward = (
            self.cfg.feet_air_time_reward_scale
            * raw_air
            * moving_mask
        )

        # =========================
        # Feet sliding penalty
        # =========================
        contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_ids]  # [N, 2]
        in_contact = contact_time > 0.0                                                   # [N, 2]

        feet_vel_xy = self.robot.data.body_lin_vel_w[:, self._feet_ids, :2]               # [N, 2, 2]
        feet_speed = torch.norm(feet_vel_xy, dim=-1)                                      # [N, 2]

        sliding = feet_speed * in_contact.float()                                         # [N, 2]
        feet_slide_penalty = self.cfg.feet_slide_scale * sliding.sum(dim=1)              # [N]

        # =========================
        # Legs relative to hip heading penalty (make feet less sideways)
        # =========================
        hip_xy = self.torso_position[:, :2]                           # [N, 2]
        feet_pos = self.robot.data.body_pos_w[:, self._feet_ids, :]   # [N, 2, 3]
        feet_xy = feet_pos[:, :, :2]                                  # [N, 2, 2]

        foot_vecs = feet_xy - hip_xy.unsqueeze(1)                     # [N, 2, 2]
        foot_vecs_norm = torch.norm(foot_vecs, dim=-1, keepdim=True) + 1e-6
        foot_dirs = foot_vecs / foot_vecs_norm                        # [N, 2, 2]

        heading_xy = self.heading_vec[:, :2]
        heading_xy = heading_xy / (torch.norm(heading_xy, dim=-1, keepdim=True) + 1e-6)  # [N, 2]

        dot = (foot_dirs * heading_xy.unsqueeze(1)).sum(dim=-1)      # [N, 2]
        foot_heading_misalignment = 1.0 - dot                        # [N, 2]

        # Use *max* instead of mean so one very sideways foot gets penalized strongly
        foot_misalignment_max = foot_heading_misalignment.max(dim=1).values  # [N]
        leg_heading_penalty = self.cfg.leg_heading_scale * foot_misalignment_max

        # =========================
        # Hip posture penalty
        # =========================
        hip_angles = self.dof_pos[:, self._hip_dof_idx]                     # [N, H]
        hip_default = self.default_joint_pos[self._hip_dof_idx]             # [H]
        hip_deviation = hip_angles - hip_default.unsqueeze(0)               # [N, H]
        hip_posture_cost = (hip_deviation ** 2).mean(dim=1)                 # [N]
        hip_posture_penalty = self.cfg.hip_posture_scale * hip_posture_cost # [N]

        # =========================
        # Continuous stance symmetry penalty (anti single-leg dominance)
        # =========================
        total_ct = contact_time.sum(dim=1, keepdim=True) + 1e-6      # [N, 1]
        ct_frac = contact_time / total_ct                            # [N, 2]
        ct_asym = torch.abs(ct_frac[:, 0] - ct_frac[:, 1])           # [N]
        symmetry_penalty = self.cfg.sym_scale * ct_asym * moving_mask  # [N]

        # =========================
        # Smooth hop / bouncing penalty (vertical COM velocity)
        # =========================
        v_z = self.velocity[:, 2]                                    # [N]
        excess_vz = torch.clamp(torch.abs(v_z) - 0.2, min=0.0)       # small dead zone
        hop_penalty = self.cfg.hop_penalty_scale * (excess_vz ** 2)  # quadratic cost

        # =========================
        # NEW: torso yaw alignment with direction-of-travel / target
        # =========================
        # yaw and angle_to_target were computed in _compute_intermediate_values
        yaw = self.yaw
        angle_to_target = self.angle_to_target
        yaw_error = normalize_angle(yaw - angle_to_target)           # [N], small when facing target
        yaw_penalty = self.cfg.yaw_penalty_scale * (yaw_error ** 2)

        # =========================
        # NEW: lateral CoM velocity penalty (discourage side-stepping)
        # =========================
        lat_vel = torch.abs(self.velocity[:, 0])                     # x-direction speed
        lateral_penalty = self.cfg.lateral_vel_scale * lat_vel

        total_reward = (
            base_reward
            + feet_air_time_reward
            - feet_slide_penalty
            - hip_posture_penalty
            - leg_heading_penalty
            - symmetry_penalty
            - hop_penalty
            - yaw_penalty
            - lateral_penalty
        )

        return total_reward






    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.torso_position[:, 2] < self.cfg.termination_height
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset actions for those envs to zero velocity
        self.actions[env_ids] = 0.0

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()

@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    motor_effort_ratio: torch.Tensor,
    velocity: torch.Tensor,
    heading_vec: torch.Tensor,
    orient_vel_weight: float,
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    # continuous upright reward; up_proj ≈ cos(angle between torso-up and world-up)
    up_clamped = torch.clamp(up_proj, min=0.0)
    up_reward = up_weight * up_clamped


    # energy penalty for movement
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    v_xy = velocity[:, :2]
    h_xy = heading_vec[:, :2]

    v_norm = torch.norm(v_xy, dim=-1) + 1e-6
    h_norm = torch.norm(h_xy, dim=-1) + 1e-6

    cos_vel_heading = torch.sum(v_xy * h_xy, dim=-1) / (v_norm * h_norm)

    # only care when actually moving
    move_mask = (v_norm > 0.1).float()
    orient_vel_reward = orient_vel_weight * cos_vel_heading * move_mask

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        + orient_vel_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )
    # adjust reward for fallen agents
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    return total_reward


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

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
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