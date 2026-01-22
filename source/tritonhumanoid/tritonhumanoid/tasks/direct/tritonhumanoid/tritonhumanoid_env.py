# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) v by the inverse of quaternion(s) q.
    q: [N,4] in (w,x,y,z)
    v: [N,3]
    returns: [N,3]
    """
    # inverse rotation by q is rotation by conjugate(q)
    w = q[:, 0:1]
    q_xyz = q[:, 1:4]
    # conjugate: (w, -x, -y, -z)
    q_xyz = -q_xyz

    # t = 2 * cross(q_xyz, v)
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    # v' = v + w*t + cross(q_xyz, t)
    return v + w * t + torch.cross(q_xyz, t, dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) v by quaternion(s) q.
    q: [N,4] in (w,x,y,z)
    v: [N,3]
    returns: [N,3]
    """
    w = q[:, 0:1]
    q_xyz = q[:, 1:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + w * t + torch.cross(q_xyz, t, dim=-1)


class LocomotionEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        obs_stack_frames = max(1, int(getattr(cfg, "obs_stack_frames", 1)))
        obs_single_dim = 1 + 3 + 3 + 3 + 3 + cfg.action_space * 3
        if cfg.use_phase_obs:
            obs_single_dim += 2
        cfg.observation_space_single = obs_single_dim
        cfg.obs_stack_frames = obs_stack_frames
        cfg.observation_space = obs_single_dim * obs_stack_frames

        super().__init__(cfg, render_mode, **kwargs)

        # actions are POSITION OFFSETS for actuated joints
        self.action_scale = self.cfg.action_scale

        actuated_joint_regex = (
            "left_hip1_joint|left_hip2_joint|left_thigh_joint|left_knee_joint|left_ankle_joint|"
            "right_hip1_joint|right_hip2_joint|right_thigh_joint|right_knee_joint|right_ankle_joint"
        )
        self._joint_dof_idx, _ = self.robot.find_joints(actuated_joint_regex)
        self.num_actions = len(self._joint_dof_idx)

        # Left/right joint pairs for symmetry penalty (action indices)
        joint_id_to_action = {int(jid): i for i, jid in enumerate(self._joint_dof_idx)}

        action_scale_by_joint = getattr(self.cfg, "action_scale_by_joint", {})
        self._action_scale_per_joint = torch.ones(self.num_actions, device=self.sim.device)
        for joint_name, scale in action_scale_by_joint.items():
            joint_ids, _ = self.robot.find_joints(joint_name)
            if len(joint_ids) == 0:
                continue
            joint_id = int(joint_ids[0])
            if joint_id not in joint_id_to_action:
                continue
            self._action_scale_per_joint[joint_id_to_action[joint_id]] = float(scale)
        sym_pairs = [
            ("left_hip1_joint", "right_hip1_joint"),
            ("left_hip2_joint", "right_hip2_joint"),
            ("left_thigh_joint", "right_thigh_joint"),
            ("left_knee_joint", "right_knee_joint"),
            ("left_ankle_joint", "right_ankle_joint"),
        ]
        sym_left = []
        sym_right = []
        for left_name, right_name in sym_pairs:
            left_ids, _ = self.robot.find_joints(left_name)
            right_ids, _ = self.robot.find_joints(right_name)
            if len(left_ids) == 0 or len(right_ids) == 0:
                continue
            left_id = int(left_ids[0])
            right_id = int(right_ids[0])
            if left_id not in joint_id_to_action or right_id not in joint_id_to_action:
                continue
            sym_left.append(joint_id_to_action[left_id])
            sym_right.append(joint_id_to_action[right_id])
        self._sym_left_action_ids = torch.tensor(sym_left, device=self.sim.device, dtype=torch.long)
        self._sym_right_action_ids = torch.tensor(sym_right, device=self.sim.device, dtype=torch.long)

        thigh_left_ids, _ = self.robot.find_joints("left_thigh_joint")
        thigh_right_ids, _ = self.robot.find_joints("right_thigh_joint")
        thigh_action_ids = []
        if len(thigh_left_ids) > 0:
            left_thigh_id = int(thigh_left_ids[0])
            if left_thigh_id in joint_id_to_action:
                thigh_action_ids.append(joint_id_to_action[left_thigh_id])
        if len(thigh_right_ids) > 0:
            right_thigh_id = int(thigh_right_ids[0])
            if right_thigh_id in joint_id_to_action:
                thigh_action_ids.append(joint_id_to_action[right_thigh_id])
        self._thigh_action_ids = torch.tensor(thigh_action_ids, device=self.sim.device, dtype=torch.long)

        # torso link is named "world" in your URDF
        torso_ids, _ = self.robot.find_bodies("world")
        self._torso_body_idx = int(torso_ids[0])

        # pre-create world up for speed (avoid allocating every step)
        self._world_up = torch.tensor([0.0, 0.0, 1.0], device=self.sim.device).unsqueeze(0).repeat(self.num_envs, 1)

        # default pose + joint limits (actuated only)
        default_joint_pos = self.robot.data.default_joint_pos[0]
        self.default_actuated_pos = default_joint_pos[self._joint_dof_idx].clone()

        self.actuated_lower = self.robot.data.soft_joint_pos_limits[0, self._joint_dof_idx, 0].clone()
        self.actuated_upper = self.robot.data.soft_joint_pos_limits[0, self._joint_dof_idx, 1].clone()

        # buffers
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.q_des = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

        # control timestep for smoothness costs
        self._control_dt = float(self.cfg.sim.dt * self.cfg.decimation)

        # previous joint velocities for acceleration cost
        self.prev_act_vel = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

        # commanded base velocity in BODY frame: [vx, vy, yaw_rate]
        self.commands = torch.zeros(self.num_envs, 3, device=self.sim.device)

        # rotate body-frame vectors to align forward axis with command frame
        self._command_yaw_offset = float(getattr(self.cfg, "command_yaw_offset", 0.0))
        self._cmd_yaw_cos = math.cos(self._command_yaw_offset)
        self._cmd_yaw_sin = math.sin(self._command_yaw_offset)
        self._cmd_yaw_inv_cos = self._cmd_yaw_cos
        self._cmd_yaw_inv_sin = -self._cmd_yaw_sin
        self._use_cmd_yaw_offset = abs(self._command_yaw_offset) > 1e-6

        # feet tracking flag
        self._feet_inited = False

        # --- Command curriculum tracking (global env-steps) ---
        self._global_env_steps = 0
        self._current_curriculum_stage = 0  # 0=forward, 1=+yaw, 2=+lateral
        self._curriculum_stage_names = ["forward-only", "forward+yaw", "forward+yaw+lateral"]

        # --- Visualization markers setup ---
        self._visualization_enabled = getattr(self.cfg, "debug_vel_vis", False)
        self._visualize_all_envs = False
        if self._visualization_enabled:
            self.visualization_markers = self.define_markers()
            self._marker_offset = torch.zeros(3, device=self.sim.device)
            self._marker_offset[2] = getattr(self.cfg, "vel_vis_height", 0.25)
            self._visualize_all_envs = bool(getattr(self.cfg, "debug_vel_vis_all_envs", False))
            if not self._visualize_all_envs:
                max_envs = int(getattr(self.cfg, "debug_vel_vis_all_envs_max_envs", 0))
                self._visualize_all_envs = max_envs > 0 and self.num_envs <= max_envs

        # observation stacking for memory
        self.obs_stack_frames = obs_stack_frames
        self._obs_single_dim = obs_single_dim
        if self.obs_stack_frames > 1:
            self.obs_stack_buf = torch.zeros(
                self.num_envs,
                self._obs_single_dim,
                self.obs_stack_frames,
                device=self.sim.device,
            )
        else:
            self.obs_stack_buf = None

        # randomized episode lengths
        self._randomize_episode_length = bool(getattr(self.cfg, "randomize_episode_length", False))
        self._min_episode_length_steps = max(
            1, int(self.cfg.min_episode_length_s / self._control_dt)
        )
        self.randomized_episode_lengths = torch.full(
            (self.num_envs,),
            self.max_episode_length,
            dtype=torch.long,
            device=self.sim.device,
        )
        self._resample_episode_lengths(self.robot._ALL_INDICES)

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
        self.prev_actions[:] = self.actions
        self.actions = actions.clone()

        # Update curriculum based on global env-steps
        self._global_env_steps += self.num_envs
        self._update_curriculum()

        if self._visualization_enabled:
            self._visualize_markers()

    def _apply_action(self):
        pos_offsets = self.action_scale * self._action_scale_per_joint.unsqueeze(0) * self.actions  # [N, 10]
        q_des = self.default_actuated_pos.unsqueeze(0) + pos_offsets
        q_des = torch.clamp(q_des, self.actuated_lower.unsqueeze(0), self.actuated_upper.unsqueeze(0))
        self.q_des = q_des
        self.robot.set_joint_position_target(q_des, joint_ids=self._joint_dof_idx)

    def _update_state(self):
        i = self._torso_body_idx

        # torso (world-frame)
        self.torso_pos_w  = self.robot.data.body_pos_w[:, i]     # [N,3]
        self.torso_quat_w = self.robot.data.body_quat_w[:, i]    # [N,4] (w,x,y,z)
        torso_lin_vel_w   = self.robot.data.body_lin_vel_w[:, i] # [N,3]
        torso_ang_vel_w   = self.robot.data.body_ang_vel_w[:, i] # [N,3]

        self.torso_lin_vel_w = torso_lin_vel_w
        self.torso_ang_vel_w = torso_ang_vel_w

        # convert velocities to torso/body frame
        self.torso_lin_vel_b = quat_rotate_inverse(self.torso_quat_w, torso_lin_vel_w)  # [N,3]
        self.torso_ang_vel_b = quat_rotate_inverse(self.torso_quat_w, torso_ang_vel_w)  # [N,3]
        if self._use_cmd_yaw_offset:
            self.torso_lin_vel_cmd = self._rotate_xy(self.torso_lin_vel_b, self._cmd_yaw_cos, self._cmd_yaw_sin)
            self.torso_ang_vel_cmd = self._rotate_xy(self.torso_ang_vel_b, self._cmd_yaw_cos, self._cmd_yaw_sin)
        else:
            self.torso_lin_vel_cmd = self.torso_lin_vel_b
            self.torso_ang_vel_cmd = self.torso_ang_vel_b

        # IMU-ish "up" expressed in body frame (up_b.z ~ 1 when upright)
        self.up_b = quat_rotate_inverse(self.torso_quat_w, self._world_up)  # [N,3]
        if self._use_cmd_yaw_offset:
            self.up_cmd = self._rotate_xy(self.up_b, self._cmd_yaw_cos, self._cmd_yaw_sin)
        else:
            self.up_cmd = self.up_b

        # joints
        self.dof_pos = self.robot.data.joint_pos
        self.dof_vel = self.robot.data.joint_vel

        self.act_pos = self.dof_pos[:, self._joint_dof_idx]
        self.act_vel = self.dof_vel[:, self._joint_dof_idx]

        lo = self.actuated_lower.unsqueeze(0)
        hi = self.actuated_upper.unsqueeze(0)
        self.act_pos_scaled = 2.0 * (self.act_pos - lo) / (hi - lo + 1e-6) - 1.0

    @staticmethod
    def define_markers() -> VisualizationMarkers:
        """Define arrow markers for command vs actual velocity."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/humanoidMarkers",
            markers={
                # 0: command (red)
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                # 1: actual velocity (green)
                "velocity": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
            },
        )
        return VisualizationMarkers(cfg=marker_cfg)

    @staticmethod
    def _rotate_xy(v: torch.Tensor, cos_yaw: float, sin_yaw: float) -> torch.Tensor:
        x = v[..., 0] * cos_yaw - v[..., 1] * sin_yaw
        y = v[..., 0] * sin_yaw + v[..., 1] * cos_yaw
        return torch.stack((x, y, v[..., 2]), dim=-1)

    def _visualize_markers(self):
        """Draw arrows for commanded velocity (red) and actual velocity (green) in world frame for single env."""
        if not self._visualization_enabled:
            return

        # Ensure state is updated before accessing torso_pos_w
        if not hasattr(self, 'torso_pos_w'):
            return

        if self._visualize_all_envs:
            env_ids = torch.arange(self.num_envs, device=self.sim.device)
        else:
            env_id = int(getattr(self.cfg, "debug_env_id", 0))
            env_id = max(0, min(env_id, self.num_envs - 1))
            env_ids = torch.tensor([env_id], device=self.sim.device)

        # Get single env data
        torso_pos = self.torso_pos_w[env_ids]  # [N,3]
        torso_quat = self.torso_quat_w[env_ids]  # [N,4]
        cmd = self.commands[env_ids]  # [N,3]
        vel_w = self.torso_lin_vel_w[env_ids]  # [N,3]

        # Marker location (lifted above robot)
        marker_loc = torso_pos + self._marker_offset  # [N,3]

        # --- 1. COMMAND VELOCITY DIRECTION IN WORLD FRAME ---
        cmd_b = torch.cat([cmd[:, :2], torch.zeros((cmd.shape[0], 1), device=self.sim.device)], dim=1)  # [N,3]
        if self._use_cmd_yaw_offset:
            cmd_b = self._rotate_xy(cmd_b, self._cmd_yaw_inv_cos, self._cmd_yaw_inv_sin)

        # Rotate into WORLD frame
        cmd_w = math_utils.quat_apply(torso_quat, cmd_b)  # [N,3]

        # Compute yaw for command
        cmd_xy = cmd_w[:, :2]
        cmd_norm = torch.norm(cmd_xy, dim=1).clamp(min=1e-6)
        cmd_dir_xy = cmd_xy / cmd_norm.unsqueeze(1)
        cmd_yaw = torch.atan2(cmd_dir_xy[:, 1], cmd_dir_xy[:, 0])

        cmd_half = 0.5 * cmd_yaw
        cmd_orient = torch.stack(
            (torch.cos(cmd_half), torch.zeros_like(cmd_half), torch.zeros_like(cmd_half), torch.sin(cmd_half)),
            dim=1,
        )  # [N,4]

        # --- 2. ACTUAL VELOCITY DIRECTION IN WORLD FRAME ---
        vel_xy = vel_w[:, :2]
        vel_norm = torch.norm(vel_xy, dim=1).clamp(min=1e-6)
        vel_dir_xy = vel_xy / vel_norm.unsqueeze(1)
        vel_yaw = torch.atan2(vel_dir_xy[:, 1], vel_dir_xy[:, 0])

        vel_half = 0.5 * vel_yaw
        vel_orient = torch.stack(
            (torch.cos(vel_half), torch.zeros_like(vel_half), torch.zeros_like(vel_half), torch.sin(vel_half)),
            dim=1,
        )  # [N,4]

        # --- 3. Visualize 2 markers: command (red) then velocity (green) ---
        loc = torch.repeat_interleave(marker_loc, repeats=2, dim=0)  # [2N,3]
        rots = torch.stack([cmd_orient, vel_orient], dim=1).reshape(-1, 4)  # [2N,4]
        indices = torch.tensor([0, 1], device=self.sim.device).repeat(marker_loc.shape[0])  # [2N]

        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _compute_single_observation(self) -> torch.Tensor:
        obs = torch.cat(
            (
                self.torso_pos_w[:, 2:3],                        # height
                self.torso_lin_vel_cmd,                          # command-frame lin vel
                self.torso_ang_vel_cmd * self.cfg.ang_vel_scale, # command-frame ang vel
                self.up_cmd,                                     # IMU-ish orientation feature
                self.commands,                                   # commanded [vx, vy, yaw_rate] in BODY frame
                self.act_pos_scaled,
                self.act_vel * self.cfg.dof_vel_scale,
                self.prev_actions,
            ),
            dim=-1,
        )

        if self.cfg.use_phase_obs:
            t = self.episode_length_buf.float() * self._control_dt
            phase = 2.0 * torch.pi * (t / self.cfg.gait_period_s)
            clock = torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)
            obs = torch.cat([obs, clock], dim=-1)

        return obs

    def _get_observations(self) -> dict:
        self._update_state()
        obs = self._compute_single_observation()

        if self.obs_stack_buf is not None:
            self.obs_stack_buf = torch.roll(self.obs_stack_buf, shifts=-1, dims=2)
            self.obs_stack_buf[:, :, -1] = obs
            obs = self.obs_stack_buf.reshape(self.num_envs, -1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._update_state()

        # --- Base velocity tracking ---
        vel_err = self.torso_lin_vel_cmd[:, :2] - self.commands[:, :2]
        yaw_err = self.torso_ang_vel_cmd[:, 2]  - self.commands[:, 2]

        r_lin = torch.exp(-torch.sum(vel_err * vel_err, dim=1) / self.cfg.lin_vel_sigma)
        r_yaw = torch.exp(-(yaw_err * yaw_err) / self.cfg.yaw_rate_sigma)

        upright = torch.clamp(self.up_b[:, 2], 0.0, 1.0)

        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        act_speed = torch.norm(self.torso_lin_vel_b[:, :2], dim=1)
        standstill = torch.clamp(self.cfg.standstill_speed_threshold - act_speed, min=0.0)
        standstill = standstill * (cmd_speed > self.cfg.command_speed_threshold).float()

        act_cost = torch.sum(self.actions * self.actions, dim=1)
        at_limit = torch.sum(torch.abs(self.act_pos_scaled) > 0.98, dim=1).float()

        pose_err = self.act_pos - self.default_actuated_pos.unsqueeze(0)
        pose_pen = (pose_err * pose_err).mean(dim=1)
        pose_return_penalty = torch.where(
            self.up_b[:, 2] > self.cfg.pose_return_upright_threshold,
            self.cfg.pose_return_scale * pose_pen,
            torch.zeros_like(pose_pen),
        )

        sym_pen = torch.zeros_like(pose_pen)
        if self._sym_left_action_ids.numel() > 0:
            left_off = self.act_pos[:, self._sym_left_action_ids] - self.default_actuated_pos[self._sym_left_action_ids].unsqueeze(0)
            right_off = self.act_pos[:, self._sym_right_action_ids] - self.default_actuated_pos[self._sym_right_action_ids].unsqueeze(0)
            sym_pen = torch.mean((torch.abs(left_off) - torch.abs(right_off)) ** 2, dim=1)

        thigh_pose_pen = torch.zeros_like(pose_pen)
        if self._thigh_action_ids.numel() > 0:
            thigh_off = self.act_pos[:, self._thigh_action_ids] - self.default_actuated_pos[self._thigh_action_ids].unsqueeze(0)
            thigh_pose_pen = torch.mean(thigh_off * thigh_off, dim=1)
            thigh_pose_pen = torch.where(
                self.up_b[:, 2] > self.cfg.pose_return_upright_threshold,
                thigh_pose_pen,
                torch.zeros_like(thigh_pose_pen),
            )

        # --- Stability costs (prevent hopping/rolling) ---
        lin_vel_z_cost = self.torso_lin_vel_b[:, 2] ** 2
        ang_vel_xy_cost = torch.sum(self.torso_ang_vel_b[:, :2] ** 2, dim=1)
        flat_ori_cost = torch.sum(self.up_b[:, :2] ** 2, dim=1)  # up_b ~= [0,0,1] when upright

        # --- Smoothness costs ---
        action_rate_cost = torch.sum((self.actions - self.prev_actions) ** 2, dim=1)
        dof_vel_cost = torch.sum(self.act_vel ** 2, dim=1)
        
        # Penalize velocity changes (not acceleration, to avoid dt sensitivity)
        dof_vel_delta = self.act_vel - self.prev_act_vel
        dof_vel_delta_cost = torch.sum(dof_vel_delta * dof_vel_delta, dim=1)

        self.prev_act_vel[:] = self.act_vel

        # --- Contact-based rewards (air-time, slip, undesired contacts) ---
        self._init_feet()

        # Latest normal forces (history index 0 is most recent)
        forces_hist = self._contact_sensor.data.net_forces_w_history  # (N,T,B,3)
        foot_forces = forces_hist[:, 0, self._feet_sensor_ids, :]     # (N,2,3)
        
        # Use vertical force component for contact detection (more robust than magnitude)
        foot_force_z = torch.abs(foot_forces[:, :, 2])  # (N,2)
        foot_contact = foot_force_z > self.cfg.foot_contact_force_thresh  # (N,2)

        # (1) air-time reward at touchdown
        air_time = self._contact_sensor.data.current_air_time[:, self._feet_sensor_ids]  # (N,2)
        touchdown = foot_contact & (~self.prev_foot_contact)
        air_rew = torch.sum(torch.clamp(air_time - self.cfg.min_air_time, min=0.0) * touchdown.float(), dim=1)

        # (2) slip penalty when in contact (horizontal foot speed)
        feet_vel_w = self.robot.data.body_lin_vel_w[:, self._feet_body_ids, :]  # (N,2,3)
        slip_speed_sq = torch.sum(feet_vel_w[..., :2] ** 2, dim=-1)            # (N,2)
        slip_cost = torch.sum(slip_speed_sq * foot_contact.float(), dim=1)     # (N,)

        # (3) penalize "non-foot contacts" (knees/shins/torso scraping)
        all_forces = forces_hist[:, 0, :, :]                 # (N,B,3)
        all_mag = torch.linalg.norm(all_forces, dim=-1)      # (N,B)
        all_mag[:, self._feet_sensor_ids] = 0.0
        undesired = (all_mag > self.cfg.undesired_contact_force_thresh).any(dim=1).float()

        self.prev_foot_contact[:] = foot_contact

        # --- Swing-phase gating for energy/smoothness costs ---
        if self.cfg.gate_smoothness_to_swing:
            swing_frac = (~foot_contact).float().mean(dim=1)
            swing_gate = (1.0 - self.cfg.swing_gate_alpha) + self.cfg.swing_gate_alpha * swing_frac
            act_cost = act_cost * swing_gate
            action_rate_cost = action_rate_cost * swing_gate
            dof_vel_cost = dof_vel_cost * swing_gate
            dof_vel_delta_cost = dof_vel_delta_cost * swing_gate

        # --- Combine all rewards ---
        reward = (
            self.cfg.lin_vel_reward_scale * r_lin
            + self.cfg.yaw_rate_reward_scale * r_yaw
            + self.cfg.upright_reward_scale * upright
            + self.cfg.alive_reward
            - self.cfg.action_cost_scale * act_cost
            - self.cfg.joint_limit_cost_scale * at_limit
            - pose_return_penalty
            - self.cfg.lin_vel_z_cost_scale * lin_vel_z_cost
            - self.cfg.ang_vel_xy_cost_scale * ang_vel_xy_cost
            - self.cfg.flat_ori_cost_scale * flat_ori_cost
            - self.cfg.action_rate_cost_scale * action_rate_cost
            - self.cfg.dof_vel_cost_scale * dof_vel_cost
            - self.cfg.dof_vel_delta_cost_scale * dof_vel_delta_cost
            - self.cfg.standstill_penalty_scale * standstill
            - self.cfg.symmetry_cost_scale * sym_pen
            - self.cfg.thigh_pose_cost_scale * thigh_pose_pen
            + self.cfg.feet_air_time_reward_scale * air_rew
            - self.cfg.foot_slip_cost_scale * slip_cost
            - self.cfg.undesired_contact_cost_scale * undesired
        )

        reward = torch.where(self.reset_terminated, torch.ones_like(reward) * self.cfg.death_cost, reward)

        # --- Logging (every N steps) ---
        if hasattr(self, 'common_step_counter') and (self.common_step_counter % 200) == 0:
            # Tracking rewards (what we want to maximize)
            self.extras["reward_tracking/r_lin"] = float(r_lin.mean().item())
            self.extras["reward_tracking/r_yaw"] = float(r_yaw.mean().item())
            self.extras["reward_tracking/upright"] = float(upright.mean().item())
            self.extras["reward_tracking/alive"] = float(self.cfg.alive_reward)
            self.extras["reward_tracking/air_time"] = float(air_rew.mean().item())

            # Penalties (what we want to minimize)
            self.extras["reward_penalties/action_cost"] = float(act_cost.mean().item())
            self.extras["reward_penalties/joint_limit"] = float(at_limit.mean().item())
            self.extras["reward_penalties/pose_return"] = float(pose_return_penalty.mean().item())
            self.extras["reward_penalties/lin_vel_z"] = float(lin_vel_z_cost.mean().item())
            self.extras["reward_penalties/ang_vel_xy"] = float(ang_vel_xy_cost.mean().item())
            self.extras["reward_penalties/flat_ori"] = float(flat_ori_cost.mean().item())
            self.extras["reward_penalties/action_rate"] = float(action_rate_cost.mean().item())
            self.extras["reward_penalties/dof_vel"] = float(dof_vel_cost.mean().item())
            self.extras["reward_penalties/dof_vel_delta"] = float(dof_vel_delta_cost.mean().item())
            self.extras["reward_penalties/standstill"] = float(standstill.mean().item())
            self.extras["reward_penalties/symmetry"] = float(sym_pen.mean().item())
            self.extras["reward_penalties/thigh_pose"] = float(thigh_pose_pen.mean().item())
            self.extras["reward_penalties/slip"] = float(slip_cost.mean().item())
            self.extras["reward_penalties/undesired_contact"] = float(undesired.mean().item())

            # Scaled contributions (actual impact on total reward)
            self.extras["reward_scaled/lin_tracking"] = float((self.cfg.lin_vel_reward_scale * r_lin).mean().item())
            self.extras["reward_scaled/yaw_tracking"] = float((self.cfg.yaw_rate_reward_scale * r_yaw).mean().item())
            self.extras["reward_scaled/upright"] = float((self.cfg.upright_reward_scale * upright).mean().item())
            self.extras["reward_scaled/pose_return"] = float(pose_return_penalty.mean().item())
            self.extras["reward_scaled/air_time"] = float((self.cfg.feet_air_time_reward_scale * air_rew).mean().item())
            self.extras["reward_scaled/slip_cost"] = float((self.cfg.foot_slip_cost_scale * slip_cost).mean().item())
            self.extras["reward_scaled/undesired_cost"] = float((self.cfg.undesired_contact_cost_scale * undesired).mean().item())
            self.extras["reward_scaled/standstill_cost"] = float((self.cfg.standstill_penalty_scale * standstill).mean().item())
            self.extras["reward_scaled/dof_vel_delta_cost"] = float((self.cfg.dof_vel_delta_cost_scale * dof_vel_delta_cost).mean().item())
            self.extras["reward_scaled/symmetry_cost"] = float((self.cfg.symmetry_cost_scale * sym_pen).mean().item())
            self.extras["reward_scaled/thigh_pose_cost"] = float((self.cfg.thigh_pose_cost_scale * thigh_pose_pen).mean().item())

            # Total reward stats
            self.extras["reward_total/mean"] = float(reward.mean().item())
            self.extras["reward_total/std"] = float(reward.std().item())
            self.extras["reward_total/min"] = float(reward.min().item())
            self.extras["reward_total/max"] = float(reward.max().item())

            # Command tracking errors (diagnostics)
            self.extras["diagnostics/vel_err_x"] = float(vel_err[:, 0].abs().mean().item())
            self.extras["diagnostics/vel_err_y"] = float(vel_err[:, 1].abs().mean().item())
            self.extras["diagnostics/yaw_err"] = float(yaw_err.abs().mean().item())
            self.extras["diagnostics/contact_rate"] = float(foot_contact.float().mean().item())
            self.extras["diagnostics/avg_air_time"] = float(air_time.mean().item())
            self.extras["diagnostics/curriculum_stage"] = float(self._current_curriculum_stage)

        return reward

    def _get_dones(self):
        self._update_state()
        time_out = self.episode_length_buf >= self.randomized_episode_lengths - 1

        fell = self.torso_pos_w[:, 2] < self.cfg.termination_height
        too_tilted = self.up_b[:, 2] < self.cfg.upright_threshold

        died = fell | too_tilted
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._resample_episode_lengths(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        root = self.robot.data.default_root_state[env_ids]
        root[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        # Sample commands based on current curriculum stage
        self._sample_commands(env_ids)

        if self._visualization_enabled or self.obs_stack_buf is not None:
            self._update_state()

        # Update state before visualization so torso_pos_w exists
        if self._visualization_enabled:
            self._visualize_markers()

        if self.obs_stack_buf is not None:
            obs0 = self._compute_single_observation()
            self.obs_stack_buf[env_ids] = obs0[env_ids].unsqueeze(-1).expand(-1, -1, self.obs_stack_frames)

    def _update_curriculum(self):
        """Check if we should advance to the next curriculum stage based on per-env steps."""
        if not self.cfg.use_curriculum:
            return

        old_stage = self._current_curriculum_stage

        # Calculate average steps per environment
        per_env_steps = self._global_env_steps / float(self.num_envs)

        # Stage progression based on per-env steps (independent of num_envs)
        if per_env_steps >= self.cfg.curriculum_stage2_steps_per_env:
            self._current_curriculum_stage = 2  # full: vx, vy, yaw
        elif per_env_steps >= self.cfg.curriculum_stage1_steps_per_env:
            self._current_curriculum_stage = 1  # vx + yaw
        else:
            self._current_curriculum_stage = 0  # vx only

        # Print when we advance
        if self._current_curriculum_stage != old_stage:
            stage_name = self._curriculum_stage_names[self._current_curriculum_stage]
            print(f"\n{'='*60}")
            print(f"[Command Curriculum] Advanced to Stage {self._current_curriculum_stage}: {stage_name}")
            print(f"  Global env-steps: {self._global_env_steps}")
            print(f"  Per-env steps: {per_env_steps:.1f}")
            print(f"{'='*60}\n")

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample velocity commands based on current curriculum stage."""
        n = len(env_ids)

        if not self.cfg.use_curriculum:
            # No curriculum: sample full ranges
            self.commands[env_ids, 0] = torch.empty(n, device=self.sim.device).uniform_(-1.0, 1.0)  # vx
            self.commands[env_ids, 1] = torch.empty(n, device=self.sim.device).uniform_(-0.5, 0.5)  # vy
            self.commands[env_ids, 2] = torch.empty(n, device=self.sim.device).uniform_(-1.0, 1.0)  # yaw rate
            return

        # Curriculum active: sample based on stage
        stage = self._current_curriculum_stage

        if stage == 0:
            # Stage 0: forward-only (vx in [0.3, 1.0], no yaw or lateral)
            # Encourage actual forward motion, not standing still
            vx_min = self.cfg.curriculum_stage0_vx_min
            vx_max = self.cfg.curriculum_stage0_vx_max
            self.commands[env_ids, 0] = torch.empty(n, device=self.sim.device).uniform_(vx_min, vx_max)
            self.commands[env_ids, 1] = 0.0
            self.commands[env_ids, 2] = 0.0

        elif stage == 1:
            # Stage 1: forward + yaw (vx in [-1, 1], yaw in [-1, 1], no lateral)
            self.commands[env_ids, 0] = torch.empty(n, device=self.sim.device).uniform_(-1.0, 1.0)
            self.commands[env_ids, 1] = 0.0
            self.commands[env_ids, 2] = torch.empty(n, device=self.sim.device).uniform_(-1.0, 1.0)

        else:  # stage == 2
            # Stage 2: full command space
            self.commands[env_ids, 0] = torch.empty(n, device=self.sim.device).uniform_(-1.0, 1.0)
            self.commands[env_ids, 1] = torch.empty(n, device=self.sim.device).uniform_(-0.5, 0.5)
            self.commands[env_ids, 2] = torch.empty(n, device=self.sim.device).uniform_(-1.0, 1.0)

    def _init_feet(self):
        """Initialize foot body and contact sensor indices (call once before using contact rewards)."""
        if self._feet_inited:
            return

        # 1) feet kinematics indices (for slip / clearance)
        foot_body_ids, foot_body_names = self.robot.find_bodies(self.cfg.foot_body_regex)
        if len(foot_body_ids) != 2:
            raise RuntimeError(f"Expected 2 feet from regex {self.cfg.foot_body_regex}, got {foot_body_names}")
        self._feet_body_ids = torch.tensor(foot_body_ids, device=self.sim.device, dtype=torch.long)

        # 2) feet contact-sensor indices (for contact forces / air time)
        sensor_names = self._contact_sensor.body_names  # list[str]
        foot_sensor_ids = []
        for n in foot_body_names:
            match = next((i for i, sn in enumerate(sensor_names) if (sn.endswith(n) or (n in sn))), None)
            if match is None:
                raise RuntimeError(f"Foot body '{n}' not found in contact_sensor.body_names.")
            foot_sensor_ids.append(match)

        self._feet_sensor_ids = torch.tensor(foot_sensor_ids, device=self.sim.device, dtype=torch.long)
        self.prev_foot_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.sim.device)
        self._feet_inited = True

    def _resample_episode_lengths(self, env_ids: torch.Tensor):
        if not self._randomize_episode_length:
            self.randomized_episode_lengths[env_ids] = self.max_episode_length
            return

        min_steps = min(self._min_episode_length_steps, self.max_episode_length)
        self.randomized_episode_lengths[env_ids] = torch.randint(
            min_steps,
            self.max_episode_length + 1,
            (env_ids.numel(),),
            dtype=torch.long,
            device=self.sim.device,
        )
