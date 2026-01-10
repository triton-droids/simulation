# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from copy import deepcopy
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


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_tuple(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (_lerp(a[0], b[0], t), _lerp(a[1], b[1], t))


class StandingADR:
    """
    Minimal ADR controller:
      - EventManager-backed: widens term cfg ranges via event_manager.get_term_cfg/set_term_cfg
      - Custom params: provides “current” values (push range, noise, latency, etc.)
    """

    def __init__(self, event_manager, adr_event_cfg_dict: dict, adr_custom_cfg_dict: dict,
                 num_increments: int):
        self.event_manager = event_manager
        self.adr_event_cfg_dict = deepcopy(adr_event_cfg_dict)
        self.adr_custom_cfg_dict = deepcopy(adr_custom_cfg_dict)
        self._num_increments_max = int(num_increments)
        self._num_increments_cur = 0

        # snapshot “base” event params (increment=0 endpoints)
        self._base_event_params: dict[str, dict[str, tuple[float, float]]] = {}
        for term_name, term_updates in self.adr_event_cfg_dict.items():
            term_cfg = self.event_manager.get_term_cfg(term_name)
            self._base_event_params[term_name] = {}
            for k in term_updates.keys():
                self._base_event_params[term_name][k] = deepcopy(term_cfg.params[k])

    def num_increments(self) -> int:
        return self._num_increments_cur

    def set_num_increments(self, n: int) -> None:
        self._num_increments_cur = int(max(0, min(self._num_increments_max, n)))
        self.apply_event_ranges()

    def difficulty(self) -> float:
        if self._num_increments_max <= 0:
            return 0.0
        return float(self._num_increments_cur) / float(self._num_increments_max)

    def increase(self, k: int = 1) -> None:
        self.set_num_increments(self._num_increments_cur + int(k))

    def decrease(self, k: int = 1) -> None:
        self.set_num_increments(self._num_increments_cur - int(k))

    def apply_event_ranges(self) -> None:
        """Update EventManager term cfgs according to current difficulty."""
        t = self.difficulty()

        for term_name, max_updates in self.adr_event_cfg_dict.items():
            term_cfg = self.event_manager.get_term_cfg(term_name)
            for param_name, max_range in max_updates.items():
                base_range = self._base_event_params[term_name][param_name]
                term_cfg.params[param_name] = _lerp_tuple(base_range, max_range, t)
            self.event_manager.set_term_cfg(term_name, term_cfg)

    def get_custom(self, group: str, key: str):
        """
        Returns "current" custom ADR value. Supports:
          - (min,max) tuples: lerp base->max
          - ((min0,max0),(min1,max1)) for push_force_range: lerp each endpoint tuple
        """
        t = self.difficulty()
        spec = self.adr_custom_cfg_dict[group][key]

        # numeric range (a, b)
        if isinstance(spec, (tuple, list)) and len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
            return _lerp(float(spec[0]), float(spec[1]), t)

        # nested endpoints: ((lo0, hi0), (lo1, hi1))  -> lerp each endpoint
        if (
            isinstance(spec, (tuple, list)) and len(spec) == 2
            and isinstance(spec[0], (tuple, list)) and isinstance(spec[1], (tuple, list))
            and len(spec[0]) == 2 and len(spec[1]) == 2
            and all(isinstance(x, (int, float)) for x in spec[0])
            and all(isinstance(x, (int, float)) for x in spec[1])
        ):
            return (
                _lerp(float(spec[0][0]), float(spec[1][0]), t),
                _lerp(float(spec[0][1]), float(spec[1][1]), t),
            )

        return spec

    def print_params(self) -> str:
        t = self.difficulty()
        lines = [f"[ADR] increments={self._num_increments_cur}/{self._num_increments_max}  difficulty={t:.3f}"]
        for term_name, max_updates in self.adr_event_cfg_dict.items():
            term_cfg = self.event_manager.get_term_cfg(term_name)
            for param_name in max_updates.keys():
                lines.append(f"  - {term_name}.{param_name} = {term_cfg.params[param_name]}")
        # common custom ones
        if "push" in self.adr_custom_cfg_dict:
            lines.append(f"  - push.push_force_range = {self.get_custom('push','push_force_range')}")
        if "action_noise" in self.adr_custom_cfg_dict:
            lines.append(f"  - action_noise.std = {self.get_custom('action_noise','std')}")
        if "latency" in self.adr_custom_cfg_dict:
            lines.append(f"  - latency.act_steps = {self.get_custom('latency','act_steps')}")
            lines.append(f"  - latency.obs_steps = {self.get_custom('latency','obs_steps')}")
        return "\n".join(lines)


class StandingEnv(DirectRLEnv):
    """Standing disturbance-rejection environment + ADR."""

    cfg: HumanoidEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = float(self.cfg.action_scale)

        actuated_joint_regex = (
            "left_hip1_joint|left_hip2_joint|left_thigh_joint|left_knee_joint|left_ankle_joint|"
            "right_hip1_joint|right_hip2_joint|right_thigh_joint|right_knee_joint|right_ankle_joint"
        )
        self._joint_dof_idx, _ = self.robot.find_joints(actuated_joint_regex)
        self._joint_dof_idx = torch.as_tensor(self._joint_dof_idx, device=self.sim.device, dtype=torch.long)
        self.num_actions = int(self._joint_dof_idx.numel())

        hip_joint_regex = "left_hip1_joint|left_hip2_joint|right_hip1_joint|right_hip2_joint"
        hip_dof_idx, _ = self.robot.find_joints(hip_joint_regex)
        self._hip_dof_idx = torch.as_tensor(hip_dof_idx, device=self.sim.device, dtype=torch.long)

        self.default_joint_pos_full = self.robot.data.default_joint_pos[0].clone()

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.filtered_actions = torch.zeros_like(self.actions)
        self.motor_effort_ratio = torch.ones(self.num_actions, dtype=torch.float32, device=self.sim.device)

        torso_body_indices, _ = self.robot.find_bodies("world")
        self._torso_body_idx = int(torso_body_indices[0])

        right_foot_body_indices, _ = self.robot.find_bodies("right_foot")
        self._right_foot_body_idx = int(right_foot_body_indices[0])

        left_foot_body_indices, _ = self.robot.find_bodies("left_foot")
        self._left_foot_body_idx = int(left_foot_body_indices[0])

        # feet indices for stance metrics (reward/termination only, not in obs)
        self._feet_ids, _ = self._contact_sensor.find_bodies("left_foot|right_foot")

        # compute default step width (for reward only)
        feet_pos_all = self.robot.data.body_pos_w[:]  # [N, B, 3]
        left_xy = feet_pos_all[:, self._left_foot_body_idx, :2]
        right_xy = feet_pos_all[:, self._right_foot_body_idx, :2]
        torso_xy = feet_pos_all[:, self._torso_body_idx, :2]
        left_lat = left_xy[:, 1] - torso_xy[:, 1]
        right_lat = right_xy[:, 1] - torso_xy[:, 1]
        self.default_step_width = torch.abs(left_lat - right_lat).clone()

        # orientation helpers (for reward/termination only)
        qz_minus_90 = torch.tensor(
            [math.cos(math.pi / 4.0), 0.0, 0.0, -math.sin(math.pi / 4.0)],
            device=self.sim.device,
            dtype=torch.float32,
        )
        self.start_rotation = qz_minus_90
        self.inv_start_rot = quat_conjugate(self.start_rotation).unsqueeze(0).repeat(self.num_envs, 1)

        self.basis_vec1 = torch.tensor([0.0, 0.0, 1.0], device=self.sim.device, dtype=torch.float32).repeat(self.num_envs, 1)
        self.basis_vec0 = torch.tensor([1.0, 0.0, 0.0], device=self.sim.device, dtype=torch.float32).repeat(self.num_envs, 1)

        self.targets = self.scene.env_origins + torch.tensor([1.0, 0.0, 0.0], device=self.sim.device)

        self.potentials = torch.zeros(self.num_envs, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

        # push timers
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        min_steps = max(1, int(self.cfg.min_push_interval_s / self.dt))
        max_steps = max(min_steps, int(self.cfg.max_push_interval_s / self.dt))
        self._push_interval_steps_min = min_steps
        self._push_interval_steps_max = max_steps
        self.push_counters = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim.device)
        self.next_push_steps = torch.randint(
            self._push_interval_steps_min,
            self._push_interval_steps_max + 1,
            (self.num_envs,),
            device=self.sim.device,
        )

        # buffers for intermediate values (reward computation)
        self.torso_position = torch.zeros(self.num_envs, 3, device=self.sim.device)
        self.torso_rotation = torch.zeros(self.num_envs, 4, device=self.sim.device)
        self.velocity = torch.zeros(self.num_envs, 3, device=self.sim.device)
        self.ang_velocity = torch.zeros(self.num_envs, 3, device=self.sim.device)

        self.dof_pos_full = torch.zeros(self.num_envs, self.robot.num_joints, device=self.sim.device)
        self.dof_vel_full = torch.zeros_like(self.dof_pos_full)

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

        self.dof_pos_scaled = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

        # cache flag for intermediate values
        self._intermediates_valid = False

        # -----------------------
        # ADR state + buffers
        # -----------------------
        self._global_policy_step = 0
        self._last_adr_update_step = 0
        self._last_adr_decrease_step = 0

        self.success_rate_ema = torch.zeros(1, device=self.sim.device)

        # allocate latency history buffers at max possible (from cfg)
        self.act_max_latency = int(getattr(self.cfg, "act_max_latency", 0))
        self.obs_max_latency = int(getattr(self.cfg, "obs_max_latency", 0))

        self.act_latency_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.sim.device)
        if self.act_max_latency > 0:
            self.act_hist_buf = torch.zeros(self.num_envs, self.num_actions, self.act_max_latency + 1, device=self.sim.device)
        else:
            self.act_hist_buf = None

        self.obs_latency_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.sim.device)
        
        # observation stacking for memory
        self.obs_stack_frames = int(getattr(self.cfg, "obs_stack_frames", 1))
        obs_single_dim = int(self.cfg.observation_space_single)  # dimension of single frame
        
        if self.obs_max_latency > 0:
            self.obs_hist_buf = torch.zeros(self.num_envs, obs_single_dim, self.obs_max_latency + 1, device=self.sim.device)
        else:
            self.obs_hist_buf = None
        
        # stacking buffer for multi-frame observations
        if self.obs_stack_frames > 1:
            self.obs_stack_buf = torch.zeros(self.num_envs, obs_single_dim, self.obs_stack_frames, device=self.sim.device)
        else:
            self.obs_stack_buf = None

        # per-env motor strength multipliers
        self.motor_strength_mult = torch.ones(self.num_envs, self.num_actions, device=self.sim.device)

        # IMU bias (gravity direction + gyro)
        self.imu_bias_gravity = torch.zeros(self.num_envs, 3, device=self.sim.device)
        self.imu_bias_gyro = torch.zeros(self.num_envs, 3, device=self.sim.device)

        # current ADR “custom” params (filled by _update_adr_custom_params)
        self._push_dv_min = float(self.cfg.push_force_range[0])
        self._push_dv_max = float(self.cfg.push_force_range[1])
        self._action_noise_std = 0.0
        self._obs_noise = {
            "gravity_std": 0.0,
            "gyro_std": 0.0,
            "joint_pos_std": 0.0,
            "joint_vel_std": 0.0,
            "joint_torque_std": 0.0,
        }

        # init ADR controller after event_manager exists
        self.adr = None
        if getattr(self.cfg, "enable_adr", False):
            self.adr = StandingADR(
                self.event_manager,
                self.cfg.adr_event_cfg_dict,
                self.cfg.adr_custom_cfg_dict,
                num_increments=self.cfg.num_adr_increments,
            )
            self.adr.set_num_increments(self.cfg.starting_adr_increments)
            self._update_adr_custom_params()
            if getattr(self.cfg, "adr_print_every_update", True):
                print(self.adr.print_params())

        # extras logging
        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

    # ----------------------------------------------------------------------
    # Scene setup
    # ----------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ----------------------------------------------------------------------
    # ADR helpers (custom params)
    # ----------------------------------------------------------------------
    def _update_adr_custom_params(self) -> None:
        if self.adr is None:
            self._push_dv_min = float(self.cfg.push_force_range[0])
            self._push_dv_max = float(self.cfg.push_force_range[1])
            self._action_noise_std = 0.0
            return

        # push dv range
        pr = self.adr.get_custom("push", "push_force_range")
        self._push_dv_min = float(pr[0])
        self._push_dv_max = float(pr[1])

        # action noise
        self._action_noise_std = float(self.adr.get_custom("action_noise", "std"))

        # obs noise
        for k in self._obs_noise.keys():
            self._obs_noise[k] = float(self.adr.get_custom("obs_noise", k))

    def _sample_custom_dr_for_resets(self, env_ids: torch.Tensor) -> None:
        """Sample per-episode random variables: latency, motor strength, imu bias."""
        if self.adr is None or env_ids.numel() == 0:
            return

        device = self.sim.device

        # motor strength multipliers per joint
        r = float(self.adr.get_custom("motor_strength", "per_joint_mult_range"))
        if r > 0.0:
            lo, hi = 1.0 - r, 1.0 + r
            self.motor_strength_mult[env_ids] = torch.empty((env_ids.numel(), self.num_actions), device=device).uniform_(lo, hi)
        else:
            self.motor_strength_mult[env_ids] = 1.0

        # action latency steps
        act_max = int(round(float(self.adr.get_custom("latency", "act_steps"))))
        act_max = int(max(0, min(self.act_max_latency, act_max)))
        if act_max > 0:
            self.act_latency_steps[env_ids] = torch.randint(0, act_max + 1, (env_ids.numel(),), device=device)
        else:
            self.act_latency_steps[env_ids] = 0

        # obs latency steps
        obs_max = int(round(float(self.adr.get_custom("latency", "obs_steps"))))
        obs_max = int(max(0, min(self.obs_max_latency, obs_max)))
        if obs_max > 0:
            self.obs_latency_steps[env_ids] = torch.randint(0, obs_max + 1, (env_ids.numel(),), device=device)
        else:
            self.obs_latency_steps[env_ids] = 0

        # IMU bias on gravity direction (small constant offset per episode)
        b_grav = float(self.adr.get_custom("imu_bias", "gravity_bias_range"))
        if b_grav > 0.0:
            self.imu_bias_gravity[env_ids] = torch.empty((env_ids.numel(), 3), device=device).uniform_(-b_grav, b_grav)
        else:
            self.imu_bias_gravity[env_ids] = 0.0

        # IMU bias on gyro (small constant offset per episode)
        b_gyro = float(self.adr.get_custom("imu_bias", "gyro_bias_range"))
        if b_gyro > 0.0:
            self.imu_bias_gyro[env_ids] = torch.empty((env_ids.numel(), 3), device=device).uniform_(-b_gyro, b_gyro)
        else:
            self.imu_bias_gyro[env_ids] = 0.0

    # ----------------------------------------------------------------------
    # Disturbance logic
    # ----------------------------------------------------------------------
    def _maybe_apply_pushes(self):
        """Apply random horizontal velocity kicks to the base at random intervals."""
        if self._push_dv_max <= 0.0:
            return

        self.push_counters += 1
        ready = self.push_counters >= self.next_push_steps
        if not torch.any(ready):
            return

        env_ids = torch.nonzero(ready, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return

        K = env_ids.numel()
        device = self.sim.device

        dirs_xy = torch.randn((K, 2), device=device)
        dirs_xy /= (torch.norm(dirs_xy, dim=-1, keepdim=True) + 1e-6)

        mags = torch.empty((K, 1), device=device).uniform_(self._push_dv_min, self._push_dv_max)
        delta_v_xy = dirs_xy * mags

        root_vel = self.robot.data.root_vel_w[env_ids].clone()  # [K, 6]
        root_vel[:, 0:2] += delta_v_xy
        
        # add random yaw/roll disturbance (angular momentum from push)
        delta_w = torch.zeros((K, 3), device=device)
        delta_w[:, 2] = torch.empty((K,), device=device).uniform_(-0.5, 0.5)  # yaw impulse
        root_vel[:, 3:6] += delta_w
        
        self.robot.write_root_velocity_to_sim(root_vel, env_ids)

        self.push_counters[env_ids] = 0
        self.next_push_steps[env_ids] = torch.randint(
            self._push_interval_steps_min,
            self._push_interval_steps_max + 1,
            (K,),
            device=device,
        )

    # ----------------------------------------------------------------------
    # RL interface
    # ----------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._global_policy_step += 1
        
        # invalidate cached intermediate values
        self._intermediates_valid = False

        # clamp
        a = actions.clone().clamp(-1.0, 1.0)

        # action noise (ADR)
        if self._action_noise_std > 0.0:
            a = a + torch.randn_like(a) * self._action_noise_std
            a = a.clamp(-1.0, 1.0)

        # action latency (ADR)
        if self.act_hist_buf is not None:
            self.act_hist_buf = torch.roll(self.act_hist_buf, shifts=-1, dims=2)
            self.act_hist_buf[:, :, -1] = a

            # select delayed action: index = -1 - latency
            idx = (self.act_max_latency - self.act_latency_steps).clamp(0, self.act_max_latency)
            # gather per-env
            gather_idx = idx.view(-1, 1, 1).expand(-1, self.num_actions, 1)
            a = torch.gather(self.act_hist_buf, dim=2, index=gather_idx).squeeze(-1)

        # first-order actuator filter (motor dynamics)
        alpha = float(getattr(self.cfg, "action_filter_alpha", 0.2))
        self.filtered_actions = (1 - alpha) * self.filtered_actions + alpha * a
        a = self.filtered_actions

        # store for action rate penalty
        self.prev_actions = self.actions.clone()
        self.actions = a

        # scheduled pushes
        self._maybe_apply_pushes()

    def _apply_action(self):
        # Position control: residual around nominal standing pose
        delta_q = float(getattr(self.cfg, "residual_pos_scale", 0.25)) * self.actions
        q_tgt = self.default_joint_pos_full[self._joint_dof_idx].unsqueeze(0) + delta_q
        q_tgt = q_tgt * self.motor_strength_mult  # per-joint strength DR
        self.robot.set_joint_position_target(q_tgt, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        # skip if already computed this step
        if self._intermediates_valid:
            return
            
        torso = self._torso_body_idx

        self.torso_position = self.robot.data.body_pos_w[:, torso]
        self.torso_rotation = self.robot.data.body_quat_w[:, torso]
        self.velocity = self.robot.data.body_lin_vel_w[:, torso]
        self.ang_velocity = self.robot.data.body_ang_vel_w[:, torso]

        self.dof_pos_full = self.robot.data.joint_pos
        self.dof_vel_full = self.robot.data.joint_vel

        dof_pos = self.dof_pos_full[:, self._joint_dof_idx]
        dof_vel = self.dof_vel_full[:, self._joint_dof_idx]

        lower = self.robot.data.soft_joint_pos_limits[0, self._joint_dof_idx, 0]
        upper = self.robot.data.soft_joint_pos_limits[0, self._joint_dof_idx, 1]

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
            dof_pos,
            lower,
            upper,
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )

        # also keep dof_vel subset handy
        self._dof_vel_act = dof_vel

        # mark as valid
        self._intermediates_valid = True

    def _build_obs_no_latency(self) -> torch.Tensor:
        """Build observation without applying latency (hardware-only sensors: IMU + joints + torques)."""
        # assumes _compute_intermediate_values already called
        
        # === IMU: gravity direction in body frame ===
        # World gravity is [0, 0, -9.81]; rotate to body frame
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.sim.device, dtype=torch.float32)
        gravity_world = gravity_world.unsqueeze(0).expand(self.num_envs, -1)
        
        # Rotate by inverse of torso rotation to get gravity in body frame
        torso_quat = self.torso_rotation
        gravity_body = quat_rotate_inverse(torso_quat, gravity_world)
        gravity_body = gravity_body + self.imu_bias_gravity  # add bias
        
        # === IMU: gyroscope (angular velocity in body frame) ===
        gyro = self.angvel_loc + self.imu_bias_gyro  # already in body frame
        
        # === Joint states (actuated only) ===
        dof_pos = self.dof_pos_full[:, self._joint_dof_idx]
        dof_vel = self.dof_vel_full[:, self._joint_dof_idx]
        
        # scale joint positions
        lower = self.robot.data.soft_joint_pos_limits[0, self._joint_dof_idx, 0]
        upper = self.robot.data.soft_joint_pos_limits[0, self._joint_dof_idx, 1]
        dof_pos_scaled = torch_utils.maths.unscale(dof_pos, lower, upper)
        
        # === Joint torques (measured/commanded effort) ===
        joint_torques = self.robot.data.applied_torque[:, self._joint_dof_idx]
        
        # === Previous actions ===
        prev_actions = self.actions
        
        # Build observation vector
        obs = torch.cat(
            (
                gravity_body,                                      # 3
                gyro * self.cfg.angular_velocity_scale,            # 3
                dof_pos_scaled,                                    # num_actions
                dof_vel * self.cfg.dof_vel_scale,                  # num_actions
                joint_torques * self.cfg.torque_scale,             # num_actions
                prev_actions,                                      # num_actions
            ),
            dim=-1,
        )
        
        # Observation noise (ADR) - dynamic indexing to avoid hardcoded slices
        if self.adr is not None:
            offset = 0
            
            # gravity noise
            if self._obs_noise["gravity_std"] > 0.0:
                obs[:, offset:offset+3] += torch.randn_like(obs[:, offset:offset+3]) * self._obs_noise["gravity_std"]
            offset += 3
            
            # gyro noise
            if self._obs_noise["gyro_std"] > 0.0:
                obs[:, offset:offset+3] += torch.randn_like(obs[:, offset:offset+3]) * self._obs_noise["gyro_std"]
            offset += 3
            
            # joint pos noise
            if self._obs_noise["joint_pos_std"] > 0.0:
                obs[:, offset:offset+self.num_actions] += torch.randn_like(obs[:, offset:offset+self.num_actions]) * self._obs_noise["joint_pos_std"]
            offset += self.num_actions
            
            # joint vel noise
            if self._obs_noise["joint_vel_std"] > 0.0:
                obs[:, offset:offset+self.num_actions] += torch.randn_like(obs[:, offset:offset+self.num_actions]) * self._obs_noise["joint_vel_std"]
            offset += self.num_actions
            
            # joint torque noise
            if self._obs_noise["joint_torque_std"] > 0.0:
                obs[:, offset:offset+self.num_actions] += torch.randn_like(obs[:, offset:offset+self.num_actions]) * self._obs_noise["joint_torque_std"]
            offset += self.num_actions
            
            # no noise on prev_actions

        return obs

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()  # only computes once per step now

        obs = self._build_obs_no_latency()

        # obs latency (ADR)
        if self.obs_hist_buf is not None:
            self.obs_hist_buf = torch.roll(self.obs_hist_buf, shifts=-1, dims=2)
            self.obs_hist_buf[:, :, -1] = obs

            idx = (self.obs_max_latency - self.obs_latency_steps).clamp(0, self.obs_max_latency)
            gather_idx = idx.view(-1, 1, 1).expand(-1, obs.shape[1], 1)
            obs = torch.gather(self.obs_hist_buf, dim=2, index=gather_idx).squeeze(-1)

        # observation stacking for memory
        if self.obs_stack_buf is not None:
            self.obs_stack_buf = torch.roll(self.obs_stack_buf, shifts=-1, dims=2)
            self.obs_stack_buf[:, :, -1] = obs
            # flatten stacked frames: [N, D, T] -> [N, D*T]
            obs = self.obs_stack_buf.reshape(self.num_envs, -1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # intermediate values already computed in _get_observations
        # self._compute_intermediate_values()  # remove this call

        actuated_dof_vel = self._dof_vel_act

        up_clamped = torch.clamp(self.up_proj, min=0.0)
        up_reward = self.cfg.up_weight * up_clamped

        height = self.torso_position[:, 2]
        height_error = height - self.cfg.target_root_height
        base_height_penalty = self.cfg.base_height_scale * (height_error**2)

        torso_xy = self.torso_position[:, :2] - self.scene.env_origins[:, :2]
        xy_dist = torch.norm(torso_xy, dim=-1)
        base_xy_penalty = self.cfg.base_xy_scale * (xy_dist**2)

        lin_vel_penalty = self.cfg.lin_vel_l2_scale * torch.sum(self.velocity**2, dim=-1)
        ang_vel_penalty = self.cfg.ang_vel_l2_scale * torch.sum(self.ang_velocity**2, dim=-1)

        # stance regularizers
        feet_pos = self.robot.data.body_pos_w[:, self._feet_ids, :]  # [N, 2, 3]
        feet_xy = feet_pos[:, :, :2]

        foot_lats = feet_xy[:, :, 1] - self.torso_position[:, 1].unsqueeze(1)  # [N, 2]
        step_width = torch.abs(foot_lats[:, 0] - foot_lats[:, 1])
        width_deviation = step_width - self.default_step_width
        step_width_penalty = self.cfg.step_width_scale * (width_deviation**2)

        heading_xy = self.heading_vec[:, :2]
        heading_xy = heading_xy / (torch.norm(heading_xy, dim=-1, keepdim=True) + 1e-6)
        foot_vecs = feet_xy - self.torso_position[:, :2].unsqueeze(1)
        forward_offsets = (foot_vecs * heading_xy.unsqueeze(1)).sum(dim=-1)
        max_forward = forward_offsets.abs().max(dim=1).values
        excess_stride = torch.clamp(max_forward - self.cfg.max_stride_length, min=0.0)
        stride_penalty = self.cfg.stride_penalty_scale * (excess_stride**2)

        hip_angles = self.dof_pos_full[:, self._hip_dof_idx]
        hip_default = self.default_joint_pos_full[self._hip_dof_idx]
        hip_deviation = hip_angles - hip_default.unsqueeze(0)
        hip_posture_penalty = self.cfg.hip_posture_scale * (hip_deviation**2).mean(dim=1)

        actions_cost = torch.sum(self.actions**2, dim=-1)
        
        # action rate penalty (smoothness)
        action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1)
        
        electricity_cost = torch.sum(
            torch.abs(self.actions * actuated_dof_vel * self.cfg.dof_vel_scale) * self.motor_effort_ratio.unsqueeze(0),
            dim=-1,
        )

        alive_reward = torch.ones_like(self.up_proj) * self.cfg.alive_reward_scale

        fell_height = self.torso_position[:, 2] < self.cfg.termination_height
        too_tilted = self.up_proj < self.cfg.termination_up_proj
        died = fell_height | too_tilted
        death_penalty = torch.where(died, torch.ones_like(self.up_proj) * self.cfg.death_cost, 0.0)

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
            - float(getattr(self.cfg, "action_rate_scale", 0.05)) * action_rate
            - self.cfg.energy_cost_scale * electricity_cost
            + death_penalty
        )

        return total_reward

    def _get_dones(self):
        # intermediate values already computed in _get_observations
        # self._compute_intermediate_values()  # remove this call

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        fell_height = self.torso_position[:, 2] < self.cfg.termination_height
        too_tilted = self.up_proj < self.cfg.termination_up_proj
        too_far_xy = torch.norm(self.torso_position[:, :2] - self.scene.env_origins[:, :2], dim=-1) > self.cfg.max_xy_displacement

        died = fell_height | too_tilted | too_far_xy
        return died, time_out

    def _maybe_update_adr(self, batch_success_rate: float) -> None:
        """Increase/decrease ADR based on EMA survival rate."""
        if self.adr is None:
            return

        ema_k = float(self.cfg.adr_ema_factor)
        self.success_rate_ema[:] = (1.0 - ema_k) * self.success_rate_ema + ema_k * batch_success_rate

        # log
        self.extras["log"]["adr_success_rate_batch"] = float(batch_success_rate)
        self.extras["log"]["adr_success_rate_ema"] = float(self.success_rate_ema.item())
        self.extras["log"]["adr_increments"] = int(self.adr.num_increments())

        # update interval gate
        if (self._global_policy_step - self._last_adr_update_step) < int(self.cfg.adr_update_interval_steps):
            return

        ema = float(self.success_rate_ema.item())

        if ema >= float(self.cfg.adr_success_rate_to_increase):
            self.adr.increase(1)
            self._last_adr_update_step = self._global_policy_step
            self._update_adr_custom_params()
            if getattr(self.cfg, "adr_print_every_update", True):
                print(self.adr.print_params())

        # optional: decrease if really failing (and cooled down)
        elif ema <= float(self.cfg.adr_success_rate_to_decrease):
            if (self._global_policy_step - self._last_adr_decrease_step) >= int(self.cfg.adr_min_steps_before_decrease):
                self.adr.decrease(1)
                self._last_adr_decrease_step = self._global_policy_step
                self._last_adr_update_step = self._global_policy_step
                self._update_adr_custom_params()
                if getattr(self.cfg, "adr_print_every_update", True):
                    print(self.adr.print_params())

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.sim.device, dtype=torch.long)

        # --- compute success rate for ADR update from episodes that ended ---
        # "success" := timed out (survived full episode), not died early
        if self.adr is not None and env_ids.numel() > 0:
            # Skip ADR update on the very first reset (startup reset), when episode_length_buf is 0.
            if self._global_policy_step > 0:
                # timeout if episode length reached max-1 (same logic as _get_dones)
                timed_out = self.episode_length_buf[env_ids] >= (self.max_episode_length - 1)
                batch_success = timed_out.float().mean().item()
                self._maybe_update_adr(batch_success)

        # --- reset robot ---
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.filtered_actions[env_ids] = 0.0

        # reset push timers
        self.push_counters[env_ids] = 0
        self.next_push_steps[env_ids] = torch.randint(
            self._push_interval_steps_min,
            self._push_interval_steps_max + 1,
            (env_ids.numel(),),
            device=self.sim.device,
        )

        # sample per-episode DR variables (latency, strength, imu bias)
        self._sample_custom_dr_for_resets(env_ids)

        # warm-start history buffers
        if self.act_hist_buf is not None:
            self.act_hist_buf[env_ids, :, :] = 0.0
        
        # invalidate cache and compute fresh
        self._intermediates_valid = False
        self._compute_intermediate_values()
        
        if self.obs_hist_buf is not None:
            obs0 = self._build_obs_no_latency()
            # fill all history slots with obs0 for the reset envs
            self.obs_hist_buf[env_ids] = obs0[env_ids].unsqueeze(-1).expand(-1, -1, self.obs_max_latency + 1)
        
        # warm-start observation stacking buffer
        if self.obs_stack_buf is not None:
            obs0 = self._build_obs_no_latency()
            self.obs_stack_buf[env_ids] = obs0[env_ids].unsqueeze(-1).expand(-1, -1, self.obs_stack_frames)

        # keep potentials consistent (unused in reward)
        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt


# ========= Helper =========

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

    prev_potentials[:] = potentials
    to_target2 = targets - torso_position
    to_target2[:, 2] = 0.0
    potentials = -torch.norm(to_target2, p=2, dim=-1) / dt

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


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by inverse of quaternion q."""
    q_conj = quat_conjugate(q)
    return quat_rotate(q_conj, v)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q."""
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c
