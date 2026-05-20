# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
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


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_tuple(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (_lerp(a[0], b[0], t), _lerp(a[1], b[1], t))


def _ramp_scale(difficulty: float, start: float, ramp: float) -> float:
    if difficulty <= start:
        return 0.0
    if ramp <= 0.0:
        return 1.0
    return float(min(1.0, (difficulty - start) / ramp))


class LocomotionADR:
    """
    Minimal ADR controller:
      - EventManager-backed: widens term cfg ranges via event_manager.get_term_cfg/set_term_cfg
      - Custom params: provides “current” values (push range, noise, etc.)
    """

    def __init__(self, event_manager, adr_event_cfg_dict: dict, adr_custom_cfg_dict: dict, num_increments: int):
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

        if isinstance(spec, (tuple, list)) and len(spec) == 2 and all(isinstance(x, (int, float)) for x in spec):
            return _lerp(float(spec[0]), float(spec[1]), t)

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
        if "push" in self.adr_custom_cfg_dict:
            lines.append(f"  - push.push_force_range = {self.get_custom('push','push_force_range')}")
        if "action_noise" in self.adr_custom_cfg_dict:
            lines.append(f"  - action_noise.std = {self.get_custom('action_noise','std')}")
        if "latency" in self.adr_custom_cfg_dict:
            lines.append(f"  - latency.act_steps = {self.get_custom('latency','act_steps')}")
            lines.append(f"  - latency.obs_steps = {self.get_custom('latency','obs_steps')}")
        if "command_scale" in self.adr_custom_cfg_dict:
            lines.append(f"  - command_scale.scale = {self.get_custom('command_scale','scale')}")
        return "\n".join(lines)


class LocomotionEnv(DirectRLEnv):
    cfg: DirectRLEnvCfg

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        obs_stack_frames = max(1, int(getattr(cfg, "obs_stack_frames", 1)))
        obs_single_dim = 3 + 3 + 3 + 3 + cfg.action_space * 3
        if cfg.use_phase_obs:
            obs_single_dim += 2
        if bool(getattr(cfg, "use_motion_reference", False)) and bool(
            getattr(cfg, "motion_reference_observation", True)
        ):
            obs_single_dim += cfg.action_space * 2
        cfg.observation_space_single = obs_single_dim
        cfg.obs_stack_frames = obs_stack_frames
        cfg.observation_space = obs_single_dim * obs_stack_frames

        # Robot/sensor state is reused across observations, rewards, and dones within one control step.
        self._state_valid = False

        super().__init__(cfg, render_mode, **kwargs)

        # actions are POSITION OFFSETS for actuated joints
        self.action_scale = self.cfg.action_scale

        actuated_joint_regex = (
            "left_hip1_joint|left_hip2_joint|left_thigh_joint|left_knee_joint|left_ankle_joint|"
            "right_hip1_joint|right_hip2_joint|right_thigh_joint|right_knee_joint|right_ankle_joint"
        )
        self._joint_dof_idx, _ = self.robot.find_joints(actuated_joint_regex)

        print("\n")
        print(self._joint_dof_idx)
        print(_)
        print("\n")
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

        # Hip1 pair for anti-phase gait reward (forward walking)
        hip1_left_ids, _ = self.robot.find_joints("left_hip1_joint")
        hip1_right_ids, _ = self.robot.find_joints("right_hip1_joint")
        self._hip1_left_action_id = None
        self._hip1_right_action_id = None
        if len(hip1_left_ids) > 0 and len(hip1_right_ids) > 0:
            left_hip1_id = int(hip1_left_ids[0])
            right_hip1_id = int(hip1_right_ids[0])
            if left_hip1_id in joint_id_to_action and right_hip1_id in joint_id_to_action:
                self._hip1_left_action_id = int(joint_id_to_action[left_hip1_id])
                self._hip1_right_action_id = int(joint_id_to_action[right_hip1_id])

        # Hip2 pair for stand posture control (avoid inward collapse at stand)
        hip2_left_ids, _ = self.robot.find_joints("left_hip2_joint")
        hip2_right_ids, _ = self.robot.find_joints("right_hip2_joint")
        hip2_action_ids = []
        if len(hip2_left_ids) > 0:
            j = int(hip2_left_ids[0])
            if j in joint_id_to_action:
                hip2_action_ids.append(joint_id_to_action[j])
        if len(hip2_right_ids) > 0:
            j = int(hip2_right_ids[0])
            if j in joint_id_to_action:
                hip2_action_ids.append(joint_id_to_action[j])
        self._hip2_action_ids = torch.tensor(hip2_action_ids, device=self.sim.device, dtype=torch.long)

        # IMU body is named "world" in your URDF
        imu_ids, _ = self.robot.find_bodies("world")
        self._imu_body_idx = int(imu_ids[0])

        # tracking site index (for velocity/height tracking)
        site_names = self.scene["ee_site"].data.target_frame_names
        if "top" not in site_names:
            raise RuntimeError(f"'top' not found in ee_site target_frame_names: {site_names}")
        self._top_frame_idx = site_names.index("top")

        # pre-create world up for speed (avoid allocating every step)
        self._world_up = torch.tensor([0.0, 0.0, 1.0], device=self.sim.device).unsqueeze(0).repeat(self.num_envs, 1)

        # default pose + joint limits (actuated only)
        default_joint_pos = self.robot.data.default_joint_pos[0]
        self.default_actuated_pos = default_joint_pos[self._joint_dof_idx].clone()
        
        print("\n")
        print(self.default_actuated_pos)
        print("\n")

        self.actuated_lower = self.robot.data.soft_joint_pos_limits[0, self._joint_dof_idx, 0].clone()
        self.actuated_upper = self.robot.data.soft_joint_pos_limits[0, self._joint_dof_idx, 1].clone()

        # Optional reference motion produced by the Holosoma retargeting converter.
        self._motion_reference_enabled = bool(getattr(self.cfg, "use_motion_reference", False))
        self._motion_reference_observation = bool(getattr(self.cfg, "motion_reference_observation", True))
        self.motion_target_joint_pos = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)
        self.motion_target_joint_vel = torch.zeros_like(self.motion_target_joint_pos)
        self.motion_target_joint_pos_error = torch.zeros_like(self.motion_target_joint_pos)
        self.motion_start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.sim.device)
        self.motion_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.sim.device)
        self._motion_num_frames = 0
        self._motion_fps = 0.0
        self._motion_joint_pos = None
        self._motion_joint_vel = None
        self._motion_debug_printed = False
        if self._motion_reference_enabled:
            self._load_motion_reference()

        # buffers
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.q_des = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)
        self.action_max_latency = int(getattr(self.cfg, "action_max_latency", 0))
        self.action_latency_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.sim.device)
        if self.action_max_latency > 0:
            self.action_hist_buf = torch.zeros(
                self.num_envs,
                self.num_actions,
                self.action_max_latency + 1,
                device=self.sim.device,
            )
        else:
            self.action_hist_buf = None

        # control timestep for smoothness costs
        self._control_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        if int(getattr(self.cfg, "command_resample_interval_steps", 0)) > 0:
            self._cmd_resample_interval_steps = int(self.cfg.command_resample_interval_steps)
        else:
            self._cmd_resample_interval_steps = max(
                1, int(float(self.cfg.command_resample_interval_s) / self._control_dt)
            )

        # previous joint velocities for acceleration cost
        self.prev_act_vel = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)

        # commanded base velocity in BODY frame: [vx, vy, yaw_rate]
        self.commands = torch.zeros(self.num_envs, 3, device=self.sim.device)
        self.phase_offset = torch.zeros(self.num_envs, device=self.sim.device)

        # rotate body-frame vectors to align forward axis with command frame
        self._command_yaw_offset = float(getattr(self.cfg, "command_yaw_offset", 0.0))
        self._cmd_yaw_cos = math.cos(self._command_yaw_offset)
        self._cmd_yaw_sin = math.sin(self._command_yaw_offset)
        self._cmd_yaw_inv_cos = self._cmd_yaw_cos
        self._cmd_yaw_inv_sin = -self._cmd_yaw_sin
        self._use_cmd_yaw_offset = abs(self._command_yaw_offset) > 1e-6

        # COM tracking config + cached masses
        self._track_com_linear = bool(getattr(self.cfg, "track_com_linear_velocity", True))
        self._refresh_runtime_masses_on_reset = bool(getattr(self.cfg, "refresh_runtime_masses_on_reset", False))
        self._body_mass = None
        self._body_mass_sum = None
        self._cache_body_masses()

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

        # debug observation printing
        self._debug_obs_print = bool(getattr(self.cfg, "debug_obs_print", False))
        self._debug_obs_print_steps = int(getattr(self.cfg, "debug_obs_print_steps", 0))
        self._debug_obs_print_every = max(1, int(getattr(self.cfg, "debug_obs_print_every", 1)))
        self._debug_obs_print_env = int(getattr(self.cfg, "debug_obs_print_env", 0))
        self._debug_obs_step = 0
        self._obs_debug_slices = []
        offset = 0
        def _add_obs_slice(name: str, size: int) -> None:
            nonlocal offset
            self._obs_debug_slices.append((name, offset, offset + size))
            offset += size
        _add_obs_slice("lin_vel_cmd", 3)
        _add_obs_slice("ang_vel_cmd_scaled", 3)
        _add_obs_slice("up_cmd", 3)
        _add_obs_slice("commands", 3)
        _add_obs_slice("act_pos_scaled", self.num_actions)
        _add_obs_slice("act_vel_scaled", self.num_actions)
        _add_obs_slice("prev_actions", self.num_actions)
        if bool(getattr(self.cfg, "use_phase_obs", False)):
            _add_obs_slice("phase_clock", 2)
        if self._motion_reference_enabled and self._motion_reference_observation:
            _add_obs_slice("motion_target_pos_error", self.num_actions)
            _add_obs_slice("motion_target_vel", self.num_actions)
        self._obs_debug_dim = offset

        if bool(getattr(self.cfg, "debug_print_orderings", False)):
            try:
                joint_names = self.robot.data.joint_names
            except Exception:
                joint_names = None
            print("[DebugOrder] action/act_pos order (index -> joint name):")
            for i, jid in enumerate(self._joint_dof_idx):
                name = joint_names[jid] if joint_names is not None and jid < len(joint_names) else str(jid)
                print(f"  {i}: {name}")
            print("[DebugOrder] commands order: [vx, vy, yaw_rate]")
            print("[DebugOrder] observation slices (single frame):")
            for name, s, e in self._obs_debug_slices:
                print(f"  {name}: [{s}, {e})")

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

        # -----------------------
        # ADR state + buffers
        # -----------------------
        self._global_policy_step = 0
        self._last_adr_update_step = 0
        self._last_adr_decrease_step = 0
        self.success_rate_ema = torch.zeros(1, device=self.sim.device)
        self.lin_err_ema = torch.zeros(1, device=self.sim.device)
        self.yaw_err_ema = torch.zeros(1, device=self.sim.device)

        self._ep_lin_err_sum = torch.zeros(self.num_envs, device=self.sim.device)
        self._ep_yaw_err_sum = torch.zeros(self.num_envs, device=self.sim.device)
        self._ep_len = torch.zeros(self.num_envs, device=self.sim.device)

        # observation latency buffers
        self.obs_max_latency = int(getattr(self.cfg, "obs_max_latency", 0))
        self.obs_latency_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.sim.device)
        if self.obs_max_latency > 0:
            self.obs_hist_buf = torch.zeros(self.num_envs, self._obs_single_dim, self.obs_max_latency + 1, device=self.sim.device)
        else:
            self.obs_hist_buf = None

        # per-env motor strength multipliers
        self.motor_strength_mult = torch.ones(self.num_envs, self.num_actions, device=self.sim.device)

        # IMU bias (gravity direction + gyro) and mount misalignment
        self.imu_bias_gravity = torch.zeros(self.num_envs, 3, device=self.sim.device)
        self.imu_bias_gyro = torch.zeros(self.num_envs, 3, device=self.sim.device) # rad/s
        self.imu_mount_axis = torch.tensor([1.0, 0.0, 0.0], device=self.sim.device).repeat(self.num_envs, 1) # unit axis vector
        self.imu_mount_ang = torch.zeros(self.num_envs, device=self.sim.device) # radians

        # continuous micro disturbances (OU process)
        self.micro_lin_acc = torch.zeros(self.num_envs, 3, device=self.sim.device)
        self.micro_ang_acc = torch.zeros(self.num_envs, 3, device=self.sim.device)

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
        self._joint_pos_obs_noise_std_rad = float(getattr(self.cfg, "joint_pos_obs_noise_std_rad", 0.0))
        self._joint_vel_obs_noise_std_rad_s = float(getattr(self.cfg, "joint_vel_obs_noise_std_rad_s", 0.0))
        self._command_scale = 1.0
        self._push_scale = 1.0
        self._micro_wrench_scale = 1.0

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

        # init ADR controller after event_manager exists
        self.adr = None
        if getattr(self.cfg, "enable_adr", False):
            self.adr = LocomotionADR(
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

        self._resample_episode_lengths(self.robot._ALL_INDICES)


        #  visualizing the site code
        # data = self.scene["ee_site"].data
        # # list names to get correct index:
        # print(data.target_frame_names)  # order may differ if regex is used
        # idx = data.target_frame_names.index("top")

        # pos_w  = data.target_pos_w[:, idx]      # world position of your site
        # quat_w = data.target_quat_w[:, idx]     # world orientation of your site


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        # add contact sensors
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._ee_site_sensor = FrameTransformer(self.cfg.ee_site)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.sensors["ee_site"] = self._ee_site_sensor

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

    def _cache_body_masses(self) -> None:
        """Cache link masses on sim device. Prefer default_mass (fast, GPU)."""
        if hasattr(self.robot.data, "default_mass") and self.robot.data.default_mass is not None:
            m = self.robot.data.default_mass
            if m.dim() == 1:
                m = m.unsqueeze(0).expand(self.num_envs, -1)
            self._body_mass = m.to(self.sim.device)
        else:
            m = self.robot.root_physx_view.get_masses()
            if not torch.is_tensor(m):
                m = torch.as_tensor(m)
            if m.dim() == 1:
                m = m.unsqueeze(0).expand(self.num_envs, -1)
            self._body_mass = m.to(self.sim.device)

        self._body_mass_sum = self._body_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)

    # ----------------------------------------------------------------------
    # ADR helpers (custom params)
    # ----------------------------------------------------------------------
    def _update_adr_custom_params(self) -> None:
        if self.adr is None:
            self._push_dv_min = float(self.cfg.push_force_range[0])
            self._push_dv_max = float(self.cfg.push_force_range[1])
            self._action_noise_std = 0.0
            for k in self._obs_noise.keys():
                self._obs_noise[k] = 0.0
            self._command_scale = 1.0
            self._push_scale = 1.0
            self._micro_wrench_scale = 1.0
            return

        pr = self.adr.get_custom("push", "push_force_range")
        self._push_dv_min = float(pr[0])
        self._push_dv_max = float(pr[1])

        difficulty = float(self.adr.difficulty())
        push_start = float(getattr(self.cfg, "adr_push_start_difficulty", 0.0))
        push_ramp = float(getattr(self.cfg, "adr_push_ramp_difficulty", 0.0))
        self._push_scale = _ramp_scale(difficulty, push_start, push_ramp)
        self._push_dv_min *= self._push_scale
        self._push_dv_max *= self._push_scale

        self._action_noise_std = float(self.adr.get_custom("action_noise", "std"))

        for k in self._obs_noise.keys():
            if "obs_noise" in self.adr.adr_custom_cfg_dict and k in self.adr.adr_custom_cfg_dict["obs_noise"]:
                self._obs_noise[k] = float(self.adr.get_custom("obs_noise", k))
            else:
                self._obs_noise[k] = 0.0

        if "command_scale" in self.adr.adr_custom_cfg_dict:
            self._command_scale = float(self.adr.get_custom("command_scale", "scale"))
        else:
            self._command_scale = 1.0

        micro_start = float(getattr(self.cfg, "adr_micro_wrench_start_difficulty", 0.0))
        micro_ramp = float(getattr(self.cfg, "adr_micro_wrench_ramp_difficulty", 0.0))
        self._micro_wrench_scale = _ramp_scale(difficulty, micro_start, micro_ramp)

    def _sample_custom_dr_for_resets(self, env_ids: torch.Tensor) -> None:
        """Sample per-episode random variables: motor strength, latency, and IMU calibration."""
        if self.adr is None or env_ids.numel() == 0:
            self.motor_strength_mult[env_ids] = 1.0
            self.action_latency_steps[env_ids] = 0
            self.obs_latency_steps[env_ids] = 0
            self.imu_bias_gravity[env_ids] = 0.0
            self.imu_bias_gyro[env_ids] = 0.0
            self.imu_mount_axis[env_ids] = torch.tensor([1.0, 0.0, 0.0], device=self.sim.device).repeat(env_ids.numel(), 1)
            self.imu_mount_ang[env_ids] = 0.0
            return

        device = self.sim.device

        r = float(self.adr.get_custom("motor_strength", "per_joint_mult_range"))
        if r > 0.0:
            lo, hi = 1.0 - r, 1.0 + r
            self.motor_strength_mult[env_ids] = torch.empty((env_ids.numel(), self.num_actions), device=device).uniform_(lo, hi)
        else:
            self.motor_strength_mult[env_ids] = 1.0

        act_max = 0
        obs_max = 0
        if "latency" in self.adr.adr_custom_cfg_dict:
            act_max = int(round(float(self.adr.get_custom("latency", "act_steps"))))
            obs_max = int(round(float(self.adr.get_custom("latency", "obs_steps"))))
        act_max = int(max(0, min(self.action_max_latency, act_max)))
        obs_max = int(max(0, min(self.obs_max_latency, obs_max)))
        if act_max > 0:
            self.action_latency_steps[env_ids] = torch.randint(0, act_max + 1, (env_ids.numel(),), device=device)
        else:
            self.action_latency_steps[env_ids] = 0
        if obs_max > 0:
            self.obs_latency_steps[env_ids] = torch.randint(0, obs_max + 1, (env_ids.numel(),), device=device)
        else:
            self.obs_latency_steps[env_ids] = 0

        b_grav = float(self.adr.get_custom("imu_bias", "gravity_bias_range"))
        if b_grav > 0.0:
            self.imu_bias_gravity[env_ids] = torch.empty((env_ids.numel(), 3), device=device).uniform_(-b_grav, b_grav)
        else:
            self.imu_bias_gravity[env_ids] = 0.0

        b_gyro = float(self.adr.get_custom("imu_bias", "gyro_bias_range"))
        if b_gyro > 0.0:
            self.imu_bias_gyro[env_ids] = torch.empty((env_ids.numel(), 3), device=device).uniform_(-b_gyro, b_gyro)
        else:
            self.imu_bias_gyro[env_ids] = 0.0

        if "sensor_extrinsics" in self.adr.adr_custom_cfg_dict:
            deg = float(self.adr.get_custom("sensor_extrinsics", "imu_mount_deg"))
        else:
            deg = 0.0
        if deg > 0.0:
            max_rad = deg * math.pi / 180.0
            axis = torch.randn((env_ids.numel(), 3), device=device)
            axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-6)
            ang = torch.empty((env_ids.numel(), 1), device=device).uniform_(-max_rad, max_rad)
            self.imu_mount_axis[env_ids] = axis
            self.imu_mount_ang[env_ids] = ang.squeeze(-1)
        else:
            self.imu_mount_axis[env_ids] = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(env_ids.numel(), 1)
            self.imu_mount_ang[env_ids] = 0.0

    def _maybe_apply_pushes(self):
        """Apply random directed velocity kicks to the base at random intervals."""
        if self._push_dv_max <= 0.0:
            return

        self.push_counters += 1
        ready = self.push_counters >= self.next_push_steps
        if not torch.any(ready):
            return

        env_ids = torch.nonzero(ready, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return

        k = env_ids.numel()
        device = self.sim.device

        dirs = torch.randn((k, 3), device=device)
        z_frac = float(getattr(self.cfg, "push_z_fraction", 0.25))
        dirs[:, 2] *= z_frac
        dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-6)

        mags = torch.empty((k, 1), device=device).uniform_(self._push_dv_min, self._push_dv_max)
        delta_v = dirs * mags

        root_vel = self.robot.data.root_vel_w[env_ids].clone()  # [K, 6]
        root_vel[:, 0:3] += delta_v

        w_scale = float(getattr(self.cfg, "push_angvel_scale", 0.8))
        delta_w = torch.randn((k, 3), device=device)
        delta_w = delta_w / (torch.norm(delta_w, dim=-1, keepdim=True) + 1e-6)
        delta_w = delta_w * (w_scale * mags)
        root_vel[:, 3:6] += delta_w

        self.robot.write_root_velocity_to_sim(root_vel, env_ids)

        self.push_counters[env_ids] = 0
        self.next_push_steps[env_ids] = torch.randint(
            self._push_interval_steps_min,
            self._push_interval_steps_max + 1,
            (k,),
            device=device,
        )

    def _apply_micro_disturbance(self):
        """Apply temporally correlated micro-accelerations (OU process)."""
        if self.adr is None:
            return

        scale = float(self._micro_wrench_scale)
        if scale <= 0.0:
            return

        rho = float(self.adr.get_custom("micro_wrench", "rho"))
        lin_std = float(self.adr.get_custom("micro_wrench", "lin_acc_std")) * scale
        ang_std = float(self.adr.get_custom("micro_wrench", "ang_acc_std")) * scale
        max_lin = float(self.adr.get_custom("micro_wrench", "max_lin_acc")) * scale
        max_ang = float(self.adr.get_custom("micro_wrench", "max_ang_acc")) * scale

        if lin_std <= 0.0 and ang_std <= 0.0:
            return

        if lin_std > 0.0:
            self.micro_lin_acc = rho * self.micro_lin_acc + (1.0 - rho) * torch.randn_like(self.micro_lin_acc) * lin_std
            self.micro_lin_acc = torch.clamp(self.micro_lin_acc, -max_lin, max_lin)

        if ang_std > 0.0:
            self.micro_ang_acc = rho * self.micro_ang_acc + (1.0 - rho) * torch.randn_like(self.micro_ang_acc) * ang_std
            self.micro_ang_acc = torch.clamp(self.micro_ang_acc, -max_ang, max_ang)

        root_vel = self.robot.data.root_vel_w.clone()  # [N, 6]
        root_vel[:, 0:3] += self.micro_lin_acc * self.dt
        root_vel[:, 3:6] += self.micro_ang_acc * self.dt
        self.robot.write_root_velocity_to_sim(root_vel)

    def _maybe_update_adr(self, batch_success_rate: float, batch_lin_err: float, batch_yaw_err: float) -> None:
        """Increase/decrease ADR based on EMA survival rate."""
        if self.adr is None:
            return

        if self._global_policy_step < int(getattr(self.cfg, "adr_warmup_steps", 0)):
            return
        if self._current_curriculum_stage < int(getattr(self.cfg, "adr_min_stage", 0)):
            return

        ema_k = float(self.cfg.adr_ema_factor)
        self.success_rate_ema[:] = (1.0 - ema_k) * self.success_rate_ema + ema_k * batch_success_rate
        self.lin_err_ema[:] = (1.0 - ema_k) * self.lin_err_ema + ema_k * batch_lin_err
        self.yaw_err_ema[:] = (1.0 - ema_k) * self.yaw_err_ema + ema_k * batch_yaw_err

        if getattr(self.cfg, "adr_debug_print", True):
            every = int(getattr(self.cfg, "adr_debug_print_every_steps", 2000))
            if (self._global_policy_step % every) == 0:
                print(
                    f"[ADR] step={self._global_policy_step} "
                    f"batch_success={batch_success_rate:.3f} "
                    f"ema={float(self.success_rate_ema.item()):.3f} "
                    f"lin_err_ema={float(self.lin_err_ema.item()):.3f} "
                    f"yaw_err_ema={float(self.yaw_err_ema.item()):.3f} "
                    f"inc={self.adr.num_increments()}/{self.cfg.num_adr_increments}",
                    flush=True,
                )

        self.extras["log"]["adr_success_rate_batch"] = float(batch_success_rate)
        self.extras["log"]["adr_success_rate_ema"] = float(self.success_rate_ema.item())
        self.extras["log"]["adr_tracking_lin_err_batch"] = float(batch_lin_err)
        self.extras["log"]["adr_tracking_yaw_err_batch"] = float(batch_yaw_err)
        self.extras["log"]["adr_tracking_lin_err_ema"] = float(self.lin_err_ema.item())
        self.extras["log"]["adr_tracking_yaw_err_ema"] = float(self.yaw_err_ema.item())
        self.extras["log"]["adr_increments"] = int(self.adr.num_increments())

        if (self._global_policy_step - self._last_adr_update_step) < int(self.cfg.adr_update_interval_steps):
            return

        ema = float(self.success_rate_ema.item())
        lin_ema = float(self.lin_err_ema.item())
        yaw_ema = float(self.yaw_err_ema.item())

        lin_inc = float(getattr(self.cfg, "adr_track_err_lin_increase_threshold", 0.0))
        lin_dec = float(getattr(self.cfg, "adr_track_err_lin_decrease_threshold", 1e6))
        yaw_inc = float(getattr(self.cfg, "adr_track_err_yaw_increase_threshold", 0.0))
        yaw_dec = float(getattr(self.cfg, "adr_track_err_yaw_decrease_threshold", 1e6))

        if (
            ema >= float(self.cfg.adr_success_rate_to_increase)
            and lin_ema <= lin_inc
            and yaw_ema <= yaw_inc
        ):
            self.adr.increase(1)
            self._last_adr_update_step = self._global_policy_step
            self._update_adr_custom_params()
            if getattr(self.cfg, "adr_print_every_update", True):
                print(self.adr.print_params())

        elif (
            ema <= float(self.cfg.adr_success_rate_to_decrease)
            or lin_ema >= lin_dec
            or yaw_ema >= yaw_dec
        ):
            if (self._global_policy_step - self._last_adr_decrease_step) >= int(self.cfg.adr_min_steps_before_decrease):
                self.adr.decrease(1)
                self._last_adr_decrease_step = self._global_policy_step
                self._last_adr_update_step = self._global_policy_step
                self._update_adr_custom_params()
                if getattr(self.cfg, "adr_print_every_update", True):
                    print(self.adr.print_params())

    def _pre_physics_step(self, actions: torch.Tensor):
        self._global_policy_step += 1
        self._invalidate_state_cache()

        a = actions.clone().clamp(-1.0, 1.0)

        # action noise (ADR)
        if self._action_noise_std > 0.0:
            a = a + torch.randn_like(a) * self._action_noise_std
            a = a.clamp(-1.0, 1.0)

        if self.action_hist_buf is not None:
            self.action_hist_buf = torch.roll(self.action_hist_buf, shifts=-1, dims=2)
            self.action_hist_buf[:, :, -1] = a
            idx = (self.action_max_latency - self.action_latency_steps).clamp(0, self.action_max_latency)
            gather_idx = idx.view(-1, 1, 1).expand(-1, a.shape[1], 1)
            a = torch.gather(self.action_hist_buf, dim=2, index=gather_idx).squeeze(-1)

        self.prev_actions[:] = self.actions
        self.actions = a

        # Update curriculum based on global env-steps
        self._global_env_steps += self.num_envs
        self._update_curriculum()

        if self._cmd_resample_interval_steps > 0:
            resample_mask = (self.episode_length_buf % self._cmd_resample_interval_steps) == 0
            resample_ids = torch.nonzero(resample_mask, as_tuple=False).squeeze(-1)
            if resample_ids.numel() > 0:
                self._sample_commands(resample_ids)

        # scheduled pushes + micro disturbances (ADR)
        self._maybe_apply_pushes()
        self._apply_micro_disturbance()

        if self._visualization_enabled:
            self._visualize_markers()

    def _invalidate_state_cache(self) -> None:
        self._state_valid = False

    def _resolve_motion_reference_path(self, motion_file: str) -> Path:
        path = Path(motion_file).expanduser()
        if path.is_absolute():
            return path

        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path

        package_root = Path(__file__).resolve().parents[3]
        package_path = package_root / path
        if package_path.exists():
            return package_path

        return package_path

    def _action_joint_names(self) -> list[str]:
        try:
            joint_names = self.robot.data.joint_names
        except Exception as exc:
            raise RuntimeError("Cannot read IsaacLab robot joint_names for motion reference remap.") from exc

        names = []
        for jid in self._joint_dof_idx:
            idx = int(jid)
            if idx < 0 or idx >= len(joint_names):
                raise RuntimeError(f"Action joint index {idx} is outside robot joint_names length {len(joint_names)}.")
            names.append(str(joint_names[idx]))
        return names

    def _load_motion_reference(self) -> None:
        motion_file = str(getattr(self.cfg, "motion_reference_file", ""))
        if not motion_file:
            raise RuntimeError("use_motion_reference=True but motion_reference_file is empty.")

        motion_path = self._resolve_motion_reference_path(motion_file)
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion reference file not found: {motion_path}")

        data = np.load(str(motion_path), allow_pickle=True)
        required_keys = {
            "fps",
            "joint_pos",
            "joint_vel",
            "joint_names",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "body_names",
        }
        missing = sorted(required_keys.difference(data.files))
        if missing:
            raise RuntimeError(f"Motion reference file is missing keys: {missing}")

        reference_joint_names = [str(name) for name in data["joint_names"].tolist()]
        action_joint_names = self._action_joint_names()
        missing_action_names = [name for name in action_joint_names if name not in reference_joint_names]
        if missing_action_names:
            raise RuntimeError(
                "Motion reference joint_names do not cover IsaacLab action joints: "
                f"{missing_action_names}. Reference names: {reference_joint_names}"
            )

        remap = [reference_joint_names.index(name) for name in action_joint_names]
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)

        expected_qpos_dim = 7 + len(reference_joint_names)
        expected_qvel_dim = 6 + len(reference_joint_names)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != expected_qpos_dim:
            raise RuntimeError(
                f"Expected joint_pos shape (T, {expected_qpos_dim}) for root qpos + joints, got {joint_pos.shape}."
            )
        if joint_vel.ndim != 2 or joint_vel.shape[1] != expected_qvel_dim:
            raise RuntimeError(
                f"Expected joint_vel shape (T, {expected_qvel_dim}) for root qvel + joints, got {joint_vel.shape}."
            )

        self._motion_joint_pos = torch.as_tensor(joint_pos[:, 7:][:, remap], device=self.sim.device)
        self._motion_joint_vel = torch.as_tensor(joint_vel[:, 6:][:, remap], device=self.sim.device)
        self._motion_num_frames = int(self._motion_joint_pos.shape[0])
        self._motion_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        self._motion_reference_path = motion_path
        self._motion_reference_joint_names = reference_joint_names
        self._motion_action_joint_names = action_joint_names
        self._motion_reference_remap = remap

        if self._motion_num_frames < 2:
            raise RuntimeError(f"Motion reference must contain at least 2 frames, got {self._motion_num_frames}.")

        if bool(getattr(self.cfg, "motion_reference_debug_print", False)):
            duration = (self._motion_num_frames - 1) / max(self._motion_fps, 1e-6)
            print(
                "[MotionReference] loaded "
                f"path={motion_path} frames={self._motion_num_frames} fps={self._motion_fps:g} "
                f"duration={duration:.3f}s",
                flush=True,
            )
            print(f"[MotionReference] reference joint_names={reference_joint_names}", flush=True)
            print(f"[MotionReference] action joint_names={action_joint_names}", flush=True)
            print(f"[MotionReference] remap reference->action={remap}", flush=True)

    def _update_motion_targets(self) -> None:
        if not self._motion_reference_enabled:
            return
        if self._motion_joint_pos is None or self._motion_joint_vel is None:
            return

        frame_stride = self._control_dt * self._motion_fps
        frame_offsets = torch.floor(self.episode_length_buf.float() * frame_stride).long()
        self.motion_frame = (self.motion_start_frame + frame_offsets) % self._motion_num_frames
        self.motion_target_joint_pos = self._motion_joint_pos[self.motion_frame]
        self.motion_target_joint_vel = self._motion_joint_vel[self.motion_frame]
        self.motion_target_joint_pos_error = self.motion_target_joint_pos - self.act_pos

        if bool(getattr(self.cfg, "motion_reference_debug_print", False)) and not self._motion_debug_printed:
            env_id = max(0, min(int(getattr(self.cfg, "debug_obs_print_env", 0)), self.num_envs - 1))
            print(
                "[MotionReference] first target "
                f"env={env_id} frame={int(self.motion_frame[env_id].item())} "
                f"target_joint_pos={self.motion_target_joint_pos[env_id].detach().cpu().tolist()} "
                f"target_joint_vel={self.motion_target_joint_vel[env_id].detach().cpu().tolist()}",
                flush=True,
            )
            self._motion_debug_printed = True

    def _apply_action(self):
        pos_offsets = (
            self.action_scale
            * self._action_scale_per_joint.unsqueeze(0)
            * self.actions
            * self.motor_strength_mult
        )  # [N, 10]
        q_des = self.default_actuated_pos.unsqueeze(0) + pos_offsets
        q_des = torch.clamp(q_des, self.actuated_lower.unsqueeze(0), self.actuated_upper.unsqueeze(0))
        self.q_des = q_des
        self.robot.set_joint_position_target(q_des, joint_ids=self._joint_dof_idx)
        self.robot.set_joint_velocity_target(torch.zeros_like(q_des), joint_ids=self._joint_dof_idx)
        self.robot.set_joint_effort_target(torch.zeros_like(q_des), joint_ids=self._joint_dof_idx)

    def _update_state(self):
        if self._state_valid:
            return

        # -------------------------------
        # A) IMU state from body "world"
        # -------------------------------
        i = self._imu_body_idx

        self.imu_pos_w = self.robot.data.body_pos_w[:, i]     # [N,3]
        self.imu_quat_w = self.robot.data.body_quat_w[:, i]   # [N,4] (w,x,y,z)
        imu_lin_vel_w = self.robot.data.body_lin_vel_w[:, i]  # [N,3]
        imu_ang_vel_w = self.robot.data.body_ang_vel_w[:, i]  # [N,3]

        self.imu_lin_vel_w = imu_lin_vel_w
        self.imu_ang_vel_w = imu_ang_vel_w

        self.imu_lin_vel_b = quat_rotate_inverse(self.imu_quat_w, imu_lin_vel_w)  # [N,3]
        self.imu_ang_vel_b = quat_rotate_inverse(self.imu_quat_w, imu_ang_vel_w)  # [N,3]
        if self._use_cmd_yaw_offset:
            self.imu_lin_vel_cmd = self._rotate_xy(self.imu_lin_vel_b, self._cmd_yaw_cos, self._cmd_yaw_sin)
            self.imu_ang_vel_cmd = self._rotate_xy(self.imu_ang_vel_b, self._cmd_yaw_cos, self._cmd_yaw_sin)
        else:
            self.imu_lin_vel_cmd = self.imu_lin_vel_b
            self.imu_ang_vel_cmd = self.imu_ang_vel_b

        # IMU "up" expressed in body frame (up_b.z ~ 1 when upright)
        self.up_b = quat_rotate_inverse(self.imu_quat_w, self._world_up)  # [N,3]
        if self._use_cmd_yaw_offset:
            self.up_cmd = self._rotate_xy(self.up_b, self._cmd_yaw_cos, self._cmd_yaw_sin)
        else:
            self.up_cmd = self.up_b

        # ------------------------------------------
        # B) Tracking state from site frame "top"
        # ------------------------------------------
        ft = self.scene["ee_site"].data
        self.track_pos_w = ft.target_pos_w[:, self._top_frame_idx]

        # ------------------------------------------
        # C) Center of mass (COM) state for tracking
        # ------------------------------------------
        body_pos_w = getattr(self.robot.data, "body_com_pos_w", None)
        if body_pos_w is None:
            body_pos_w = self.robot.data.body_pos_w
        body_lin_vel_w = getattr(self.robot.data, "body_com_lin_vel_w", None)
        if body_lin_vel_w is None:
            body_lin_vel_w = self.robot.data.body_lin_vel_w

        if self._track_com_linear:
            m = self._body_mass
            msum = self._body_mass_sum
            self.com_pos_w = (body_pos_w * m.unsqueeze(-1)).sum(dim=1) / msum
            self.com_lin_vel_w = (body_lin_vel_w * m.unsqueeze(-1)).sum(dim=1) / msum
            self.com_lin_vel_b = quat_rotate_inverse(self.imu_quat_w, self.com_lin_vel_w)
            if self._use_cmd_yaw_offset:
                self.com_lin_vel_cmd = self._rotate_xy(self.com_lin_vel_b, self._cmd_yaw_cos, self._cmd_yaw_sin)
            else:
                self.com_lin_vel_cmd = self.com_lin_vel_b
        else:
            self.com_lin_vel_cmd = self.imu_lin_vel_cmd

        self.com_ang_vel_cmd = self.imu_ang_vel_cmd

        # joints
        self.dof_pos = self.robot.data.joint_pos
        self.dof_vel = self.robot.data.joint_vel

        self.act_pos = self.dof_pos[:, self._joint_dof_idx]
        self.act_vel = self.dof_vel[:, self._joint_dof_idx]
        self._update_motion_targets()

        lo = self.actuated_lower.unsqueeze(0)
        hi = self.actuated_upper.unsqueeze(0)
        self.act_pos_scaled = 2.0 * (self.act_pos - lo) / (hi - lo + 1e-6) - 1.0
        self._state_valid = True

    def _compute_single_observation(self) -> torch.Tensor:
        up_cmd = self.up_cmd
        ang_vel_cmd = self.imu_ang_vel_cmd
        act_pos = self.act_pos
        act_vel = self.act_vel

        # IMU biases + mount misalignment (ADR)
        up_cmd = up_cmd + self.imu_bias_gravity
        ang_vel_cmd = ang_vel_cmd + self.imu_bias_gyro

        theta = self.imu_mount_axis * self.imu_mount_ang.unsqueeze(-1)
        up_cmd = up_cmd + torch.cross(theta, up_cmd, dim=-1)
        ang_vel_cmd = ang_vel_cmd + torch.cross(theta, ang_vel_cmd, dim=-1)

        # Observation noise (ADR)
        if self.adr is not None:
            if self._obs_noise["gravity_std"] > 0.0:
                up_cmd = up_cmd + torch.randn_like(up_cmd) * self._obs_noise["gravity_std"]
            if self._obs_noise["gyro_std"] > 0.0:
                ang_vel_cmd = ang_vel_cmd + torch.randn_like(ang_vel_cmd) * self._obs_noise["gyro_std"]
            if self._obs_noise["joint_pos_std"] > 0.0:
                act_pos = act_pos + torch.randn_like(act_pos) * self._obs_noise["joint_pos_std"]
            if self._obs_noise["joint_vel_std"] > 0.0:
                act_vel = act_vel + torch.randn_like(act_vel) * self._obs_noise["joint_vel_std"]

        if self._joint_pos_obs_noise_std_rad > 0.0:
            act_pos = act_pos + torch.randn_like(act_pos) * self._joint_pos_obs_noise_std_rad
        if self._joint_vel_obs_noise_std_rad_s > 0.0:
            act_vel = act_vel + torch.randn_like(act_vel) * self._joint_vel_obs_noise_std_rad_s

        lo = self.actuated_lower.unsqueeze(0)
        hi = self.actuated_upper.unsqueeze(0)
        act_pos_scaled = 2.0 * (act_pos - lo) / (hi - lo + 1e-6) - 1.0

        obs = torch.cat(
            (
                self.imu_lin_vel_cmd,                            # command-frame lin vel
                ang_vel_cmd * self.cfg.ang_vel_scale,            # command-frame ang vel
                up_cmd,                                          # IMU-ish orientation feature
                self.commands,                                   # commanded [vx, vy, yaw_rate] in BODY frame
                act_pos_scaled,
                act_vel * self.cfg.dof_vel_scale,
                self.prev_actions,
            ),
            dim=-1,
        )

        if self.cfg.use_phase_obs:
            t = self.episode_length_buf.float() * self._control_dt
            phase = 2.0 * torch.pi * (t / self.cfg.gait_period_s) + self.phase_offset

            if self.cfg.freeze_phase_when_standing:
                stand_mask = (
                    torch.norm(self.commands[:, :2], dim=1) < self.cfg.stand_phase_lin_threshold
                ) & (torch.abs(self.commands[:, 2]) < self.cfg.stand_phase_yaw_threshold)
                phase = torch.where(
                    stand_mask,
                    torch.full_like(phase, self.cfg.stand_phase_value),
                    phase,
                )

            clock = torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)
            obs = torch.cat([obs, clock], dim=-1)

        if self._motion_reference_enabled and self._motion_reference_observation:
            obs = torch.cat(
                [
                    obs,
                    self.motion_target_joint_pos_error
                    * float(getattr(self.cfg, "motion_reference_pos_error_scale", 1.0)),
                    self.motion_target_joint_vel
                    * float(getattr(self.cfg, "motion_reference_vel_scale", self.cfg.dof_vel_scale)),
                ],
                dim=-1,
            )


        return obs

    def _get_observations(self) -> dict:
        self._update_state()
        obs = self._compute_single_observation()

        if self.obs_hist_buf is not None:
            self.obs_hist_buf = torch.roll(self.obs_hist_buf, shifts=-1, dims=2)
            self.obs_hist_buf[:, :, -1] = obs

            idx = (self.obs_max_latency - self.obs_latency_steps).clamp(0, self.obs_max_latency)
            gather_idx = idx.view(-1, 1, 1).expand(-1, obs.shape[1], 1)
            obs = torch.gather(self.obs_hist_buf, dim=2, index=gather_idx).squeeze(-1)

        if self.obs_stack_buf is not None:
            self.obs_stack_buf = torch.roll(self.obs_stack_buf, shifts=-1, dims=2)
            self.obs_stack_buf[:, :, -1] = obs
            obs = self.obs_stack_buf.reshape(self.num_envs, -1)

        if self._debug_obs_print and self._debug_obs_step < self._debug_obs_print_steps:
            if (self._debug_obs_step % self._debug_obs_print_every) == 0:
                env_id = max(0, min(self._debug_obs_print_env, self.num_envs - 1))
                if self.obs_stack_buf is not None:
                    frames = self.obs_stack_buf[env_id].detach().cpu()
                    for i in range(self.obs_stack_frames):
                        frame = frames[:, i]
                        print(f"[ObsDebug] env={env_id} step={self._debug_obs_step} frame={i}")
                        for name, s, e in self._obs_debug_slices:
                            vals = frame[s:e].tolist()
                            print(f"  {name}: {vals}")
                else:
                    frame = obs[env_id].detach().cpu()
                    print(f"[ObsDebug] env={env_id} step={self._debug_obs_step} frame=0")
                    for name, s, e in self._obs_debug_slices:
                        vals = frame[s:e].tolist()
                        print(f"  {name}: {vals}")
            self._debug_obs_step += 1

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._update_state()

        # --- Base velocity tracking ---
        vel_err = self.com_lin_vel_cmd[:, :2] - self.commands[:, :2]
        yaw_err = self.com_ang_vel_cmd[:, 2]  - self.commands[:, 2]

        r_lin = torch.exp(-torch.sum(vel_err * vel_err, dim=1) / self.cfg.lin_vel_sigma)
        r_yaw = torch.exp(-(yaw_err * yaw_err) / self.cfg.yaw_rate_sigma)

        vel_err_norm = torch.norm(vel_err, dim=1)
        yaw_err_abs = torch.abs(yaw_err)
        self._ep_lin_err_sum += vel_err_norm
        self._ep_yaw_err_sum += yaw_err_abs
        self._ep_len += 1.0

        upright = torch.clamp(self.up_b[:, 2], 0.0, 1.0)

        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        t = self.episode_length_buf.float() * self._control_dt
        phase = 2.0 * torch.pi * (t / self.cfg.gait_period_s) + self.phase_offset
        air_time_gate = (cmd_speed > self.cfg.air_time_command_speed_threshold).float()
        act_speed = torch.norm(self.com_lin_vel_b[:, :2], dim=1)
        gait_gate = (
            (cmd_speed > self.cfg.anti_phase_min_speed)
            & (act_speed > self.cfg.gait_actual_speed_thresh)
            & (self.up_b[:, 2] > self.cfg.gait_upright_thresh)
        ).float()
        yaw_cmd_mag = torch.abs(self.commands[:, 2])
        yaw_gate = (yaw_cmd_mag > self.cfg.yaw_cmd_reward_thresh).float()
        r_yaw = r_yaw * yaw_gate
        standstill = torch.clamp(self.cfg.standstill_speed_threshold - act_speed, min=0.0)
        standstill = standstill * (cmd_speed > self.cfg.command_speed_threshold).float()
        speed_shortfall = torch.clamp(cmd_speed - act_speed, min=0.0)

        # Stand command mask (explicit "should be standing")
        stand_cmd = (
            torch.norm(self.commands[:, :2], dim=1) < self.cfg.stand_cmd_lin_thresh
        ) & (
            torch.abs(self.commands[:, 2]) < self.cfg.stand_cmd_yaw_thresh
        )
        stand_cmd_f = stand_cmd.float()

        # Stand posture reward/cost terms
        pose_err_all = self.act_pos - self.default_actuated_pos.unsqueeze(0)
        stand_pose_err = torch.mean(pose_err_all * pose_err_all, dim=1)
        stand_pose_rew = torch.exp(-stand_pose_err / (self.cfg.stand_pose_sigma + 1e-6))

        stand_upright_rew = torch.clamp(self.up_b[:, 2], 0.0, 1.0)

        # Penalize motion while standing
        stand_vel_cost = (
            torch.sum(self.com_lin_vel_cmd[:, :2] ** 2, dim=1)
            + 0.5 * (self.com_ang_vel_cmd[:, 2] ** 2)
        )

        stand_action_cost = torch.sum(self.actions * self.actions, dim=1)

        # Stronger anti-inward penalty on hip2 during stand
        stand_hip2_pen = torch.zeros_like(stand_pose_err)
        if self._hip2_action_ids.numel() > 0:
            hip2_off = self.act_pos[:, self._hip2_action_ids] - self.default_actuated_pos[self._hip2_action_ids].unsqueeze(0)
            stand_hip2_pen = torch.mean(hip2_off * hip2_off, dim=1)

        walk_hip2_pen = torch.zeros_like(r_lin)
        if self._hip2_action_ids.numel() > 0:
            hip2_off_walk = self.act_pos[:, self._hip2_action_ids] - self.default_actuated_pos[self._hip2_action_ids].unsqueeze(0)
            walk_mask = (cmd_speed > self.cfg.walk_cmd_speed_thresh).float()
            walk_hip2_pen = torch.mean(hip2_off_walk * hip2_off_walk, dim=1) * walk_mask

        # --- Anti-phase gait reward (HIP1 joints, forward-walk gated) ---
        anti_phase_rew = torch.zeros_like(r_lin)
        if (self._hip1_left_action_id is not None) and (self._hip1_right_action_id is not None):
            target = torch.sin(phase)  # desired left hip1 profile

            # hip1 offsets around default pose
            left_off = self.act_pos[:, self._hip1_left_action_id] - self.default_actuated_pos[self._hip1_left_action_id]
            right_off = self.act_pos[:, self._hip1_right_action_id] - self.default_actuated_pos[self._hip1_right_action_id]

            # normalize offsets to bounded range for stable matching
            k = float(self.cfg.anti_phase_pos_gain)
            left_norm = torch.tanh(k * left_off)
            right_norm = torch.tanh(k * right_off)

            # anti-phase match: left follows +target, right follows -target
            sig = float(self.cfg.anti_phase_sigma)
            left_match = torch.exp(-((left_norm - target) ** 2) / (sig + 1e-6))
            right_match = torch.exp(-((right_norm + target) ** 2) / (sig + 1e-6))
            anti_phase_rew = 0.5 * (left_match + right_match)

            anti_phase_rew = anti_phase_rew * gait_gate

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
        lin_vel_z_cost = self.imu_lin_vel_b[:, 2] ** 2
        ang_vel_xy_cost = torch.sum(self.imu_ang_vel_b[:, :2] ** 2, dim=1)
        flat_ori_cost = torch.sum(self.up_b[:, :2] ** 2, dim=1)  # up_b ~= [0,0,1] when upright

        # --- Smoothness costs ---
        action_rate_cost = torch.sum((self.actions - self.prev_actions) ** 2, dim=1)
        dof_vel_cost = torch.sum(self.act_vel ** 2, dim=1)
        
        # Penalize velocity changes (not acceleration, to avoid dt sensitivity)
        dof_vel_delta = self.act_vel - self.prev_act_vel
        dof_vel_delta_cost = torch.sum(dof_vel_delta * dof_vel_delta, dim=1)

        joint_torques = self.robot.data.applied_torque[:, self._joint_dof_idx]
        energy_cost = torch.sum(torch.abs(joint_torques * self.act_vel), dim=1)

        self.prev_act_vel[:] = self.act_vel

        # --- Contact-based rewards (air-time, slip, undesired contacts) ---
        self._init_feet()

        # Latest normal forces (history index 0 is most recent)
        forces_hist = self._contact_sensor.data.net_forces_w_history  # (N,T,B,3)
        foot_forces = forces_hist[:, 0, self._feet_sensor_ids, :]     # (N,2,3)
        
        # Use positive vertical force component for contact detection
        foot_force_z = torch.clamp(foot_forces[:, :, 2], min=0.0)  # (N,2)
        foot_contact = foot_force_z > self.cfg.foot_contact_force_thresh  # (N,2)

        # (1) air-time reward at touchdown
        air_time = self._contact_sensor.data.current_air_time[:, self._feet_sensor_ids]  # (N,2)
        touchdown = foot_contact & (~self.prev_foot_contact)
        air_rew = torch.sum(torch.clamp(air_time - self.cfg.min_air_time, min=0.0) * touchdown.float(), dim=1)
        air_time_sym_pen = (air_time[:, 0] - air_time[:, 1]) ** 2
        air_rew = air_rew * air_time_gate
        air_time_sym_pen = air_time_sym_pen * air_time_gate

        # (2) slip penalty when in contact (horizontal foot speed)
        feet_vel_w = self.robot.data.body_lin_vel_w[:, self._feet_body_ids, :]  # (N,2,3)
        slip_speed_sq = torch.sum(feet_vel_w[..., :2] ** 2, dim=-1)            # (N,2)
        slip_cost = torch.sum(slip_speed_sq * foot_contact.float(), dim=1)     # (N,)

        # --- Anti-stomp: penalize hard touchdowns ---
        downward_speed = torch.clamp(-feet_vel_w[..., 2], min=0.0)  # (N,2)
        v_ref = float(getattr(self.cfg, "touchdown_vel_ref", 0.6))
        touchdown_vel_cost = torch.sum(
            ((downward_speed / (v_ref + 1e-6)) ** 2) * touchdown.float(),
            dim=1,
        )

        speed_gate = (cmd_speed > float(getattr(self.cfg, "touchdown_min_cmd_speed", 0.15))).float()
        touchdown_vel_cost = touchdown_vel_cost * speed_gate

        touchdown_force_scale = float(getattr(self.cfg, "touchdown_force_cost_scale", 0.0))
        touchdown_force_cost = torch.zeros_like(touchdown_vel_cost)
        if touchdown_force_scale > 0.0:
            f_thresh = float(getattr(self.cfg, "touchdown_force_thresh", 120.0))
            excess_fz = torch.clamp(foot_force_z - f_thresh, min=0.0)
            touchdown_force_cost = torch.sum(
                ((excess_fz / (f_thresh + 1e-6)) ** 2) * touchdown.float(),
                dim=1,
            ) * speed_gate

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

        foot_force_z = torch.clamp(foot_forces[:, :, 2], min=0.0)  # (N,2)
        foot_contact = foot_force_z > self.cfg.foot_contact_force_thresh  # (N,2)

        # left-right contact signal in {-1, 0, +1}
        contact_signal = foot_contact[:, 0].float() - foot_contact[:, 1].float()
        contact_target = torch.sin(phase)
        contact_phase_rew = torch.exp(-((contact_signal - contact_target) ** 2) / (self.cfg.contact_phase_sigma + 1e-6))
        contact_phase_rew = contact_phase_rew * gait_gate

        # (0) explicit no-fly penalty: both feet off ground
        no_fly = (foot_contact.sum(dim=1) == 0).float()  # (N,)

        # --- Combine all rewards ---
        reward = (
            self.cfg.lin_vel_reward_scale * r_lin # reward for linear velocity tracking
            + self.cfg.yaw_rate_reward_scale * r_yaw # reward for yaw tracking
            + self.cfg.upright_reward_scale * upright # reward for staying upright (measured from IMU)
            + self.cfg.alive_reward # reward for liveness
            # - self.cfg.action_cost_scale * act_cost # penalty for large actions
            - self.cfg.joint_limit_cost_scale * at_limit # penalty for being at joint limits
            # - pose_return_penalty # penalty for deviating from default pose (encourages natural stance and self-righting)
            # - self.cfg.lin_vel_z_cost_scale * lin_vel_z_cost # no jumping/hopping: penalize vertical velocity
            # # - self.cfg.ang_vel_xy_cost_scale * ang_vel_xy_cost 
            # # - self.cfg.flat_ori_cost_scale * flat_ori_cost
            # - self.cfg.action_rate_cost_scale * action_rate_cost
            # # - self.cfg.dof_vel_cost_scale * dof_vel_cost
            # # - self.cfg.dof_vel_delta_cost_scale * dof_vel_delta_cost
            # - self.cfg.energy_cost_scale * energy_cost
            # # - self.cfg.standstill_penalty_scale * standstill # helps exploration early on by rewarding any movement, but eventually encourages matching the command speed
            # # - self.cfg.speed_shortfall_cost_scale * speed_shortfall # penalty for not matching cmd speed
            # - self.cfg.symmetry_cost_scale * sym_pen
            # # - self.cfg.thigh_pose_cost_scale * thigh_pose_pen
            + self.cfg.feet_air_time_reward_scale * air_rew
            # + self.cfg.anti_phase_reward_scale * anti_phase_rew
            # + self.cfg.contact_phase_reward_scale * contact_phase_rew
            # - self.cfg.air_time_symmetry_cost_scale * air_time_sym_pen
            # # - self.cfg.walk_hip2_cost_scale * walk_hip2_pen
            # - self.cfg.foot_slip_cost_scale * slip_cost
            # - self.cfg.undesired_contact_cost_scale * undesired
            # - self.cfg.touchdown_cost_scale * touchdown_vel_cost
            # - touchdown_force_scale * touchdown_force_cost
            # - self.cfg.no_fly_cost_scale * no_fly
            # + stand_cmd_f
            # * (
                # self.cfg.stand_pose_reward_scale * stand_pose_rew
                # + self.cfg.stand_upright_reward_scale * stand_upright_rew
                # - self.cfg.stand_vel_cost_scale * stand_vel_cost
                # - self.cfg.stand_action_cost_scale * stand_action_cost
                # - self.cfg.stand_hip2_cost_scale * stand_hip2_pen
            # )
        )

        reward = torch.where(self.reset_terminated, torch.ones_like(reward) * self.cfg.death_cost, reward)

        # --- Logging (every N steps) ---
        if (
            bool(getattr(self.cfg, "enable_reward_logging", False))
            and hasattr(self, "common_step_counter")
            and (self.common_step_counter % 200) == 0
        ):
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
            self.extras["reward_penalties/energy"] = float(energy_cost.mean().item())
            self.extras["reward_penalties/standstill"] = float(standstill.mean().item())
            self.extras["reward_penalties/symmetry"] = float(sym_pen.mean().item())
            self.extras["reward_penalties/thigh_pose"] = float(thigh_pose_pen.mean().item())
            self.extras["reward_penalties/air_time_symmetry"] = float(air_time_sym_pen.mean().item())
            self.extras["reward_penalties/slip"] = float(slip_cost.mean().item())
            self.extras["reward_penalties/undesired_contact"] = float(undesired.mean().item())
            self.extras["reward_penalties/touchdown_vel"] = float(touchdown_vel_cost.mean().item())
            self.extras["reward_penalties/touchdown_force"] = float(touchdown_force_cost.mean().item())

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
            self.extras["reward_scaled/energy_cost"] = float((self.cfg.energy_cost_scale * energy_cost).mean().item())
            self.extras["reward_scaled/symmetry_cost"] = float((self.cfg.symmetry_cost_scale * sym_pen).mean().item())
            self.extras["reward_scaled/thigh_pose_cost"] = float((self.cfg.thigh_pose_cost_scale * thigh_pose_pen).mean().item())
            self.extras["reward_scaled/air_time_symmetry_cost"] = float((self.cfg.air_time_symmetry_cost_scale * air_time_sym_pen).mean().item())
            self.extras["reward_scaled/touchdown_vel_cost"] = float((self.cfg.touchdown_cost_scale * touchdown_vel_cost).mean().item())
            self.extras["reward_scaled/touchdown_force_cost"] = float((touchdown_force_scale * touchdown_force_cost).mean().item())

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
            self.extras["diagnostics/touchdown_rate"] = float(touchdown.float().mean().item())
            self.extras["diagnostics/downward_speed_at_touchdown"] = float(
                (downward_speed * touchdown.float()).sum(dim=1).mean().item()
            )
            self.extras["diagnostics/curriculum_stage"] = float(self._current_curriculum_stage)

        return reward

    def _get_dones(self):
        self._update_state()
        time_out = self.episode_length_buf >= self.randomized_episode_lengths - 1

        fell = self.track_pos_w[:, 2] < self.cfg.termination_height
        too_tilted = self.up_b[:, 2] < self.cfg.upright_threshold

        died = fell | too_tilted

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.sim.device, dtype=torch.long)

        if self.adr is not None and env_ids.numel() > 0:
            if self._global_policy_step > 0:
                timed_out = self.episode_length_buf[env_ids] >= (self.randomized_episode_lengths[env_ids] - 1)
                batch_success = timed_out.float().mean().item()
                ep_len = self._ep_len[env_ids].clamp(min=1.0)
                batch_lin_err = (self._ep_lin_err_sum[env_ids] / ep_len).mean().item()
                batch_yaw_err = (self._ep_yaw_err_sum[env_ids] / ep_len).mean().item()
                self._maybe_update_adr(batch_success, batch_lin_err, batch_yaw_err)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._resample_episode_lengths(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        # Initial pose randomization (ADR)
        if self.adr is not None:
            jpos_w = float(self.adr.get_custom("robot_spawn", "joint_pos_noise"))
            jvel_w = float(self.adr.get_custom("robot_spawn", "joint_vel_noise"))

            if jpos_w > 0.0:
                noise = torch.empty_like(joint_pos).uniform_(-jpos_w, jpos_w)
                joint_pos = joint_pos + noise

                lo = self.robot.data.soft_joint_pos_limits[0, :, 0].unsqueeze(0)
                hi = self.robot.data.soft_joint_pos_limits[0, :, 1].unsqueeze(0)
                joint_pos = torch.clamp(joint_pos, lo, hi)

            if jvel_w > 0.0:
                joint_vel = joint_vel + torch.empty_like(joint_vel).uniform_(-jvel_w, jvel_w)

        root = self.robot.data.default_root_state[env_ids]
        root[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._invalidate_state_cache()

        if self._refresh_runtime_masses_on_reset:
            self._cache_body_masses()

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.action_latency_steps[env_ids] = 0
        if self.action_hist_buf is not None:
            self.action_hist_buf[env_ids] = 0.0
        self.prev_act_vel[env_ids] = 0.0
        self._ep_lin_err_sum[env_ids] = 0.0
        self._ep_yaw_err_sum[env_ids] = 0.0
        self._ep_len[env_ids] = 0.0

        if self._motion_reference_enabled:
            if bool(getattr(self.cfg, "motion_reference_random_start", True)):
                self.motion_start_frame[env_ids] = torch.randint(
                    0,
                    self._motion_num_frames,
                    (env_ids.numel(),),
                    device=self.sim.device,
                )
            else:
                self.motion_start_frame[env_ids] = 0
            self.motion_frame[env_ids] = self.motion_start_frame[env_ids]

        # reset push timers
        self.push_counters[env_ids] = 0
        self.next_push_steps[env_ids] = torch.randint(
            self._push_interval_steps_min,
            self._push_interval_steps_max + 1,
            (env_ids.numel(),),
            device=self.sim.device,
        )

        # sample per-episode DR variables (strength, imu bias)
        self._sample_custom_dr_for_resets(env_ids)

        # Sample commands based on current curriculum stage
        self._sample_commands(env_ids)

        if bool(getattr(self.cfg, "randomize_phase", False)):
            self.phase_offset[env_ids] = torch.rand(env_ids.numel(), device=self.sim.device) * (2.0 * torch.pi)
        else:
            self.phase_offset[env_ids] = 0.0

        if self._visualization_enabled or self.obs_stack_buf is not None or self.obs_hist_buf is not None:
            self._update_state()

        # Update state before visualization so tracking/IMU state exists
        if self._visualization_enabled:
            self._visualize_markers()

        obs0 = None
        if self.obs_hist_buf is not None or self.obs_stack_buf is not None:
            obs0 = self._compute_single_observation()

        if self.obs_hist_buf is not None:
            self.obs_hist_buf[env_ids] = obs0[env_ids].unsqueeze(-1).expand(-1, -1, self.obs_max_latency + 1)

        if self.obs_stack_buf is not None:
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
        zero_prob = float(getattr(self.cfg, "zero_command_probability", 0.0))
        turn_prob = float(getattr(self.cfg, "turn_in_place_probability", 0.0))
        turn_min_stage = int(getattr(self.cfg, "turn_in_place_min_stage", 1))
        turn_mask = None
        zero_mask = None
        if turn_prob > 0.0 and self._current_curriculum_stage >= turn_min_stage:
            turn_mask = torch.rand(n, device=self.sim.device) < turn_prob
        if zero_prob > 0.0:
            zero_mask = torch.rand(n, device=self.sim.device) < zero_prob
            if turn_mask is not None:
                zero_mask = zero_mask & (~turn_mask)

        if not self.cfg.use_curriculum:
            # No curriculum: sample full ranges
            self.commands[env_ids, 0] = torch.empty(n, device=self.sim.device).uniform_(-1.0, 1.0)  # vx
            self.commands[env_ids, 1] = torch.empty(n, device=self.sim.device).uniform_(-0.5, 0.5)  # vy
            self.commands[env_ids, 2] = torch.empty(n, device=self.sim.device).uniform_(-1.0, 1.0)  # yaw rate
            if zero_mask is not None:
                self.commands[env_ids[zero_mask]] = 0.0
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

        command_scale = float(self._command_scale)
        min_stage = int(getattr(self.cfg, "adr_command_scale_min_stage", 0))
        if self._current_curriculum_stage < min_stage:
            command_scale = 1.0
        if command_scale != 1.0:
            self.commands[env_ids] *= command_scale

        if zero_mask is not None:
            self.commands[env_ids[zero_mask]] = 0.0

        if turn_mask is not None:
            yaw_min = float(getattr(self.cfg, "turn_in_place_yaw_min", 0.3))
            yaw_max = float(getattr(self.cfg, "turn_in_place_yaw_max", 1.0))
            yaw = torch.empty(n, device=self.sim.device).uniform_(-yaw_max, yaw_max)
            yaw = torch.where(torch.abs(yaw) < yaw_min, torch.sign(yaw) * yaw_min, yaw)
            self.commands[env_ids[turn_mask], 0] = 0.0
            self.commands[env_ids[turn_mask], 1] = 0.0
            self.commands[env_ids[turn_mask], 2] = yaw[turn_mask]

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

        # Ensure state is updated before accessing tracking/IMU state
        if not hasattr(self, "track_pos_w") or not hasattr(self, "imu_quat_w"):
            return

        if self._visualize_all_envs:
            env_ids = torch.arange(self.num_envs, device=self.sim.device)
        else:
            env_id = int(getattr(self.cfg, "debug_env_id", 0))
            env_id = max(0, min(env_id, self.num_envs - 1))
            env_ids = torch.tensor([env_id], device=self.sim.device)

        # Get single env data
        torso_pos = self.track_pos_w[env_ids]  # [N,3]
        torso_quat = self.imu_quat_w[env_ids]  # [N,4]
        cmd = self.commands[env_ids]  # [N,3]
        if self._track_com_linear and hasattr(self, "com_lin_vel_w"):
            vel_w = self.com_lin_vel_w[env_ids]  # [N,3]
        else:
            vel_w = self.imu_lin_vel_w[env_ids]  # [N,3]

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
