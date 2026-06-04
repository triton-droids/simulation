# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from copy import deepcopy
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


def _lerp_range(a, b, t: float):
    if isinstance(a, dict) and isinstance(b, dict):
        return {key: _lerp_range(a[key], b[key], t) for key in a.keys()}
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(_lerp_range(x, y, t) for x, y in zip(a, b))
    if isinstance(a, list) and isinstance(b, list):
        return [_lerp_range(x, y, t) for x, y in zip(a, b)]
    return _lerp(float(a), float(b), t)


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
                term_cfg.params[param_name] = _lerp_range(base_range, max_range, t)
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

        self._hip_action_ids = self._action_ids_for_joint_names(
            [
                "left_hip1_joint", "right_hip1_joint",
                "left_hip2_joint", "right_hip2_joint",
                "left_thigh_joint", "right_thigh_joint",
            ],
            joint_id_to_action,
        )
        self._knee_action_ids = self._action_ids_for_joint_names(
            ["left_knee_joint", "right_knee_joint"],
            joint_id_to_action,
        )
        self._ankle_action_ids = self._action_ids_for_joint_names(
            ["left_ankle_joint", "right_ankle_joint"],
            joint_id_to_action,
        )

        # IMU body is named "world" in your URDF
        imu_ids, _ = self.robot.find_bodies("world")
        self._imu_body_idx = int(imu_ids[0])

        self._use_frame_transformer_sensor = bool(getattr(self.cfg, "use_frame_transformer_sensor", False))
        if self._use_frame_transformer_sensor:
            site_names = self.scene["ee_site"].data.target_frame_names
            if "top" not in site_names:
                raise RuntimeError(f"'top' not found in ee_site target_frame_names: {site_names}")
            self._top_frame_idx = site_names.index("top")
        else:
            torso_ids, _ = self.robot.find_bodies(getattr(self.cfg, "track_body_name", "torso"))
            if len(torso_ids) == 0:
                raise RuntimeError(f"Tracking body '{getattr(self.cfg, 'track_body_name', 'torso')}' was not found.")
            self._track_body_idx = int(torso_ids[0])
            self._track_offset_b = torch.tensor(self.cfg.track_body_offset, device=self.sim.device).unsqueeze(0)

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
        self.last_reward_terms: dict[str, torch.Tensor] = {}
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
        self._obs_debug_dim = offset

        if bool(getattr(self.cfg, "debug_print_orderings", False)):
            try:
                joint_names = self.robot.data.joint_names
            except Exception:
                joint_names = None
            print("[DebugOrder] joint dof ids:", self._joint_dof_idx)
            print("[DebugOrder] default actuated pose:", self.default_actuated_pos)
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
        self.forward_progress_vel = torch.zeros(self.num_envs, device=self.sim.device)
        self.forward_progress_reward = torch.zeros(self.num_envs, device=self.sim.device)
        self.forward_stuck_progress_ema = torch.zeros(self.num_envs, device=self.sim.device)
        self.forward_stuck_penalty = torch.zeros(self.num_envs, device=self.sim.device)
        self.forward_stuck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)
        self._forward_progress_prev_pos_w = torch.zeros(self.num_envs, 3, device=self.sim.device)
        self._forward_progress_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)
        self._forward_progress_metric_step = -1
        self._forward_stuck_warmup_steps = max(
            1,
            int(float(getattr(self.cfg, "forward_stuck_warmup_s", 0.75)) / self._control_dt),
        )
        window_s = max(float(getattr(self.cfg, "forward_stuck_window_s", 0.5)), self._control_dt)
        self._forward_stuck_ema_alpha = min(1.0, self._control_dt / window_s)

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
            self._domain_randomization_mode = str(
                getattr(self.cfg, "domain_randomization_mode", "adaptive")
            ).lower()
            if self._domain_randomization_mode not in ("adaptive", "fixed"):
                raise ValueError(
                    "domain_randomization_mode must be 'adaptive' or 'fixed', "
                    f"got {self._domain_randomization_mode!r}."
                )
            self.adr = LocomotionADR(
                self.event_manager,
                self.cfg.adr_event_cfg_dict,
                self.cfg.adr_custom_cfg_dict,
                num_increments=self.cfg.num_adr_increments,
            )
            if self._domain_randomization_mode == "fixed":
                self.adr.set_num_increments(self.cfg.num_adr_increments)
            else:
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

        # Contact sensors are optional. Locomotion rewards are kinematic/distance-based by default.
        self._contact_sensor = None
        if bool(getattr(self.cfg, "enable_contact_sensor", False)):
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.articulations["robot"] = self.robot
        if self._contact_sensor is not None:
            self.scene.sensors["contact_sensor"] = self._contact_sensor
        if bool(getattr(self.cfg, "use_frame_transformer_sensor", False)):
            self._ee_site_sensor = FrameTransformer(self.cfg.ee_site)
            self.scene.sensors["ee_site"] = self._ee_site_sensor

        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        self._disable_non_foot_collisions()
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _disable_non_foot_collisions(self) -> None:
        """Disable collider prims for non-foot links in the referenced USD asset."""
        if not bool(getattr(self.cfg, "disable_non_foot_collisions", True)):
            return

        from pxr import UsdPhysics

        stage = sim_utils.get_current_stage()
        root_regex = self.cfg.robot.prim_path
        for body_name in getattr(self.cfg, "non_foot_collision_body_names", ()):
            body_paths = sim_utils.find_matching_prim_paths(f"{root_regex}/{body_name}")
            for body_path in body_paths:
                body_prim = stage.GetPrimAtPath(body_path)
                if not body_prim.IsValid():
                    continue
                prims = [body_prim]
                while prims:
                    prim = prims.pop(0)
                    if prim.IsInstance():
                        continue
                    collision_api = UsdPhysics.CollisionAPI(prim)
                    if collision_api:
                        collision_api.GetCollisionEnabledAttr().Set(False)
                    prims.extend(prim.GetChildren())

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
        if getattr(self, "_domain_randomization_mode", "adaptive") == "fixed":
            self.extras["log"]["adr_increments"] = int(self.adr.num_increments())
            self.extras["log"]["adr_difficulty"] = float(self.adr.difficulty())
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
        self.extras["log"]["adr_difficulty"] = float(self.adr.difficulty())

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
        self._capture_forward_progress_reference()
        self._invalidate_state_cache()
        self._forward_progress_metric_step = -1

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
        # B) Tracking state from torso-offset point
        # ------------------------------------------
        if self._use_frame_transformer_sensor:
            ft = self.scene["ee_site"].data
            self.track_pos_w = ft.target_pos_w[:, self._top_frame_idx]
        else:
            torso_pos_w = self.robot.data.body_pos_w[:, self._track_body_idx]
            torso_quat_w = self.robot.data.body_quat_w[:, self._track_body_idx]
            self.track_pos_w = torso_pos_w + quat_rotate(torso_quat_w, self._track_offset_b.expand(self.num_envs, -1))

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

        lo = self.actuated_lower.unsqueeze(0)
        hi = self.actuated_upper.unsqueeze(0)
        self.act_pos_scaled = 2.0 * (self.act_pos - lo) / (hi - lo + 1e-6) - 1.0
        self._state_valid = True

    def _forward_progress_position_w(self) -> torch.Tensor:
        if self._track_com_linear and hasattr(self, "com_pos_w"):
            return self.com_pos_w
        return self.track_pos_w

    def _capture_forward_progress_reference(self) -> None:
        self._update_state()
        self._forward_progress_prev_pos_w[:] = self._forward_progress_position_w()
        self._forward_progress_initialized[:] = True

    def _update_forward_progress_metrics(self) -> None:
        if self._forward_progress_metric_step == self._global_policy_step:
            return

        self._update_state()
        pos_w = self._forward_progress_position_w()
        initialized = self._forward_progress_initialized
        delta_w = torch.where(
            initialized.unsqueeze(-1),
            pos_w - self._forward_progress_prev_pos_w,
            torch.zeros_like(pos_w),
        )
        delta_b = quat_rotate_inverse(self.imu_quat_w, delta_w)
        if self._use_cmd_yaw_offset:
            delta_cmd = self._rotate_xy(delta_b, self._cmd_yaw_cos, self._cmd_yaw_sin)
        else:
            delta_cmd = delta_b

        progress_vel = delta_cmd[:, 0] / self._control_dt
        cmd_threshold = float(getattr(self.cfg, "forward_progress_command_threshold", 0.15))
        forward_cmd = self.commands[:, 0] > cmd_threshold
        denom = torch.clamp(self.commands[:, 0], min=cmd_threshold)
        progress_reward = torch.clamp(progress_vel / denom, min=-1.0, max=1.0) * forward_cmd.float()

        alpha = float(self._forward_stuck_ema_alpha)
        next_ema = (1.0 - alpha) * self.forward_stuck_progress_ema + alpha * progress_vel
        self.forward_stuck_progress_ema = torch.where(forward_cmd, next_ema, torch.zeros_like(next_ema))

        min_vel = float(getattr(self.cfg, "forward_progress_min_vel", 0.08))
        warm = self.episode_length_buf >= self._forward_stuck_warmup_steps
        stuck = forward_cmd & warm & (self.forward_stuck_progress_ema < min_vel)
        if not bool(getattr(self.cfg, "forward_stuck_termination_enabled", True)):
            stuck = torch.zeros_like(stuck)

        self.forward_progress_vel = progress_vel
        self.forward_progress_reward = progress_reward
        self.forward_stuck = stuck
        self.forward_stuck_penalty = (forward_cmd & warm & (self.forward_stuck_progress_ema < min_vel)).float()
        self._forward_progress_metric_step = self._global_policy_step

    def _action_ids_for_joint_names(self, names: list[str], joint_id_to_action: dict[int, int]) -> torch.Tensor:
        ids = []
        for name in names:
            joint_ids, _ = self.robot.find_joints(name)
            if len(joint_ids) == 0:
                continue
            joint_id = int(joint_ids[0])
            if joint_id in joint_id_to_action:
                ids.append(joint_id_to_action[joint_id])
        return torch.tensor(ids, device=self.sim.device, dtype=torch.long)

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

    def _compute_reward_terms(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._update_state()
        self._update_forward_progress_metrics()

        # --- Command and fixed-height tracking ---
        x_err = self.com_lin_vel_cmd[:, 0] - self.commands[:, 0]
        y_err = self.com_lin_vel_cmd[:, 1] - self.commands[:, 1]
        yaw_err = self.com_ang_vel_cmd[:, 2] - self.commands[:, 2]

        tracking_sigma = self.cfg.tracking_sigma + 1e-6
        r_x = torch.exp(-(x_err * x_err) / tracking_sigma)
        r_y = torch.exp(-(y_err * y_err) / tracking_sigma)
        r_yaw = torch.exp(-(yaw_err * yaw_err) / tracking_sigma)
        height_err = self.track_pos_w[:, 2] - self.cfg.base_height_target
        r_height = torch.exp(-(height_err * height_err) / (self.cfg.height_tracking_sigma + 1e-6))

        self._ep_lin_err_sum += torch.sqrt(x_err * x_err + y_err * y_err)
        self._ep_yaw_err_sum += torch.abs(yaw_err)
        self._ep_len += 1.0

        # --- Base stability ---
        lin_vel_z_cost = self.imu_lin_vel_b[:, 2] ** 2
        ang_vel_xy_cost = torch.sum(self.imu_ang_vel_b[:, :2] ** 2, dim=1)
        flat_ori_cost = torch.sum(self.up_b[:, :2] ** 2, dim=1)

        # --- Joint posture regularization ---
        pose_err = self.act_pos - self.default_actuated_pos.unsqueeze(0)
        hip_dev = torch.zeros(self.num_envs, device=self.sim.device)
        if self._hip_action_ids.numel() > 0:
            hip_dev = torch.mean(pose_err[:, self._hip_action_ids] ** 2, dim=1)

        knee_dev = torch.zeros(self.num_envs, device=self.sim.device)
        if self._knee_action_ids.numel() > 0:
            knee_dev = torch.mean(pose_err[:, self._knee_action_ids] ** 2, dim=1)

        ankle_dev = torch.zeros(self.num_envs, device=self.sim.device)
        if self._ankle_action_ids.numel() > 0:
            ankle_dev = torch.mean(pose_err[:, self._ankle_action_ids] ** 2, dim=1)

        # --- Smoothness / effort ---
        action_rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=1)
        dof_vel = torch.sum(self.act_vel ** 2, dim=1)
        dof_acc = torch.sum(((self.act_vel - self.prev_act_vel) / self._control_dt) ** 2, dim=1)

        joint_torques = self.robot.data.applied_torque[:, self._joint_dof_idx]
        torques = torch.sum(joint_torques ** 2, dim=1)
        joint_power = torch.sum(torch.abs(joint_torques * self.act_vel), dim=1)

        self.prev_act_vel[:] = self.act_vel

        # --- Limits ---
        limit_excess = torch.clamp(torch.abs(self.act_pos_scaled) - self.cfg.soft_dof_pos_limit, min=0.0)
        dof_pos_limits = torch.sum(limit_excess ** 2, dim=1)

        # --- Distance-based feet shaping ---
        self._init_feet()

        feet_pos_w = self.robot.data.body_pos_w[:, self._feet_body_ids, :]
        feet_vel_w = self.robot.data.body_lin_vel_w[:, self._feet_body_ids, :]
        foot_height = feet_pos_w[:, :, 2] - float(self.cfg.foot_ground_height_target)
        foot_speed_xy = torch.linalg.norm(feet_vel_w[:, :, :2], dim=-1)

        min_foot_height = torch.min(torch.abs(foot_height), dim=1).values
        r_support_height = torch.exp(-(min_foot_height * min_foot_height) / (self.cfg.foot_height_sigma + 1e-6))

        swing_weight = torch.clamp(
            (foot_speed_xy - self.cfg.foot_swing_speed_threshold)
            / (self.cfg.foot_swing_speed_threshold + 1e-6),
            min=0.0,
            max=1.0,
        )
        clearance_err = foot_height - self.cfg.foot_clearance_target
        feet_clearance_cost = torch.sum(swing_weight * clearance_err * clearance_err, dim=1)

        near_ground_weight = torch.exp(
            -(foot_height * foot_height) / (self.cfg.foot_height_sigma + 1e-6)
        )
        feet_near_ground_velocity = torch.sum(near_ground_weight * foot_speed_xy * foot_speed_xy, dim=1)

        feet_distance = torch.linalg.norm(feet_pos_w[:, 0, :2] - feet_pos_w[:, 1, :2], dim=1)
        feet_distance_cost = (
            torch.clamp(self.cfg.feet_distance_min - feet_distance, min=0.0) ** 2
            + torch.clamp(feet_distance - self.cfg.feet_distance_max, min=0.0) ** 2
        )

        knee_pos_w = self.robot.data.body_pos_w[:, self._knee_body_ids, :]
        knee_distance = torch.linalg.norm(knee_pos_w[:, 0, :2] - knee_pos_w[:, 1, :2], dim=1)
        knee_distance_cost = (
            torch.clamp(self.cfg.knee_distance_min - knee_distance, min=0.0) ** 2
            + torch.clamp(knee_distance - self.cfg.knee_distance_max, min=0.0) ** 2
        )

        # --- Stand behavior for explicit zero commands ---
        stand_cmd = (
            torch.norm(self.commands[:, :2], dim=1) < self.cfg.stand_cmd_lin_thresh
        ) & (
            torch.abs(self.commands[:, 2]) < self.cfg.stand_cmd_yaw_thresh
        )

        stand_still_cost = (
            torch.sum(self.com_lin_vel_cmd[:, :2] ** 2, dim=1)
            + 0.5 * self.com_ang_vel_cmd[:, 2] ** 2
            + 0.05 * torch.sum(self.actions ** 2, dim=1)
        ) * stand_cmd.float()

        terms = {
            "tracking_x": r_x,
            "tracking_y": r_y,
            "tracking_yaw": r_yaw,
            "tracking_height": r_height,
            "lin_vel_z": lin_vel_z_cost,
            "ang_vel_xy": ang_vel_xy_cost,
            "orientation": flat_ori_cost,
            "hip_dev": hip_dev,
            "knee_dev": knee_dev,
            "ankle_dev": ankle_dev,
            "action_rate": action_rate,
            "dof_vel": dof_vel,
            "dof_acc": dof_acc,
            "torques": torques,
            "joint_power": joint_power,
            "dof_pos_limits": dof_pos_limits,
            "feet_support_height": r_support_height,
            "feet_clearance": feet_clearance_cost,
            "feet_near_ground_velocity": feet_near_ground_velocity,
            "feet_distance": feet_distance,
            "feet_distance_cost": feet_distance_cost,
            "knee_distance": knee_distance,
            "knee_distance_cost": knee_distance_cost,
            "stand_still": stand_still_cost,
            "forward_progress_vel": self.forward_progress_vel,
            "forward_progress": self.forward_progress_reward,
            "forward_stuck_progress_ema": self.forward_stuck_progress_ema,
            "forward_stuck_penalty": self.forward_stuck_penalty,
            "forward_stuck": self.forward_stuck.float(),
            "foot_height_left": foot_height[:, 0],
            "foot_height_right": foot_height[:, 1],
            "x_err_abs": torch.abs(x_err),
            "y_err_abs": torch.abs(y_err),
            "yaw_err_abs": torch.abs(yaw_err),
            "height_err_abs": torch.abs(height_err),
        }

        scaled_terms = {
            "scaled_tracking_x": self.cfg.tracking_x_vel_scale * r_x,
            "scaled_tracking_y": self.cfg.tracking_y_vel_scale * r_y,
            "scaled_tracking_yaw": self.cfg.tracking_ang_vel_scale * r_yaw,
            "scaled_tracking_height": self.cfg.tracking_base_height_scale * r_height,
            "scaled_lin_vel_z": self.cfg.lin_vel_z_scale * lin_vel_z_cost,
            "scaled_ang_vel_xy": self.cfg.ang_vel_xy_scale * ang_vel_xy_cost,
            "scaled_orientation": self.cfg.orientation_scale * flat_ori_cost,
            "scaled_hip_dev": self.cfg.deviation_hip_joint_scale * hip_dev,
            "scaled_knee_dev": self.cfg.deviation_knee_joint_scale * knee_dev,
            "scaled_ankle_dev": self.cfg.deviation_ankle_joint_scale * ankle_dev,
            "scaled_action_rate": self.cfg.action_rate_scale * action_rate,
            "scaled_dof_vel": self.cfg.dof_vel_reward_scale * dof_vel,
            "scaled_dof_acc": self.cfg.dof_acc_scale * dof_acc,
            "scaled_torques": self.cfg.torques_scale * torques,
            "scaled_joint_power": self.cfg.joint_power_scale * joint_power,
            "scaled_dof_pos_limits": self.cfg.dof_pos_limits_scale * dof_pos_limits,
            "scaled_feet_support_height": self.cfg.feet_support_height_scale * r_support_height,
            "scaled_feet_clearance": self.cfg.feet_clearance_scale * feet_clearance_cost,
            "scaled_feet_near_ground_velocity": (
                self.cfg.feet_near_ground_velocity_scale * feet_near_ground_velocity
            ),
            "scaled_feet_distance": self.cfg.feet_distance_scale * feet_distance_cost,
            "scaled_knee_distance": self.cfg.knee_distance_scale * knee_distance_cost,
            "scaled_stand_still": self.cfg.stand_still_scale * stand_still_cost,
            "scaled_forward_progress": self.cfg.forward_progress_scale * self.forward_progress_reward,
            "scaled_forward_stuck": self.cfg.forward_stuck_penalty_scale * self.forward_stuck_penalty,
        }
        terms.update(scaled_terms)

        reward = sum(scaled_terms.values())

        reward = torch.where(self.reset_terminated, torch.ones_like(reward) * self.cfg.death_cost, reward)
        terms["total"] = reward
        terms["terminated"] = self.reset_terminated.float()
        return reward, terms

    def _get_rewards(self) -> torch.Tensor:
        reward, terms = self._compute_reward_terms()
        self.last_reward_terms = terms

        # --- Logging (every N steps) ---
        if (
            bool(getattr(self.cfg, "enable_reward_logging", False))
            and hasattr(self, "common_step_counter")
            and (self.common_step_counter % 200) == 0
        ):
            self.extras["reward_tracking/x"] = float(terms["tracking_x"].mean().item())
            self.extras["reward_tracking/y"] = float(terms["tracking_y"].mean().item())
            self.extras["reward_tracking/yaw"] = float(terms["tracking_yaw"].mean().item())
            self.extras["reward_tracking/height"] = float(terms["tracking_height"].mean().item())
            self.extras["reward_tracking/feet_support_height"] = float(terms["feet_support_height"].mean().item())

            self.extras["reward_penalties/lin_vel_z"] = float(terms["lin_vel_z"].mean().item())
            self.extras["reward_penalties/ang_vel_xy"] = float(terms["ang_vel_xy"].mean().item())
            self.extras["reward_penalties/flat_ori"] = float(terms["orientation"].mean().item())
            self.extras["reward_penalties/hip_dev"] = float(terms["hip_dev"].mean().item())
            self.extras["reward_penalties/knee_dev"] = float(terms["knee_dev"].mean().item())
            self.extras["reward_penalties/ankle_dev"] = float(terms["ankle_dev"].mean().item())
            self.extras["reward_penalties/action_rate"] = float(terms["action_rate"].mean().item())
            self.extras["reward_penalties/dof_vel"] = float(terms["dof_vel"].mean().item())
            self.extras["reward_penalties/dof_acc"] = float(terms["dof_acc"].mean().item())
            self.extras["reward_penalties/torques"] = float(terms["torques"].mean().item())
            self.extras["reward_penalties/joint_power"] = float(terms["joint_power"].mean().item())
            self.extras["reward_penalties/dof_pos_limits"] = float(terms["dof_pos_limits"].mean().item())
            self.extras["reward_penalties/feet_clearance"] = float(terms["feet_clearance"].mean().item())
            self.extras["reward_penalties/near_ground_velocity"] = float(
                terms["feet_near_ground_velocity"].mean().item()
            )
            self.extras["reward_penalties/feet_distance"] = float(terms["feet_distance_cost"].mean().item())
            self.extras["reward_penalties/knee_distance"] = float(terms["knee_distance_cost"].mean().item())
            self.extras["reward_penalties/stand_still"] = float(terms["stand_still"].mean().item())
            self.extras["reward_penalties/forward_stuck"] = float(terms["forward_stuck_penalty"].mean().item())

            self.extras["reward_scaled/x_tracking"] = float(terms["scaled_tracking_x"].mean().item())
            self.extras["reward_scaled/y_tracking"] = float(terms["scaled_tracking_y"].mean().item())
            self.extras["reward_scaled/yaw_tracking"] = float(terms["scaled_tracking_yaw"].mean().item())
            self.extras["reward_scaled/height_tracking"] = float(terms["scaled_tracking_height"].mean().item())
            self.extras["reward_scaled/feet_support_height"] = float(
                terms["scaled_feet_support_height"].mean().item()
            )
            self.extras["reward_scaled/feet_clearance"] = float(terms["scaled_feet_clearance"].mean().item())
            self.extras["reward_scaled/near_ground_velocity"] = float(
                terms["scaled_feet_near_ground_velocity"].mean().item()
            )
            self.extras["reward_scaled/feet_distance"] = float(terms["scaled_feet_distance"].mean().item())
            self.extras["reward_scaled/knee_distance"] = float(terms["scaled_knee_distance"].mean().item())
            self.extras["reward_scaled/stand_still"] = float(terms["scaled_stand_still"].mean().item())
            self.extras["reward_scaled/forward_progress"] = float(terms["scaled_forward_progress"].mean().item())
            self.extras["reward_scaled/forward_stuck"] = float(terms["scaled_forward_stuck"].mean().item())

            self.extras["reward_total/mean"] = float(reward.mean().item())
            self.extras["reward_total/std"] = float(reward.std().item())
            self.extras["reward_total/min"] = float(reward.min().item())
            self.extras["reward_total/max"] = float(reward.max().item())

            self.extras["diagnostics/x_err"] = float(terms["x_err_abs"].mean().item())
            self.extras["diagnostics/y_err"] = float(terms["y_err_abs"].mean().item())
            self.extras["diagnostics/yaw_err"] = float(terms["yaw_err_abs"].mean().item())
            self.extras["diagnostics/height_err"] = float(terms["height_err_abs"].mean().item())
            self.extras["diagnostics/feet_distance"] = float(terms["feet_distance"].mean().item())
            self.extras["diagnostics/knee_distance"] = float(terms["knee_distance"].mean().item())
            self.extras["diagnostics/torques"] = float(terms["torques"].mean().item())
            self.extras["diagnostics/joint_power"] = float(terms["joint_power"].mean().item())
            self.extras["diagnostics/curriculum_stage"] = float(self._current_curriculum_stage)
            self.extras["diagnostics/forward_progress_vel"] = float(terms["forward_progress_vel"].mean().item())
            self.extras["diagnostics/forward_stuck_progress_ema"] = float(
                terms["forward_stuck_progress_ema"].mean().item()
            )
            self.extras["diagnostics/forward_stuck"] = float(terms["forward_stuck"].mean().item())

        return reward

    def _get_dones(self):
        self._update_state()
        self._update_forward_progress_metrics()
        time_out = self.episode_length_buf >= self.randomized_episode_lengths - 1

        fell = self.track_pos_w[:, 2] < self.cfg.termination_height
        too_tilted = self.up_b[:, 2] < self.cfg.upright_threshold

        died = fell | too_tilted | self.forward_stuck

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
        self.forward_progress_vel[env_ids] = 0.0
        self.forward_progress_reward[env_ids] = 0.0
        self.forward_stuck_progress_ema[env_ids] = 0.0
        self.forward_stuck_penalty[env_ids] = 0.0
        self.forward_stuck[env_ids] = False
        self._forward_progress_prev_pos_w[env_ids] = 0.0
        self._forward_progress_initialized[env_ids] = False
        self._forward_progress_metric_step = -1

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
            vx_min, vx_max = self.cfg.lin_vel_x_range
            vy_min, vy_max = self.cfg.lin_vel_y_range
            yaw_min, yaw_max = self.cfg.ang_vel_yaw_range
            self.commands[env_ids, 0] = torch.empty(n, device=self.sim.device).uniform_(vx_min, vx_max)
            self.commands[env_ids, 1] = torch.empty(n, device=self.sim.device).uniform_(vy_min, vy_max)
            self.commands[env_ids, 2] = torch.empty(n, device=self.sim.device).uniform_(yaw_min, yaw_max)
            if zero_mask is not None:
                self.commands[env_ids[zero_mask]] = 0.0
            return

        # Curriculum active: sample based on stage
        stage = self._current_curriculum_stage

        if stage == 0:
            # Stage 0: forward-only, no yaw or lateral.
            # Encourage actual forward motion, not standing still
            vx_min = self.cfg.curriculum_stage0_vx_min
            vx_max = self.cfg.curriculum_stage0_vx_max
            self.commands[env_ids, 0] = torch.empty(n, device=self.sim.device).uniform_(vx_min, vx_max)
            self.commands[env_ids, 1] = 0.0
            self.commands[env_ids, 2] = 0.0

        elif stage == 1:
            # Stage 1: forward + yaw, no lateral.
            vx_min, vx_max = self.cfg.lin_vel_x_range
            yaw_min, yaw_max = self.cfg.ang_vel_yaw_range
            self.commands[env_ids, 0] = torch.empty(n, device=self.sim.device).uniform_(vx_min, vx_max)
            self.commands[env_ids, 1] = 0.0
            self.commands[env_ids, 2] = torch.empty(n, device=self.sim.device).uniform_(yaw_min, yaw_max)

        else:  # stage == 2
            # Stage 2: full command space.
            vx_min, vx_max = self.cfg.lin_vel_x_range
            vy_min, vy_max = self.cfg.lin_vel_y_range
            yaw_min, yaw_max = self.cfg.ang_vel_yaw_range
            self.commands[env_ids, 0] = torch.empty(n, device=self.sim.device).uniform_(vx_min, vx_max)
            self.commands[env_ids, 1] = torch.empty(n, device=self.sim.device).uniform_(vy_min, vy_max)
            self.commands[env_ids, 2] = torch.empty(n, device=self.sim.device).uniform_(yaw_min, yaw_max)

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
            yaw_sign = torch.where(yaw >= 0.0, torch.ones_like(yaw), -torch.ones_like(yaw))
            yaw = torch.where(torch.abs(yaw) < yaw_min, yaw_sign * yaw_min, yaw)
            self.commands[env_ids[turn_mask], 0] = 0.0
            self.commands[env_ids[turn_mask], 1] = 0.0
            self.commands[env_ids[turn_mask], 2] = yaw[turn_mask]

    def _init_feet(self):
        """Initialize body indices used by distance-based locomotion rewards."""
        if self._feet_inited:
            return

        foot_body_ids, foot_body_names = self.robot.find_bodies(self.cfg.foot_body_regex)
        if len(foot_body_ids) != 2:
            raise RuntimeError(f"Expected 2 feet from regex {self.cfg.foot_body_regex}, got {foot_body_names}")
        foot_pairs = sorted(
            zip(foot_body_ids, foot_body_names),
            key=lambda pair: (0 if "left" in pair[1] else 1, pair[1]),
        )
        self._feet_body_ids = torch.tensor([pair[0] for pair in foot_pairs], device=self.sim.device, dtype=torch.long)

        knee_body_ids, knee_body_names = self.robot.find_bodies(self.cfg.knee_body_regex)
        if len(knee_body_ids) != 2:
            raise RuntimeError(f"Expected 2 knee bodies from regex {self.cfg.knee_body_regex}, got {knee_body_names}")
        knee_pairs = sorted(
            zip(knee_body_ids, knee_body_names),
            key=lambda pair: (0 if "left" in pair[1] else 1, pair[1]),
        )
        self._knee_body_ids = torch.tensor([pair[0] for pair in knee_pairs], device=self.sim.device, dtype=torch.long)
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
