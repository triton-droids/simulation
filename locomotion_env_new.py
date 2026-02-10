"""
Humanoid Locomotion Validation Environment (Pure MuJoCo)
For validating IsaacSim-trained policies on CPU MuJoCo.

Policy order (interleaved, what policy observes/outputs):
  [left_hip1, right_hip1, left_hip2, right_hip2, left_thigh, right_thigh,
   left_knee, right_knee, left_ankle, right_ankle]

MuJoCo actuator order (left block then right block):
  [left_hip1, left_hip2, left_thigh, left_knee, left_ankle,
   right_hip1, right_hip2, right_thigh, right_knee, right_ankle]
"""

import math
import numpy as np
import mujoco


class HumanoidLocomotionEnv:
    """Pure MuJoCo locomotion validation environment."""

    def __init__(
        self,
        xml_path: str,
        frame_stack: int = 3,
        use_phase_obs: bool = False,
        gait_period_s: float = 1.0,
        include_height: bool = False,
        disturbance_force_max: float = 5.0,
        disturbance_torque_max: float = 2.0,
        disturbance_prob: float = 0.00,
        action_scale: float = 1.0,
        action_scale_by_joint: dict[str, float] = {
            "left_thigh_joint": 0.3,
            "right_thigh_joint": 0.3,
        },
        action_noise_std: float = 0.0,
        act_max_latency: int = 5,
        act_latency_steps: int = 0,
        obs_max_latency: int = 3,
        obs_latency_steps: int = 0,
        imu_bias_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        imu_bias_gyro: tuple[float, float, float] = (0.0, 0.0, 0.0),
        imu_mount_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        imu_mount_ang: float = 0.0,
        obs_noise_gravity_std: float = 0.0,
        obs_noise_gyro_std: float = 0.0,
        obs_noise_joint_pos_std: float = 0.0,
        obs_noise_joint_vel_std: float = 0.0,
        dt: float = 1 / 50,
        sim_dt: float = 1 / 250,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Timing matches IsaacLab setup: 250 Hz sim, 50 Hz control, decimation=5.
        self.model.opt.timestep = sim_dt
        self.dt = dt
        self.n_substeps = int(round(self.dt / self.model.opt.timestep))
        if not np.isclose(self.n_substeps * self.model.opt.timestep, self.dt):
            raise ValueError(
                f"control dt ({self.dt}) must be an integer multiple of sim dt ({self.model.opt.timestep})"
            )

        self._frame_stack = max(1, int(frame_stack))
        self._use_phase_obs = bool(use_phase_obs)
        self._gait_period_s = float(gait_period_s)
        self._include_height = bool(include_height)

        self._action_scale = float(action_scale)
        self._disturbance_force_max = float(disturbance_force_max)
        self._disturbance_torque_max = float(disturbance_torque_max)
        self._disturbance_prob = float(disturbance_prob)

        self._action_noise_std = float(action_noise_std)
        self._act_max_latency = max(0, int(act_max_latency))
        self._act_latency_steps = int(np.clip(act_latency_steps, 0, self._act_max_latency))

        self._obs_max_latency = max(0, int(obs_max_latency))
        self._obs_latency_steps = int(np.clip(obs_latency_steps, 0, self._obs_max_latency))

        self._imu_bias_gravity = np.asarray(imu_bias_gravity, dtype=float).copy()
        self._imu_bias_gyro = np.asarray(imu_bias_gyro, dtype=float).copy()
        self._imu_mount_axis = np.asarray(imu_mount_axis, dtype=float).copy()
        axis_norm = np.linalg.norm(self._imu_mount_axis)
        if axis_norm < 1e-8:
            self._imu_mount_axis[:] = np.array([1.0, 0.0, 0.0])
        else:
            self._imu_mount_axis /= axis_norm
        self._imu_mount_ang = float(imu_mount_ang)

        self._obs_noise_gravity_std = float(obs_noise_gravity_std)
        self._obs_noise_gyro_std = float(obs_noise_gyro_std)
        self._obs_noise_joint_pos_std = float(obs_noise_joint_pos_std)
        self._obs_noise_joint_vel_std = float(obs_noise_joint_vel_std)

        self._torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._torso_top_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")

        self._nu = self.model.nu
        self._nq = self.model.nq
        self._nv = self.model.nv

        self._q_joint_start = 7
        self._qd_joint_start = 6

        self._policy_joint_order = [
            "left_hip1_joint",
            "right_hip1_joint",
            "left_hip2_joint",
            "right_hip2_joint",
            "left_thigh_joint",
            "right_thigh_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_ankle_joint",
            "right_ankle_joint",
        ]

        if len(self._policy_joint_order) != self._nu:
            raise ValueError(
                f"policy_joint_order length ({len(self._policy_joint_order)}) must equal num actuators ({self._nu})"
            )

        self._model_joint_order = []
        for actuator_id in range(self._nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            self._model_joint_order.append(joint_name)

        self._policy_to_mj = np.array(
            [self._model_joint_order.index(jn) for jn in self._policy_joint_order],
            dtype=int,
        )
        self._mj_to_policy = np.argsort(self._policy_to_mj)

        self._joint_action_scales = np.ones(self._nu, dtype=float)
        if action_scale_by_joint:
            for actuator_id in range(self._nu):
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
                joint_id = int(self.model.actuator_trnid[actuator_id, 0])
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name in action_scale_by_joint:
                    self._joint_action_scales[actuator_id] = float(action_scale_by_joint[joint_name])
                elif actuator_name in action_scale_by_joint:
                    self._joint_action_scales[actuator_id] = float(action_scale_by_joint[actuator_name])

        standing_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "locomotion_standing_pose")
        self._standing_qpos = self.model.key_qpos[standing_key_id].copy()

        # Hard limits in *policy (interleaved)* order, read from model by joint name.
        # This keeps obs scaling and action clipping aligned with the policy indexing.
        self._joint_range_lower_policy = np.array(
            [
                float(
                    self.model.jnt_range[
                        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name), 0
                    ]
                )
                for joint_name in self._policy_joint_order
            ],
            dtype=float,
        )
        self._joint_range_upper_policy = np.array(
            [
                float(
                    self.model.jnt_range[
                        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name), 1
                    ]
                )
                for joint_name in self._policy_joint_order
            ],
            dtype=float,
        )

        self._soft_joint_limit_factor = 0.95
        self._joint_range_lower_mj = self._joint_range_lower_policy[self._mj_to_policy]
        self._joint_range_upper_mj = self._joint_range_upper_policy[self._mj_to_policy]

        self._ang_vel_scale = 0.25
        self._dof_vel_scale = 0.1

        # Commands are expected in body/command convention [vx, vy, yaw_rate].
        self._commands = np.array([1.0, 0.0, 0.0], dtype=float)

        self._command_yaw_offset = -math.pi / 2.0
        self._cmd_yaw_cos = math.cos(self._command_yaw_offset)
        self._cmd_yaw_sin = math.sin(self._command_yaw_offset)
        self._use_cmd_yaw_offset = abs(self._command_yaw_offset) > 1e-6

        self._single_frame_size = 3 + 3 + 3 + 3 + self._nu + self._nu + self._nu
        if self._use_phase_obs:
            self._single_frame_size += 2
        self._obs_size = self._single_frame_size * self._frame_stack

        # Action buffers.
        self._actions = np.zeros(self._nu, dtype=float)
        self._last_act = np.zeros(self._nu, dtype=float)
        self._act_hist_buf = (
            np.zeros((self._nu, self._act_max_latency + 1), dtype=float)
            if self._act_max_latency > 0
            else None
        )

        # Observation latency + stack buffers.
        self._obs_hist_buf = (
            np.zeros((self._single_frame_size, self._obs_max_latency + 1), dtype=float)
            if self._obs_max_latency > 0
            else None
        )
        self._obs_stack_buf = (
            np.zeros((self._single_frame_size, self._frame_stack), dtype=float)
            if self._frame_stack > 1
            else None
        )

        self._step_count = 0

        self._torso_mass = 3.175
        self._approx_inertia = 0.02

        # Cached state tensors used for observation construction.
        self.torso_quat_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.torso_lin_vel_w = np.zeros(3, dtype=float)
        self.torso_ang_vel_w = np.zeros(3, dtype=float)
        self.torso_lin_vel_b = np.zeros(3, dtype=float)
        self.torso_ang_vel_b = np.zeros(3, dtype=float)
        self.torso_lin_vel_cmd = np.zeros(3, dtype=float)
        self.torso_ang_vel_cmd = np.zeros(3, dtype=float)
        self.up_b = np.array([0.0, 0.0, 1.0], dtype=float)
        self.up_cmd = np.array([0.0, 0.0, 1.0], dtype=float)
        self.act_pos = np.zeros(self._nu, dtype=float)
        self.act_vel = np.zeros(self._nu, dtype=float)
        self.act_pos_scaled = np.zeros(self._nu, dtype=float)

        print("=" * 60)
        print("Humanoid Locomotion Validation (MuJoCo CPU)")
        print("=" * 60)
        print(f"Number of actuators: {self._nu}")
        print(f"Number of position DOFs: {self._nq}")
        print(f"Number of velocity DOFs: {self._nv}")
        print(f"Torso body ID: {self._torso_body_id}")
        print(f"Torso_top site ID: {self._torso_top_site_id}")
        print(f"Action order: {self._policy_joint_order}")
        print(f"Action scale: +/-{self._action_scale} rad")
        print(f"Control frequency: {1 / self.dt:.1f} Hz")
        print(f"Physics timestep: {self.model.opt.timestep}s")
        print(f"Substeps per control: {self.n_substeps}")
        print(f"Frame stack: {self._frame_stack} frames")
        print(f"Use phase obs: {self._use_phase_obs}")
        print(f"Single frame obs size: {self._single_frame_size}")
        print(f"Total obs size: {self._obs_size}")
        print(f"Action noise std: {self._action_noise_std}")
        print(f"Action latency steps: {self._act_latency_steps} (max {self._act_max_latency})")
        print(f"Obs latency steps: {self._obs_latency_steps} (max {self._obs_max_latency})")
        print(f"Disturbance probability: {self._disturbance_prob:.1%}")
        if self._include_height:
            print("Note: include_height=True is ignored for IsaacLab-compatible policy observations.")
        if action_scale_by_joint:
            print("\nPer-joint action scale multipliers:")
            for actuator_id in range(self._nu):
                if self._joint_action_scales[actuator_id] != 1.0:
                    joint_id = int(self.model.actuator_trnid[actuator_id, 0])
                    joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                    print(f"  {joint_name}: {self._joint_action_scales[actuator_id]:.2f}")
        print("=" * 60)

    def reset(self) -> np.ndarray:
        """Reset environment to standing pose and prefill observation histories."""
        self.data.qpos[:] = self._standing_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._standing_qpos[self._q_joint_start:self._q_joint_start + self._nu]

        mujoco.mj_forward(self.model, self.data)

        self._actions[:] = 0.0
        self._last_act[:] = 0.0
        if self._act_hist_buf is not None:
            self._act_hist_buf[:, :] = 0.0

        self._step_count = 0

        self._update_state()
        obs0 = self._compute_single_observation()

        if self._obs_hist_buf is not None:
            self._obs_hist_buf[:, :] = obs0[:, None]
        delayed_obs = obs0

        if self._obs_stack_buf is not None:
            self._obs_stack_buf[:, :] = delayed_obs[:, None]
            obs = self._obs_stack_buf.reshape(-1)
        else:
            obs = delayed_obs

        if self._torso_top_site_id >= 0:
            init_height = float(self.data.site_xpos[self._torso_top_site_id, 2])
        else:
            init_height = float(self.data.xpos[self._torso_body_id, 2])
        print(f"\nReset complete. Initial torso height: {init_height:.3f}m")
        return obs.copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        """Step the environment by one policy step."""
        self._pre_physics_step(action)
        self._apply_action()

        self._apply_disturbance()
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        return self._get_obs()

    def _pre_physics_step(self, action: np.ndarray):
        """Isaac-style action preprocessing: clip -> noise -> latency -> prev/current update."""
        action_policy = np.asarray(action, dtype=float).reshape(-1)
        if action_policy.shape != (self._nu,):
            raise ValueError(f"expected action shape {(self._nu,)}, got {action_policy.shape}")

        a = np.clip(action_policy, -1.0, 1.0)

        if self._action_noise_std > 0.0:
            a = np.clip(a + np.random.randn(*a.shape) * self._action_noise_std, -1.0, 1.0)

        if self._act_hist_buf is not None:
            self._act_hist_buf = np.roll(self._act_hist_buf, shift=-1, axis=1)
            self._act_hist_buf[:, -1] = a
            idx = int(np.clip(self._act_max_latency - self._act_latency_steps, 0, self._act_max_latency))
            a = self._act_hist_buf[:, idx].copy()

        self._last_act = self._actions.copy()
        self._actions = a

    def _apply_action(self):
        """Apply position-offset actions around the standing keyframe with soft-limit clipping."""
        action_mj = self._actions[self._mj_to_policy]
        standing_joint_pos = self._standing_qpos[self._q_joint_start:self._q_joint_start + self._nu]

        pos_offsets = action_mj * self._action_scale * self._joint_action_scales
        target_positions = standing_joint_pos + pos_offsets

        soft_lo = self._joint_range_lower_mj * self._soft_joint_limit_factor
        soft_hi = self._joint_range_upper_mj * self._soft_joint_limit_factor
        self.data.ctrl[:] = np.clip(target_positions, soft_lo, soft_hi)

    def _apply_disturbance(self):
        """Apply random disturbance forces and torques."""
        if np.random.uniform() > self._disturbance_prob:
            return

        force_magnitude = np.random.uniform(0, self._disturbance_force_max)
        force_direction = np.random.randn(3)
        force_direction /= (np.linalg.norm(force_direction) + 1e-8)
        disturbance_force = force_direction * force_magnitude

        disturbance_velocity = disturbance_force * self.dt / self._torso_mass
        self.data.qvel[:3] += disturbance_velocity

        torque_magnitude = np.random.uniform(0, self._disturbance_torque_max)
        torque_direction = np.random.randn(3)
        torque_direction /= (np.linalg.norm(torque_direction) + 1e-8)
        disturbance_torque = torque_direction * torque_magnitude

        disturbance_ang_velocity = disturbance_torque * self.dt / self._approx_inertia
        self.data.qvel[3:6] += disturbance_ang_velocity

    def _update_state(self):
        """Compute state terms used by the IsaacLab observation function."""
        self.torso_quat_w = self.data.xquat[self._torso_body_id].copy()  # (w, x, y, z)

        self.torso_lin_vel_w = self.data.qvel[:3].copy()
        self.torso_ang_vel_w = self.data.qvel[3:6].copy()

        self.torso_lin_vel_b = self._rotate_vector(self.torso_lin_vel_w, self.torso_quat_w, inverse=True)
        self.torso_ang_vel_b = self._rotate_vector(self.torso_ang_vel_w, self.torso_quat_w, inverse=True)

        if self._use_cmd_yaw_offset:
            self.torso_lin_vel_cmd = self._rotate_xy(self.torso_lin_vel_b, self._cmd_yaw_cos, self._cmd_yaw_sin)
            self.torso_ang_vel_cmd = self._rotate_xy(self.torso_ang_vel_b, self._cmd_yaw_cos, self._cmd_yaw_sin)
        else:
            self.torso_lin_vel_cmd = self.torso_lin_vel_b.copy()
            self.torso_ang_vel_cmd = self.torso_ang_vel_b.copy()

        up_world = np.array([0.0, 0.0, 1.0], dtype=float)
        self.up_b = self._rotate_vector(up_world, self.torso_quat_w, inverse=True)
        self.up_cmd = self.up_b.copy()

        joint_pos_mj = self.data.qpos[self._q_joint_start:self._q_joint_start + self._nu]
        joint_vel_mj = self.data.qvel[self._qd_joint_start:self._qd_joint_start + self._nu]

        # Convert model actuator order -> policy order.
        self.act_pos = joint_pos_mj[self._policy_to_mj]
        self.act_vel = joint_vel_mj[self._policy_to_mj]

        lo = self._joint_range_lower_policy * self._soft_joint_limit_factor
        hi = self._joint_range_upper_policy * self._soft_joint_limit_factor
        self.act_pos_scaled = 2.0 * (self.act_pos - lo) / (hi - lo + 1e-6) - 1.0

    def _compute_single_observation(self) -> np.ndarray:
        """Build one observation frame using IsaacLab ordering and transforms."""
        up_cmd = self.up_cmd.copy()
        ang_vel_cmd = self.torso_ang_vel_cmd.copy()
        act_pos_scaled = self.act_pos_scaled.copy()
        act_vel = self.act_vel.copy()

        # IMU biases.
        up_cmd = up_cmd + self._imu_bias_gravity
        ang_vel_cmd = ang_vel_cmd + self._imu_bias_gyro

        # IMU mount misalignment small-angle approx: v' ~= v + theta x v.
        theta = self._imu_mount_axis * self._imu_mount_ang
        up_cmd = up_cmd + np.cross(theta, up_cmd)
        ang_vel_cmd = ang_vel_cmd + np.cross(theta, ang_vel_cmd)

        # Observation noise.
        if self._obs_noise_gravity_std > 0.0:
            up_cmd = up_cmd + np.random.randn(*up_cmd.shape) * self._obs_noise_gravity_std
        if self._obs_noise_gyro_std > 0.0:
            ang_vel_cmd = ang_vel_cmd + np.random.randn(*ang_vel_cmd.shape) * self._obs_noise_gyro_std
        if self._obs_noise_joint_pos_std > 0.0:
            act_pos_scaled = act_pos_scaled + np.random.randn(*act_pos_scaled.shape) * self._obs_noise_joint_pos_std
        if self._obs_noise_joint_vel_std > 0.0:
            act_vel = act_vel + np.random.randn(*act_vel.shape) * self._obs_noise_joint_vel_std

        obs = np.concatenate(
            [
                self.torso_lin_vel_cmd,
                ang_vel_cmd * self._ang_vel_scale,
                up_cmd,
                self._commands,
                act_pos_scaled,
                act_vel * self._dof_vel_scale,
                self._last_act,
            ]
        )

        if self._use_phase_obs:
            if self._gait_period_s <= 0.0:
                raise ValueError("gait_period_s must be > 0 when use_phase_obs is enabled")
            t = float(self._step_count) * self.dt
            phase = 2.0 * math.pi * (t / self._gait_period_s)
            clock = np.array([math.sin(phase), math.cos(phase)], dtype=float)
            obs = np.concatenate([obs, clock])

        return obs

    def _get_obs(self) -> np.ndarray:
        """Isaac-style observation path: state -> single obs -> latency -> stack."""
        self._update_state()
        obs = self._compute_single_observation()

        if self._obs_hist_buf is not None:
            self._obs_hist_buf = np.roll(self._obs_hist_buf, shift=-1, axis=1)
            self._obs_hist_buf[:, -1] = obs

            idx = int(np.clip(self._obs_max_latency - self._obs_latency_steps, 0, self._obs_max_latency))
            obs = self._obs_hist_buf[:, idx].copy()

        if self._obs_stack_buf is not None:
            self._obs_stack_buf = np.roll(self._obs_stack_buf, shift=-1, axis=1)
            self._obs_stack_buf[:, -1] = obs
            # Feature-major flatten from [D, F] -> [D*F].
            obs = self._obs_stack_buf.reshape(-1)

        return obs.copy()

    def _rotate_vector(self, vec: np.ndarray, quat: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Rotate a vector by a quaternion in (w, x, y, z) convention."""
        q = quat
        if inverse:
            q = np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)

        rot_mat = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(rot_mat, q)
        rot_mat = rot_mat.reshape(3, 3)
        return rot_mat @ vec

    def _rotate_xy(self, v: np.ndarray, cos_yaw: float, sin_yaw: float) -> np.ndarray:
        """Rotate xy components of vector by yaw angle."""
        x = v[..., 0] * cos_yaw - v[..., 1] * sin_yaw
        y = v[..., 0] * sin_yaw + v[..., 1] * cos_yaw
        z = v[..., 2] if v.shape[-1] > 2 else np.zeros_like(x)
        return np.stack([x, y, z], axis=-1)

    def render(self, mode: str = "rgb_array", width: int = 640, height: int = 480, camera_name: str | None = None):
        """Render the environment."""
        if mode != "rgb_array":
            raise NotImplementedError(f"Render mode '{mode}' not supported")

        renderer = mujoco.Renderer(self.model, height=height, width=width)
        if camera_name:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            renderer.update_scene(self.data, camera=camera_id)
        else:
            renderer.update_scene(self.data)
        return renderer.render()
