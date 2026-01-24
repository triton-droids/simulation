"""
Humanoid Locomotion Validation Environment (Pure MuJoCo)
For validating IsaacSim-trained policies on CPU MuJoCo.
"""

from typing import Dict, Tuple
import math
import numpy as np
import mujoco


class HumanoidLocomotionEnv:
    """Pure MuJoCo locomotion validation environment."""

    def __init__(
        self,
        xml_path: str,
        frame_stack: int = 3,
        #Significantly reduced disturbances for mujoco
        disturbance_force_max: float = 5.0,
        disturbance_torque_max: float = 2.0,
        disturbance_prob: float = 0.00,
        action_scale: float = 1.0,
        action_scale_by_joint: dict[str, float] = {
            "left_thigh_act": 0.3,
            "right_thigh_act": 0.3,
        },
        dt: float = 1/120,  # 60 Hz control
    ):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Set timestep
        self.model.opt.timestep = 1/240
        self.dt = dt
        self.n_substeps = int(dt / self.model.opt.timestep)
        
        # Parameters
        self._action_scale = action_scale
        self._frame_stack = frame_stack
        self._disturbance_force_max = disturbance_force_max
        self._disturbance_torque_max = disturbance_torque_max
        self._disturbance_prob = disturbance_prob
        
        # Find torso body
        self._torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso'
        )

        # Find torso_top site for height measurement
        self._torso_top_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, 'torso_top'
        )
        
        # Get DOF info
        self._nu = self.model.nu  # Number of actuators
        self._nq = self.model.nq  # Position DOFs
        self._nv = self.model.nv  # Velocity DOFs
        
        # Joint indices (skip freejoint)
        self._q_joint_start = 7   # Skip 3 pos + 4 quat
        self._qd_joint_start = 6  # Skip 3 linear + 3 angular
        # MuJoCo -> Isaac observation order mapping
        self._mj_to_isaac = np.array([0, 5, 1, 6, 2, 7, 3, 8, 4, 9])
        self._isaac_to_mj = np.argsort(self._mj_to_isaac)

        # Build per-joint action scale array
        self._joint_action_scales = np.ones(self._nu)
        if action_scale_by_joint:
            for i in range(self._nu):
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if actuator_name in action_scale_by_joint:
                    self._joint_action_scales[i] = action_scale_by_joint[actuator_name]

        # Load standing pose from keyframe
        standing_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, 'locomotion_standing_pose'
        )
        self._standing_qpos = self.model.key_qpos[standing_key_id].copy()
        
        # Joint limits (in IsaacSim observation order: lh1, rh1, lh2, rh2, lt, rt, lk, rk, la, ra)
        self._joint_range_lower_isaac = np.array([
            -1.57,      # left_hip1_joint
            -1.57,      # right_hip1_joint
            -1.57,      # left_hip2_joint
            -0.436332,  # right_hip2_joint
            -0.785398,  # left_thigh_joint
            -0.785398,  # right_thigh_joint
            -2.0944,    # left_knee_joint
            -2.0944,    # right_knee_joint
            -0.6,       # left_ankle_joint
            -0.6        # right_ankle_joint
        ])
        self._joint_range_upper_isaac = np.array([
            1.57,       # left_hip1_joint
            1.57,       # right_hip1_joint
            0.436332,   # left_hip2_joint
            1.57,       # right_hip2_joint
            0.785398,   # left_thigh_joint
            0.785398,   # right_thigh_joint
            0,          # left_knee_joint
            0,          # right_knee_joint
            0.6,        # left_ankle_joint
            0.6         # right_ankle_joint
        ])
        self._joint_range_lower_mj = self._joint_range_lower_isaac[self._isaac_to_mj]
        self._joint_range_upper_mj = self._joint_range_upper_isaac[self._isaac_to_mj]
        self._soft_joint_limit_factor = 0.95

        
        # Observation size calculation
        # Single frame: 1 (height) + 3 (lin_vel_cmd) + 3 (ang_vel_cmd) + 3 (up_cmd) + 3 (commands) + nu (joint_pos) + nu (joint_vel) + nu (last_actions)
        self._single_frame_size = 1 + 3 + 3 + 3 + 3 + self._nu + self._nu + self._nu
        self._obs_size = self._single_frame_size * self._frame_stack

        # Scaling factors
        self._ang_vel_scale = 0.25
        self._dof_vel_scale = 0.1

        # Commands (vx, vy, yaw_rate) - set to zero for standing
        self._commands = np.zeros(3)
        
        # State tracking
        self._obs_history = np.zeros(self._obs_size)
        self._last_act = np.zeros(self._nu)
        self._step_count = 0
        
        # Torso mass (from MJCF)
        self._torso_mass = 3.175
        self._approx_inertia = 0.02

        self._command_yaw_offset = -math.pi / 2.0  # -90 degrees
        self._cmd_yaw_cos = math.cos(self._command_yaw_offset)  # 0
        self._cmd_yaw_sin = math.sin(self._command_yaw_offset)  # -1
        self._use_cmd_yaw_offset = True
        
        # For inverse rotation (command -> body)
        self._cmd_yaw_inv_cos = self._cmd_yaw_cos   # 0
        self._cmd_yaw_inv_sin = -self._cmd_yaw_sin  # 1
        
        print("=" * 60)
        print("Humanoid Locomotion Validation (MuJoCo CPU)")
        print("=" * 60)
        print(f"Number of actuators: {self._nu}")
        print(f"Number of position DOFs: {self._nq}")
        print(f"Number of velocity DOFs: {self._nv}")
        print(f"Torso body ID: {self._torso_body_id}")
        print(f"Torso_top site ID: {self._torso_top_site_id}")
        print(f"Action scale: ±{self._action_scale} rad")
        print(f"Control frequency: {1/self.dt:.1f} Hz")
        print(f"Physics timestep: {self.model.opt.timestep}s")
        print(f"Substeps per control: {self.n_substeps}")
        print(f"Frame stack: {self._frame_stack} frames")
        print(f"Single frame obs size: {self._single_frame_size}")
        print(f"Total obs size: {self._obs_size}")
        print(f"Max disturbance force: {self._disturbance_force_max} N")
        print(f"Max disturbance torque: {self._disturbance_torque_max} Nm")
        print(f"Disturbance probability: {self._disturbance_prob:.1%}")
        if action_scale_by_joint:
            print("\nPer-joint action scale multipliers:")
            for i in range(self._nu):
                if self._joint_action_scales[i] != 1.0:
                    actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    print(f"  {actuator_name}: {self._joint_action_scales[i]:.2f}")
        print("=" * 60)

    def reset(self) -> np.ndarray:
        """
        Reset environment to standing pose.
        """
        # Set to standing pose
        self.data.qpos[:] = self._standing_qpos
        self.data.qvel[:] = 0.0
        
        #Set control to match standing joint positions
        self.data.ctrl[:] = self._standing_qpos[self._q_joint_start:self._q_joint_start + self._nu]
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Reset tracking variables
        self._obs_history = np.zeros(self._obs_size)
        self._last_act = np.zeros(self._nu)
        self._step_count = 0
        
        # Get initial observation
        obs = self._get_obs()

        # Replicate the first frame across all frames in the stack
        first_frame = obs[-self._single_frame_size:]  # Get the newest frame (at the end)
        for i in range(self._frame_stack):
            start_idx = i * self._single_frame_size
            end_idx = (i + 1) * self._single_frame_size
            obs[start_idx:end_idx] = first_frame

        # Update the history with the replicated observation
        self._obs_history = obs.copy()

        print(f"\nReset complete. Initial torso_top height: {self.data.site_xpos[self._torso_top_site_id, 2]:.3f}m")

        return obs

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Step the environment.

        Args:
            action: Joint position offsets normalized to [-1, 1]
                    These are offsets from the standing pose, NOT absolute positions.

        Returns:
            obs: Observation array
        """
        # Actions are Isaac-ordered; keep that for obs and convert for MuJoCo actuators
        action_isaac = action.copy()
        self._last_act = np.clip(action_isaac, -1.0, 1.0)
        action_mj = action_isaac[self._isaac_to_mj]

        # Extract actuated joint positions from standing pose
        # standing_qpos = [root_pos(3), root_quat(4), joint_pos(10)]
        standing_joint_pos = self._standing_qpos[self._q_joint_start:self._q_joint_start + self._nu]
        
        # Scale actions to position offsets (in radians)
        # action_scale controls the maximum offset magnitude
        # Per-joint multipliers further modulate specific joints
        pos_offsets = action_mj * self._action_scale * self._joint_action_scales
        
        # Compute target positions: standing_pose + offset
        target_positions = standing_joint_pos + pos_offsets
        
        # Clip to joint limits for safety
        soft_lo = self._joint_range_lower_mj * self._soft_joint_limit_factor
        soft_hi = self._joint_range_upper_mj * self._soft_joint_limit_factor
        target_positions = np.clip(target_positions, soft_lo, soft_hi)

        # Set control signals (position targets for MuJoCo's position actuators)
        self.data.ctrl[:] = target_positions
        
        # Apply random disturbances (if enabled)
        self._apply_disturbance()
        
        # Step physics multiple times per control step
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Update state tracking
        self._step_count += 1
        
        # Get observation
        obs = self._get_obs()

        return obs

    def _apply_disturbance(self):
        """Apply random disturbance forces and torques."""
        if np.random.uniform() > self._disturbance_prob:
            return
        
        # Linear force disturbance
        force_magnitude = np.random.uniform(0, self._disturbance_force_max)
        force_direction = np.random.randn(3)
        force_direction /= (np.linalg.norm(force_direction) + 1e-8)
        disturbance_force = force_direction * force_magnitude
        
        # Apply as velocity impulse
        disturbance_velocity = disturbance_force * self.dt / self._torso_mass
        self.data.qvel[:3] += disturbance_velocity
        
        # Angular torque disturbance
        torque_magnitude = np.random.uniform(0, self._disturbance_torque_max)
        torque_direction = np.random.randn(3)
        torque_direction /= (np.linalg.norm(torque_direction) + 1e-8)
        disturbance_torque = torque_direction * torque_magnitude
        
        # Apply as angular velocity impulse
        disturbance_ang_velocity = disturbance_torque * self.dt / self._approx_inertia
        self.data.qvel[3:6] += disturbance_ang_velocity

    def _get_obs(self) -> np.ndarray:
        # Get torso orientation (quaternion)
        torso_quat = self.data.xquat[self._torso_body_id].copy()

        # Height (torso_top site z-position in world frame)
        height = np.array([self.data.site_xpos[self._torso_top_site_id, 2]])

        # Linear velocity in BODY frame first
        lin_vel_world = self.data.qvel[:3].copy()
        torso_lin_vel_b = self._rotate_vector(lin_vel_world, torso_quat, inverse=True)
        
        # Then rotate to COMMAND frame
        torso_lin_vel_cmd = self._rotate_xy(
            torso_lin_vel_b, 
            self._cmd_yaw_cos, 
            self._cmd_yaw_sin
        )

        # Angular velocity in BODY frame first
        ang_vel_world = self.data.qvel[3:6].copy()
        ang_vel_b = self._rotate_vector(ang_vel_world, torso_quat, inverse=True)
        
        # Then rotate to COMMAND frame
        ang_vel_cmd = self._rotate_xy(
            ang_vel_b,
            self._cmd_yaw_cos,
            self._cmd_yaw_sin
        )
        ang_vel_cmd_scaled = ang_vel_cmd * self._ang_vel_scale

        # Up vector in BODY frame first
        up_world = np.array([0, 0, 1.0])
        up_b = self._rotate_vector(up_world, torso_quat, inverse=True)
        
        # Then rotate to COMMAND frame
        up_cmd = self._rotate_xy(
            up_b,
            self._cmd_yaw_cos,
            self._cmd_yaw_sin
        )

        # Keep commands in the same frame as lin/ang vel + up (command frame)
        commands = self._commands.copy()
        if self._use_cmd_yaw_offset:
            commands = self._rotate_xy(
                commands,
                self._cmd_yaw_cos,
                self._cmd_yaw_sin
            )

        # Joint states
        joint_pos = self.data.qpos[self._q_joint_start:self._q_joint_start + self._nu]
        joint_vel = self.data.qvel[self._qd_joint_start:self._qd_joint_start + self._nu]

        # Reorder: 0 5 1 6 2 7 3 8 4 9
        joint_pos = joint_pos[self._mj_to_isaac]
        joint_vel = joint_vel[self._mj_to_isaac]

        # Scale observations (FIX: use proper scaling like IsaacLab)
        lo = self._joint_range_lower_isaac * self._soft_joint_limit_factor
        hi = self._joint_range_upper_isaac * self._soft_joint_limit_factor
        act_pos_scaled = 2.0 * (joint_pos - lo) / (hi - lo + 1e-6) - 1.0
        #act_pos_scaled = joint_pos / 1.57  # Normalized by joint range
        act_vel = joint_vel * self._dof_vel_scale

        # Construct single frame
        obs_frame = np.concatenate([
            height,                  # 1
            torso_lin_vel_cmd,       # 3
            ang_vel_cmd_scaled,      # 3
            up_cmd,                  # 3
            commands,                # 3
            act_pos_scaled,          # num_dofs
            act_vel,                 # num_dofs
            self._last_act,          # num_dofs
        ])

        # Clip
        obs_frame = np.clip(obs_frame, -10.0, 10.0)

        # Stack with history - NEW frame goes at the END (like IsaacLab)
        self._obs_history = np.roll(self._obs_history, -self._single_frame_size)  # Note the negative!
        self._obs_history[-self._single_frame_size:] = obs_frame  # Put at end, not beginning

        return self._obs_history.copy()
    def _rotate_vector(self, vec: np.ndarray, quat: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Rotate a vector by a quaternion."""
        if inverse:
            # Conjugate quaternion for inverse rotation
            quat = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        
        # Quaternion multiplication: q * v * q^-1
        # Using MuJoCo's rotation matrix conversion
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)
        
        return rot_mat @ vec

    def _rotate_xy(self, v: np.ndarray, cos_yaw: float, sin_yaw: float) -> np.ndarray:
        """Rotate xy components of vector by yaw angle."""
        x = v[..., 0] * cos_yaw - v[..., 1] * sin_yaw
        y = v[..., 0] * sin_yaw + v[..., 1] * cos_yaw
        z = v[..., 2] if v.shape[-1] > 2 else np.zeros_like(x)
        return np.stack([x, y, z], axis=-1)

    def render(self, mode='rgb_array', width=640, height=480, camera_name=None):
        """Render the environment."""
        if mode == 'rgb_array':
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            
            if camera_name:
                camera_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
                )
                renderer.update_scene(self.data, camera=camera_id)
            else:
                renderer.update_scene(self.data)
            
            return renderer.render()
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported")
