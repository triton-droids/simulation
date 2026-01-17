"""
Humanoid Disturbance Rejection Validation Environment (Pure MuJoCo)
For validating IsaacSim-trained policies on CPU MuJoCo.
"""

from typing import Dict, Tuple
import numpy as np
import mujoco


class HumanoidDisturbanceEnv:
    """Pure MuJoCo disturbance rejection validation environment."""

    def __init__(
        self,
        xml_path: str,
        frame_stack: int = 3,
        disturbance_force_max: float = 25.0,
        disturbance_torque_max: float = 10.0,
        disturbance_prob: float = 0.02,
        action_scale: float = 1.0,
        dt: float = 0.02,  # 50 Hz control
    ):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Set timestep
        self.model.opt.timestep = 0.002
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
        
        # Get DOF info
        self._nu = self.model.nu  # Number of actuators
        self._nq = self.model.nq  # Position DOFs
        self._nv = self.model.nv  # Velocity DOFs
        
        # Joint indices (skip freejoint)
        self._q_joint_start = 7   # Skip 3 pos + 4 quat
        self._qd_joint_start = 6  # Skip 3 linear + 3 angular
        
        # Load standing pose from keyframe
        standing_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, 'standing_pose'
        )
        self._standing_qpos = self.model.key_qpos[standing_key_id].copy()
        
        # Joint limits
        self._joint_range_lower = np.full(self._nu, -1.57)
        self._joint_range_upper = np.full(self._nu, 1.57)
        
        # Observation size calculation
        # Single frame: 3 (gravity) + 3 (ang_vel) + nu (joint_pos) + nu (joint_vel) + nu (torques) + nu (last_actions)
        self._single_frame_size = 3 + 3 + self._nu + self._nu + self._nu + self._nu
        self._obs_size = self._single_frame_size * self._frame_stack
        
        # State tracking
        self._obs_history = np.zeros(self._obs_size)
        self._last_act = np.zeros(self._nu)
        self._step_count = 0
        
        # Torso mass (from MJCF)
        self._torso_mass = 3.175
        self._approx_inertia = 0.02
        
        print("=" * 60)
        print("Humanoid Disturbance Rejection Validation (MuJoCo CPU)")
        print("=" * 60)
        print(f"Number of actuators: {self._nu}")
        print(f"Number of position DOFs: {self._nq}")
        print(f"Number of velocity DOFs: {self._nv}")
        print(f"Torso body ID: {self._torso_body_id}")
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
        print("=" * 60)

    def reset(self) -> np.ndarray:
        """Reset environment to standing pose."""
        # Set to standing pose
        self.data.qpos[:] = self._standing_qpos
        self.data.qvel[:] = 0.0
        
        self.data.ctrl[:] = self._standing_qpos[self._q_joint_start:]
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Reset tracking variables
        self._obs_history = np.zeros(self._obs_size)
        self._last_act = np.zeros(self._nu)
        self._step_count = 0
        
        # Get initial observation
        obs = self._get_obs()
        
        print(f"\nReset complete. Initial torso height: {self.data.xpos[self._torso_body_id, 2]:.3f}m")
        
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step the environment.
        
        Args:
            action: Joint positions normalized to [-1, 1]
        
        Returns:
            obs
        """
        # Scale and clip actions
        target_positions = action * self._action_scale
        target_positions = np.clip(
            target_positions,
            self._joint_range_lower,
            self._joint_range_upper
        )
        
        # Set control
        self.data.ctrl[:] = target_positions
        
        # Apply disturbance
        self._apply_disturbance()
        
        # Step physics (multiple substeps)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Update state
        self._last_act = action.copy()
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
        """
        Get observation.
        
        Observation structure (single frame):
        - 3: gravity in body frame
        - 3: angular velocity in body frame
        - num_dofs: joint positions (scaled)
        - num_dofs: joint velocities (scaled)
        - num_dofs: joint torques (scaled)
        - num_dofs: last actions
        """
        # Get torso orientation (quaternion)
        torso_quat = self.data.xquat[self._torso_body_id].copy()
        
        # Gravity in body frame
        gravity_world = np.array([0, 0, -9.81])
        gravity_body = self._rotate_vector(gravity_world, torso_quat, inverse=True)
        gravity_body_norm = gravity_body / 9.81
        
        # Angular velocity in body frame
        ang_vel_world = self.data.qvel[3:6].copy()
        ang_vel_body = self._rotate_vector(ang_vel_world, torso_quat, inverse=True)
        ang_vel_body_scaled = ang_vel_body * 0.1
        
        # Joint states
        joint_pos = self.data.qpos[self._q_joint_start:self._q_joint_start + self._nu]
        joint_vel = self.data.qvel[self._qd_joint_start:self._qd_joint_start + self._nu]
        
        # Joint torques (actuator forces)
        joint_torques = self.data.qfrc_actuator[self._qd_joint_start:self._qd_joint_start + self._nu]
        
        # Scale observations
        joint_pos_scaled = joint_pos / 1.57
        joint_vel_scaled = joint_vel * 0.1
        joint_torques_scaled = joint_torques * 0.01
        
        # Construct single frame
        obs_frame = np.concatenate([
            gravity_body_norm,
            ang_vel_body_scaled,
            joint_pos_scaled,
            joint_vel_scaled,
            joint_torques_scaled,
            self._last_act,
        ])
        
        # Clip
        obs_frame = np.clip(obs_frame, -10.0, 10.0)
        
        # Stack with history
        self._obs_history = np.roll(self._obs_history, self._single_frame_size)
        self._obs_history[:self._single_frame_size] = obs_frame
        
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

    def get_state(self) -> Dict:
        """Get current state for debugging."""
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'torso_pos': self.data.xpos[self._torso_body_id].copy(),
            'torso_quat': self.data.xquat[self._torso_body_id].copy(),
            'step_count': self._step_count,
        }