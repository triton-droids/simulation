# Copyright 2025 Triton Droids

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Joystick task for Default Humanoid."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from .base import DefaultHumanoidEnv
from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from .default_humanoid_constants import *

class Joystick(DefaultHumanoidEnv):
    """ Track a joystick command"""
    def __init__(
        self,
        terrain: str ="flat_terrain",
        config: config_dict.ConfigDict = None,
    ):
        super().__init__(
            xml_path= task_to_xml(terrain).as_posix(),
            config = config,
        )
        self._post_init()

    def _post_init(self) -> None:
        """ 
        Stores inital state information along with address' of joints, geoms, and sensors for tracking rewards and visualization. 
        Introduces noise to joint positions. 
        
        """
        #STORE ANY INFORMATION THAT MAY BE USEFUL FOR REWARDS, VISUALIZATION, ETC
        
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos) #Set initial joint positions standing
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

        #Note: First joint is freejoint
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T 
        c = (self._lowers + self._uppers) / 2 
        r = self._uppers - self._lowers 
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor #soft position limits. Help revent extreme joint positions
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        #Obtain address of hip and knee joints (to be used for reward functions)
        hip_indices = []
        hip_joint_names = ["HR", "HAA"]
        for side in ["LL", "LR"]:
            for joint_name in hip_joint_names:
                hip_indices.append(
                    self._mj_model.joint(f"{side}_{joint_name}").qposadr - 7 
                )
        self._hip_indices = jp.array(hip_indices)

        knee_indices = []
        for side in ["LL", "LR"]:
            knee_indices.append(self._mj_model.joint(f"{side}_KFE").qposadr - 7)
        self._knee_indices = jp.array(knee_indices)

        #Set weights for each joint in the default pose  
        self._weights = jp.array([
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # left leg.
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # right leg.
    ])

        self._torso_body_id = self._mj_model.body(ROOT_BODY).id 
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        #Obtain id of feet sites to save information about feet / leg movement
        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in FEET_SITES]
    )
        #Obtain id of floor and feet geoms for contact detection. e.x. Does foot hit floor?
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.body(name).id for name in FEET_BODIES]
    )
        #Obtain linear velocity of feet sensors
        foot_linvel_sensor_adr = []
        for site in FEET_SITES:  
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id] 
            sensor_dim = self._mj_model.sensor_dim[sensor_id] 
            foot_linvel_sensor_adr.append( 
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr) 

        qpos_noise_scale = np.zeros(12)
        hip_ids = [0, 1, 2, 6, 7, 8]
        kfe_ids = [3, 9]
        ffe_ids = [4, 10]
        faa_ids = [5, 11]
        qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
        qpos_noise_scale[kfe_ids] = self._config.noise_config.scales.kfe_pos
        qpos_noise_scale[ffe_ids] = self._config.noise_config.scales.ffe_pos
        qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """ 
        Resets the environment to an initial state and includes some randomization to improve generalization and robustness. 
        The translations from the starting point, orientation, joint positions, velocities, phase and gate frequency of the
        humanoid are sampled from a uniform distribution using independent random number generators. 

        Args:
            rng: Random number generator.

        Returns:
            mjx_env.State : data, obs, reward, done, metrics, info
        """
        qpos = self._init_q 
        qvel = jp.zeros(self.mjx_model.nv)

        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5) 
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)  #qpos[0:2]: The global x and y position of the humanoid's root
        rng, key = jax.random.split(rng) 
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat) 
        qpos = qpos.at[3:7].set(new_quat) #qpos[3:7]: The quaternion representing the orientation (yaw, pitch, roll) of the humanoid's root

        # qpos[7:]=*U(0.5, 1.5)
        #qpos[7:]: The joint positions within the humanoid
        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(
            qpos[7:] * jax.random.uniform(key, (12,), minval=0.5, maxval=1.5) 
        )

        # d(xyzrpy)=U(-0.5, 0.5)
        #qvel[0:6]: The generalized velocities of the humanoid's base
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5) 
        )

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        # Phase, freq=U(1.25, 1.5) : Gate cycle
        rng, key = jax.random.split(rng)
        gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
        phase_dt = 2 * jp.pi * self.dt * gait_freq #Change in phase per time step (how gait phase evolves over time)
        phase = jp.array([0, jp.pi]) #One leg starts at beginning of gait cycle, other starts at mid-gait cycle (natural alternation between two legs)

        #Generate a random command
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng) #[lin_vel_x, lin_vel_y, ang_vel_yaw]

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Phase related.
            "phase_dt": phase_dt,
            "phase": phase,
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
            metrics["swing_peak"] = jp.zeros(())

        contact = jp.array([
            geoms_colliding(data, geom_id, self._floor_geom_id)
            for geom_id in self._feet_geom_id
        ])
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """
        Applies the action to the humanoid and updates the state. Peforms push on the humanoid if flag is enabled.
        Stores state and reward information for visualization and reward computation.

        Args:
            state: mjx_env.state that defines the current state of the environment.
            action: jax.Array with dim size(DOFS) to be applied to the joints of the humanoid.

        Returns:
            mjx_env.State : data, obs, reward, done, metrics, info
        
        """
        state.info["rng"], push1_rng, push2_rng = jax.random.split(
            state.info["rng"], 3
        )

        #Apply a push to the humanoid if flag is enabled. 
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
            == 0
        )
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        #Action is applied to the humanoid with a scalar value
        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )
        state.info["motor_targets"] = motor_targets

        #Contact status of the humanoid's feet with the floor
        contact : bool = jp.array([
            geoms_colliding(data, geom_id, self._floor_geom_id)
            for geom_id in self._feet_geom_id
        ])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1] #Extract z position of the feet of foot position
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        #Get observations, rewards, and termination status
        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(
        data, action, state.info, done, first_contact, contact
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        #Update state information:
        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        phase_tp1 = state.info["phase"] + state.info["phase_dt"] #Compute robot's phase at next time step
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi #Keep phase within a range of -pi, pi for gait. 
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        #If step > 500 --> explore. 
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        #Check if terminated to reset step size. 
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
        )
        #Contact indicates if robot's feet are in contact with the floor. We update the air time and swing peak based on this. 
        state.info["feet_air_time"] *= ~ contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact

        #Store reward information.
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
            state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state


    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        return (
            fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        )

    def _get_obs(
        self, 
        data: mjx.Data, 
        info: dict[str, Any], 
        contact: jax.Array
    ) -> mjx_env.Observation:
        
        gyro = self.get_gyro(data) 
        info["rng"], noise_rng = jax.random.split(info["rng"])

        #Introduce some noise into our sesnor readings.
        noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )
        
        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        joint_angles = data.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        joint_vel = data.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        cos = jp.cos(info["phase"])
        sin = jp.sin(info["phase"])
        phase = jp.concatenate([cos, sin])

        linvel = self.get_local_linvel(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (
            linvel
            + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.linvel
        )

        state = jp.hstack([
        noisy_linvel,  # 3 (dimensions)
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        phase,
    ])
        
        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[2]

        privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel,
        root_height,  # 1
        data.actuator_force,  # 12
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
    ])
        
        return {
        "state": state,
        "privileged_state": privileged_state,
    }
        

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
  ) -> dict[str, jax.Array]:
        """ Computes all rewards for the current state of the environment. """
        return {
            # Tracking rewards.
            "tracking_lin_vel": self._reward_tracking_lin_vel(
                info["command"], self.get_local_linvel(data)
            ),
            "tracking_ang_vel": self._reward_tracking_ang_vel(
                info["command"], self.get_gyro(data)
            ),
            # Base-related rewards.
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "orientation": self._cost_orientation(self.get_gravity(data)),
            "base_height": self._cost_base_height(data.qpos[2]),
            # Energy related rewards.
            #"torques": self._cost_torques(data.actuator_force),
            #"action_rate": self._cost_action_rate(
            #    action, info["last_act"], info["last_last_act"]
            #),
            #"energy": self._cost_energy(data.qvel[6:], data.actuator_force),
            # Feet related rewards.
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "feet_clearance": self._cost_feet_clearance(data, info),
            "feet_height": self._cost_feet_height(
                info["swing_peak"], first_contact, info
            ),
            "feet_air_time": self._reward_feet_air_time(
                info["feet_air_time"], first_contact, info["command"]
            ),
            "feet_phase": self._reward_feet_phase(
                data,
                info["phase"],
                self._config.reward_config.max_foot_height,
                info["command"],
            ),
            # Other rewards.
            "alive": self._reward_alive(),
            "termination": self._cost_termination(done),
            "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
            # Pose related rewards.
            "joint_deviation_hip": self._cost_joint_deviation_hip(
                data.qpos[7:], info["command"]
            ),
            "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "pose": self._cost_pose(data.qpos[7:]),
        }
    
    # Tracking rewards.
    """
    The following rewards are based on how well the linear and angular velocities of the humanoid in simulation match the target velocities.
    The target velocities are sampled from a uniform distribution between -1 and 1 with a 10% chance of being 0.
    """

    def _reward_tracking_lin_vel(
        self,
        commands: jax.Array,
        local_vel: jax.Array,
    ) -> jax.Array:
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        ang_vel: jax.Array,
    ) -> jax.Array:
        ang_vel_error = jp.square(commands[2] - ang_vel[2])
        return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

    # Base-related rewards.
    """ 
    The following cost functions produce negative rewards based on how large the z axis linear velocity, xy angular velocity are. They also produce a 
    negative reward if there is deviation from the target base height or torso is not aligned with the z axis. 
    """
     
    def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
        return jp.square(global_linvel[2])

    def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
        return jp.sum(jp.square(global_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torso_zaxis[:2]))

    def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
        return jp.square(
            base_height - self._config.reward_config.base_height_target
        )

    # Energy related rewards.
    """
    The following cost functions are based on the magnitude of torques applies, mechanical energy expended by actuator, 
    and the rate of change of actions. 
    """
    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(torques))

    def _cost_energy(
        self, qvel: jax.Array, qfrc_actuator: jax.Array
    ) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        c1 = jp.sum(jp.square(act - last_act))
        return c1

    # Other rewards.
    """
    The following cost functions are based on a soft bound for joint positions and deviations from the current pose  """
    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    def _cost_stand_still(
        self,
        commands: jax.Array,
        qpos: jax.Array,
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.1)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done

    def _reward_alive(self) -> jax.Array:
        return jp.array(1.0)

    # Pose-related rewards.
    """
    The following cost functions produce a negative reward when the joint positions deviates from the default pose (knees bent athletic) to encourage walking and stability. 
    """
    def _cost_joint_deviation_hip(
        self, qpos: jax.Array, cmd: jax.Array
    ) -> jax.Array:
        cost = jp.sum(
            jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices])
        )
        cost *= jp.abs(cmd[1]) > 0.1
        return cost

    def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
        return jp.sum(
            jp.abs(
                qpos[self._knee_indices] - self._default_pose[self._knee_indices]
            )
        )

    def _cost_pose(self, qpos: jax.Array) -> jax.Array:
        return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

    # Feet related rewards.
    """
    The following cost functions produce a negative reward for the velocity in the x-y place in contact with the ground, impromper foot clearnace by feet 
    height and velocities, swing height of the feet by a predfined maximum. Includes a reward for feet air time upon first contact and correct phase during foot swing. 
    """
    def _cost_feet_slip(
        self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        del info  # Unused.
        body_vel = self.get_global_linvel(data)[:2]
        reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
        return reward

    def _cost_feet_clearance(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        del info  # Unused.
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
        return jp.sum(delta * vel_norm)

    def _cost_feet_height(
        self,
        swing_peak: jax.Array,
        first_contact: jax.Array,
        info: dict[str, Any],
    ) -> jax.Array:
        del info  # Unused.
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact)

    def _reward_feet_air_time(
        self,
        air_time: jax.Array,
        first_contact: jax.Array,
        commands: jax.Array,
        threshold_min: float = 0.2,
        threshold_max: float = 0.5,
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        air_time = (air_time - threshold_min) * first_contact
        air_time = jp.clip(air_time, max=threshold_max - threshold_min)
        reward = jp.sum(air_time)
        reward *= cmd_norm > 0.1  # No reward for zero commands.
        return reward

    def _reward_feet_phase(
        self,
        data: mjx.Data,
        phase: jax.Array,
        foot_height: jax.Array,
        commands: jax.Array,
    ) -> jax.Array:
        # Reward for tracking the desired foot height.
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1] #Actual z-axis height
        rz = gait.get_rz(phase, swing_height=foot_height) #Desired z-axis height
        error = jp.sum(jp.square(foot_z - rz))
        reward = jp.exp(-error / 0.01)
        cmd_norm = jp.linalg.norm(commands)
        reward *= cmd_norm > 0.1  # No reward for zero commands.
        return reward

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """ 
        Sample random commands with 10% chance of setting everything to 0. 
        Used for exploration, robust learning, and generalization. 
        """
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        #x,y velocity should be between [-1, 1]
        lin_vel_x = jax.random.uniform(
            rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self._config.ang_vel_yaw[0],
            maxval=self._config.ang_vel_yaw[1],
        )

        # With 10% chance, set everything to zero.
        return jp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jp.zeros(3),
            jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
        )

       