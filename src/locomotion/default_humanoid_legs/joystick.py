from dataclasses import asdict
from typing import Any, Callable, List
import jax
import mujoco
import numpy as np
from omegaconf import OmegaConf
import scipy
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jp
from mujoco import mjx
from mujoco.mjx._src import support  # type: ignore
from src.utils.file_utils import find_robot_file_path
from src.utils.math_utils import quat2euler, quat_inv, rotate_vec
from src.rewards import get_reward_function
from src.tools.collision import check_feet_contact
from mujoco.mjx._src import math

from src.locomotion.default_humanoid_legs.base import DefaultHumanoidEnv


class Joystick(DefaultHumanoidEnv):
    def __init__(
            self,
            name: str,
            robot: any,
            scene: str,
            cfg,
            **kwargs: Any,
    ):
        """ Initalizes the environment with the specified configuration and robot parameters"""

        super().__init__(name, robot, scene, cfg, **kwargs)

        self._init_env()
        self._init_reward()

    def _init_env(self) -> None:
        """Initializes the environment by setting up the system and its components."""

        self.init_q = jp.array(self.sys.mj_model.keyframe("home").qpos)
        self.default_pose = jp.array(self.sys.mj_model.keyframe("home").qpos)[7:]

        self.nu = self.sys.nu
        self.nq = self.sys.nq
        self.nv = self.sys.nv

        feet_link_mask = jp.array(
            np.char.find(self.sys.link_names, "foot") >= 0
        )
        self.feet_link_ids = jp.arange(self.sys.num_links())[feet_link_mask]

        self._weights = jp.array([
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # left leg.
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # right leg.
    ])

        foot_linvel_sensor_adr = []
        for site in ["l_foot", "r_foot"]:
            sensor_adr = int(self.robot.sensors[f"{site}_global_linvel"]["sensor_adr"])
            sensor_dim = int(self.robot.sensors[f"{site}_global_linvel"]["sensor_dim"])
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self.feet_pos_sensor_names = ("l_foot", "r_foot")

        hip_indices = []
        hip_joint_names = ["HR", "HAA"]
        for side in ["LL", "LR"]:
            for joint_name in hip_joint_names:
                hip_indices.append(
                    self.sys.mj_model.joint(f"{side}_{joint_name}").qposadr - 7
                )
        self.hip_indices = jp.array(hip_indices)

        knee_indices = []
        for side in ["LL", "LR"]:
            knee_indices.append(self.sys.mj_model.joint(f"{side}_KFE").qposadr - 7)
        self.knee_indices = jp.array(knee_indices)



        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)
        
        self.feet_body_ids = jp.array(
            [
                support.name2id(self.sys, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ["foot_left", "foot_right"]
            ]
        )
        self.torso_body_id = support.name2id(self.sys, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.torso_sensor_id = support.name2id(self.sys, mujoco.mjtObj.mjOBJ_SENSOR, "torso")


        #Observations 
        self.num_obs_history = self.cfg.obs.frame_stack
        self.num_privileged_obs_history = self.cfg.obs.c_frame_stack
        self.obs_size = self.cfg.obs.num_single_obs
        self.privileged_obs_size = self.cfg.obs.num_single_privileged_obs

        #Joystick related. Sample command
        self.resample_time = self.cfg.commands.resample_time
        self.resample_steps = int(self.resample_time / self.dt)
        self.reset_time = self.cfg.commands.reset_time
        self.reset_steps = int(self.reset_time / self.dt)

        #Domain randomization
        self.add_noise = self.cfg.noise.add_noise
        self.stack_obs = self.cfg.obs.stack_obs
        self.add_domain_rand = self.cfg.domain_rand.add_domain_rand
        self.add_push = self.cfg.push.add_push
        
        self.lin_vel_x = self.cfg.domain_rand.lin_vel_x
        self.lin_vel_y = self.cfg.domain_rand.lin_vel_y
        self.ang_vel_yaw = self.cfg.domain_rand.ang_vel_yaw

        #Push related
        self.push_interval_range = self.cfg.push.interval_range
        self.push_magnitude_range = self.cfg.push.magnitude_range

        qpos_noise_scale = np.zeros(12)
        hip_ids = [0, 1, 2, 6, 7, 8]
        kfe_ids = [3, 9]
        ffe_ids = [4, 10]
        faa_ids = [5, 11]
        qpos_noise_scale[hip_ids] = self.cfg.noise.hip_pos
        qpos_noise_scale[kfe_ids] = self.cfg.noise.kfe_pos
        qpos_noise_scale[ffe_ids] = self.cfg.noise.ffe_pos
        qpos_noise_scale[faa_ids] = self.cfg.noise.faa_pos
        self.qpos_noise_scale = jp.array(qpos_noise_scale)


    def _init_reward(self) -> None:
        """Initializes the reward system by filtering and scaling reward components.

        This method processes the reward scales configuration by removing any components with a scale of zero and scaling the remaining components by a time factor. It then prepares a list of reward function names and their corresponding scales, which are stored for later use in reward computation. Additionally, it sets parameters related to health and tracking rewards.
        """
        reward_scale_dict = OmegaConf.to_container(self.cfg.reward_scales)
        # Remove zero scales and multiply non-zero ones by dt
        for key in list(reward_scale_dict.keys()):
            if reward_scale_dict[key] == 0:
                reward_scale_dict.pop(key)

        # prepare list of functions
        self.reward_names = list(reward_scale_dict.keys())
        self.reward_functions = []
        self.reward_scales = jp.zeros(len(reward_scale_dict))
        for i, (name, scale) in enumerate(reward_scale_dict.items()):
            self.reward_functions.append(get_reward_function(name))
            self.reward_scales = self.reward_scales.at[i].set(scale)


        self.healthy_z_range = self.cfg.rewards.healthy_z_range
        self.max_foot_height = self.cfg.rewards.max_foot_height
        self.tracking_sigma = self.cfg.rewards.tracking_sigma
        self.base_height_target = self.cfg.rewards.base_height_target


    def reset(self, rng: jp.ndarray) -> State:
        """ Resets the environment to the initial state"""
        qpos = self.init_q.copy() 
        qvel = jp.zeros(self.nv)

        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw) 
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # qpos[7:]=*U(0.5, 1.5)
        rng, key = jax.random.split(rng)
        qpos = qpos.at[7:].set(
            qpos[7:] * jax.random.uniform(key, (12,), minval=0.5, maxval=1.5)
        )

        # d(xyzrpy)=U(-0.5, 0.5)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        #Here we can include some randomizations like joint positions, velocities of base, and position of base and orientation.

        pipeline_state = self.pipeline_init(qpos, qvel) 

        # We can take either a phase based walking approach or a trajectory based walking approach. (ZMP)
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
            minval=self.push_interval_range[0],
            maxval=self.push_interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)


        state_info = {
            "rng": rng,
            "step": 0,
            "done": False,
            "command": cmd,
            "last_act": jp.zeros(self.nu),
            "last_last_act": jp.zeros(self.nu),
            "motor_targets": jp.zeros(self.nu),
            "feet_air_time": jp.zeros(2),
            "first_contact": jp.zeros(2, dtype=jp.float32),
            "last_contact": jp.zeros(2, dtype=jp.float32),
            "swing_peak": jp.zeros(2),
            # Phase related.
            "phase_dt": phase_dt,
            "phase": phase,
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
        }

        obs_history = jp.zeros(self.num_obs_history * self.obs_size)
        privileged_obs_history = jp.zeros(
            self.num_privileged_obs_history * self.privileged_obs_size
        )

        obs = self._get_obs(
            pipeline_state,
            state_info,
            obs_history,
            privileged_obs_history
        )

        reward, done, zero = jp.zeros(3)

        metrics = {}
        for k in self.reward_names:
            metrics[k] = zero
        metrics["swing_peak"] = jp.zeros(())
        
        return State(
            pipeline_state, obs, reward, done, metrics, state_info
        )

    def step(self, state: State, action: jax.Array) -> State:
        """ Performs a step in the environment"""
        
        
        state.info["rng"], push1_rng, push2_rng = jax.random.split(
            state.info["rng"], 3
        )
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self.push_magnitude_range[0],
            maxval=self.push_magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
            == 0
        )
        push *= self.add_push
        qvel = state.pipeline_state.qd
        qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
        state.tree_replace({"pipeline_state.qd": qvel})


        motor_targets = self.default_pose + action * self.cfg.action.action_scale
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        state.info["motor_targets"] = motor_targets

        contact = check_feet_contact(pipeline_state, self.feet_link_ids)

        contact_bool = contact.astype(bool)
        last_contact_bool = state.info["last_contact"].astype(bool)
        contact_filt = contact_bool | last_contact_bool
        state.info["first_contact"] = ((state.info["feet_air_time"] > 0.0) * contact_filt).astype(jp.float32)
        state.info["feet_air_time"] += self.dt
        #check these values
        p_f = pipeline_state.x.pos[self.feet_link_ids] 
        p_fz = p_f[...,-1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact.astype(jp.float32)  
        state.info["swing_peak"] *= ~contact
        
        obs = self._get_obs(
            pipeline_state,
            state.info,
            state.obs['state'],
            state.obs['privileged_state'],
        )        

        done = self.get_termination(pipeline_state)
        state.info["done"] = done

        reward_dict = self._compute_reward(pipeline_state, state.info, action)
        reward = jp.clip(sum(reward_dict.values()) * self.dt, 0.0, 10000.0)
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1

        phase_tp1 = state.info["phase"] + state.info["phase_dt"]
        state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action.copy()
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["command"] = jax.lax.cond(
            state.info["step"] % self.resample_steps == 0,
            lambda: self.sample_command(cmd_rng),
            lambda: state.info["command"],
        )

        # reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > self.reset_steps), 0, state.info["step"]
        )

        state.metrics.update(reward_dict)

        return state.replace(
            pipeline_state = pipeline_state,
            obs = obs,
            reward = reward,
            done = done.astype(jp.float32),
        )
    
    def get_termination(self, pipeline_state: base.State) -> jax.Array:
        """Returns a boolean array indicating whether the termination condition is met."""
        torso_height = pipeline_state.xpos[self.torso_body_id, 2] 
        
        return jp.logical_or(
            torso_height < self.healthy_z_range[0],
            torso_height > self.healthy_z_range[1],
        )
    def _get_obs(
            self,
            pipeline_state: base.State,
            info: dict[str, Any],
            obs_history: jax.Array,
            privileged_obs_history: jax.Array,
    ) -> jp.ndarray:
        """ Returns the observation"""
        gyro = self.get_gyro(pipeline_state)
        info["rng"], noise_rng = jax.random.split(info["rng"], 2)
        noise = (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1) * self.cfg.noise.level * self.cfg.noise.gyro
        noisy_gyro = jp.where(self.add_noise, gyro + noise, gyro)
        
        gravity = pipeline_state.site_xmat[self.sys.mj_model.site("imu").id] @ jp.array([0,0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"], 2)
        noise = (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1) * self.cfg.noise.level * self.cfg.noise.gravity
        noisy_gravity = jp.where(self.add_noise, gravity + noise, gravity)

        joint_angles = pipeline_state.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"], 2)
        noise = (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1) * self.cfg.noise.level * self.qpos_noise_scale
        noisy_joint_angles = jp.where(self.add_noise, joint_angles + noise, joint_angles)

        joint_vel = pipeline_state.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"], 2)
        noise = (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1) * self.cfg.noise.level * self.cfg.noise.joint_vel
        noisy_joint_vel = jp.where(self.add_noise, joint_vel + noise, joint_vel)

        lin_vel = self.get_local_linvel(pipeline_state)
        info["rng"], noise_rng = jax.random.split(info["rng"], 2)
        noise = (2 * jax.random.uniform(noise_rng, shape=lin_vel.shape) - 1) * self.cfg.noise.level * self.cfg.noise.lin_vel
        noisy_lin_vel = jp.where(self.add_noise, lin_vel + noise, lin_vel)

        cos = jp.cos(info["phase"])
        sin = jp.sin(info["phase"])
        phase = jp.concatenate([cos, sin])

        obs = jp.hstack([
            noisy_lin_vel, #(3,)
            noisy_gyro, #(3,)
            noisy_gravity, #(3,)
            info["command"], #(3,)
            noisy_joint_angles - self.default_pose, #(12,)
            noisy_joint_vel,
            info["last_act"], #(12,)
            phase, #(2,)
        ])

        acceleromter = self.get_accelerometer(pipeline_state) #(3,)
        global_ang_vel = self.get_global_angvel(pipeline_state) #(3,)
        feet_vel = pipeline_state.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = pipeline_state.qpos[2]

        actuator_forces = pipeline_state.actuator_force
    
        privileged_obs = jp.hstack([
            obs, #(50,)
            gyro, #(3,)
            acceleromter, #(3,)
            gravity, #(3,)
            lin_vel, #(3,)
            global_ang_vel, #(3,)
            joint_angles - self.default_pose, #(12,)
            joint_vel, #(12,)
            root_height, #(1,)
            actuator_forces, #(12,)
            feet_vel, #(2,)
            info["feet_air_time"] #(2,)
        ])
    
        if self.stack_obs:
            obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

            privileged_obs = (
                jp.roll(privileged_obs_history, privileged_obs.size)
                .at[: privileged_obs.size]
                .set(privileged_obs)
            )
            
        return {
            "state": obs, 
            "privileged_state": privileged_obs
        }
    
    def _compute_reward(
        self, pipeline_state: base.State, info: dict[str, Any], action: jax.Array
    ):
        """Computes a dictionary of rewards based on the current pipeline state, additional information, and the action taken.

        Args:
            pipeline_state (base.State): The current state of the pipeline.
            info (dict[str, Any]): Additional information that may be required for reward computation.
            action (jax.Array): The action taken, which influences the reward calculation.

        Returns:
            Dict[str, jax.Array]: A dictionary where keys are reward names and values are the computed rewards as JAX arrays.
        """
        # Create an array of indices to map over
        indices = jp.arange(len(self.reward_names))
        
        # Create a list of partial functions that each include self as the first argument
        reward_fns_with_self = [
            lambda ps, inf, act, fn=fn: fn(self, ps, inf, act)
            for fn in self.reward_functions
        ]
        
        # Use jax.lax.map to compute rewards
        reward_arr = jax.lax.map(
            lambda i: jax.lax.switch(
                i,
                reward_fns_with_self,
                pipeline_state,
                info,
                action,
            ) * self.reward_scales[i],
            indices,
        )

        reward_dict = {}
        for i, name in enumerate(self.reward_names):
            reward_dict[name] = reward_arr[i]

        return reward_dict            

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """ 
        Sample random commands with 10% chance of setting everything to 0. 
        Used for exploration, robust learning, and generalization. 
        """
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        #x,y velocity should be between [-1, 1]
        lin_vel_x = jax.random.uniform(
            rng1, minval=self.lin_vel_x[0], maxval=self.lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            rng2, minval=self.lin_vel_y[0], maxval=self.lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            rng3,
            minval=self.ang_vel_yaw[0],
            maxval=self.ang_vel_yaw[1],
        )

        # With 10% chance, set everything to zero.
        return jp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jp.zeros(3),
            jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
        )

