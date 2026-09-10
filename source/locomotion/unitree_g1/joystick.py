# Copyright 2026 Triton Droids
# Behavioral design adapted from MuJoCo Playground commit
# 8a4b4642d8eba8a80ac99ed125cb62c16e1457ad (Apache-2.0).
"""Unitree G1 MJX joystick environment.

Phase 3 supplies deterministic reset/step, verified position-target actions,
contact state, observations, and termination. Phase 4 adds the task reward.
"""

from __future__ import annotations

from typing import Any

from brax import base
from brax.envs.base import State
import jax
import jax.numpy as jp
from mujoco.mjx._src import math

from source.locomotion.unitree_g1.base import UnitreeG1Env
from source.locomotion.unitree_g1 import rewards as reward_terms


def rigid_body_point_velocity(
    body_position: jax.Array,
    body_linear_velocity: jax.Array,
    body_angular_velocity: jax.Array,
    point_position: jax.Array,
) -> jax.Array:
    """Return world velocity of a rigidly attached world-frame point."""

    return body_linear_velocity + jp.cross(
        body_angular_velocity, point_position - body_position
    )


class Joystick(UnitreeG1Env):
    """Flat-ground 29-actuator G1 environment with joystick commands."""

    def __init__(self, name: str, robot: Any, scene: str, cfg: Any, **kwargs: Any):
        super().__init__(name, robot, scene, cfg, **kwargs)
        self.init_q = jp.asarray(self.metadata.default_qpos)
        self.default_pose = self.init_q[7:]
        self.ctrl_lower = jp.asarray(self.sys.actuator.ctrl_range[:, 0])
        self.ctrl_upper = jp.asarray(self.sys.actuator.ctrl_range[:, 1])
        self.soft_joint_lower = self._soft_limit(
            self.ctrl_lower, self.ctrl_upper, take_lower=True
        )
        self.soft_joint_upper = self._soft_limit(
            self.ctrl_lower, self.ctrl_upper, take_lower=False
        )

        self.foot_site_ids = jp.asarray(
            [self.sys.mj_model.site(name).id for name in self.metadata.foot_sites]
        )
        self.foot_link_ids = jp.asarray(self.metadata.foot_link_ids)
        self.foot_geom_ids = tuple(jp.asarray(side) for side in self.metadata.foot_geom_ids)
        self.cross_contact_foot_geom_ids = jp.asarray(
            self.metadata.cross_contact_foot_geom_ids
        )
        self.shin_geom_ids = tuple(jp.asarray(side) for side in self.metadata.shin_geom_ids)
        self.hand_geom_ids = jp.asarray(self.metadata.hand_geom_ids)
        self.thigh_geom_ids = jp.asarray(self.metadata.thigh_geom_ids)
        self.floor_geom_id = self.metadata.floor_geom_id
        self.pelvis_body_id = self.metadata.pelvis_body_id
        self.torso_body_id = self.metadata.torso_body_id
        self.pelvis_imu_site_id = self.metadata.pelvis_imu_site_id

        self.resample_steps = max(1, int(self.cfg.commands.resample_time / self.dt))
        self.add_noise = self.cfg.noise.add_noise
        self.add_push = self.cfg.push.add_push
        self.push_interval_range = self.cfg.push.interval_range
        self.push_magnitude_range = self.cfg.push.magnitude_range
        self.reward_scales = {
            name: float(value) for name, value in self.cfg.reward_scales.items()
        }

    def _soft_limit(
        self, lower: jax.Array, upper: jax.Array, *, take_lower: bool
    ) -> jax.Array:
        center = (lower + upper) * 0.5
        radius = (upper - lower) * 0.5 * self.cfg.rewards.soft_joint_pos_limit_factor
        return center - radius if take_lower else center + radius

    def action_to_targets(self, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Map normalized actions to legal absolute position targets."""

        clipped_action = jp.clip(jp.asarray(action), -1.0, 1.0)
        targets = self.default_pose + clipped_action * self.cfg.action.action_scale
        targets = jp.clip(targets, self.ctrl_lower, self.ctrl_upper)
        return clipped_action, targets

    def get_foot_global_linvel(self, pipeline_state: base.State) -> jax.Array:
        """Return world-frame velocity at each named foot site.

        ``pipeline_state.xd.vel`` is the linear velocity of the body-frame
        origin, not of an offset site.  Playground defines this quantity with
        MuJoCo ``framelinvel`` site sensors, so include the rigid-body
        ``omega x r`` contribution explicitly for the Menagerie model.
        """

        body_position = pipeline_state.x.pos[self.foot_link_ids]
        body_motion = jax.tree_util.tree_map(
            lambda value: value[self.foot_link_ids], pipeline_state.xd
        )
        return rigid_body_point_velocity(
            body_position,
            body_motion.vel,
            body_motion.ang,
            pipeline_state.site_xpos[self.foot_site_ids],
        )

    def reset(self, rng: jax.Array) -> State:
        qpos = self.init_q
        qvel = jp.zeros(self.nv)

        rng, xy_rng, yaw_rng, joint_rng, velocity_rng = jax.random.split(rng, 5)
        if self.cfg.reset.randomize:
            qpos = qpos.at[:2].add(
                jax.random.uniform(
                    xy_rng,
                    (2,),
                    minval=-self.cfg.reset.xy_range,
                    maxval=self.cfg.reset.xy_range,
                )
            )
            yaw = jax.random.uniform(
                yaw_rng,
                (),
                minval=-self.cfg.reset.yaw_range,
                maxval=self.cfg.reset.yaw_range,
            )
            yaw_quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
            qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], yaw_quat))
            qpos = qpos.at[7:].set(
                qpos[7:]
                * jax.random.uniform(
                    joint_rng,
                    (self.nu,),
                    minval=self.cfg.reset.joint_scale_range[0],
                    maxval=self.cfg.reset.joint_scale_range[1],
                )
            )
            qpos = qpos.at[7:].set(jp.clip(qpos[7:], self.ctrl_lower, self.ctrl_upper))
            qvel = qvel.at[:6].set(
                jax.random.uniform(
                    velocity_rng,
                    (6,),
                    minval=-self.cfg.reset.base_velocity_range,
                    maxval=self.cfg.reset.base_velocity_range,
                )
            )

        # Position actuators must initially hold the pose they were forwarded
        # from.  Brax otherwise defaults ``ctrl`` to zero, creating a large,
        # artificial reset impulse and inconsistent critic observation.
        pipeline_state = self.pipeline_init(qpos, qvel, ctrl=qpos[7:])
        contact, undesired, self_collision, nonfoot_ground = self.contact_details(
            pipeline_state
        )
        collision = self_collision | nonfoot_ground
        rng, frequency_rng, command_rng, push_rng = jax.random.split(rng, 4)
        gait_frequency = jax.random.uniform(frequency_rng, (), minval=1.25, maxval=1.5)
        phase = jp.array([0.0, jp.pi])
        push_interval = jax.random.uniform(
            push_rng,
            (),
            minval=self.push_interval_range[0],
            maxval=self.push_interval_range[1],
        )
        info = {
            "rng": rng,
            "step": jp.array(0, dtype=jp.int32),
            "command": self.sample_command(command_rng),
            "last_act": jp.zeros(self.nu),
            "last_last_act": jp.zeros(self.nu),
            "motor_targets": qpos[7:],
            "feet_air_time": jp.zeros(2),
            "first_contact": jp.zeros(2, dtype=bool),
            "last_contact": contact,
            "swing_peak": jp.zeros(2),
            "phase": phase,
            "phase_dt": 2.0 * jp.pi * self.dt * gait_frequency,
            "push": jp.zeros(2),
            "push_step": jp.array(0, dtype=jp.int32),
            "push_interval_steps": jp.maximum(
                1, jp.round(push_interval / self.dt).astype(jp.int32)
            ),
            "undesired_contact": undesired,
            "collision_contact": collision,
            "self_collision_contact": self_collision,
            "nonfoot_ground_contact": nonfoot_ground,
        }
        obs = self._get_obs(pipeline_state, info, contact)
        metrics = {
            "foot_contact_left": contact[0].astype(jp.float32),
            "foot_contact_right": contact[1].astype(jp.float32),
            "undesired_contact": undesired.astype(jp.float32),
            "collision_contact": collision.astype(jp.float32),
            "self_collision_contact": self_collision.astype(jp.float32),
            "nonfoot_ground_contact": nonfoot_ground.astype(jp.float32),
        }
        metrics.update(
            {f"reward/{name}": jp.zeros(()) for name in self.reward_scales}
        )
        zero = jp.zeros(())
        return State(pipeline_state, obs, zero, zero, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        state, push = self._apply_scheduled_push(state)
        # Treat ``info`` and ``metrics`` as next-state values.  Mutating the
        # dictionaries carried by the input State aliases rollout history and
        # makes observation timing depend on tracing details.
        info = dict(state.info)
        metrics = dict(state.metrics)
        clipped_action, motor_targets = self.action_to_targets(action)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        contact, undesired, self_collision, nonfoot_ground = self.contact_details(
            pipeline_state
        )
        collision = self_collision | nonfoot_ground

        contact_filtered = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0.0) & contact_filtered
        feet_air_time = info["feet_air_time"] + self.dt
        foot_z = pipeline_state.site_xpos[self.foot_site_ids, 2]
        swing_peak = jp.maximum(info["swing_peak"], foot_z)

        # Rewards belong to the transition just taken: use the command, phase,
        # previous action and *unreset* air time that governed that transition.
        reward_info = {
            **info,
            "motor_targets": motor_targets,
            "first_contact": first_contact,
            "feet_air_time": feet_air_time,
            "swing_peak": swing_peak,
            "undesired_contact": undesired,
            "collision_contact": collision,
            "self_collision_contact": self_collision,
            "nonfoot_ground_contact": nonfoot_ground,
        }
        done = self.get_termination(pipeline_state, undesired)
        raw_rewards = self._reward_terms(
            pipeline_state,
            clipped_action,
            reward_info,
            done,
            first_contact,
            contact,
            undesired,
            collision,
        )
        weighted_rewards = {
            name: raw_rewards[name] * scale
            for name, scale in self.reward_scales.items()
        }
        reward = reward_terms.aggregate_weighted_terms(weighted_rewards, self.dt)

        # Build the state observed by the next policy call only after the
        # transition reward has been evaluated.
        next_info = dict(reward_info)
        next_info["push"] = push
        next_info["push_step"] = info["push_step"] + 1
        next_info["step"] = info["step"] + 1
        next_info["phase"] = jp.fmod(
            info["phase"] + info["phase_dt"] + jp.pi, 2.0 * jp.pi
        ) - jp.pi
        next_info["last_last_act"] = info["last_act"]
        next_info["last_act"] = clipped_action
        next_info["last_contact"] = contact
        next_info["feet_air_time"] = feet_air_time * ~contact
        next_info["swing_peak"] = swing_peak * ~contact
        next_info["rng"], command_rng = jax.random.split(info["rng"])
        next_info["command"] = jax.lax.cond(
            next_info["step"] % self.resample_steps == 0,
            lambda: self.sample_command(command_rng),
            lambda: info["command"],
        )
        obs = self._get_obs(pipeline_state, next_info, contact)

        metrics["foot_contact_left"] = contact[0].astype(jp.float32)
        metrics["foot_contact_right"] = contact[1].astype(jp.float32)
        metrics["undesired_contact"] = undesired.astype(jp.float32)
        metrics["collision_contact"] = collision.astype(jp.float32)
        metrics["self_collision_contact"] = self_collision.astype(jp.float32)
        metrics["nonfoot_ground_contact"] = nonfoot_ground.astype(jp.float32)
        for name, value in weighted_rewards.items():
            metrics[f"reward/{name}"] = value
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
            metrics=metrics,
            info=next_info,
        )

    def _reward_terms(
        self,
        pipeline_state: base.State,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
        undesired_contact: jax.Array,
        collision_contact: jax.Array,
    ) -> dict[str, jax.Array]:
        """Compute all unweighted Phase 4 reward/cost terms."""

        joint_position = pipeline_state.qpos[7:]
        joint_velocity = pipeline_state.qvel[6:]
        foot_velocity = self.get_foot_global_linvel(pipeline_state)
        return {
            "tracking_lin_vel": reward_terms.tracking_linear_velocity(
                info["command"],
                self.get_local_linvel(pipeline_state, "pelvis"),
                self.cfg.rewards.tracking_sigma,
            ),
            "tracking_ang_vel": reward_terms.tracking_yaw_rate(
                info["command"],
                self.get_gyro(pipeline_state, "pelvis"),
                self.cfg.rewards.tracking_sigma,
            ),
            "lin_vel_z": reward_terms.vertical_velocity_cost(
                self.get_global_linvel(pipeline_state, "pelvis"),
                self.get_global_linvel(pipeline_state, "torso"),
            ),
            "ang_vel_xy": reward_terms.roll_pitch_angular_velocity_cost(
                self.get_global_angvel(pipeline_state, "torso")
            ),
            "orientation": reward_terms.upright_orientation_cost(
                self.get_gravity(pipeline_state, "torso")
            ),
            "torques": reward_terms.actuator_effort_cost(
                pipeline_state.actuator_force
            ),
            "energy": reward_terms.mechanical_power_cost(
                joint_velocity, pipeline_state.actuator_force
            ),
            "action_rate": reward_terms.action_rate_cost(
                action, info["last_act"]
            ),
            "dof_acc": reward_terms.joint_acceleration_cost(
                pipeline_state.qacc[6:]
            ),
            "dof_pos_limits": reward_terms.joint_limit_cost(
                joint_position, self.soft_joint_lower, self.soft_joint_upper
            ),
            "feet_slip": reward_terms.foot_slip_cost(
                foot_velocity[:, :2], contact
            ),
            "collision": reward_terms.undesired_contact_cost(collision_contact),
            "termination": reward_terms.termination_cost(done),
            "alive": reward_terms.alive_reward(),
            "feet_air_time": reward_terms.foot_air_time_reward(
                info["feet_air_time"], first_contact, info["command"]
            ),
            "feet_phase": reward_terms.foot_phase_reward(
                pipeline_state.site_xpos[self.foot_site_ids, 2],
                info["phase"],
                self.cfg.rewards.max_foot_height,
                info["command"],
            ),
            "stand_still": reward_terms.stand_still_cost(
                info["command"], joint_position, self.default_pose
            ),
            "pose": reward_terms.pose_cost(joint_position, self.default_pose),
        }

    def contact_state(
        self, pipeline_state: base.State
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Classify foot support, terminal contacts, and penalized collisions.

        The pinned Menagerie scene uses three capsule geoms per foot for floor
        support and a distinct box geom per foot for explicit cross-leg pairs.
        Playground terminates cross-foot/cross-shin contacts; non-foot ground
        and same-side hand/thigh contacts are penalized here without making a
        still-upright recovery impossible.
        """

        contact, termination, self_collision, nonfoot_ground = self.contact_details(
            pipeline_state
        )
        return contact, termination, self_collision | nonfoot_ground

    def contact_details(
        self, pipeline_state: base.State
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return support, terminal cross-leg, self, and ground contacts."""

        geom_pairs = pipeline_state.contact.geom
        active = pipeline_state.contact.dist < 0.0
        geom1 = geom_pairs[:, 0]
        geom2 = geom_pairs[:, 1]
        foot_contacts = []
        allowed = jp.zeros_like(active, dtype=bool)
        for foot_geoms in self.foot_geom_ids:
            match = (
                (geom1 == self.floor_geom_id) & jp.isin(geom2, foot_geoms)
            ) | ((geom2 == self.floor_geom_id) & jp.isin(geom1, foot_geoms))
            allowed |= match
            foot_contacts.append(jp.any(active & match))
        floor_contact = (geom1 == self.floor_geom_id) | (geom2 == self.floor_geom_id)
        nonfoot_ground = active & floor_contact & ~allowed

        def pair_matches(first_set: jax.Array, second_set: jax.Array) -> jax.Array:
            return (
                jp.isin(geom1, first_set) & jp.isin(geom2, second_set)
            ) | (
                jp.isin(geom2, first_set) & jp.isin(geom1, second_set)
            )

        left_cross_foot = self.cross_contact_foot_geom_ids[:1]
        right_cross_foot = self.cross_contact_foot_geom_ids[1:]
        dangerous_cross_leg = pair_matches(left_cross_foot, right_cross_foot)
        dangerous_cross_leg |= pair_matches(
            left_cross_foot, self.shin_geom_ids[1]
        )
        dangerous_cross_leg |= pair_matches(
            right_cross_foot, self.shin_geom_ids[0]
        )
        self_collision = pair_matches(
            self.hand_geom_ids[:1], self.thigh_geom_ids[:1]
        )
        self_collision |= pair_matches(
            self.hand_geom_ids[1:], self.thigh_geom_ids[1:]
        )

        termination_contact = jp.any(active & dangerous_cross_leg)
        self_collision_contact = jp.any(active & self_collision)
        nonfoot_ground_contact = jp.any(nonfoot_ground)
        return (
            jp.stack(foot_contacts),
            termination_contact,
            self_collision_contact,
            nonfoot_ground_contact,
        )

    def get_termination(
        self, pipeline_state: base.State, undesired_contact: jax.Array
    ) -> jax.Array:
        """Terminate on an invalid pose/contact or any non-finite state."""

        # The pelvis carries the free joint, so q[2] is its verified world z.
        pelvis_height = pipeline_state.q[2]
        torso_up_z = self.get_gravity(pipeline_state, "torso")[2]
        invalid = ~jp.isfinite(pipeline_state.q).all() | ~jp.isfinite(
            pipeline_state.qd
        ).all()
        return (
            (pelvis_height < self.cfg.termination.min_pelvis_height)
            | (pelvis_height > self.cfg.termination.max_pelvis_height)
            | (torso_up_z < self.cfg.termination.min_torso_up_z)
            | undesired_contact
            | invalid
        )

    def _apply_scheduled_push(self, state: State) -> tuple[State, jax.Array]:
        info = dict(state.info)
        info["rng"], theta_rng, magnitude_rng = jax.random.split(info["rng"], 3)
        theta = jax.random.uniform(theta_rng, (), maxval=2.0 * jp.pi)
        magnitude = jax.random.uniform(
            magnitude_rng,
            (),
            minval=self.push_magnitude_range[0],
            maxval=self.push_magnitude_range[1],
        )
        scheduled = (
            (info["push_step"] + 1) % info["push_interval_steps"] == 0
        ) & self.add_push
        push = jp.array([jp.cos(theta), jp.sin(theta)]) * magnitude * scheduled
        qd = state.pipeline_state.qd.at[:2].add(push)
        return (
            state.replace(
                pipeline_state=state.pipeline_state.replace(qd=qd), info=info
            ),
            push,
        )

    def _get_obs(
        self, pipeline_state: base.State, info: dict[str, Any], contact: jax.Array
    ) -> dict[str, jax.Array]:
        gyro = self.get_gyro(pipeline_state, "pelvis")
        projected_gravity = (
            pipeline_state.site_xmat[self.pelvis_imu_site_id].T
            @ jp.array([0.0, 0.0, -1.0])
        )
        joint_pos = pipeline_state.qpos[7:]
        joint_vel = pipeline_state.qvel[6:]
        local_linvel = self.get_local_linvel(pipeline_state, "pelvis")

        noisy_linvel = self._with_noise(info, local_linvel, self.cfg.noise.lin_vel)
        noisy_gyro = self._with_noise(info, gyro, self.cfg.noise.gyro)
        noisy_gravity = self._with_noise(info, projected_gravity, self.cfg.noise.gravity)
        noisy_joint_pos = self._with_noise(info, joint_pos, self.cfg.noise.joint_pos)
        noisy_joint_vel = self._with_noise(info, joint_vel, self.cfg.noise.joint_vel)

        phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])
        actor = jp.concatenate(
            [
                noisy_linvel,
                noisy_gyro,
                noisy_gravity,
                info["command"],
                noisy_joint_pos - self.default_pose,
                noisy_joint_vel,
                info["last_act"],
                phase,
            ]
        )

        accelerometer = self.get_accelerometer(pipeline_state, "pelvis")
        global_angvel = self.get_global_angvel(pipeline_state, "pelvis")
        foot_vel = self.get_foot_global_linvel(pipeline_state).reshape(-1)
        privileged = jp.concatenate(
            [
                actor,
                gyro,
                accelerometer,
                projected_gravity,
                local_linvel,
                global_angvel,
                joint_pos - self.default_pose,
                joint_vel,
                pipeline_state.qpos[2:3],
                pipeline_state.actuator_force,
                contact.astype(jp.float32),
                foot_vel,
                info["feet_air_time"],
            ]
        )
        return {"state": actor, "privileged_state": privileged}

    def _with_noise(
        self, info: dict[str, Any], value: jax.Array, scale: float
    ) -> jax.Array:
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noise = jax.random.uniform(noise_rng, value.shape, minval=-1.0, maxval=1.0)
        return jp.where(
            self.add_noise,
            value + noise * self.cfg.noise.level * scale,
            value,
        )

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample `[vx, vy, yaw_rate]` inside the conservative Playground ranges."""

        x_rng, y_rng, yaw_rng, zero_rng = jax.random.split(rng, 4)
        command = jp.array(
            [
                jax.random.uniform(
                    x_rng,
                    (),
                    minval=self.cfg.commands.lin_vel_x[0],
                    maxval=self.cfg.commands.lin_vel_x[1],
                ),
                jax.random.uniform(
                    y_rng,
                    (),
                    minval=self.cfg.commands.lin_vel_y[0],
                    maxval=self.cfg.commands.lin_vel_y[1],
                ),
                jax.random.uniform(
                    yaw_rng,
                    (),
                    minval=self.cfg.commands.ang_vel_yaw[0],
                    maxval=self.cfg.commands.ang_vel_yaw[1],
                ),
            ]
        )
        return jp.where(
            jax.random.bernoulli(zero_rng, self.cfg.commands.zero_probability),
            jp.zeros(3),
            command,
        )
