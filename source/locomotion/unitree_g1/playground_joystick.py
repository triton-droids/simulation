# Copyright 2026 Triton Droids
# Thin adapter around MuJoCo Playground commit
# 8a4b4642d8eba8a80ac99ed125cb62c16e1457ad (Apache-2.0).
"""Pinned authoritative Unitree G1 joystick adapter."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jp
from mujoco import mjx

from source.locomotion.unitree_g1.playground_source import (
    load_playground_g1_modules,
    resolve_playground_source,
)
from source.locomotion.unitree_g1.training_wrapper import wrap_for_brax_training
from source.locomotion.unitree_g1.foot_separation import narrow_foot_cost
from source.locomotion.unitree_g1.recovery_reset import sample_recovery_reset, backward_lateral_score


def synchronize_transition_observation(
    observation: dict[str, jax.Array], info: dict[str, jax.Array]
) -> dict[str, jax.Array]:
    """Synchronize upstream's stale history slices with returned next state."""

    phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])

    def update_actor_prefix(value: jax.Array) -> jax.Array:
        value = value.at[9:12].set(info["command"])
        value = value.at[70:99].set(info["last_act"])
        return value.at[99:103].set(phase)

    state = update_actor_prefix(observation["state"])
    privileged = update_actor_prefix(observation["privileged_state"])
    privileged = privileged.at[-2:].set(info["feet_air_time"])
    return {"state": state, "privileged_state": privileged}


def contact_phase_reward(contact: jax.Array, phase: jax.Array, command: jax.Array) -> jax.Array:
    """Reward support near phase pi and swing near phase zero.

    Phase is the pre-transition phase, as in upstream's height reward.
    Antipodal phases give both planted and both airborne feet zero net credit;
    correct alternating support is positive, opposite support is negative.
    Zero command disables this optional gait-acquisition signal.
    """
    signed_contact = 2.0 * contact.astype(phase.dtype) - 1.0
    alignment = jp.mean(-signed_contact * jp.cos(phase))
    single_support = jp.sum(contact) == 1
    return alignment * single_support * (jp.linalg.norm(command) > 0.01)


def standing_command_sampler(rng, *, original, probability):
    """Increase upstream's zero-command probability using its same fourth key."""
    command = original(rng)
    zero_key = jax.random.split(rng, 4)[3]
    return jp.where(jax.random.bernoulli(zero_key, p=probability), jp.zeros_like(command), command)


class Joystick:
    """Repository-facing wrapper around the exact pinned Playground G1 task."""

    _feet_contact_phase_scale = 0.0
    _narrow_feet_scale = 0.0
    _stand_yaw_scale = 0.0
    _phase_reward_command_only = False
    _airtime_reward_command_only = False

    def __init__(self, name: str, robot: Any, scene: str, cfg: Any, **_: Any):
        if name != "unitree_g1":
            raise ValueError(f"Playground G1 adapter received robot name {name!r}")
        if scene != "flat":
            raise ValueError("The pinned Playground G1 adapter supports flat terrain only.")
        if cfg.playground.implementation != "jax":
            raise ValueError("The baseline adapter requires Playground's JAX MJX implementation.")
        if cfg.domain_rand.add_domain_rand:
            raise NotImplementedError(
                "Playground domain randomization remains disabled during baseline diagnosis."
            )

        source = resolve_playground_source(
            source_root=cfg.playground.source_root,
            cache_root=cfg.playground.cache_root,
            fetch=cfg.playground.fetch_source,
        )
        mjx_env_module, wrapper_module, joystick_module = load_playground_g1_modules(
            source
        )
        mjx_env_module.MENAGERIE_PATH = mjx_env_module.epath.Path(
            robot.resolution.scene_path.parents[1]
        )

        effective = joystick_module.default_config()
        effective.impl = "jax"
        effective.sim_dt = float(cfg.sim.timestep)
        effective.ctrl_dt = float(cfg.sim.timestep * cfg.action.n_frames)
        effective.action_scale = float(cfg.action.action_scale)
        effective.soft_joint_pos_limit_factor = float(
            cfg.rewards.soft_joint_pos_limit_factor
        )
        effective.reward_config.tracking_sigma = float(cfg.rewards.tracking_sigma)
        effective.reward_config.max_foot_height = float(cfg.rewards.max_foot_height)
        effective.reward_config.max_contact_force = float(cfg.rewards.max_contact_force)
        for reward_name in effective.reward_config.scales.keys():
            effective.reward_config.scales[reward_name] = float(
                cfg.reward_scales[reward_name]
            )
        effective.noise_config.level = (
            float(cfg.noise.level) if cfg.noise.add_noise else 0.0
        )
        noise_fields = {
            "joint_pos": "joint_pos",
            "joint_vel": "joint_vel",
            "gravity": "gravity",
            "linvel": "lin_vel",
            "gyro": "gyro",
        }
        for playground_name, repository_name in noise_fields.items():
            effective.noise_config.scales[playground_name] = float(
                cfg.noise[repository_name]
            )
        effective.push_config.enable = bool(cfg.push.add_push)
        effective.push_config.interval_range = list(cfg.push.interval_range)
        effective.push_config.magnitude_range = list(cfg.push.magnitude_range)
        effective.lin_vel_x = list(cfg.commands.lin_vel_x)
        effective.lin_vel_y = list(cfg.commands.lin_vel_y)
        effective.ang_vel_yaw = list(cfg.commands.ang_vel_yaw)

        self.name = name
        self.robot = robot
        self.cfg = cfg
        self.metadata = robot.metadata
        self.model_source = robot.resolution
        self.playground_source = source
        self.add_domain_rand = False
        self._reset_randomized = bool(cfg.reset.randomize)
        self._recovery_reset_candidates = int(getattr(cfg.playground, "recovery_reset_candidates", 1))
        if not 1 <= self._recovery_reset_candidates <= 8:
            raise ValueError("Reset candidates must be between 1 and 8")
        self._reset_disturbance_scale = float(getattr(cfg.playground, "reset_disturbance_scale", 1.0))
        if not 0.0 <= self._reset_disturbance_scale <= 1.0:
            raise ValueError("Reset disturbance scale must be in [0, 1]")
        # Old saved configs have no local shaping field and remain unchanged.
        self._feet_contact_phase_scale = float(getattr(cfg.reward_scales, "feet_contact_phase", 0.0))
        self._phase_reward_command_only = bool(getattr(cfg.playground, "phase_reward_command_only", False))
        self._airtime_reward_command_only = bool(getattr(cfg.playground, "airtime_reward_command_only", False))
        self._narrow_feet_scale = float(getattr(cfg.reward_scales, "narrow_feet", 0.0))
        self._stand_yaw_scale = float(getattr(cfg.reward_scales, "stand_yaw", 0.0))
        self._mjx_env_module = mjx_env_module
        self._env = joystick_module.Joystick(task="flat_terrain", config=effective)
        stand_probability = float(getattr(cfg.playground, "standing_command_probability", 0.1))
        if not 0.1 <= stand_probability <= 1.0:
            raise ValueError("Standing command probability must be in [0.1, 1]")
        if stand_probability != 0.1:
            self._env.sample_command = functools.partial(
                standing_command_sampler, original=self._env.sample_command,
                probability=stand_probability)
        # Playground's default fast auto-reset restores only data/observations
        # and deliberately carries transition history across episodes.  That
        # makes the command, phase, contact, and action history disagree with
        # the restored reset observation.  Use its supported full-reset mode
        # so every training episode starts from one coherent state.
        self.brax_training_wrapper = functools.partial(
            wrap_for_brax_training, wrapper_module=wrapper_module, full_reset=True
        )
        self.effective_config = effective.to_dict()
        self.source_record = {
            **source.to_dict(),
            "scene": str(self._env.xml_path),
            "model": robot.source_record,
            "adaptations": [
                "functional info/metrics dictionaries",
                "returned command/action/phase/air-time observation synchronization",
                "normalized action clipping",
                "qpos/qvel finite termination",
                "optional nominal reset for diagnostics",
                "full-state auto-reset metadata synchronization",
                "preserve Brax terminal and timeout bookkeeping through full reset",
            ],
        }
        self.effective_config["standing_command_probability"] = stand_probability
        if stand_probability != 0.1:
            self.source_record["adaptations"].append("increase sampled zero-command exposure using upstream RNG key")
        if self._feet_contact_phase_scale != 0.0:
            self.effective_config["local_reward_scales"] = {
                "feet_contact_phase": self._feet_contact_phase_scale,
            }
            self.source_record["adaptations"].append(
                "optional local contact-phase reward using pre-transition phase and command"
            )
        if self._narrow_feet_scale != 0.0:
            self.effective_config.setdefault("local_reward_scales", {})["narrow_feet"] = self._narrow_feet_scale
            self.source_record["adaptations"].append("optional bounded narrow-foot cost, heading-frame width .16m")
        if self._stand_yaw_scale != 0.0:
            self.effective_config.setdefault("local_reward_scales", {})["stand_yaw"] = self._stand_yaw_scale
            self.source_record["adaptations"].append("zero-command-only squared pelvis yaw rate cost")
        if self._phase_reward_command_only:
            self.effective_config["phase_reward_command_only"] = True
            self.source_record["adaptations"].append("disable phase-height reward at zero command")
        if self._airtime_reward_command_only:
            self.effective_config["airtime_reward_command_only"] = True
            self.source_record["adaptations"].append("disable air-time reward at zero command")
        self.nq = self._env.mj_model.nq
        self.nv = self._env.mj_model.nv
        self.nu = self._env.mj_model.nu
        self.obs_size = cfg.obs.num_single_obs
        self.privileged_obs_size = cfg.obs.num_single_privileged_obs

    def __getattr__(self, name: str):
        if name == "__setstate__":
            raise AttributeError(name)
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)

    def _contact(self, data: Any) -> jax.Array:
        return jp.asarray(
            [
                data.sensordata[self._env.mj_model.sensor_adr[sensor_id]] > 0
                for sensor_id in self._env._feet_floor_found_sensor
            ]
        )

    def action_to_targets(self, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        applied = jp.clip(jp.asarray(action), -1.0, 1.0)
        targets = self._env._default_pose + applied * self._env._config.action_scale
        ranges = jp.asarray(self._env.mj_model.actuator_ctrlrange)
        return applied, jp.clip(targets, ranges[:, 0], ranges[:, 1])

    def reset(self, rng: jax.Array):
        state = sample_recovery_reset(
            self._env.reset,
            lambda s: backward_lateral_score(self._env.get_local_linvel(s.data, "pelvis")),
            rng, self._recovery_reset_candidates if self._reset_randomized else 1)
        info = dict(state.info)
        metrics = dict(state.metrics)
        data = state.data
        if not self._reset_randomized or self._reset_disturbance_scale != 1.0:
            if not self._reset_randomized:
                qpos = self._env._init_q
                qvel = jp.zeros(self.nv)
            else:
                scale = self._reset_disturbance_scale
                qpos = data.qpos.at[7:].set(self._env._init_q[7:] + scale * (data.qpos[7:] - self._env._init_q[7:]))
                qvel = data.qvel * scale
            data = self._mjx_env_module.make_data(
                self._env.mj_model,
                qpos=qpos,
                qvel=qvel,
                ctrl=qpos[7:],
                impl=self._env.mjx_model.impl.value,
                naconmax=self._env._config.naconmax,
                njmax=self._env._config.njmax,
            )
            data = mjx.forward(self._env.mjx_model, data)
            contact = self._contact(data)
            info["last_contact"] = contact
            info["feet_air_time"] = jp.zeros(2)
            info["swing_peak"] = jp.zeros(2)
            observation_info = dict(info)
            obs = self._env._get_obs(data, observation_info, contact)
            info = observation_info
        else:
            obs = state.obs
        info["motor_targets"] = data.ctrl
        if self._feet_contact_phase_scale != 0.0:
            metrics["reward/feet_contact_phase"] = jp.zeros_like(state.reward)
        if self._narrow_feet_scale != 0.0:
            metrics["reward/narrow_feet"] = jp.zeros_like(state.reward)
        if self._stand_yaw_scale != 0.0:
            metrics["reward/stand_yaw"] = jp.zeros_like(state.reward)
        return state.replace(data=data, obs=obs, info=info, metrics=metrics)

    def step(self, state: Any, action: jax.Array):
        functional_state = state.replace(
            info=dict(state.info), metrics=dict(state.metrics)
        )
        applied_action = jp.clip(jp.asarray(action), -1.0, 1.0)
        next_state = self._env.step(functional_state, applied_action)
        info = dict(next_state.info)
        metrics = dict(next_state.metrics)
        reward = next_state.reward
        if self._phase_reward_command_only:
            old = metrics["reward/feet_phase"]
            new = old * (jp.linalg.norm(state.info["command"]) > 0.01)
            metrics["reward/feet_phase"] = new
            reward = reward + (new - old) * self.dt
        if self._airtime_reward_command_only:
            old = metrics["reward/feet_air_time"]
            new = old * (jp.linalg.norm(state.info["command"]) > 0.01)
            metrics["reward/feet_air_time"] = new
            reward = reward + (new - old) * self.dt
        if self._feet_contact_phase_scale != 0.0:
            weighted = self._feet_contact_phase_scale * contact_phase_reward(
                self._contact(next_state.data), state.info["phase"], state.info["command"]
            )
            metrics["reward/feet_contact_phase"] = weighted
            reward = reward + weighted * self.dt
        if self._narrow_feet_scale != 0.0:
            weighted = self._narrow_feet_scale * narrow_foot_cost(
                next_state.data.site_xpos[self._env._feet_site_id], next_state.data.qpos[3:7])
            metrics["reward/narrow_feet"] = weighted
            reward = reward + weighted * self.dt
        if self._stand_yaw_scale != 0.0:
            weighted = self._stand_yaw_scale * jp.square(self._env.get_gyro(next_state.data, "pelvis")[2]) * (jp.linalg.norm(state.info["command"]) < 0.01)
            metrics["reward/stand_yaw"] = weighted
            reward = reward + weighted * self.dt
        obs = synchronize_transition_observation(next_state.obs, info)
        invalid = ~jp.isfinite(next_state.data.qpos).all()
        invalid |= ~jp.isfinite(next_state.data.qvel).all()
        done = next_state.done.astype(bool) | invalid
        return next_state.replace(
            reward=reward,
            obs=obs,
            done=done.astype(next_state.reward.dtype),
            info=info,
            metrics=metrics,
        )
