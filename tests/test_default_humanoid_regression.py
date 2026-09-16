"""Regression contract for the pre-existing 12-actuator locomotion task."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf
import pytest

from source.config.sim import MJXConfig
from source.locomotion.default_humanoid_legs.joystick import Joystick
from source.rewards import locomotion_rewards
from source.robots.robot import Robot


@pytest.fixture(scope="module")
def default_env():
    cfg = OmegaConf.structured(MJXConfig())
    cfg.noise.add_noise = False
    cfg.domain_rand.add_domain_rand = False
    cfg.push.add_push = False
    cfg.obs.stack_obs = False
    return Joystick(
        "default_humanoid_legs",
        Robot("default_humanoid_legs"),
        "flat",
        cfg,
    )


@pytest.fixture(scope="module")
def reset_state(default_env):
    state = jax.jit(default_env.reset)(jax.random.PRNGKey(123))
    jax.block_until_ready(state.reward)
    return state


def test_model_and_robot_specific_shapes_are_preserved(default_env):
    assert (default_env.nq, default_env.nv, default_env.nu) == (19, 18, 12)
    assert default_env.default_pose.shape == (12,)
    assert default_env.qpos_noise_scale.shape == (12,)
    assert default_env._weights.shape == (12,)
    assert tuple(default_env.feet_pos_sensor_names) == ("l_foot", "r_foot")
    assert default_env.sys.mj_model.body("foot_left").id >= 0
    assert default_env.sys.mj_model.body("foot_right").id >= 0


def test_seeded_reset_and_zero_action_step_are_finite_and_shaped(default_env, reset_state):
    repeated = jax.jit(default_env.reset)(jax.random.PRNGKey(123))
    np.testing.assert_allclose(reset_state.pipeline_state.q, repeated.pipeline_state.q)
    np.testing.assert_allclose(reset_state.pipeline_state.qd, repeated.pipeline_state.qd)
    np.testing.assert_allclose(reset_state.info["command"], repeated.info["command"])

    assert reset_state.obs["state"].shape == (default_env.obs_size,) == (52,)
    assert reset_state.obs["privileged_state"].shape == (
        default_env.privileged_obs_size,
    ) == (112,)

    stepped = jax.jit(default_env.step)(reset_state, jp.zeros(default_env.nu))
    jax.block_until_ready(stepped.reward)
    for value in (
        stepped.pipeline_state.q,
        stepped.pipeline_state.qd,
        stepped.obs["state"],
        stepped.obs["privileged_state"],
        stepped.reward,
        stepped.done,
    ):
        assert np.isfinite(np.asarray(value)).all()


def test_sampled_commands_are_three_dimensional_and_bounded(default_env):
    keys = jax.random.split(jax.random.PRNGKey(9), 512)
    commands = jax.vmap(default_env.sample_command)(keys)
    assert commands.shape == (512, 3)
    lower = np.asarray([default_env.lin_vel_x[0], default_env.lin_vel_y[0], default_env.ang_vel_yaw[0]])
    upper = np.asarray([default_env.lin_vel_x[1], default_env.lin_vel_y[1], default_env.ang_vel_yaw[1]])
    assert np.all(np.asarray(commands) >= lower)
    assert np.all(np.asarray(commands) <= upper)


def test_existing_tracking_rewards_improve_as_error_decreases():
    class RewardEnv:
        tracking_sigma = 0.5

        def __init__(self, linvel, gyro):
            self.linvel = jp.asarray(linvel)
            self.gyro = jp.asarray(gyro)

        def get_local_linvel(self, _state):
            return self.linvel

        def get_gyro(self, _state):
            return self.gyro

    info = {"command": jp.array([0.6, -0.2, 0.4])}
    matched = RewardEnv([0.6, -0.2, 0.0], [0.0, 0.0, 0.4])
    wrong = RewardEnv([-0.6, 0.5, 0.0], [0.0, 0.0, -0.5])
    dummy_state = SimpleNamespace()
    action = jp.zeros(12)

    assert locomotion_rewards._lin_vel(matched, dummy_state, info, action) > locomotion_rewards._lin_vel(
        wrong, dummy_state, info, action
    )
    assert locomotion_rewards._ang_vel(matched, dummy_state, info, action) > locomotion_rewards._ang_vel(
        wrong, dummy_state, info, action
    )


def test_action_rate_and_orientation_costs_increase_with_error():
    action = jp.ones(12)
    info = {"last_act": jp.zeros(12)}
    assert locomotion_rewards._action_rate(None, None, info, action) > locomotion_rewards._action_rate(
        None, None, info, jp.zeros(12)
    )

    class GravityEnv:
        def __init__(self, gravity):
            self.gravity = jp.asarray(gravity)

        def get_gravity(self, _state):
            return self.gravity

    upright = GravityEnv([0.0, 0.0, -1.0])
    tilted = GravityEnv([0.5, -0.25, -0.8])
    assert locomotion_rewards._orientation(tilted, None, None, None) > locomotion_rewards._orientation(
        upright, None, None, None
    )


def test_scheduled_push_changes_planar_velocity(default_env, reset_state):
    default_env.add_push = True
    default_env.push_magnitude_range = (1.0, 1.0)
    state = reset_state.replace(
        info={**reset_state.info, "push_step": jp.array(0), "push_interval_steps": jp.array(1)}
    )
    before = state.pipeline_state.qd.copy()
    pushed, push = default_env._apply_scheduled_push(state)
    default_env.add_push = False

    assert np.linalg.norm(np.asarray(push)) > 0.99
    np.testing.assert_allclose(
        np.asarray(pushed.pipeline_state.qd[:2] - before[:2]),
        np.asarray(push),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(pushed.pipeline_state.qd[2:], before[2:])


def test_manual_smoke_reads_pipeline_state_fields() -> None:
    smoke_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "locomotion"
        / "test_joystick.py"
    )
    source = smoke_path.read_text(encoding="utf-8")

    assert "print(state.pipeline_state.q)" in source
    assert "print(state.pipeline_state.qd)" in source
    assert "print(state.q)" not in source
    assert "print(state.qd)" not in source
