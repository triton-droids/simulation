"""Gate 3 deterministic smoke tests for the Unitree G1 MJX environment."""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from omegaconf import OmegaConf
import pytest

from source.config.g1 import G1MJXConfig
from source.locomotion.unitree_g1 import rewards
from source.locomotion.unitree_g1.joystick import Joystick, rigid_body_point_velocity
from source.robots.unitree_g1 import UnitreeG1Model


def _make_g1_env(*, randomize: bool = False, resample_time: float = 10.0):
    cfg = OmegaConf.structured(G1MJXConfig())
    cfg.reset.randomize = randomize
    cfg.commands.resample_time = resample_time
    cfg.noise.add_noise = False
    cfg.push.add_push = False
    cfg.domain_rand.add_domain_rand = False
    return Joystick("unitree_g1", UnitreeG1Model(fetch=False), "flat", cfg)


def test_g1_rejects_unimplemented_domain_randomization():
    cfg = OmegaConf.structured(G1MJXConfig())
    cfg.domain_rand.add_domain_rand = True

    with pytest.raises(NotImplementedError, match="tested G1 torso/contact mapping"):
        Joystick("unitree_g1", UnitreeG1Model(fetch=False), "flat", cfg)


@pytest.fixture(scope="module")
def g1_env():
    return _make_g1_env()


@pytest.fixture(scope="module")
def g1_randomized_env():
    return _make_g1_env(randomize=True)


@pytest.fixture(scope="module")
def g1_resample_env():
    return _make_g1_env(resample_time=0.02)


@pytest.fixture(scope="module")
def g1_resample_reset(g1_resample_env):
    return jax.jit(g1_resample_env.reset)


@pytest.fixture(scope="module")
def g1_resample_step(g1_resample_env):
    return jax.jit(g1_resample_env.step)


@pytest.fixture(scope="module")
def g1_reset(g1_env):
    return jax.jit(g1_env.reset)


@pytest.fixture(scope="module")
def g1_step(g1_env):
    return jax.jit(g1_env.step)


@pytest.fixture(scope="module")
def g1_state(g1_reset):
    state = g1_reset(jax.random.PRNGKey(17))
    jax.block_until_ready(state.reward)
    return state


def test_seeded_reset_is_reproducible_finite_and_named(g1_env, g1_state, g1_reset):
    repeated = g1_reset(jax.random.PRNGKey(17))
    np.testing.assert_allclose(g1_state.pipeline_state.q, repeated.pipeline_state.q)
    np.testing.assert_allclose(g1_state.pipeline_state.qd, repeated.pipeline_state.qd)
    np.testing.assert_allclose(g1_state.info["command"], repeated.info["command"])
    assert np.isfinite(np.asarray(g1_state.pipeline_state.q)).all()
    assert np.isfinite(np.asarray(g1_state.pipeline_state.qd)).all()
    assert g1_state.obs["state"].shape == (g1_env.obs_size,) == (103,)
    assert g1_state.obs["privileged_state"].shape == (
        g1_env.privileged_obs_size,
    ) == (216,)
    assert g1_env.nu == g1_env.action_size == 29


def test_reset_initializes_pd_targets_to_the_actual_joint_pose(g1_env, g1_state):
    np.testing.assert_allclose(
        g1_state.pipeline_state.ctrl, g1_state.pipeline_state.qpos[7:], atol=1e-7
    )
    np.testing.assert_allclose(
        g1_state.info["motor_targets"], g1_state.pipeline_state.qpos[7:], atol=1e-7
    )
    assert np.max(np.abs(np.asarray(g1_state.pipeline_state.actuator_force))) < 1e-4


def test_randomized_reset_is_seeded_and_bounded(g1_randomized_env):
    g1_env = g1_randomized_env
    reset = jax.jit(g1_env.reset)
    first = reset(jax.random.PRNGKey(1700))
    repeated = reset(jax.random.PRNGKey(1700))
    different = reset(jax.random.PRNGKey(1701))
    jax.block_until_ready(different.reward)

    np.testing.assert_allclose(first.pipeline_state.q, repeated.pipeline_state.q)
    np.testing.assert_allclose(first.pipeline_state.qd, repeated.pipeline_state.qd)
    assert not np.allclose(first.pipeline_state.q, different.pipeline_state.q)
    xy_delta = np.asarray(first.pipeline_state.q[:2] - g1_env.init_q[:2])
    assert np.all(np.abs(xy_delta) <= g1_env.cfg.reset.xy_range + 1e-6)
    assert np.all(
        np.abs(np.asarray(first.pipeline_state.qd[:6]))
        <= g1_env.cfg.reset.base_velocity_range + 1e-6
    )
    assert np.all(np.asarray(first.pipeline_state.q[7:]) >= np.asarray(g1_env.ctrl_lower))
    assert np.all(np.asarray(first.pipeline_state.q[7:]) <= np.asarray(g1_env.ctrl_upper))


def test_standing_contacts_match_verified_foot_geoms(g1_env, g1_state):
    contact, undesired, collision = g1_env.contact_state(g1_state.pipeline_state)
    np.testing.assert_array_equal(contact, np.array([True, True]))
    assert not bool(undesired)
    assert not bool(collision)


def test_foot_site_velocity_includes_rigid_body_angular_motion(g1_env):
    body_position = jp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    body_velocity = jp.array([[0.2, -0.1, 0.3], [0.0, 0.4, -0.2]])
    angular_velocity = jp.array([[0.0, 0.0, 2.0], [1.0, 0.0, 0.0]])
    site_position = jp.array([[1.5, 2.0, 3.0], [-1.0, 1.5, 0.0]])

    actual = rigid_body_point_velocity(
        body_position, body_velocity, angular_velocity, site_position
    )

    np.testing.assert_allclose(
        actual,
        np.array([[0.2, 0.9, 0.3], [0.0, 0.4, 0.8]]),
        atol=1e-7,
    )
    assert tuple(g1_env.metadata.foot_link_ids) == (6, 12)
    assert tuple(len(side) for side in g1_env.metadata.foot_geom_ids) == (3, 3)


def test_velocity_frames_and_positive_yaw_sign(g1_env):
    model = g1_env.sys.mj_model
    data = mujoco.MjData(model)

    def sensor(name: str) -> np.ndarray:
        sensor_id = model.sensor(name).id
        start = model.sensor_adr[sensor_id]
        return data.sensordata[start : start + model.sensor_dim[sensor_id]].copy()

    qpos = np.asarray(g1_env.init_q).copy()
    qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    qvel = np.zeros(g1_env.nv)
    qvel[5] = 0.4
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = qpos[7:]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(sensor("gyro_pelvis"), [0.0, 0.0, 0.4], atol=1e-6)

    yaw = np.pi / 2.0
    qpos[3:7] = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    qvel[:] = 0.0
    qvel[1] = 0.4  # World +Y is pelvis-local +X at +90 degrees yaw.
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = qpos[7:]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        sensor("global_linvel_pelvis"),
        [0.0, 0.4, 0.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sensor("local_linvel_pelvis"),
        [0.4, 0.0, 0.0],
        atol=1e-6,
    )


def _pipeline_with_only_contact_pair(pipeline_state, first: int, second: int):
    contact = pipeline_state.contact
    contact = contact.replace(
        geom=contact.geom.at[0].set(jp.array([first, second])),
        dist=jp.ones_like(contact.dist).at[0].set(-0.01),
    )
    return pipeline_state.replace(contact=contact)


def test_contact_classes_match_playground_semantics_and_ground_safety(
    g1_env, g1_state
):
    metadata = g1_env.metadata

    foot_floor = _pipeline_with_only_contact_pair(
        g1_state.pipeline_state, metadata.floor_geom_id, metadata.foot_geom_ids[0][0]
    )
    contact, terminal, collision = g1_env.contact_state(foot_floor)
    _, _, self_collision, nonfoot_ground = g1_env.contact_details(foot_floor)
    np.testing.assert_array_equal(contact, np.array([True, False]))
    assert not bool(terminal)
    assert not bool(collision)
    assert not bool(self_collision)
    assert not bool(nonfoot_ground)

    hand_thigh = _pipeline_with_only_contact_pair(
        g1_state.pipeline_state, metadata.hand_geom_ids[0], metadata.thigh_geom_ids[0]
    )
    _, terminal, collision = g1_env.contact_state(hand_thigh)
    _, _, self_collision, nonfoot_ground = g1_env.contact_details(hand_thigh)
    assert not bool(terminal)
    assert bool(collision)
    assert bool(self_collision)
    assert not bool(nonfoot_ground)

    shin_ground = _pipeline_with_only_contact_pair(
        g1_state.pipeline_state, metadata.floor_geom_id, metadata.shin_geom_ids[0][0]
    )
    _, terminal, collision = g1_env.contact_state(shin_ground)
    _, _, self_collision, nonfoot_ground = g1_env.contact_details(shin_ground)
    assert not bool(terminal)
    assert bool(collision)
    assert not bool(self_collision)
    assert bool(nonfoot_ground)

    cross_leg = _pipeline_with_only_contact_pair(
        g1_state.pipeline_state,
        metadata.cross_contact_foot_geom_ids[0],
        metadata.shin_geom_ids[1][0],
    )
    _, terminal, _ = g1_env.contact_state(cross_leg)
    assert bool(terminal)


def test_actions_are_clipped_and_targets_respect_actuator_ranges(g1_env):
    action = jp.linspace(-4.0, 4.0, g1_env.nu)
    clipped, targets = g1_env.action_to_targets(action)
    assert np.all(np.asarray(clipped) >= -1.0)
    assert np.all(np.asarray(clipped) <= 1.0)
    assert np.all(np.asarray(targets) >= np.asarray(g1_env.ctrl_lower))
    assert np.all(np.asarray(targets) <= np.asarray(g1_env.ctrl_upper))
    # The pinned ankle-roll range is narrower than +/- action_scale and must clip.
    _, upper_targets = g1_env.action_to_targets(jp.ones(g1_env.nu))
    assert np.isclose(
        float(upper_targets[5]), float(g1_env.ctrl_upper[5]), atol=1e-7
    )


def test_zero_action_step_is_finite(g1_env, g1_state, g1_step):
    stepped = g1_step(g1_state, jp.zeros(g1_env.nu))
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
    assert stepped.info["motor_targets"].shape == (29,)
    reward_metrics = {
        name: value
        for name, value in stepped.metrics.items()
        if name.startswith("reward/")
    }
    assert set(reward_metrics) == {
        f"reward/{name}" for name in g1_env.reward_scales
    }
    assert all(np.isfinite(float(value)) for value in reward_metrics.values())
    assert np.isclose(
        float(stepped.reward),
        sum(float(value) for value in reward_metrics.values()) * g1_env.dt,
        atol=1e-6,
    )


def test_step_observation_contains_just_applied_previous_action(
    g1_env, g1_reset, g1_step
):
    state = g1_reset(jax.random.PRNGKey(1701))
    action = jp.linspace(-0.2, 0.2, g1_env.nu)
    clipped, _ = g1_env.action_to_targets(action)
    stepped = g1_step(state, action)
    jax.block_until_ready(stepped.reward)
    previous_action_start = 3 + 3 + 3 + 3 + g1_env.nu + g1_env.nu
    observed_previous_action = stepped.obs["state"][
        previous_action_start : previous_action_start + g1_env.nu
    ]
    np.testing.assert_allclose(observed_previous_action, clipped, atol=1e-6)


def test_resampled_command_matches_returned_observation(
    g1_resample_env, g1_resample_reset, g1_resample_step
):
    g1_env = g1_resample_env
    state = g1_resample_reset(jax.random.PRNGKey(1702))
    state = state.replace(
        info={**state.info, "command": jp.array([9.0, 9.0, 9.0])}
    )
    stepped = g1_resample_step(state, jp.zeros(g1_env.nu))
    jax.block_until_ready(stepped.reward)
    np.testing.assert_allclose(
        stepped.obs["state"][9:12], stepped.info["command"], atol=1e-6
    )


def test_transition_reward_uses_command_that_governed_action(
    g1_resample_env, g1_resample_reset, g1_resample_step
):
    g1_env = g1_resample_env
    state = g1_resample_reset(jax.random.PRNGKey(1703))
    transition_command = jp.array([-0.5, -0.3, -0.5])
    state = state.replace(
        info={**state.info, "command": transition_command}
    )
    stepped = g1_resample_step(state, jp.zeros(g1_env.nu))
    jax.block_until_ready(stepped.reward)

    local_velocity = g1_env.get_local_linvel(stepped.pipeline_state, "pelvis")
    expected = (
        rewards.tracking_linear_velocity(
            transition_command, local_velocity, g1_env.cfg.rewards.tracking_sigma
        )
        * g1_env.reward_scales["tracking_lin_vel"]
    )
    resampled = (
        rewards.tracking_linear_velocity(
            stepped.info["command"],
            local_velocity,
            g1_env.cfg.rewards.tracking_sigma,
        )
        * g1_env.reward_scales["tracking_lin_vel"]
    )
    actual = stepped.metrics["reward/tracking_lin_vel"]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert not np.isclose(float(actual), float(resampled), atol=1e-4)


def test_returned_observation_contains_advanced_phase(g1_env, g1_reset, g1_step):
    state = g1_reset(jax.random.PRNGKey(1704))
    initial_phase = jp.array([-0.7, 1.2])
    state = state.replace(info={**state.info, "phase": initial_phase})
    stepped = g1_step(state, jp.zeros(g1_env.nu))
    jax.block_until_ready(stepped.reward)

    expected_phase = jp.fmod(
        initial_phase + state.info["phase_dt"] + jp.pi, 2.0 * jp.pi
    ) - jp.pi
    np.testing.assert_allclose(stepped.info["phase"], expected_phase, atol=1e-6)
    np.testing.assert_allclose(
        stepped.obs["state"][99:103],
        jp.concatenate([jp.cos(expected_phase), jp.sin(expected_phase)]),
        atol=1e-6,
    )


def test_touchdown_reward_uses_air_time_before_contact_reset(
    g1_env, g1_reset, g1_step
):
    state = g1_reset(jax.random.PRNGKey(1705))
    state = state.replace(
        info={
            **state.info,
            "command": jp.array([0.4, 0.0, 0.0]),
            "feet_air_time": jp.array([0.4, 0.0]),
            "last_contact": jp.array([False, True]),
        }
    )
    stepped = g1_step(state, jp.zeros(g1_env.nu))
    jax.block_until_ready(stepped.reward)

    assert bool(stepped.info["first_contact"][0])
    assert float(stepped.metrics["reward/feet_air_time"]) > 0.0
    assert float(stepped.info["feet_air_time"][0]) == 0.0


def test_step_does_not_mutate_input_state_history(g1_env, g1_reset, g1_step):
    state = g1_reset(jax.random.PRNGKey(1706))
    original_command = np.asarray(state.info["command"]).copy()
    original_last_act = np.asarray(state.info["last_act"]).copy()
    original_step = int(state.info["step"])

    stepped = g1_step(state, jp.full(g1_env.nu, 0.1))
    jax.block_until_ready(stepped.reward)

    np.testing.assert_allclose(state.info["command"], original_command)
    np.testing.assert_allclose(state.info["last_act"], original_last_act)
    assert int(state.info["step"]) == original_step
    assert int(stepped.info["step"]) == original_step + 1


def test_commands_are_bounded(g1_env):
    keys = jax.random.split(jax.random.PRNGKey(23), 1024)
    commands = np.asarray(jax.vmap(g1_env.sample_command)(keys))
    lower = np.array(
        [
            g1_env.cfg.commands.lin_vel_x[0],
            g1_env.cfg.commands.lin_vel_y[0],
            g1_env.cfg.commands.ang_vel_yaw[0],
        ]
    )
    upper = np.array(
        [
            g1_env.cfg.commands.lin_vel_x[1],
            g1_env.cfg.commands.lin_vel_y[1],
            g1_env.cfg.commands.ang_vel_yaw[1],
        ]
    )
    assert commands.shape == (1024, 3)
    assert np.all(commands >= lower)
    assert np.all(commands <= upper)
    assert np.any(np.all(commands == 0.0, axis=1))


def test_termination_catches_height_contact_and_nonfinite_state(g1_env, g1_state):
    state = g1_state.pipeline_state
    low = state.replace(q=state.q.at[2].set(g1_env.cfg.termination.min_pelvis_height - 0.01))
    high = state.replace(q=state.q.at[2].set(g1_env.cfg.termination.max_pelvis_height + 0.01))
    nan_q = state.replace(q=state.q.at[7].set(jp.nan))
    inf_q = state.replace(q=state.q.at[7].set(jp.inf))
    inf_qd = state.replace(qd=state.qd.at[6].set(-jp.inf))
    assert bool(g1_env.get_termination(low, jp.array(False)))
    assert bool(g1_env.get_termination(high, jp.array(False)))
    assert bool(g1_env.get_termination(state, jp.array(True)))
    assert bool(g1_env.get_termination(nan_q, jp.array(False)))
    assert bool(g1_env.get_termination(inf_q, jp.array(False)))
    assert bool(g1_env.get_termination(inf_qd, jp.array(False)))
    assert not bool(g1_env.get_termination(state, jp.array(False)))


def test_scheduled_push_changes_only_planar_base_velocity(g1_env, g1_state):
    g1_env.add_push = True
    g1_env.push_magnitude_range = (1.0, 1.0)
    state = g1_state.replace(
        info={
            **g1_state.info,
            "push_step": jp.array(0),
            "push_interval_steps": jp.array(1),
        }
    )
    before = state.pipeline_state.qd.copy()
    pushed, push = g1_env._apply_scheduled_push(state)
    g1_env.add_push = False
    np.testing.assert_allclose(pushed.pipeline_state.qd[:2] - before[:2], push, atol=1e-6)
    np.testing.assert_allclose(pushed.pipeline_state.qd[2:], before[2:], atol=1e-6)
    assert np.linalg.norm(np.asarray(push)) > 0.99


def test_one_thousand_bounded_steps_have_no_nan_or_shape_error(g1_env, g1_state):
    def rollout(initial_state):
        def one_step(state, index):
            action = 0.02 * jp.sin(0.035 * index + jp.arange(g1_env.nu) * 0.17)
            state = g1_env.step(state, action)
            finite = (
                jp.isfinite(state.pipeline_state.q).all()
                & jp.isfinite(state.pipeline_state.qd).all()
                & jp.isfinite(state.obs["state"]).all()
                & jp.isfinite(state.obs["privileged_state"]).all()
            )
            return state, finite

        return jax.lax.scan(one_step, initial_state, jp.arange(1000))

    final_state, finite = jax.jit(rollout)(g1_state)
    jax.block_until_ready(finite)
    assert bool(jp.all(finite))
    assert final_state.obs["state"].shape == (103,)
    assert final_state.obs["privileged_state"].shape == (216,)
