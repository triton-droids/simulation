"""Gate 3 deterministic smoke tests for the Unitree G1 MJX environment."""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf
import pytest

from source.config.g1 import G1MJXConfig
from source.locomotion.unitree_g1.joystick import Joystick
from source.robots.unitree_g1 import UnitreeG1Model


@pytest.fixture(scope="module")
def g1_env():
    cfg = OmegaConf.structured(G1MJXConfig())
    cfg.reset.randomize = False
    cfg.noise.add_noise = False
    cfg.push.add_push = False
    cfg.domain_rand.add_domain_rand = False
    return Joystick("unitree_g1", UnitreeG1Model(fetch=False), "flat", cfg)


@pytest.fixture(scope="module")
def g1_state(g1_env):
    state = jax.jit(g1_env.reset)(jax.random.PRNGKey(17))
    jax.block_until_ready(state.reward)
    return state


def test_seeded_reset_is_reproducible_finite_and_named(g1_env, g1_state):
    repeated = jax.jit(g1_env.reset)(jax.random.PRNGKey(17))
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


def test_randomized_reset_is_seeded_and_bounded(g1_env):
    previous = bool(g1_env.cfg.reset.randomize)
    g1_env.cfg.reset.randomize = True
    try:
        reset = jax.jit(g1_env.reset)
        first = reset(jax.random.PRNGKey(1700))
        repeated = reset(jax.random.PRNGKey(1700))
        different = reset(jax.random.PRNGKey(1701))
        jax.block_until_ready(different.reward)
    finally:
        g1_env.cfg.reset.randomize = previous

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
    assert tuple(g1_env.metadata.foot_link_ids) == (6, 12)
    assert tuple(len(side) for side in g1_env.metadata.foot_geom_ids) == (3, 3)


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
    np.testing.assert_array_equal(contact, np.array([True, False]))
    assert not bool(terminal)
    assert not bool(collision)

    hand_thigh = _pipeline_with_only_contact_pair(
        g1_state.pipeline_state, metadata.hand_geom_ids[0], metadata.thigh_geom_ids[0]
    )
    _, terminal, collision = g1_env.contact_state(hand_thigh)
    assert not bool(terminal)
    assert bool(collision)

    shin_ground = _pipeline_with_only_contact_pair(
        g1_state.pipeline_state, metadata.floor_geom_id, metadata.shin_geom_ids[0][0]
    )
    _, terminal, collision = g1_env.contact_state(shin_ground)
    assert not bool(terminal)
    assert bool(collision)

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


def test_zero_action_step_is_finite(g1_env, g1_state):
    stepped = jax.jit(g1_env.step)(g1_state, jp.zeros(g1_env.nu))
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Gate 4 audit: step constructs the returned observation before advancing "
        "last_act; fix only as part of a new training/evaluation family"
    ),
)
def test_step_observation_contains_just_applied_previous_action(g1_env):
    state = jax.jit(g1_env.reset)(jax.random.PRNGKey(1701))
    action = jp.linspace(-0.2, 0.2, g1_env.nu)
    clipped, _ = g1_env.action_to_targets(action)
    stepped = jax.jit(g1_env.step)(state, action)
    jax.block_until_ready(stepped.reward)
    previous_action_start = 3 + 3 + 3 + 3 + g1_env.nu + g1_env.nu
    observed_previous_action = stepped.obs["state"][
        previous_action_start : previous_action_start + g1_env.nu
    ]
    np.testing.assert_allclose(observed_previous_action, clipped, atol=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Gate 4 audit: step constructs the returned observation before command "
        "resampling; fix only as part of a new training/evaluation family"
    ),
)
def test_resampled_command_matches_returned_observation(g1_env):
    state = jax.jit(g1_env.reset)(jax.random.PRNGKey(1702))
    state = state.replace(
        info={**state.info, "command": jp.array([9.0, 9.0, 9.0])}
    )
    previous_resample_steps = g1_env.resample_steps
    g1_env.resample_steps = 1
    try:
        stepped = jax.jit(g1_env.step)(state, jp.zeros(g1_env.nu))
        jax.block_until_ready(stepped.reward)
    finally:
        g1_env.resample_steps = previous_resample_steps
    np.testing.assert_allclose(
        stepped.obs["state"][9:12], stepped.info["command"], atol=1e-6
    )


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


def test_termination_catches_height_contact_and_nan(g1_env, g1_state):
    state = g1_state.pipeline_state
    low = state.replace(q=state.q.at[2].set(g1_env.cfg.termination.min_pelvis_height - 0.01))
    high = state.replace(q=state.q.at[2].set(g1_env.cfg.termination.max_pelvis_height + 0.01))
    invalid = state.replace(q=state.q.at[7].set(jp.nan))
    assert bool(g1_env.get_termination(low, jp.array(False)))
    assert bool(g1_env.get_termination(high, jp.array(False)))
    assert bool(g1_env.get_termination(state, jp.array(True)))
    assert bool(g1_env.get_termination(invalid, jp.array(False)))
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
