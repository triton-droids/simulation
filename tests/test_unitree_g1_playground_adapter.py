"""Tests for the pinned authoritative Playground G1 adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
import functools
import json

from hydra import compose, initialize
import jax
import jax.numpy as jp
import pytest
from omegaconf import OmegaConf

import source.config  # noqa: F401 - register structured Hydra configs.
from source.config.g1_playground import G1PlaygroundMJXConfig
from source.locomotion import get_env_class
from source.locomotion.unitree_g1.playground_joystick import (
    Joystick,
    contact_phase_reward,
    synchronize_transition_observation,
)
from source.locomotion.unitree_g1.playground_source import (
    PLAYGROUND_COMMIT,
    REQUIRED_PATHS,
    load_playground_g1_modules,
    resolve_playground_source,
)
from source.robots.unitree_g1 import UnitreeG1Model
from source.locomotion.unitree_g1.training_wrapper import wrap_for_brax_training


def test_playground_config_matches_pinned_g1_defaults_and_round_trips() -> None:
    cfg = G1PlaygroundMJXConfig()

    assert cfg.commands.lin_vel_x == (-1.0, 1.0)
    assert cfg.commands.lin_vel_y == (-0.5, 0.5)
    assert cfg.commands.ang_vel_yaw == (-1.0, 1.0)
    assert cfg.reward_scales.tracking_lin_vel == 1.0
    assert cfg.reward_scales.tracking_ang_vel == 0.75
    assert cfg.reward_scales.alive == 0.0
    assert cfg.reward_scales.contact_force == -0.01
    assert cfg.reward_scales.feet_contact_phase == 0.0
    assert cfg.push.add_push is True
    assert cfg.playground.implementation == "jax"

    structured = OmegaConf.structured(cfg)
    restored = OmegaConf.create(json.loads(json.dumps(OmegaConf.to_container(structured))))
    assert OmegaConf.to_container(restored) == OmegaConf.to_container(structured)


def test_hydra_registers_playground_adapter_separately_from_native() -> None:
    with initialize(version_base=None, config_path=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "env=unitree_g1_playground",
                "robot=unitree_g1",
                "sim=unitree_g1_playground",
            ],
        )

    assert cfg.env.name == "unitree_g1_playground"
    assert cfg.robot.name == "unitree_g1"
    assert get_env_class(cfg.env.name) is Joystick


def test_playground_source_resolves_exact_pin_without_duplicate_assets() -> None:
    source = resolve_playground_source(fetch=False)

    assert source.pinned is True
    assert source.revision == PLAYGROUND_COMMIT
    assert all((source.root / path).is_file() for path in REQUIRED_PATHS)
    assert not (source.root / "mujoco_menagerie" / "unitree_g1").exists()


def test_transition_observation_uses_returned_command_action_phase_and_air_time() -> None:
    observation = {
        "state": jp.zeros(103),
        "privileged_state": jp.zeros(216),
    }
    info = {
        "command": jp.array([0.2, -0.3, 0.4]),
        "last_act": jp.linspace(-1.0, 1.0, 29),
        "phase": jp.array([0.25, -0.75]),
        "feet_air_time": jp.array([0.0, 0.42]),
    }

    synced = synchronize_transition_observation(observation, info)

    for value in synced.values():
        assert jp.allclose(value[9:12], info["command"])
        assert jp.allclose(value[70:99], info["last_act"])
        assert jp.allclose(
            value[99:103],
            jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])]),
        )
    assert jp.allclose(synced["privileged_state"][-2:], info["feet_air_time"])


@dataclass(frozen=True)
class _FakeData:
    qpos: jp.ndarray
    qvel: jp.ndarray


@dataclass(frozen=True)
class _FakeState:
    data: _FakeData
    obs: dict[str, jp.ndarray]
    reward: jp.ndarray
    done: jp.ndarray
    metrics: dict[str, jp.ndarray]
    info: dict[str, jp.ndarray]

    def replace(self, **updates):
        return replace(self, **updates)


class _MutatingUpstream:
    def step(self, state, action):
        state.info["command"] = jp.array([0.4, 0.1, -0.2])
        state.info["last_act"] = action
        state.info["phase"] = jp.array([0.4, -0.4])
        state.info["feet_air_time"] = jp.array([0.0, 0.3])
        return state.replace(obs={"state": jp.zeros(103), "privileged_state": jp.zeros(216)})


def test_contact_phase_rewards_alternation_not_stance_flight_or_wrong_phase() -> None:
    phase = jp.array([0.0, jp.pi])
    command = jp.array([0.5, 0.0, 0.0])
    reward = jax.jit(contact_phase_reward)
    assert float(reward(jp.array([False, True]), phase, command)) == pytest.approx(1.0)
    assert float(reward(jp.array([True, False]), phase, command)) == pytest.approx(-1.0)
    assert float(reward(jp.array([True, True]), phase, command)) == pytest.approx(0.0)
    assert float(reward(jp.array([False, False]), phase, command)) == pytest.approx(0.0)
    assert float(reward(jp.array([False, True]), phase, jp.zeros(3))) == pytest.approx(0.0)
    assert float(reward(jp.array([True, False]), phase + jp.pi, command)) == pytest.approx(1.0)
    phases = jp.stack([jp.linspace(-jp.pi, jp.pi, 101), jp.linspace(0, 2*jp.pi, 101)], axis=1)
    both = jax.vmap(lambda p: reward(jp.array([True, True]), p, command))(phases)
    assert jp.allclose(both, 0, atol=2e-7)


def test_contact_phase_uses_old_phase_command_and_control_dt() -> None:
    adapter = Joystick.__new__(Joystick)
    adapter._env = _MutatingUpstream()
    adapter._feet_contact_phase_scale = 2.0
    adapter.dt = .02
    adapter._contact = lambda _data: jp.array([False, True])
    state = _FakeState(
        data=_FakeData(jp.zeros(36), jp.zeros(35)),
        obs={"state": jp.zeros(103), "privileged_state": jp.zeros(216)},
        reward=jp.asarray(.5), done=jp.zeros(()),
        metrics={"reward/feet_contact_phase": jp.zeros(())},
        info={"command": jp.array([.5, 0, 0]), "last_act": jp.zeros(29),
              "phase": jp.array([0., jp.pi]), "feet_air_time": jp.zeros(2)},
    )
    next_state = adapter.step(state, jp.zeros(29))
    assert float(next_state.reward) == pytest.approx(.54)
    assert float(next_state.metrics["reward/feet_contact_phase"]) == pytest.approx(2.)
    assert float(state.reward) == pytest.approx(.5)
    stopped = state.replace(info={**state.info, "command": jp.zeros(3)})
    assert float(adapter.step(stopped, jp.zeros(29)).reward) == pytest.approx(.5)


def test_adapter_step_is_functional_clips_action_and_repairs_upstream_staleness() -> None:
    adapter = Joystick.__new__(Joystick)
    adapter._env = _MutatingUpstream()
    original_info = {
        "command": jp.zeros(3),
        "last_act": jp.zeros(29),
        "phase": jp.array([0.0, jp.pi]),
        "feet_air_time": jp.zeros(2),
    }
    state = _FakeState(
        data=_FakeData(jp.zeros(36), jp.zeros(35)),
        obs={"state": jp.zeros(103), "privileged_state": jp.zeros(216)},
        reward=jp.zeros(()),
        done=jp.zeros(()),
        metrics={"kept": jp.ones(())},
        info=original_info,
    )

    next_state = adapter.step(state, jp.full(29, 2.0))

    assert jp.all(original_info["last_act"] == 0.0)
    assert jp.all(next_state.info["last_act"] == 1.0)
    assert jp.all(next_state.obs["state"][70:99] == 1.0)
    assert jp.allclose(next_state.obs["state"][9:12], next_state.info["command"])
    assert next_state.metrics is not state.metrics


def test_adapter_constructs_official_feet_only_model_from_existing_assets() -> None:
    cfg = OmegaConf.structured(G1PlaygroundMJXConfig())
    cfg.playground.fetch_source = False
    cfg.reset.randomize = False
    cfg.noise.add_noise = False
    robot = UnitreeG1Model(fetch=False)

    env = Joystick("unitree_g1", robot, "flat", cfg)

    assert (env.nq, env.nv, env.nu) == (36, 35, 29)
    assert (env.mj_model.ngeom, env.mj_model.npair, env.mj_model.nsensor) == (72, 5, 29)
    assert env.source_record["revision"] == PLAYGROUND_COMMIT
    assert env.source_record["model"]["revision"] == robot.resolution.revision
    assert env.mj_model.opt.integrator == 0  # mujoco.mjtIntegrator.mjINT_EULER
    assert isinstance(env.brax_training_wrapper, functools.partial)
    assert env.brax_training_wrapper.keywords["full_reset"] is True


@pytest.mark.parametrize("physical_terminal", [True, False])
@pytest.mark.parametrize("episode_length", [1, 3])
def test_playground_training_auto_reset_restores_matching_history(physical_terminal, episode_length) -> None:
    source = resolve_playground_source(fetch=False)
    mjx_env_module, wrapper_module, _ = load_playground_g1_modules(source)

    class ForcedDoneEnvironment:
        action_size = 29

        @property
        def observation_size(self):
            return {"state": 103, "privileged_state": 216}

        @property
        def unwrapped(self):
            return self

        @staticmethod
        def _observation(info):
            return synchronize_transition_observation(
                {"state": jp.zeros(103), "privileged_state": jp.zeros(216)},
                info,
            )

        def reset(self, rng):
            token = jax.random.uniform(rng, ())
            info = {
                "command": jp.array([token, -token, token / 2]),
                "last_act": jp.zeros(29),
                "phase": jp.array([token, token + jp.pi]),
                "feet_air_time": jp.zeros(2),
            }
            return mjx_env_module.State(
                jp.array([token]),
                self._observation(info),
                jp.zeros(()),
                jp.zeros(()),
                {"progress": jp.zeros(())},
                info,
            )

        def step(self, state, action):
            del action
            info = dict(state.info)
            info.update(
                command=jp.full(3, 9.0),
                last_act=jp.ones(29),
                phase=jp.ones(2),
                feet_air_time=jp.ones(2),
            )
            return state.replace(
                data=jp.array([9.0]),
                obs=self._observation(info),
                done=jp.asarray(float(physical_terminal)),
                reward=jp.asarray(2.0),
                metrics={"progress": jp.asarray(3.0)},
                info=info,
            )

    wrapped = wrap_for_brax_training(
        ForcedDoneEnvironment(),
        episode_length=episode_length,
        action_repeat=1,
        full_reset=True,
        wrapper_module=wrapper_module,
    )
    reset_keys = jax.random.split(jax.random.PRNGKey(0), 2)
    state = jax.jit(wrapped.reset)(reset_keys)
    expected_length = 1 if physical_terminal else episode_length
    for elapsed in range(1, expected_length):
        state = jax.jit(wrapped.step)(state, jp.zeros((2, 29)))
        assert jp.all(state.done == 0)
        assert jp.all(state.info["episode_metrics"]["sum_reward"] == 2 * elapsed)
    next_state = jax.jit(wrapped.step)(state, jp.zeros((2, 29)))

    assert jp.all(next_state.done == 1)
    assert jp.all(next_state.data[:, 0] != 9.0)
    assert jp.allclose(next_state.obs["state"][:, 9:12], next_state.info["command"])
    assert jp.allclose(next_state.obs["state"][:, 70:99], next_state.info["last_act"])
    expected_phase = jp.concatenate(
        [jp.cos(next_state.info["phase"]), jp.sin(next_state.info["phase"])], axis=-1
    )
    assert jp.allclose(next_state.obs["state"][:, 99:103], expected_phase)
    assert jp.all(next_state.info["steps"] == expected_length)
    assert jp.all(next_state.info["truncation"] == float(not physical_terminal))
    assert jp.all(next_state.info["episode_done"] == 1)
    assert jp.all(next_state.info["episode_metrics"]["sum_reward"] == 2 * expected_length)
    assert jp.all(next_state.info["episode_metrics"]["length"] == expected_length)
    assert jp.all(next_state.info["episode_metrics"]["progress"] == 3 * expected_length)
    following = jax.jit(wrapped.step)(next_state, jp.zeros((2, 29)))
    assert jp.all(following.info["episode_metrics"]["sum_reward"] == 2)
    assert jp.all(following.info["episode_metrics"]["length"] == 1)


@pytest.mark.parametrize("contact_phase_scale", [0.0, 2.0])
def test_adapter_jitted_reset_step_and_resampling_invariants(contact_phase_scale) -> None:
    cfg = OmegaConf.structured(G1PlaygroundMJXConfig())
    cfg.playground.fetch_source = False
    cfg.reset.randomize = False
    cfg.noise.add_noise = False
    cfg.push.add_push = False
    cfg.reward_scales.feet_contact_phase = contact_phase_scale
    env = Joystick("unitree_g1", UnitreeG1Model(fetch=False), "flat", cfg)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    state = reset(jax.random.PRNGKey(1707))
    next_state = step(state, jp.full(env.action_size, 0.01))
    jax.block_until_ready(next_state.obs["state"])

    assert jp.isfinite(next_state.data.qpos).all()
    assert jp.allclose(state.info["last_act"], jp.zeros(env.action_size))
    assert jp.allclose(next_state.info["last_act"], 0.01)
    assert jp.allclose(next_state.obs["state"][70:99], next_state.info["last_act"])
    assert jp.allclose(next_state.obs["state"][9:12], next_state.info["command"])
    assert jp.allclose(state.info["motor_targets"], state.data.ctrl)
    assert ("reward/feet_contact_phase" in state.metrics) == (contact_phase_scale != 0)
    if contact_phase_scale:
        expected = contact_phase_scale * contact_phase_reward(
            env._contact(next_state.data), state.info["phase"], state.info["command"]
        )
        assert jp.allclose(next_state.metrics["reward/feet_contact_phase"], expected)
        assert jp.allclose(next_state.reward, env.dt * sum(
            v for k, v in next_state.metrics.items() if k.startswith("reward/")
        ), atol=1e-6)

    boundary_info = dict(state.info)
    boundary_info["step"] = jp.asarray(500, dtype=jp.int32)
    boundary_state = state.replace(info=boundary_info)
    resampled = step(boundary_state, jp.zeros(env.action_size))
    jax.block_until_ready(resampled.obs["state"])

    assert int(resampled.info["step"]) == 0
    assert jp.allclose(resampled.obs["state"][9:12], resampled.info["command"])


def test_command_only_phase_reward_removes_exact_term_only_when_stopped():
    class PhaseUpstream(_MutatingUpstream):
        def step(self, state, action):
            result = super().step(state, action)
            return result.replace(reward=jp.asarray(.5), metrics={"reward/feet_phase": jp.asarray(2.)})
    adapter = Joystick.__new__(Joystick)
    adapter._env = PhaseUpstream()
    adapter._phase_reward_command_only = True
    adapter.dt = .02
    for command, expected in [(jp.zeros(3), .46), (jp.array([.45,0,0]), .5)]:
        state = _FakeState(data=_FakeData(jp.zeros(36), jp.zeros(35)),
            obs={"state": jp.zeros(103), "privileged_state": jp.zeros(216)},
            reward=jp.zeros(()), done=jp.zeros(()), metrics={},
            info={"command": command, "last_act": jp.zeros(29), "phase": jp.zeros(2), "feet_air_time": jp.zeros(2)})
        result = adapter.step(state, jp.zeros(29))
        assert float(result.reward) == pytest.approx(expected)
        assert float(result.metrics["reward/feet_phase"]) == (0. if expected < .5 else 2.)


def test_standing_command_sampler_preserves_moving_samples():
    from source.locomotion.unitree_g1.playground_joystick import standing_command_sampler
    import jax
    import jax.numpy as jp
    import numpy as np
    def original(key):
        a,b,c,z = jax.random.split(key,4)
        cmd = jp.array([jax.random.uniform(a),jax.random.uniform(b),jax.random.uniform(c)])
        return jp.where(jax.random.bernoulli(z,p=0.1),jp.zeros(3),cmd)
    keys = jax.random.split(jax.random.PRNGKey(19),10000)
    baseline = jax.jit(jax.vmap(original))(keys)
    def sample(p):
        return jax.jit(jax.vmap(lambda k: standing_command_sampler(k,original=original,probability=p)))(keys)
    np.testing.assert_array_equal(sample(0.1),baseline)
    increased = np.asarray(sample(0.3)); moving=np.any(increased!=0,axis=1)
    np.testing.assert_array_equal(increased[moving],np.asarray(baseline)[moving])
    assert 0.28 < np.mean(~moving) < 0.32
    assert np.all(increased[np.all(np.asarray(baseline)==0,axis=1)]==0)
    assert np.all(np.asarray(sample(1.0))==0)


def test_real_upstream_standing_sampler_override():
    import numpy as np
    cfg = OmegaConf.structured(G1PlaygroundMJXConfig())
    cfg.playground.fetch_source = False
    cfg.playground.standing_command_probability = 0.3
    env = Joystick("unitree_g1", UnitreeG1Model(fetch=False), "flat", cfg)
    keys = jax.random.split(jax.random.PRNGKey(71),10000)
    sample = jax.jit(jax.vmap(env._env.sample_command))(keys)
    original = jax.jit(jax.vmap(env._env.sample_command.keywords["original"]))(keys)
    moving = np.any(np.asarray(sample)!=0,axis=1)
    assert 0.28 < np.mean(~moving) < 0.32
    np.testing.assert_array_equal(np.asarray(sample)[moving],np.asarray(original)[moving])
    assert env.effective_config["standing_command_probability"] == 0.3
