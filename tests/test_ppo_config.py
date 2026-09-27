"""Regression tests for safety-relevant PPO configuration plumbing."""

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from source.config.agents import G1PPOCorrectiveConfig, PPOConfig
from source.utils.ppo_config import trainer_runtime_controls


def test_repository_runtime_controls_override_brax_defaults():
    cfg = SimpleNamespace(
        action_repeat=1.0,
        normalize_observations=True,
        max_grad_norm=1.0,
    )

    assert trainer_runtime_controls(cfg) == {
        "action_repeat": 1,
        "normalize_observations": True,
        "max_grad_norm": 1.0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (("action_repeat", 0), ("action_repeat", 1.5), ("max_grad_norm", 0.0)),
)
def test_invalid_runtime_controls_are_rejected(field, value):
    values = {
        "action_repeat": 1,
        "normalize_observations": True,
        "max_grad_norm": 1.0,
    }
    values[field] = value

    with pytest.raises(ValueError):
        trainer_runtime_controls(SimpleNamespace(**values))


def test_corrective_g1_profile_preserves_playground_ppo_semantics():
    cfg = G1PPOCorrectiveConfig()

    assert cfg.unroll_length == 20
    assert cfg.batch_size * cfg.num_minibatches == cfg.num_envs
    assert cfg.num_updates_per_batch == 4
    assert cfg.discounting == 0.97
    assert cfg.learning_rate == 3e-4
    assert cfg.entropy_cost == 0.005
    assert cfg.reward_scaling == 1.0
    assert cfg.policy_hidden_layer_sizes == (512, 256, 128)
    assert cfg.value_hidden_layer_sizes == (512, 256, 128)


def test_brax_pmap_reset_axis_workaround_remains_enabled() -> None:
    assert PPOConfig().use_pmap_on_reset is True
    assert G1PPOCorrectiveConfig().use_pmap_on_reset is True


def test_pytest_runner_is_declared_for_clean_install() -> None:
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    declared = requirements.read_text(encoding="utf-8").splitlines()

    assert "pytest==8.4.2" in declared


def test_corrective_g1_profile_yaml_round_trip_is_lossless():
    structured = OmegaConf.structured(G1PPOCorrectiveConfig())
    serialized = OmegaConf.to_yaml(structured, resolve=True)
    restored = OmegaConf.create(serialized)

    assert OmegaConf.to_container(restored, resolve=True) == OmegaConf.to_container(
        structured, resolve=True
    )

def test_adaptive_kl_runtime_forwarding_and_installed_optimizer():
    import inspect
    import jax
    import jax.numpy as jp
    import optax
    from brax.training.agents.ppo import optimizer, train
    cfg = G1PPOCorrectiveConfig(learning_rate_schedule='ADAPTIVE_KL')
    controls = trainer_runtime_controls(cfg)
    assert all(k in inspect.signature(train.train).parameters for k in controls)
    assert controls['desired_kl'] == .01
    assert controls['learning_rate_schedule_min_lr'] == 1e-5
    assert controls['learning_rate_schedule_max_lr'] == 3e-4
    opt = optax.chain(optax.clip_by_global_norm(1.), optax.inject_hyperparams(optax.adam)(learning_rate=cfg.learning_rate))
    state = opt.init(jp.zeros(2))
    def update(state, kl):
        return optimizer.adaptive_kl_learning_rate(state, kl, controls['desired_kl'], min_learning_rate=controls['learning_rate_schedule_min_lr'], max_learning_rate=controls['learning_rate_schedule_max_lr'])
    step = jax.jit(update)
    for _ in range(30): state, rate = step(state, jp.array(.08))
    assert float(rate) == pytest.approx(1e-5)
    for _ in range(30): state, rate = step(state, jp.array(.001))
    assert float(rate) == pytest.approx(3e-4)
    _, unchanged = step(state, jp.array(.01))
    assert float(unchanged) == pytest.approx(float(rate))


@pytest.mark.parametrize('field,value', [('learning_rate_schedule','invalid'), ('desired_kl',float('nan')), ('learning_rate_schedule_min_lr',.001), ('learning_rate_schedule_max_lr',-1)])
def test_adaptive_kl_invalid_controls(field, value):
    cfg = G1PPOCorrectiveConfig(learning_rate_schedule='ADAPTIVE_KL')
    setattr(cfg, field, value)
    with pytest.raises(ValueError): trainer_runtime_controls(cfg)
