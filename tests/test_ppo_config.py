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
