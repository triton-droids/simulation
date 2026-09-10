"""Regression tests for safety-relevant PPO configuration plumbing."""

from types import SimpleNamespace

import pytest

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
